 
from pathlib import Path
from uuid import uuid4
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse

from database import db
from ._semantic import encode_async

router = APIRouter(prefix="/backgrounds", tags=["backgrounds"])

# === Local storage config ===
ASSETS_ROOT = Path("assets")
BGS_DIR = ASSETS_ROOT / "backgrounds"
BGS_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image content-types → file extension fallback
ALLOWED_IMAGE_TYPES = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/svg+xml": ".svg",
}

def _ext_from_upload(file: UploadFile) -> str:
    """Use filename extension if present, else infer from content-type."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix:
        return suffix
    return ALLOWED_IMAGE_TYPES.get(file.content_type, ".bin")

def _public_url(request: Request, rel_path: str) -> str:
    """Create absolute URL for client consumption (keeps HttpUrl happy)."""
    base = str(request.base_url).rstrip("/")
    rel = rel_path if rel_path.startswith("/") else f"/{rel_path}"
    return f"{base}{rel}"

def _save_upload_locally(file: UploadFile) -> tuple[str, str]:
    """
    Save the uploaded file to assets/backgrounds/<uuid>.<ext>.
    Returns (public_relative_path, local_fs_path).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. "
                   f"Allowed: {', '.join(ALLOWED_IMAGE_TYPES.keys())}",
        )

    ext = _ext_from_upload(file)
    fname = f"{uuid4().hex}{ext}"
    local_path = BGS_DIR / fname

    file.file.seek(0)
    with local_path.open("wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    rel_public = f"/assets/backgrounds/{fname}"
    return rel_public, str(local_path)

def _delete_local_file(local_path: Optional[str]) -> None:
    if not local_path:
        return
    try:
        p = Path(local_path)
        if p.is_file():
            p.unlink()
    except Exception:
        # swallow errors to avoid 500s; optionally log
        pass


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_bg(
    request: Request,
    name: str = Form(..., description="Name of the background"),
    file: UploadFile = File(..., description="Image file to upload (PNG, JPG, WEBP, GIF, SVG)")
):
    # Save locally
    rel_public, local_fs = _save_upload_locally(file)
    background_url = _public_url(request, rel_public)  # absolute URL (HttpUrl-friendly)

    # Embedding for semantic search
    emb = encode_async(name).tolist()

    result = await db["backgrounds"].insert_one({
        "name": name,
        "background_url": background_url,  # public absolute URL for clients
        "background_path": local_fs,       # server-only path for housekeeping
        "embedding": emb
    })

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "Background created successfully",
            "background": {
                "_id": str(result.inserted_id),
                "name": name,
                "background_url": background_url
            }
        }
    )


@router.put("/{bg_id}", status_code=status.HTTP_200_OK)
async def update_bg(
    request: Request,
    bg_id: str,
    name: str = Form(None, description="Optional new name for the background"),
    file: UploadFile = File(None, description="Optional new image file to upload")
):
    if not ObjectId.is_valid(bg_id):
        raise HTTPException(status_code=400, detail="Invalid Background ID")

    existing = await db["backgrounds"].find_one({"_id": ObjectId(bg_id)})
    if not existing:
        raise HTTPException(status_code=404, detail="Background not found")

    update_data = {}

    if name is not None:
        update_data["name"] = name
        update_data["embedding"] = encode_async(name).tolist()

    if file is not None:
        rel_public, local_fs = _save_upload_locally(file)
        update_data["background_url"] = _public_url(request, rel_public)
        update_data["background_path"] = local_fs

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided (name or file required)")

    result = await db["backgrounds"].update_one({"_id": ObjectId(bg_id)}, {"$set": update_data})

    # If we replaced the file, remove the old one from disk
    if result.modified_count and "background_path" in update_data:
        _delete_local_file(existing.get("background_path"))

    updated_bg = await db["backgrounds"].find_one({"_id": ObjectId(bg_id)})

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "Background updated successfully",
            "background": {
                "_id": bg_id,
                "name": updated_bg.get("name"),
                "background_url": updated_bg.get("background_url")
            }
        }
    )


@router.get("/", status_code=status.HTTP_200_OK)
async def list_backgrounds():
    docs = [doc async for doc in db["backgrounds"].find()]
    result = [
        {
            "_id": str(doc["_id"]),
            "name": doc.get("name"),
            "background_url": doc.get("background_url"),  # absolute URL via /assets
        }
        for doc in docs
    ]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": 200, "backgrounds": result}
    )


@router.delete("/{bg_id}", status_code=status.HTTP_200_OK)
async def delete_background(bg_id: str):
    if not ObjectId.is_valid(bg_id):
        raise HTTPException(status_code=400, detail="Invalid Background ID")

    deleted = await db["backgrounds"].find_one_and_delete({"_id": ObjectId(bg_id)})
    if not deleted:
        raise HTTPException(status_code=404, detail="Background not found")

    # Remove file from disk (best-effort)
    _delete_local_file(deleted.get("background_path"))

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "Background deleted successfully",
            "deleted_background": {
                "_id": bg_id,
                "name": deleted.get("name"),
                "background_url": deleted.get("background_url")
            }
        }
    )
