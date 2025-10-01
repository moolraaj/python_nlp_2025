import json
from pathlib import Path
from uuid import uuid4
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse

from database import db
from ._semantic import encode_async

router = APIRouter(prefix="/svgs", tags=["svgs"])

# === Local storage config ===
ASSETS_ROOT = Path("assets")
SVGS_DIR = ASSETS_ROOT / "svgs"
SVGS_DIR.mkdir(parents=True, exist_ok=True)

# Allowed content types → file extension fallback
ALLOWED_IMAGE_TYPES = {
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

def _ext_from_upload(file: UploadFile) -> str:
    """Pick extension from filename if present, otherwise from content-type."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix:
        return suffix
    return ALLOWED_IMAGE_TYPES.get(file.content_type, ".bin")

def _public_url(request: Request, rel_path: str) -> str:
    """Build absolute URL (keeps your HttpUrl model working)."""
    # request.base_url already includes a trailing slash
    base = str(request.base_url).rstrip("/")
    rel = rel_path if rel_path.startswith("/") else f"/{rel_path}"
    return f"{base}{rel}"

def _save_upload_locally(file: UploadFile) -> tuple[str, str]:
    """
    Save the uploaded file to assets/svgs/<uuid>.<ext>.
    Returns (public_relative_path, local_fs_path).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        # You can relax this if you want any image/*, but this keeps things explicit.
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{file.content_type}'. Allowed: {', '.join(ALLOWED_IMAGE_TYPES.keys())}",
        )

    ext = _ext_from_upload(file)
    fname = f"{uuid4().hex}{ext}"
    local_path = SVGS_DIR / fname

    # stream to disk
    file.file.seek(0)
    with local_path.open("wb") as out:
        # chunked copy (avoid reading entire file into memory)
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    rel_public = f"/assets/svgs/{fname}"
    return rel_public, str(local_path)

def _delete_local_file(local_path: Optional[str]) -> None:
    if not local_path:
        return
    try:
        p = Path(local_path)
        if p.is_file():
            p.unlink()
    except Exception:
        # swallow to avoid 500s on missing files; log if you have a logger
        pass


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_svg(
    request: Request,
    tags: str = Form(..., description="JSON array of tags as strings, e.g. ['boy','nurse']"),
    file: UploadFile = File(..., description="Image file to upload (SVG, PNG, JPG, WEBP, GIF)"),
):
    # Validate tags
    try:
        tags_list = json.loads(tags)
        if not isinstance(tags_list, list) or not all(isinstance(t, str) for t in tags_list):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="`tags` must be a JSON array of strings, e.g. ['boy','nurse']",
        )

    # Save locally
    rel_public, local_fs = _save_upload_locally(file)
    svg_url = _public_url(request, rel_public)  # absolute URL (HttpUrl)

    # Embedding
    tags_text = " ".join(tags_list)
    emb = encode_async(tags_text).tolist()

    # Store both public URL and local path (local path helps on delete/update)
    result = await db["svgs"].insert_one(
        {
            "tags": tags_list,
            "svg_url": svg_url,     # public, absolute URL for clients (and your HttpUrl model)
            "svg_path": local_fs,   # internal server-side path for housekeeping
            "embedding": emb,
        }
    )

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "SVG created successfully",
            "svg": {
                "_id": str(result.inserted_id),
                "tags": tags_list,
                "svg_url": svg_url,
            },
        },
    )


@router.put("/{svg_id}", status_code=status.HTTP_200_OK)
async def update_svg(
    request: Request,
    svg_id: str,
    tags: str = Form(None, description="Optional JSON array of tags as strings"),
    file: UploadFile = File(None, description="Optional new image file to upload"),
):
    if not ObjectId.is_valid(svg_id):
        raise HTTPException(status_code=400, detail="Invalid SVG ID")

    # Load existing to cleanup old file if replaced
    existing = await db["svgs"].find_one({"_id": ObjectId(svg_id)})
    if not existing:
        raise HTTPException(status_code=404, detail="SVG not found")

    update_data = {}

    # Tags update
    if tags is not None:
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list) or not all(isinstance(t, str) for t in tags_list):
                raise ValueError()
            update_data["tags"] = tags_list
            tags_text = " ".join(tags_list)
            update_data["embedding"] = encode_async(tags_text).tolist()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="`tags` must be a JSON array of strings, e.g. ['boy','nurse']",
            )

    # File update
    if file is not None:
        rel_public, local_fs = _save_upload_locally(file)
        update_data["svg_url"] = _public_url(request, rel_public)
        update_data["svg_path"] = local_fs

    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided (tags or file required)")

    result = await db["svgs"].update_one({"_id": ObjectId(svg_id)}, {"$set": update_data})

    if result.modified_count == 0:
        # nothing changed
        updated = await db["svgs"].find_one({"_id": ObjectId(svg_id)})
    else:
        updated = await db["svgs"].find_one({"_id": ObjectId(svg_id)})

        # If we changed the file, delete the old one
        if "svg_path" in update_data:
            _delete_local_file(existing.get("svg_path"))

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "SVG updated successfully",
            "svg": {
                "_id": svg_id,
                "tags": updated.get("tags"),
                "svg_url": updated.get("svg_url"),
            },
        },
    )


@router.get("/", status_code=status.HTTP_200_OK)
async def list_svgs():
    docs = [doc async for doc in db["svgs"].find()]
    result = [
        {
            "_id": str(doc["_id"]),
            "tags": doc.get("tags", []),
            "svg_url": doc.get("svg_url"),   
        }
        for doc in docs
    ]
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": 200, "svgs": result})


@router.delete("/{svg_id}", status_code=status.HTTP_200_OK)
async def delete_svg(svg_id: str):
    if not ObjectId.is_valid(svg_id):
        raise HTTPException(status_code=400, detail="Invalid SVG ID")

    # Find and delete DB
    deleted = await db["svgs"].find_one_and_delete({"_id": ObjectId(svg_id)})
    if not deleted:
        raise HTTPException(status_code=404, detail="SVG not found")

    # Remove local file if present
    _delete_local_file(deleted.get("svg_path"))

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "SVG deleted successfully",
            "deleted_svg": {
                "_id": svg_id,
                "tags": deleted.get("tags"),
                "svg_url": deleted.get("svg_url"),
            },
        },
    )
