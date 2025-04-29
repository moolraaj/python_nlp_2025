# routers/backgrounds.py
import cloudinary_config  
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from cloudinary.uploader import upload as cloud_upload
from database import db
from ._semantic import encode

router = APIRouter(prefix="/backgrounds", tags=["backgrounds"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_bg(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")
    file.file.seek(0)
    res = cloud_upload(
        file.file,
        resource_type="image",
        folder="o_h_app",
        overwrite=True
    )
    
    emb = encode(name).tolist()

    result = await db["backgrounds"].insert_one({
        "name": name,
        "background_url": res["secure_url"],
        "embedding": emb
    })
    bg = {
        "_id": str(result.inserted_id),
        "name": name,
        "background_url": res["secure_url"]
    }
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "Background created successfully",
            "background": bg
        }
    )
