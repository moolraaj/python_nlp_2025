
import json
import cloudinary_config
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from cloudinary.uploader import upload as cloud_upload
from database import db
from ._semantic import encode

router = APIRouter(prefix="/svgs", tags=["svgs"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_svg(
    tags: str = Form(..., description="JSON array of tags as strings, e.g. ['boy','nurse']"),
    file: UploadFile = File(..., description="SVG file to upload")
):
    if file.content_type != "image/svg+xml":
        raise HTTPException(status_code=400, detail="Only SVG files accepted.")
    
    try:
        tags_list = json.loads(tags)
        if not isinstance(tags_list, list) or not all(isinstance(t, str) for t in tags_list):
            raise ValueError()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="`tags` must be a JSON array of strings, e.g. ['boy','nurse']"
        )

    file.file.seek(0)
    res = cloud_upload(
        file.file,
        resource_type="image",
        folder="o_h_app",
        overwrite=True
    )
  
    tags_text = " ".join(tags_list)
    emb = encode(tags_text).tolist()

    result = await db["svgs"].insert_one({
        "tags": tags_list,
        "svg_url": res["secure_url"],
        "embedding": emb
    })
    
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "SVG created successfully",
            "svg": {
                "_id": str(result.inserted_id),
                "tags": tags_list,
                "svg_url": res["secure_url"]
            }
        }
    )


@router.get("/", status_code=status.HTTP_200_OK)
async def list_svgs():
    docs = [doc async for doc in db["svgs"].find()]
    result = [
        {
            "_id": str(doc["_id"]),
            "tags": doc.get("tags", []),
            "svg_url": doc.get("svg_url")
        }
        for doc in docs
    ]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": 200, "svgs": result}
    )