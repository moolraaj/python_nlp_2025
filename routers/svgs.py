import json
from bson import ObjectId
import cloudinary_config
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from cloudinary.uploader import upload as cloud_upload, destroy as cloud_destroy
from database import db
from ._semantic import encode

router = APIRouter(prefix="/svgs", tags=["svgs"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_svg(
    tags: str = Form(..., description="JSON array of tags as strings, e.g. ['boy','nurse']"),
    file: UploadFile = File(..., description="Image file to upload (SVG, PNG, JPG, WEBP, GIF)")
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files accepted.")
    
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

@router.put("/{svg_id}", status_code=status.HTTP_200_OK)
async def update_svg(
    svg_id: str,
    tags: str = Form(None, description="Optional JSON array of tags as strings"),
    file: UploadFile = File(None, description="Optional new image file to upload")
):
    # Validate SVG ID
    if not ObjectId.is_valid(svg_id):
        raise HTTPException(status_code=400, detail="Invalid SVG ID")
    
    update_data = {}
    
    # Check if tags are provided
    if tags is not None:
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list) or not all(isinstance(t, str) for t in tags_list):
                raise ValueError()
            update_data["tags"] = tags_list
            tags_text = " ".join(tags_list)
            update_data["embedding"] = encode(tags_text).tolist()
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="`tags` must be a JSON array of strings, e.g. ['boy','nurse']"
            )
    
    # Check if file is provided
    if file is not None:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image files accepted.")
        
        file.file.seek(0)
        res = cloud_upload(
            file.file,
            resource_type="image",
            folder="o_h_app",
            overwrite=True
        )
        update_data["svg_url"] = res["secure_url"]
    
    if not update_data:
        raise HTTPException(
            status_code=400,
            detail="No update data provided (tags or file required)"
        )
    
    result = await db["svgs"].update_one(
        {"_id": ObjectId(svg_id)},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=404,
            detail="SVG not found or no changes made"
        )
    
    updated_svg = await db["svgs"].find_one({"_id": ObjectId(svg_id)})
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "SVG updated successfully",
            "svg": {
                "_id": svg_id,
                "tags": updated_svg.get("tags"),
                "svg_url": updated_svg.get("svg_url")
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


@router.delete("/{svg_id}", status_code=status.HTTP_200_OK)
async def delete_svg(svg_id: str):
    # Validate SVG ID
    if not ObjectId.is_valid(svg_id):
        raise HTTPException(status_code=400, detail="Invalid SVG ID")
    
    # Find and delete the SVG
    result = await db["svgs"].find_one_and_delete({"_id": ObjectId(svg_id)})
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail="SVG not found"
        )
    
 
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "SVG deleted successfully",
            "deleted_svg": {
                "_id": svg_id,
                "tags": result.get("tags"),
                "svg_url": result.get("svg_url")
            }
        }
    )