from bson import ObjectId
import cloudinary_config  
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from cloudinary.uploader import upload as cloud_upload, destroy as cloud_destroy
from database import db
from ._semantic import encode

router = APIRouter(prefix="/backgrounds", tags=["backgrounds"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_bg(
    name: str = Form(..., description="Name of the background"),
    file: UploadFile = File(..., description="Image file to upload (PNG, JPG, WEBP, GIF, SVG)")
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
    
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "Background created successfully",
            "background": {
                "_id": str(result.inserted_id),
                "name": name,
                "background_url": res["secure_url"]
            }
        }
    )

@router.put("/{bg_id}", status_code=status.HTTP_200_OK)
async def update_bg(
    bg_id: str,
    name: str = Form(None, description="Optional new name for the background"),
    file: UploadFile = File(None, description="Optional new image file to upload")
):
    # Validate Background ID
    if not ObjectId.is_valid(bg_id):
        raise HTTPException(status_code=400, detail="Invalid Background ID")
    
    update_data = {}
    
    # Check if name is provided
    if name is not None:
        update_data["name"] = name
        update_data["embedding"] = encode(name).tolist()
    
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
        update_data["background_url"] = res["secure_url"]
    
    if not update_data:
        raise HTTPException(
            status_code=400,
            detail="No update data provided (name or file required)"
        )
    
    result = await db["backgrounds"].update_one(
        {"_id": ObjectId(bg_id)},
        {"$set": update_data}
    )
    
    if result.modified_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Background not found or no changes made"
        )
    
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
            "background_url": doc.get("background_url")
        }
        for doc in docs
    ]
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": 200, "backgrounds": result}
    )


@router.delete("/{bg_id}", status_code=status.HTTP_200_OK)
async def delete_background(bg_id: str):
    # Validate Background ID
    if not ObjectId.is_valid(bg_id):
        raise HTTPException(status_code=400, detail="Invalid Background ID")
    
    # Find and delete the background
    result = await db["backgrounds"].find_one_and_delete({"_id": ObjectId(bg_id)})
    
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Background not found"
        )
    
 
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": 200,
            "message": "Background deleted successfully",
            "deleted_background": {
                "_id": bg_id,
                "name": result.get("name"),
                "background_url": result.get("background_url")
            }
        }
    )