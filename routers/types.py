 
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from database import db
from models import TypeCreate
from ._semantic import encode_async

router = APIRouter(prefix="/types", tags=["types"])

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_type(data: TypeCreate):
    emb = encode_async(data.name).tolist()
    result = await db["types"].insert_one({"name": data.name, "embedding": emb})
    type_doc = {"_id": str(result.inserted_id), "name": data.name}
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "status": 201,
            "message": "Type created successfully",
            "type": type_doc
        }
    )
