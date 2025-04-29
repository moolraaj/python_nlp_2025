from pydantic import BaseModel, Field, HttpUrl
from typing import List
from bson import ObjectId

class DocumentInDB(BaseModel):
    id: str = Field(..., alias="_id")

    class Config:
        allow_population_by_field_name = True
        orm_mode = True
        json_encoders = {ObjectId: str}

#  Backgrounds 
class BackgroundCreate(BaseModel):
    name: str

class BackgroundInDB(BackgroundCreate, DocumentInDB):
    background_url: HttpUrl

# SVGs  
class SVGCreate(BaseModel):
    tags: List[str]

class SVGInDB(SVGCreate, DocumentInDB):
    svg_url: HttpUrl

#   Types 
class TypeCreate(BaseModel):
    name: str

class TypeInDB(TypeCreate, DocumentInDB):
    pass