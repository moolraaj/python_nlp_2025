# routers/search.py

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any

from ._semantic import find_assets

class MultiSearchRequest(BaseModel):
    texts: List[str]

router = APIRouter(prefix="/search", tags=["search"])

@router.post("", status_code=status.HTTP_200_OK)
async def search(req: MultiSearchRequest):
    results: List[Any] = []

    for text in req.texts:
        txt = text.strip()
        if not txt:
            results.append({
                "svgs":        [],
                "backgrounds": [],
                "animations":  [],
            })
            continue

        assets = await find_assets(txt)
        results.append({
            "svgs":        assets["gifs"],
            "backgrounds": assets["backgrounds"],
            "animations":  assets["animations"],
        })

    return JSONResponse({"results": results})

