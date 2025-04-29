# routers/search.py

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

from ._semantic import find_assets

class MultiSearchRequest(BaseModel):
    texts: List[str]

router = APIRouter(prefix="/search", tags=["search"])

@router.post("", status_code=status.HTTP_200_OK)
async def search(req: MultiSearchRequest):
    results = []
    for text in req.texts:
        txt = text.strip()
        if not txt:
            continue

        assets = await find_assets(txt)
        # take only the first item (or None) from each list
        results.append({
            "svg_url":    assets["gifs"][0]["svg_url"]         if assets["gifs"]         else None,
            "background": assets["backgrounds"][0]["background_url"] if assets["backgrounds"] else None,
            "types":      assets["animations"][0]["name"]      if assets["animations"]    else None,
        })

    return JSONResponse({"results": results})
