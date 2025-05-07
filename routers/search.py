import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any

from ._semantic import find_assets

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/search", tags=["search"])

class MultiSearchRequest(BaseModel):
    texts: List[str]

@router.post("", status_code=status.HTTP_200_OK)
async def search(req: MultiSearchRequest):
    try:
        results: List[Any] = []
        for text in req.texts:
            txt = text.strip()
            if not txt:
                results.append({"svgs":[], "backgrounds":[], "animations":[]})
                continue
            assets = await find_assets(txt)
            results.append({
                "svgs":        assets["gifs"],
                "backgrounds": assets["backgrounds"],
                "animations":  assets["animations"],
            })
        return JSONResponse({"results": results})
    except Exception:
        logger.exception("💥 /search failed")
        return JSONResponse({"error":"Internal server error"}, status_code=500)
