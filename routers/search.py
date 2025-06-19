import logging
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Any
from datetime import datetime

from ._semantic import find_assets
from .generate_audio import generate_tts_audio, clear_audio_folder
from database import db

logger = logging.getLogger("uvicorn.error")
router = APIRouter(prefix="/search", tags=["search"])

class SearchDocument(BaseModel):
    texts: List[str]
    results: List[dict]
    created_at: datetime
    updated_at: datetime

class MultiSearchRequest(BaseModel):
    texts: List[str]

@router.post("", status_code=status.HTTP_200_OK)
async def search(req: MultiSearchRequest):
    try:
        # 🔄 Clear previous search records (optional)
        await db.searches.delete_many({})

        # 🧹 Step 1: Clear old audio files
        clear_audio_folder()

        results: List[Any] = []
        seen_texts = set()

        for text in req.texts:
            txt = text.strip()

            if not txt:
                results.append({
                    "text": [],
                    "tts_audio_url": None,
                    "svgs": [],
                    "backgrounds": [],
                    "animations": []
                })
                continue

            # 🔍 Find semantic assets
            assets = await find_assets(txt)

            # 🔊 Step 2: Generate TTS only once per unique text
            if txt not in seen_texts:
                tts_audio_urls = [generate_tts_audio(txt)]
                seen_texts.add(txt)
            else:
                from hashlib import md5
                tts_audio_urls = f"/static/audio/{md5(txt.encode()).hexdigest()}.mp3"

            # 🧾 Compose response
            results.append({
                "text": [txt],
                "tts_audio_url": tts_audio_urls,
                "svgs": assets["gifs"],
                "backgrounds": assets["backgrounds"],
                "animations": assets["animations"],
            })

        # 💾 Save in database
        search_doc = {
            "texts": req.texts,
            "results": results,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        await db.searches.insert_one(search_doc)

        return JSONResponse({"texts": req.texts, "results": results})

    except Exception:
        logger.exception("💥 /search failed")
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
