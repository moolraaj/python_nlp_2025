# routers/search.py
 
import logging
 
from fastapi import APIRouter, status
 
from fastapi.responses import JSONResponse
 
from pydantic import BaseModel
 
from typing import List, Any
 
from datetime import datetime
 
import asyncio
 
from ._semantic import find_assets
 
from .generate_audio import generate_tts_audio, clear_audio_folder
 
from database import db
 
logger = logging.getLogger(__name__)
 
router = APIRouter(prefix="/search", tags=["search"])
 
class MultiSearchRequest(BaseModel):
 
    texts: List[str]
 
@router.post("", status_code=status.HTTP_200_OK)
 
async def search(req: MultiSearchRequest):
 
    try:
 
        logger.info(f"📨 Received search request with {len(req.texts)} texts")
 
        clear_audio_folder()
 
        results: List[Any] = []
 
        seen_texts = set()
 
        for i, text in enumerate(req.texts):
 
            txt = text.strip()
 
            logger.info(f"🔄 Processing {i+1}/{len(req.texts)}: '{txt}'")
 
            if not txt:
 
                results.append({
 
                    "text": [],
 
                    "tts_audio_url": [],  # ✅ Return empty array
 
                    "svgs": [],
 
                    "backgrounds": [],
 
                    "animations": []
 
                })
 
                continue
 
            try:
 
                assets = await asyncio.wait_for(find_assets(txt), timeout=45.0)
 
                # Generate TTS with timeout protection
 
                tts_audio_urls = []  # ✅ Initialize as array
 
                if txt not in seen_texts:
 
                    try:
 
                        # Run TTS in thread with timeout
 
                        loop = asyncio.get_event_loop()
 
                        audio_url = await asyncio.wait_for(
 
                            loop.run_in_executor(None, generate_tts_audio, txt),
 
                            timeout=10.0
 
                        )
 
                        tts_audio_urls = [audio_url]  # ✅ Wrap in array
 
                        seen_texts.add(txt)
 
                    except asyncio.TimeoutError:
 
                        logger.error(f"⏰ TTS timeout for: '{txt}'")
 
                        tts_audio_urls = []  # ✅ Empty array on timeout
 
                    except Exception as e:
 
                        logger.error(f"❌ TTS failed for '{txt}': {e}")
 
                        tts_audio_urls = []  # ✅ Empty array on error
 
                else:
 
                    from hashlib import md5
 
                    audio_url = f"/static/audio/{md5(txt.encode()).hexdigest()}.mp3"
 
                    tts_audio_urls = [audio_url]  # ✅ Wrap in array
 
                results.append({
 
                    "text": [txt],
 
                    "tts_audio_url": tts_audio_urls,  # ✅ Now it's an array
 
                    "svgs": assets["gifs"],
 
                    "backgrounds": assets["backgrounds"],
 
                    "animations": assets["animations"],
 
                })
 
                logger.info(f"✅ Processed text {i+1} successfully")
 
            except asyncio.TimeoutError:
 
                logger.error(f"⏰ Search timeout for: '{txt}'")
 
                results.append({
 
                    "text": [txt],
 
                    "tts_audio_url": [],  # ✅ Empty array on timeout
 
                    "svgs": [],
 
                    "backgrounds": [],
 
                    "animations": []
 
                })
 
            except Exception as e:
 
                logger.error(f"❌ Error processing '{txt}': {e}")
 
                results.append({
 
                    "text": [txt],
 
                    "tts_audio_url": [],  # ✅ Empty array on error
 
                    "svgs": [],
 
                    "backgrounds": [],
 
                    "animations": []
 
                })
 
        search_doc = {
 
            "texts": req.texts,
 
            "results": results,
 
            "created_at": datetime.utcnow(),
 
            "updated_at": datetime.utcnow()
 
        }
 
        await db.searches.insert_one(search_doc)
 
        return JSONResponse({
 
            "texts": req.texts,
 
            "results": results,
 
            "message": f"Processed {len(results)} search requests"
 
        })
 
    except Exception as e:
 
        logger.exception("💥 Search endpoint failed completely")
 
        return JSONResponse(
 
            {"error": "Internal server error", "details": str(e)},
 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
 
        )