from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from embeddings import find_assets, suggest_random
from gtts import gTTS
from gtts.tts import gTTSError
import uvicorn
import os

app = FastAPI()

 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:4500",
        "https://editor-2025-part-2.vercel.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

 
class SearchRequest(BaseModel):
    text: str

class TTSRequest(BaseModel):
    text: str

class MultiSearchRequest(BaseModel):
    texts: list[str]

 
@app.post("/search")
async def search(req: MultiSearchRequest):
    results = []
    for text in req.texts:
        if not text.strip():
            results.append({'error': 'Text is empty'})
            continue
        assets = find_assets(text)
        results.append(assets)
    return results

@app.get("/suggest", tags=["feedback"])
async def suggest():
    text, assets = suggest_random()
    return {"suggestion": text, "assets": assets}

@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is healthy"}

@app.post("/speak", response_class=StreamingResponse)
async def speak(req: TTSRequest):
    """
    Generates spoken audio for the given text using Google TTS and returns an MP3 stream.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be non-empty")

    mp3_fp = BytesIO()
    try:
        tts = gTTS(text=req.text, lang="en", tld="co.uk", slow=False)
        tts.write_to_fp(mp3_fp)
    except gTTSError:
       
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable (rate‑limit hit). Please try again later."
        )

    mp3_fp.seek(0)
    return StreamingResponse(mp3_fp, media_type="audio/mpeg")

 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
