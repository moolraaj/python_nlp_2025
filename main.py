
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from embeddings import find_assets, suggest_random
import uvicorn
import os
import tempfile
from pathlib import Path
import pyttsx3

 
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:4500",
    "https://editor-2025-part-2.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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
            results.append({"error": "Text is empty"})
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

 
engine = pyttsx3.init()
 
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

 
@app.post("/speak", response_class=StreamingResponse)
async def speak(req: TTSRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="`text` must be non-empty")

  
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

 
    engine.save_to_file(text, str(tmp_path))
    engine.runAndWait()

  
    def iterfile():
        with tmp_path.open("rb") as f:
            yield from f
        tmp_path.unlink()

    return StreamingResponse(iterfile(), media_type="audio/wav")


 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
