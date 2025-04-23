from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from embeddings import find_assets,suggest_random
from gtts import gTTS
from gtts.tts import gTTSError
import uvicorn
import zipfile
from io import BytesIO
from fastapi.responses import StreamingResponse,JSONResponse
import os
import requests
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import re
from urllib.parse import urlparse

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
    results: list[dict] = []
    suggestions: dict[int, dict] = {}

    for idx, text in enumerate(req.texts):
        if not text.strip():
            results.append({"error": "Text is empty"})
            s_text, s_assets = suggest_random()
            suggestions[idx] = {"suggestion": s_text, "assets": s_assets}
            continue

        assets = find_assets(text)
        results.append(assets)

      
        if not (assets["animations"] or assets["backgrounds"] or assets["gifs"]):
            s_text, s_assets = suggest_random()
            suggestions[idx] = {"suggestion": s_text, "assets": s_assets}

    response = {"results": results}
    if suggestions:
        response["suggestions"] = suggestions
    return JSONResponse(response)






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




'download images'

@app.post("/download-all-images", response_class=StreamingResponse)
async def download_all_images(req: MultiSearchRequest):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for scene_idx, text in enumerate(req.texts, start=1):
            assets = find_assets(text)
            urls_and_labels = [
                (g["gif_url"], (g.get("tags") or [g["id"]])[0])
                for g in assets["gifs"]
            ] + [
                (b["background_url"], b["name"])
                for b in assets["backgrounds"]
            ]

            for img_idx, (url, label) in enumerate(urls_and_labels, start=1):
                try:
                    resp = requests.get(url, timeout=5)
                    resp.raise_for_status()
                    parsed = urlparse(url)
                    ext = os.path.splitext(parsed.path)[1] or ".jpg"
                    safe_label = re.sub(r'[^A-Za-z0-9_-]+', '_', label)
                    zip_path = f"scene_{scene_idx}/img{img_idx}_{safe_label}{ext}"
                    zf.writestr(zip_path, resp.content)
                except Exception:
                    continue

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/x-zip-compressed",
        headers={
            "Content-Disposition": "attachment; filename=all_images.zip"
        }
    )






'download pdf'

@app.post("/download-scenes-pdf", response_class=StreamingResponse)
async def download_scenes_pdf(req: MultiSearchRequest):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    total_scenes = len(req.texts)
    for idx, text in enumerate(req.texts, start=1):
      
        y_pos = height - 50

      
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, y_pos, f"Scene {idx}: {text}")
        y_pos -= 30

      
        assets = find_assets(text)
        urls_and_labels = [
            (g["gif_url"], (g.get("tags") or [g["id"]])[0])
            for g in assets["gifs"]
        ] + [
            (b["background_url"], b["name"])
            for b in assets["backgrounds"]
        ]

        
        for url, label in urls_and_labels:
            try:
                
                c.setFont("Helvetica", 12)
                c.drawString(30, y_pos, label)
                y_pos -= 18

             
                resp = requests.get(url, timeout=5)
                resp.raise_for_status()
                img = ImageReader(BytesIO(resp.content))
                img_w, img_h = img.getSize()

              
                max_w, max_h = width - 60, 150
                scale = min(max_w / img_w, max_h / img_h, 1)
                w, h = img_w * scale, img_h * scale

                c.drawImage(img, 30, y_pos - h, width=w, height=h)
                y_pos -= (h + 20)

            except Exception:
           
                continue

    
        if idx < total_scenes:
            c.showPage()

   
    c.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=scenes.pdf"}
    )



 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
