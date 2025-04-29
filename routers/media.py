# routers/media.py

from io import BytesIO
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from gtts import gTTS
from gtts.tts import gTTSError
import zipfile, os, requests, re
from urllib.parse import urlparse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

from .search import find_assets

router = APIRouter(prefix="", tags=["media"])

class TTSRequest(BaseModel):
    text: str

class MultiSearchRequest(BaseModel):
    texts: list[str]

@router.post("/speak", response_class=StreamingResponse)
async def speak(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be non-empty")

    mp3_fp = BytesIO()
    try:
        tts = gTTS(text=req.text, lang="en", tld="co.uk", slow=False)
        tts.write_to_fp(mp3_fp)
    except gTTSError:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable"
        )
    mp3_fp.seek(0)
    return StreamingResponse(mp3_fp, media_type="audio/mpeg")


@router.post("/download-all-images", response_class=StreamingResponse)
async def download_all_images(req: MultiSearchRequest):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_idx, text in enumerate(req.texts, start=1):
            assets = await find_assets(text)
         
            entries = [
                (g["svg_url"], g["tags"][0] if g["tags"] else "gif")
                for g in assets["gifs"]
            ] + [
                (b["background_url"], b["name"])
                for b in assets["backgrounds"]
            ]
            for img_idx, (url, label) in enumerate(entries, start=1):
                try:
                    resp = requests.get(url, timeout=5); resp.raise_for_status()
                    ext = os.path.splitext(urlparse(url).path)[1] or ".jpg"
                    safe = re.sub(r'[^A-Za-z0-9_-]+', "_", label)
                    path = f"scene_{scene_idx}/img{img_idx}_{safe}{ext}"
                    zf.writestr(path, resp.content)
                except:
                    continue

    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition":"attachment; filename=all_images.zip"}
    )


@router.post("/download-scenes-pdf", response_class=StreamingResponse)
async def download_scenes_pdf(req: MultiSearchRequest):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4; total = len(req.texts)

    for idx, text in enumerate(req.texts, start=1):
        y = h - 50
        c.setFont("Helvetica-Bold", 14)
        c.drawString(30, y, f"Scene {idx}: {text}")
        y -= 30

        assets = await find_assets(text)
        entries = [
            (g["svg_url"], g["tags"][0] if g["tags"] else "gif")
            for g in assets["gifs"]
        ] + [
            (b["background_url"], b["name"])
            for b in assets["backgrounds"]
        ]

        for url, label in entries:
            try:
                c.setFont("Helvetica", 12)
                c.drawString(30, y, label); y -= 18
                resp = requests.get(url, timeout=5); resp.raise_for_status()
                img = ImageReader(BytesIO(resp.content))
                iw, ih = img.getSize()
                max_w, max_h = w - 60, 150
                scale = min(max_w/iw, max_h/ih, 1)
                c.drawImage(img, 30, y-ih*scale, width=iw*scale, height=ih*scale)
                y -= ih*scale + 20
            except:
                continue

        if idx < total:
            c.showPage()

    c.save()
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition":"attachment; filename=scenes.pdf"}
    )
