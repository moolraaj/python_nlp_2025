import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from database import db
from routers.backgrounds import router as backgrounds_router
from routers.svgs        import router as svgs_router
from routers.types       import router as types_router
from routers.search      import router as search_router
from routers.media       import router as media_router
from routers._semantic   import encode  

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

 
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://editor-2025-part-2.vercel.app",
        "http://localhost:4500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)

 

async def _encode_async(text: str) -> list[float]:
    """
    Run the sync encode() in a thread so we don't block the event loop.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: encode(text).tolist())

async def backfill_embeddings():
 
    async for doc in db["backgrounds"].find({"embedding": {"$exists": False}}):
        name = doc.get("name", "")
        emb = await _encode_async(name.lower())
        await db["backgrounds"].update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb}})

  
    async for doc in db["svgs"].find({"embedding": {"$exists": False}}):
        tags = [str(tag).lower() for tag in doc.get("tags", [])]
        emb = await _encode_async(" ".join(tags))
        await db["svgs"].update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb}})

    
    async for doc in db["types"].find({"embedding": {"$exists": False}}):
        name = doc.get("name", "")
        emb = await _encode_async(name.lower())
        await db["types"].update_one({"_id": doc["_id"]}, {"$set": {"embedding": emb}})

 


@app.on_event("startup")
async def startup():
 
    try:
        await db.client.admin.command("ping")
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error("❌ MongoDB connection failed: %s", e)

 
    need_bg  = await db["backgrounds"].count_documents({"embedding": {"$exists": False}}, limit=1)
    need_svg = await db["svgs"].count_documents({"embedding": {"$exists": False}}, limit=1)
    need_ty  = await db["types"].count_documents({"embedding": {"$exists": False}}, limit=1)

    if need_bg or need_svg or need_ty:
     
        asyncio.create_task(backfill_embeddings())
        logger.info("🔄 Embedding backfill scheduled in background")
    else:
        logger.info("🟢 Embeddings already present. Skipping backfill.")


 
app.include_router(backgrounds_router)
app.include_router(svgs_router)
app.include_router(types_router)
app.include_router(search_router)
app.include_router(media_router)

 
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
else:
    logger.warning("⚠️ 'static/' directory not found. Skipping mount.")


@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is healthy"}
