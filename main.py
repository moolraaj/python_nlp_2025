
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

async def backfill_embeddings():
    async for doc in db["backgrounds"].find({"embedding": {"$exists": False}}):
        name = doc.get("name", "")
        emb = encode(name.lower()).tolist()
        await db["backgrounds"].update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": emb}}
        )
    async for doc in db["svgs"].find({"embedding": {"$exists": False}}):
        tags = [str(tag).lower() for tag in doc.get("tags", [])]
        emb = encode(" ".join(tags)).tolist()
        await db["svgs"].update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": emb}}
        )
    async for doc in db["types"].find({"embedding": {"$exists": False}}):
        name = doc.get("name", "")
        emb = encode(name.lower()).tolist()
        await db["types"].update_one(
            {"_id": doc["_id"]},
            {"$set": {"embedding": emb}}
        )

@app.on_event("startup")
async def startup():
    try:
        await db.client.admin.command("ping")
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error("❌ MongoDB connection failed: %s", e)

    await backfill_embeddings()
    logger.info("🔄 Embedding backfill done")

app.include_router(backgrounds_router)
app.include_router(svgs_router)
app.include_router(types_router)
app.include_router(search_router)
app.include_router(media_router)

@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is healthy"}
