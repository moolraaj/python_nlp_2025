import logging
import asyncio

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

def create_app() -> FastAPI:
    app = FastAPI()

    # ─── 1️⃣ CORS MIDDLEWARE ────────────────────────────────
    # Must come immediately after FastAPI() and before include_router()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # or ["https://editor-2025-part-2.vercel.app"]
        allow_methods=["*"],      # GET, POST, OPTIONS, etc.
        allow_headers=["*"],      # Content-Type, Authorization, etc.
        # allow_credentials=True, # only if you use cookies/auth
    )

    @app.on_event("startup")
    async def startup():
        # check MongoDB
        try:
            await db.client.admin.command("ping")
            logger.info("✅ MongoDB connected")
        except Exception as e:
            logger.error("❌ MongoDB connection failed: %s", e)

        # backfill embeddings for any missing docs
        async def backfill_embeddings():
            async for doc in db["backgrounds"].find({"embedding": {"$exists": False}}):
                emb = encode(doc.get("name", "").lower()).tolist()
                await db["backgrounds"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": emb}},
                )

            async for doc in db["svgs"].find({"embedding": {"$exists": False}}):
                tags_text = " ".join(str(tag).lower() for tag in doc.get("tags", []))
                emb = encode(tags_text).tolist()
                await db["svgs"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": emb}},
                )

            async for doc in db["types"].find({"embedding": {"$exists": False}}):
                emb = encode(doc.get("name", "").lower()).tolist()
                await db["types"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": emb}},
                )

            logger.info("🔄 Embedding backfill done")

        await backfill_embeddings()

    # ─── 2️⃣ ROUTERS ────────────────────────────────────────
    app.include_router(backgrounds_router)
    app.include_router(svgs_router)
    app.include_router(types_router)
    app.include_router(search_router)
    app.include_router(media_router)

    @app.get("/", tags=["health"])
    async def health_check():
        return {"status": "ok", "message": "API is healthy"}

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
