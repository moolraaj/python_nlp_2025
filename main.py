import logging
import nltk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import db
from routers.backgrounds import router as backgrounds_router
from routers.svgs        import router as svgs_router
from routers.types       import router as types_router
from routers.search      import router as search_router
from routers.media       import router as media_router
from routers._semantic   import backfill_embeddings, prep_nltk  # new imports

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# your existing CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    # 1) download your NLTK data once at startup
    prep_nltk()

    # 2) make sure MongoDB is alive
    try:
        await db.client.admin.command("ping")
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.error("❌ MongoDB connection failed: %s", e)

    # 3) backfill any missing embeddings
    await backfill_embeddings()
    logger.info("🔄 Embedding backfill done")

# include all your routers exactly as before
app.include_router(backgrounds_router)
app.include_router(svgs_router)
app.include_router(types_router)
app.include_router(search_router)
app.include_router(media_router)

@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is healthy"}
