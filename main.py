# main.py
 
import os
import asyncio
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from datetime import datetime
 
from database import db
from routers.backgrounds import router as backgrounds_router
from routers.svgs import router as svgs_router
from routers.types import router as types_router
from routers.search import router as search_router
from routers.media import router as media_router
 
logger = logging.getLogger("uvicorn.error")
 
app = FastAPI(title="Search API", version="1.0.0")
 
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://editor-2025-part-2.vercel.app",
        "http://localhost:4500",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:4500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)
 
@app.on_event("startup")
async def startup():
    """Initialize application"""
    try:
        # Test MongoDB connection
        await db.client.admin.command("ping")
        logger.info("✅ MongoDB connected successfully")
        logger.info("🟢 Search API ready with thread pooling")
        
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
 
# Include routers
app.include_router(backgrounds_router)
app.include_router(svgs_router)
app.include_router(types_router)
app.include_router(search_router)
app.include_router(media_router)
 
# Static files
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")
    logger.info("✅ Static files mounted at /static")
else:
    logger.warning("⚠️ 'static/' directory not found")
 
# Create assets directories
os.makedirs("assets/svgs", exist_ok=True)
os.makedirs("assets/backgrounds", exist_ok=True)
os.makedirs("static/audio", exist_ok=True)
 
if os.path.isdir("assets"):
    app.mount("/assets", StaticFiles(directory="assets"), name="assets")
    logger.info("✅ Assets mounted at /assets")
 
@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "API is healthy"}
 
@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check"""
    try:
        await db.client.admin.command("ping")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {e}"
 
    return {
        "status": "ok",
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )