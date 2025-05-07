import asyncio
import logging
from sentence_transformers import SentenceTransformer
from database import db

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backfill")

# Model setup
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

async def backfill():
    # Define collections and text extractor
    tasks = [
        ("backgrounds", lambda d: d.get("name","")),
        ("svgs",        lambda d: " ".join(d.get("tags",[]))),
        ("types",       lambda d: d.get("name","")),
    ]
    for coll, fn in tasks:
        log.info(f"→ Backfilling {coll}")
        async for doc in db[coll].find({"embedding": {"$exists": False}}):
            text = fn(doc).lower()
            emb = model.encode([text], convert_to_numpy=True)[0].tolist()
            await db[coll].update_one({"_id":doc["_id"]},{"$set":{"embedding":emb}})
    log.info("✓ Backfill complete")

if __name__ == "__main__":
    asyncio.run(backfill())
