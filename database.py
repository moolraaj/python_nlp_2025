import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

 
load_dotenv()

 
MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME     = os.getenv("MONGODB_DB_NAME", "test")

client = AsyncIOMotorClient(MONGODB_URI)
db     = client[DB_NAME]