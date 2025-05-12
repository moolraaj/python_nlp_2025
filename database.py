import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

load_dotenv('.env')  

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB_NAME", "test")

 
print("DEBUG URI:", repr(MONGODB_URI))  

 
MONGODB_URI = MONGODB_URI.replace(r'\x3a', ':')

if not MONGODB_URI:
    raise ValueError("MONGODB_URI not set!")
if not MONGODB_URI.startswith(('mongodb://', 'mongodb+srv://')):
    raise ValueError(f"Invalid URI format: {MONGODB_URI}")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]