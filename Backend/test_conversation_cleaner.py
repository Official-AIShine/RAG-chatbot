"""
Quick script to clear conversation collection for testing.
"""
from dotenv import load_dotenv
import os
from pymongo import MongoClient

load_dotenv()

uri = os.getenv("MONGO_DB_URI")
db_name = os.getenv("DB_NAME")

client = MongoClient(uri)
db = client[db_name]

# Delete all conversations
result = db.conversations.delete_many({})
print(f"✅ Deleted {result.deleted_count} conversations")

client.close()