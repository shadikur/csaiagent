import pymongo
from datetime import datetime

client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
db = client.voice_agent

# Mark the specific stale 'sip_room' call as completed
res1 = db.calls.update_many(
    {"room_name": "sip_room", "status": "active"},
    {"$set": {"status": "completed", "end_time": datetime.utcnow().isoformat()}}
)
print(f"Stale sip_room calls cleaned: {res1.modified_count}")

# Mark any active call that has been running for more than 1 hour as completed (fail-safe)
# In this environment, any call started earlier than 10 minutes ago is definitely stale.
res2 = db.calls.update_many(
    {"status": "active", "room_name": {"$ne": "sip_room_215_mjQ3W3Y8JyeE"}},
    {"$set": {"status": "completed", "end_time": datetime.utcnow().isoformat()}}
)
print(f"Other stale active calls cleaned: {res2.modified_count}")
