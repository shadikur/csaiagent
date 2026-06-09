import io
import os
import time
import json
import asyncio
import logging
import httpx
import hashlib
import datetime
import tempfile
import subprocess
from fastapi import FastAPI, Request, Response, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants, LiveKitAPI
from livekit import api
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_bridge")

# Global http client for connection pooling to optimize latency
http_client = httpx.AsyncClient(timeout=30.0)

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    async def periodic_sip_sync():
        while True:
            try:
                logger.info("Running periodic SIP trunks sync...")
                from sync_sip import sync
                await sync()
                logger.info("Periodic SIP trunks sync completed successfully.")
            except Exception as e:
                logger.error(f"Error in periodic SIP trunks sync: {e}")
            await asyncio.sleep(600)  # Run every 10 minutes

    asyncio.create_task(periodic_sip_sync())

    # Pre-warm the connection to Kokoro TTS so the first real call has no TCP setup cost
    async def _warmup_kokoro():
        await asyncio.sleep(5)  # Give Kokoro a moment to be ready after service start
        for attempt in range(12):  # Try for up to ~60s
            try:
                r = await http_client.get("http://127.0.0.1:8880/health", timeout=5.0)
                if r.status_code < 500:
                    logger.info("✅ Kokoro TTS pre-warmed and ready")
                    return
            except Exception:
                pass
            await asyncio.sleep(5)
        logger.warning("Kokoro TTS warmup timed out — it may still be loading")

    asyncio.create_task(_warmup_kokoro())

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB
mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
db = mongo_client["voice_agent"]
appointments_col = db["appointments"]
businesses_col = db["businesses"]
users_col = db["users"]

JWT_SECRET = "compusource_jwt_secret_998246"
JWT_ALGORITHM = "HS256"

# Helper for secure password hashing
def hash_password(password: str, salt: str = "compusource_salt_127849") -> str:
    return hashlib.sha256((password + salt).encode()).hexdigest()

# Helper for JWT creation
def create_jwt_token(data: dict, expires_in_hours: int = 24) -> str:
    import jwt
    payload = data.copy()
    payload["exp"] = datetime.datetime.utcnow() + datetime.timedelta(hours=expires_in_hours)
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

# Helper for JWT validation
def decode_jwt_token(token: str) -> dict:
    import jwt
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise Exception("Token expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")

def load_keys():
    keys_path = "/home/compusource/voice-agent/keys.yaml"
    if os.path.exists(keys_path):
        try:
            with open(keys_path) as f:
                content = f.read().strip()
                if content and ":" in content:
                    key, secret = content.split(":", 1)
                    return key.strip(), secret.strip()
        except Exception as e:
            logger.error(f"Error loading keys.yaml: {e}")
    return "devkey1778495864", "devsecret1778495864"

# Helper to automatically register Inbound SIP Trunk & Dispatch Rule in LiveKit
def register_livekit_sip_trunk_and_dispatch(extension: str):
    try:
        key, secret = load_keys()
        env = os.environ.copy()
        env["LIVEKIT_URL"] = "http://127.0.0.1:7800"
        env["LIVEKIT_API_KEY"] = key
        env["LIVEKIT_API_SECRET"] = secret

        # 1. Create Inbound Trunk JSON
        trunk_data = {
            "trunk": {
                "name": f"Trunk-{extension}",
                "numbers": [extension],
                "allowedAddresses": ["199.47.47.106"]
            }
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(trunk_data, f)
            temp_trunk_path = f.name
        
        # Execute trunk creation via lk CLI
        res = subprocess.run(["lk", "sip", "inbound", "create", temp_trunk_path], env=env, capture_output=True, text=True, check=True)
        trunk_id = None
        for line in res.stdout.splitlines():
            if "SIPTrunkID:" in line or "SipTrunkID" in line:
                trunk_id = line.split(":")[-1].strip()
        
        os.unlink(temp_trunk_path)
        if not trunk_id:
            logger.error(f"Failed to extract Inbound SIP Trunk ID for extension {extension}")
            return None, None
            
        # 2. Create Inbound Dispatch Rule via robust CLI options
        res = subprocess.run([
            "lk", "sip", "dispatch", "create",
            "--name", f"Route-{extension}",
            "--trunks", trunk_id,
            "--individual", "sip_room",
            "--randomize"
        ], env=env, capture_output=True, text=True, check=True)
        
        dispatch_id = None
        for line in res.stdout.splitlines():
            if "SIPDispatchRuleID:" in line or "SipDispatchRuleID" in line:
                dispatch_id = line.split(":")[-1].strip()
                
        logger.info(f"Successfully configured dynamic SIP mapping: extension={extension}, trunk={trunk_id}, rule={dispatch_id}")
        return trunk_id, dispatch_id
    except Exception as e:
        logger.error(f"Failed to register dynamic SIP dialplan for extension {extension}: {e}")
        return None, None

# Helper to automatically clean up Inbound SIP Trunk & Dispatch Rule
def cleanup_livekit_sip_trunk_and_dispatch(extension: str):
    try:
        key, secret = load_keys()
        env = os.environ.copy()
        env["LIVEKIT_URL"] = "http://127.0.0.1:7800"
        env["LIVEKIT_API_KEY"] = key
        env["LIVEKIT_API_SECRET"] = secret

        # Delete dispatch rule by matching name Route-{extension}
        res = subprocess.run(["lk", "sip", "dispatch", "list"], env=env, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if f"Route-{extension}" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("SDR_"):
                        subprocess.run(["lk", "sip", "dispatch", "delete", part], env=env)
                        logger.info(f"Deleted SIP dispatch rule: {part}")
                        
        # Delete inbound trunk by matching name Trunk-{extension}
        res = subprocess.run(["lk", "sip", "inbound", "list"], env=env, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if f"Trunk-{extension}" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("ST_"):
                        subprocess.run(["lk", "sip", "inbound", "delete", part], env=env)
                        logger.info(f"Deleted SIP inbound trunk: {part}")
    except Exception as e:
        logger.error(f"Failed to clean up SIP dialplan for extension {extension}: {e}")

# ----------------- JWT Authentication Endpoints -----------------

@app.post("/v1/auth/login")
async def login(request: Request):
    req_json = await request.json()
    username = req_json.get("username", "").strip()
    password = req_json.get("password", "").strip()
    
    if not (username and password):
        return JSONResponse(status_code=400, content={"error": "Username and password required"})
        
    user = await users_col.find_one({"username": username})
    if not user:
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
        
    hashed = hash_password(password)
    if user["password_hash"] != hashed:
        return JSONResponse(status_code=401, content={"error": "Invalid username or password"})
        
    token = create_jwt_token({"username": user["username"], "role": user.get("role", "admin")})
    return {"token": token, "username": user["username"]}

@app.get("/v1/auth/me")
async def get_me(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Authorization token required"})
        
    token = authorization.split("Bearer ")[-1].strip()
    try:
        payload = decode_jwt_token(token)
        return {"username": payload["username"], "role": payload.get("role", "admin")}
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})

@app.post("/v1/auth/change-password")
async def change_password(request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Authorization token required"})
    try:
        payload = decode_jwt_token(authorization.split("Bearer ")[-1].strip())
        username = payload["username"]
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
        
    req_json = await request.json()
    old_password = req_json.get("old_password", "").strip()
    new_password = req_json.get("new_password", "").strip()
    
    if not (old_password and new_password):
        return JSONResponse(status_code=400, content={"error": "Old and new password required"})
        
    user = await users_col.find_one({"username": username})
    if not user:
        return JSONResponse(status_code=404, content={"error": "User not found"})
        
    if user["password_hash"] != hash_password(old_password):
        return JSONResponse(status_code=400, content={"error": "Incorrect old password"})
        
    await users_col.update_one(
        {"username": username},
        {"$set": {"password_hash": hash_password(new_password)}}
    )
    return {"status": "success", "message": "Password changed successfully"}

# ----------------- Business / Agent Studio CRUD Endpoints -----------------

@app.get("/v1/businesses")
async def list_businesses():
    businesses = []
    try:
        cursor = businesses_col.find().sort("name", 1)
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            businesses.append(doc)
        return businesses
    except Exception as e:
        logger.error(f"Error querying businesses: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/businesses")
async def create_business(request: Request, authorization: str = Header(None)):
    # Authenticate admin first
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Authorization token required"})
    try:
        decode_jwt_token(authorization.split("Bearer ")[-1].strip())
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
        
    req_json = await request.json()
    name = req_json.get("name", "").strip()
    extension = req_json.get("extension", "").strip()
    voice = req_json.get("voice", "af_bella").strip()
    teams_webhook_url = req_json.get("teams_webhook_url", "").strip()
    skills = req_json.get("skills", ["appointments"])
    prompt = req_json.get("prompt", "").strip()
    
    if not (name and extension and prompt):
        return JSONResponse(status_code=400, content={"error": "Missing name, extension or prompt"})
        
    # Prevent duplicate extensions
    existing = await businesses_col.find_one({"extension": extension})
    if existing:
        return JSONResponse(status_code=400, content={"error": f"Extension {extension} is already assigned"})
        
    try:
        # Create SIP trunk/rule programmatically
        trunk_id, dispatch_id = register_livekit_sip_trunk_and_dispatch(extension)
        
        doc = {
            "name": name,
            "extension": extension,
            "voice": voice,
            "teams_webhook_url": teams_webhook_url,
            "skills": skills,
            "prompt": prompt,
            "sip_trunk_id": trunk_id,
            "sip_dispatch_id": dispatch_id,
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        res = await businesses_col.insert_one(doc)
        return {"status": "success", "id": str(res.inserted_id)}
    except Exception as e:
        logger.error(f"Error creating business agent: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.put("/v1/businesses/{id}")
async def update_business(id: str, request: Request, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Authorization token required"})
    try:
        decode_jwt_token(authorization.split("Bearer ")[-1].strip())
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
        
    req_json = await request.json()
    name = req_json.get("name", "").strip()
    voice = req_json.get("voice", "af_bella").strip()
    teams_webhook_url = req_json.get("teams_webhook_url", "").strip()
    skills = req_json.get("skills", ["appointments"])
    prompt = req_json.get("prompt", "").strip()
    
    if not (name and prompt):
        return JSONResponse(status_code=400, content={"error": "Missing name or prompt"})
        
    try:
        res = await businesses_col.update_one(
            {"_id": ObjectId(id)},
            {"$set": {
                "name": name,
                "voice": voice,
                "teams_webhook_url": teams_webhook_url,
                "skills": skills,
                "prompt": prompt
            }}
        )
        if res.matched_count > 0:
            return {"status": "success"}
        return JSONResponse(status_code=404, content={"error": "Business not found"})
    except Exception as e:
        logger.error(f"Error updating business agent: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/v1/businesses/{id}")
async def delete_business(id: str, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"error": "Authorization token required"})
    try:
        decode_jwt_token(authorization.split("Bearer ")[-1].strip())
    except Exception as e:
        return JSONResponse(status_code=401, content={"error": str(e)})
        
    try:
        business = await businesses_col.find_one({"_id": ObjectId(id)})
        if not business:
            return JSONResponse(status_code=404, content={"error": "Business not found"})
            
        # Clean up registered SIP trunks/rules in LiveKit
        cleanup_livekit_sip_trunk_and_dispatch(business["extension"])
        
        # Clean up database records
        await businesses_col.delete_one({"_id": ObjectId(id)})
        await appointments_col.delete_many({"business_id": ObjectId(id)})
        
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error deleting business agent: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------- Proxy TTS Endpoint (Chunked Streaming) -----------------

@app.post("/v1/audio/speech")
async def text_to_speech(request: Request):
    req_json = await request.json()
    voice = req_json.get("voice", "af_bella")
    
    # Check if we should use gTTS or Kokoro
    use_gtts = False
    gtts_lang = "en"
    voice_lower = voice.lower()

    # Non-English 2-letter language codes fall back to gTTS
    if voice_lower.startswith("de") or voice_lower == "german":
        use_gtts = True
        gtts_lang = "de"
    elif len(voice_lower) == 2 and voice_lower not in ["en"]:
        use_gtts = True
        gtts_lang = voice_lower

    if use_gtts:
        try:
            from gtts import gTTS
            import io
            import subprocess
            
            text = req_json.get("input", "")
            logger.info(f"Generating gTTS speech for lang={gtts_lang}: {text[:40]}...")
            
            tts = gTTS(text=text, lang=gtts_lang)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_bytes = mp3_fp.getvalue()
            
            # Convert MP3 to WAV using ffmpeg
            process = subprocess.Popen(
                ["ffmpeg", "-i", "pipe:0", "-f", "wav", "pipe:1"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            wav_bytes, err = process.communicate(input=mp3_bytes)
            
            if process.returncode != 0:
                logger.error(f"ffmpeg conversion failed: {err.decode(errors='ignore')}")
                return Response(status_code=500, content="Audio conversion error")
                
            return StreamingResponse(
                io.BytesIO(wav_bytes),
                media_type="audio/wav"
            )
        except Exception as e:
            logger.error(f"gTTS generation failed: {e}")
            return Response(status_code=500, content=f"gTTS Error: {str(e)}")

    # Proxy to Kokoro TTS
    logger.info(f"Proxying TTS to Kokoro: voice={voice_lower!r} text={req_json.get('input', '')[:40]!r}")
    # Ensure voice is set to 'default' if not already mapped
    if "voice" not in req_json or not req_json["voice"]:
        req_json["voice"] = "default"
        
    try:
        req = http_client.build_request(
            "POST",
            "http://127.0.0.1:8880/v1/audio/speech",
            json=req_json
        )
        resp = await http_client.send(req, stream=True)
        
        async def stream_generator():
            try:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            finally:
                await resp.aclose()
                
        return StreamingResponse(
            stream_generator(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "audio/wav")
        )
    except Exception as e:
        logger.error(f"Failed to proxy to Kokoro-FastAPI: {e}")
        return Response(status_code=500, content=f"TTS Proxy Error: {str(e)}")

# ----------------- Token Generation -----------------

@app.get("/token")
async def get_token(request: Request):
    extension = request.query_params.get("extension", "499").strip()
    key, secret = load_keys()
    room_name = f"sip_room_{int(time.time())}"
    logger.info(f"Generating access token for dynamic room: {room_name} mapped to extension: {extension}")
    
    grant = VideoGrants(
        room_join=True,
        room=room_name,
        room_admin=True,
        room_create=True,
        room_list=True,
        can_update_own_metadata=True
    )
    # Use builder methods so metadata claims are correctly compiled into the JWT instead of ignored/empty
    token = (AccessToken(key, secret)
             .with_grants(grant)
             .with_identity(f"web-user-{int(time.time())}")
             .with_name("web-test-user")
             .with_metadata(json.dumps({"agent_to_dispatch": "*", "extension": extension})))
    
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return JSONResponse(
        content={
            "token": token.to_jwt(),
            "room": room_name
        },
        headers=headers
    )

# ----------------- Outbound Telephony Dialing Endpoint -----------------

@app.post("/v1/call")
async def make_outbound_call(request: Request):
    req_json = await request.json()
    phone_number = req_json.get("phone", "").strip()
    room_name = req_json.get("room", "sip_room").strip()
    
    if not phone_number:
        return JSONResponse(status_code=400, content={"error": "Phone number is required"})
        
    key, secret = load_keys()
    logger.info(f"Initiating outbound call to {phone_number} in room '{room_name}'...")
    
    try:
        # Use LiveKitAPI context manager to create a SIP participant
        async with LiveKitAPI(url="http://localhost:7800", api_key=key, api_secret=secret) as lkapi:
            participant = await lkapi.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(
                    room_name=room_name,
                    sip_trunk_id="ST_fKws9WD2CkHC",
                    sip_call_to=phone_number,
                    participant_identity=f"phone_{int(time.time())}",
                    participant_name="Outbound Agent Call"
                )
            )
            logger.info(f"SIP Call successfully placed: {participant}")
            return {"status": "success", "participant": str(participant.sip_call_id)}
            
    except Exception as e:
        logger.error(f"Failed to place outbound call: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------- Appointments Isolated Directory Endpoints -----------------

@app.get("/v1/appointments")
async def list_appointments(request: Request):
    business_id = request.query_params.get("business_id")
    query = {}
    if business_id:
        try:
            query["business_id"] = ObjectId(business_id)
        except:
            pass
            
    appointments = []
    try:
        cursor = appointments_col.find(query).sort("time", 1)
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "business_id" in doc:
                doc["business_id"] = str(doc["business_id"])
            appointments.append(doc)
        return appointments
    except Exception as e:
        logger.error(f"Error querying appointments: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/v1/appointments")
async def create_appointment(request: Request):
    req_json = await request.json()
    name = req_json.get("name", "").strip()
    phone = req_json.get("phone", "").strip()
    time_str = req_json.get("time", "").strip()
    reason = req_json.get("reason", "").strip()
    business_id = req_json.get("business_id", "").strip()
    
    if not (name and phone and time_str and business_id):
        return JSONResponse(status_code=400, content={"error": "Missing name, phone, time or business_id"})
        
    try:
        b_id = ObjectId(business_id)
        # Check if already booked for this business
        existing = await appointments_col.find_one({"business_id": b_id, "time": time_str, "status": "scheduled"})
        if existing:
            return JSONResponse(status_code=400, content={"error": "Slot already taken"})
            
        doc = {
            "business_id": b_id,
            "name": name,
            "phone": phone,
            "time": time_str,
            "reason": reason,
            "status": "scheduled"
        }
        await appointments_col.insert_one(doc)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error creating appointment: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.delete("/v1/appointments/{appointment_id}")
async def cancel_appointment(appointment_id: str):
    try:
        res = await appointments_col.update_one(
            {"_id": ObjectId(appointment_id)},
            {"$set": {"status": "cancelled"}}
        )
        if res.modified_count > 0:
            return {"status": "success"}
        return JSONResponse(status_code=404, content={"error": "Appointment not found"})
    except Exception as e:
        logger.error(f"Error cancelling appointment: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------- Call Logs & Transcripts Endpoints -----------------

@app.get("/v1/calls/active")
async def list_active_calls():
    # Dynamically clean up stale active calls by checking LiveKit active rooms
    try:
        from livekit.protocol.room import ListRoomsRequest
        key, secret = load_keys()
        async with LiveKitAPI("http://127.0.0.1:7800", key, secret) as lkapi:
            rooms_res = await lkapi.room.list_rooms(ListRoomsRequest())
            active_room_names = {r.name for r in rooms_res.rooms}
            
            db_active_calls = await db["calls"].find({"status": "active"}).to_list(length=1000)
            for call in db_active_calls:
                room_name = call.get("room_name")
                if room_name not in active_room_names:
                    end_time = datetime.datetime.utcnow()
                    start_time_str = call.get("start_time")
                    duration = 0.0
                    if start_time_str:
                        try:
                            start_time_clean = start_time_str.replace("Z", "+00:00")
                            start_time_parsed = datetime.datetime.fromisoformat(start_time_clean)
                            if start_time_parsed.tzinfo:
                                end_time_aware = datetime.datetime.now(datetime.timezone.utc)
                                duration = (end_time_aware - start_time_parsed).total_seconds()
                            else:
                                duration = (end_time - start_time_parsed).total_seconds()
                        except Exception as parse_err:
                            logger.error(f"Error parsing start_time {start_time_str}: {parse_err}")
                    
                    await db["calls"].update_one(
                        {"_id": call["_id"]},
                        {"$set": {
                            "status": "completed",
                            "end_time": end_time.isoformat(),
                            "duration_seconds": max(0.0, duration)
                        }}
                    )
                    logger.info(f"Dynamically cleaned up stale call {call['_id']} (room {room_name})")
    except Exception as e:
        logger.error(f"Error during dynamic active calls cleanup: {e}")

    calls = []
    try:
        cursor = db["calls"].find({"status": "active"}).sort("start_time", -1)
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "business_id" in doc:
                doc["business_id"] = str(doc["business_id"])
            calls.append(doc)
        return calls
    except Exception as e:
        logger.error(f"Error querying active calls: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/calls/history")
async def list_call_history(request: Request):
    business_id = request.query_params.get("business_id")
    query = {"status": "completed"}
    if business_id:
        try:
            query["business_id"] = ObjectId(business_id)
        except:
            pass
            
    calls = []
    try:
        cursor = db["calls"].find(query).sort("start_time", -1).limit(50)
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            if "business_id" in doc:
                doc["business_id"] = str(doc["business_id"])
            calls.append(doc)
        return calls
    except Exception as e:
        logger.error(f"Error querying call history: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/calls/{call_id}")
async def get_call_details(call_id: str):
    try:
        doc = await db["calls"].find_one({"_id": ObjectId(call_id)})
        if not doc:
            return JSONResponse(status_code=404, content={"error": "Call not found"})
        doc["_id"] = str(doc["_id"])
        if "business_id" in doc:
            doc["business_id"] = str(doc["business_id"])
        return doc
    except Exception as e:
        logger.error(f"Error getting call details: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="/home/compusource/voice-web-ui", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10201)

