import asyncio
import logging
import os
import json
from datetime import datetime
from bson import ObjectId

# Manually load .env file if present before loading livekit SDK configs
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from livekit.agents import JobContext, WorkerOptions, JobRequest, cli, llm
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)

class AppointmentTools:
    def __init__(self, business_id: ObjectId):
        self.client = AsyncIOMotorClient("mongodb://localhost:27017")
        self.db = self.client["voice_agent"]
        self.appointments = self.db["appointments"]
        self.business_id = business_id

    @llm.function_tool
    async def check_availability(self, time: str) -> str:
        """Check if an appointment slot is available at the requested date and time.

        Args:
            time: The date and time string (e.g., '2026-06-01 10:00').
        """
        existing = await self.appointments.find_one({"business_id": self.business_id, "time": time, "status": "scheduled"})
        if existing:
            return f"No, that slot at {time} is already booked."
        return f"Yes, the slot at {time} is available."

    @llm.function_tool
    async def book_appointment(self, name: str, phone: str, time: str, reason: str) -> str:
        """Book a new appointment. You must check availability first.

        Args:
            name: The caller's name.
            phone: The caller's phone number.
            time: The appointment date and time (e.g., '2026-06-01 10:00').
            reason: The reason for the appointment.
        """
        existing = await self.appointments.find_one({"business_id": self.business_id, "time": time, "status": "scheduled"})
        if existing:
            return f"Cannot book. The slot at {time} is already taken."
        
        appointment = {
            "business_id": self.business_id,
            "name": name,
            "phone": phone,
            "time": time,
            "reason": reason,
            "status": "scheduled"
        }
        await self.appointments.insert_one(appointment)
        return f"Success! Appointment booked for {name} ({phone}) at {time} for '{reason}'."

    @llm.function_tool
    async def cancel_appointment(self, phone: str, time: str) -> str:
        """Cancel an existing scheduled appointment using the phone number and scheduled time.

        Args:
            phone: The caller's phone number.
            time: The scheduled date and time (e.g., '2026-06-01 10:00').
        """
        res = await self.appointments.update_many(
            {"business_id": self.business_id, "phone": phone, "time": time, "status": "scheduled"},
            {"$set": {"status": "cancelled"}}
        )
        if res.modified_count > 0:
            return f"Success! The appointment at {time} for phone {phone} has been cancelled."
        return f"No scheduled appointment was found for phone {phone} at {time}."

    @llm.function_tool
    async def make_outbound_call(self, phone_number: str) -> str:
        """Make an outbound call to schedule or follow up on an appointment.

        Args:
            phone_number: The phone number to call.
        """
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post("http://localhost:10201/v1/call", json={"phone": phone_number})
                if resp.status_code == 200:
                    return f"Successfully initiated outbound call to {phone_number}."
                else:
                    return f"Failed to initiate call. Status: {resp.status_code}."
        except Exception as e:
            return f"Error triggering outbound call: {str(e)}"

async def request_handler(request: JobRequest) -> None:
    logging.info(f"Received an incoming job request for room: {request.room.name}")
    await request.accept()

async def entrypoint(ctx: JobContext):
    logging.info(f"Connecting worker process to room session: {ctx.room.name}")
    await ctx.connect()
    
    # Wait briefly for participant attributes/metadata to populate
    await asyncio.sleep(0.5)
    
    # Resolve dialed extension from remote participants
    extension = "499"  # Default fallback extension (Compusource)
    for identity, participant in ctx.room.remote_participants.items():
        # Check custom or trunk extension attributes (for SIP inbound calls)
        if "extension" in participant.attributes:
            extension = participant.attributes["extension"]
            logging.info(f"SIP participant matched extension via attributes: {extension}")
            break
        elif "sip.trunkPhoneNumber" in participant.attributes:
            extension = participant.attributes["sip.trunkPhoneNumber"]
            logging.info(f"SIP participant matched extension via trunk number: {extension}")
            break
        # Check metadata attributes (for Web UI Sandbox sessions)
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
                if "extension" in meta:
                    extension = str(meta["extension"])
                    logging.info(f"Web Sandbox participant matched extension: {extension}")
                    break
            except Exception as e:
                logging.warning(f"Failed to parse participant metadata: {e}")
                
    logging.info(f"Resolved extension routing to: {extension}")
    
    # Dynamic Business/Agent Lookup
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["voice_agent"]
    
    business = await db["businesses"].find_one({"extension": extension})
    if business:
        logging.info(f"Dynamically loading profile for business: {business['name']}")
        agent_name = business.get("name", "Gravity")
        agent_prompt = business["prompt"]
        voice_model = business.get("voice", "af_bella")
        skills = business.get("skills", ["appointments"])
        business_id = business["_id"]
    else:
        logging.warning(f"No registered business found for extension {extension}. Falling back to default.")
        agent_name = "Gravity"
        agent_prompt = (
            "You are 'Gravity', a professional, friendly, and natural real-time voice receptionist "
            "for Compusource. Your voice is hyper-realistic and highly engaging.\n\n"
            "Keep answers very short, concise, and natural (usually one or two sentences). "
            "You can check slot availability, book appointments, or cancel appointments. "
            "Always check slot availability before booking. Our office hours are 9 AM to 5 PM, Mon-Fri."
        )
        voice_model = "af_bella"
        skills = ["appointments"]
        business_id = ObjectId("6659f13ba97312fba0a91e5c") # Static default
        
    vad_model = silero.VAD.load()
    stt_model = openai.STT(base_url="http://localhost:8000/v1", api_key="local", model="base.en")
    llm_model = openai.LLM(base_url="http://localhost:11434/v1", model="llama3.1:8b-instruct-q4_K_M", api_key="local")
    
    # Point to the local FastAPI BFF gateway proxy on port 10201
    tts_model = openai.TTS(
        base_url="http://localhost:10201/v1",
        api_key="not-needed",
        model="tts-1",
        voice=voice_model,
        response_format="mp3"
    )
    
    # Dynamically compile tools based on skills
    from livekit.agents.llm import find_function_tools
    tools_list = []
    if "appointments" in skills:
        tools_list = find_function_tools(AppointmentTools(business_id))
        
    agent = Agent(
        vad=vad_model,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        tools=tools_list,
        instructions=agent_prompt,
        allow_interruptions=True
    )
    
    session = AgentSession(
        vad=vad_model,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        allow_interruptions=True
    )
    
    logging.info(f"Warming up task matrices... Dynamic AI Receptionist [{agent_name}] is active!")
    
    # Initialize dynamic call logging record in MongoDB
    call_record = {
        "room_name": ctx.room.name,
        "extension": extension,
        "business_id": business_id,
        "business_name": agent_name,
        "phone_number": "web-sandbox",  # Fallback
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "duration_seconds": 0.0,
        "status": "active",
        "transcript": []
    }
    
    # Try to extract the caller's phone number from remote participants
    for identity, p in ctx.room.remote_participants.items():
        if "sip.phoneNumber" in p.attributes:
            call_record["phone_number"] = p.attributes["sip.phoneNumber"]
            break
            
    await db["calls"].insert_one(call_record)
    logging.info(f"Initialized live call record in MongoDB for room {ctx.room.name}")
    
    room_options = RoomOptions(close_on_disconnect=False)
    await session.start(agent, room=ctx.room, room_options=room_options)
    
    # Real-time dialog observation using conversation_item_added
    @session.on("conversation_item_added")
    def on_conversation_item_added(event):
        from livekit.agents.llm import ChatMessage
        if isinstance(event.item, ChatMessage):
            role = event.item.role
            text = event.item.text_content
            if text and text.strip():
                dialog = {
                    "role": "user" if role == "user" else "agent",
                    "text": text,
                    "time": datetime.utcnow().isoformat()
                }
                logging.info(f"Dialog Turn Added [{role}]: {text}")
                asyncio.create_task(db["calls"].update_one(
                    {"room_name": ctx.room.name, "status": "active"},
                    {"$push": {"transcript": dialog}}
                ))
    
    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        logging.info(f"Participant disconnected: {participant.identity}")
        if not ctx.room.remote_participants:
            logging.info("Room is empty, shutting down session...")
            ctx.shutdown()
            
    # Close call record upon exit
    async def close_call_record():
        try:
            end_time = datetime.utcnow()
            start_time_parsed = datetime.fromisoformat(call_record["start_time"])
            duration = (end_time - start_time_parsed).total_seconds()
            
            await db["calls"].update_one(
                {"room_name": ctx.room.name, "status": "active"},
                {"$set": {
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "status": "completed"
                }}
            )
            logging.info(f"Archived call history in MongoDB for room {ctx.room.name}. Duration: {duration}s")
        except Exception as e:
            logging.error(f"Error archiving call: {e}")
        
    ctx.add_shutdown_callback(close_call_record)
    
    # Wait for the first subscriber to connect and subscribe to our audio track
    try:
        if session.room_io and session.room_io.subscribed_fut:
            logging.info("Waiting for participant track subscription...")
            await session.room_io.subscribed_fut
            logging.info("Participant subscribed! Saying greeting...")
    except Exception as e:
        logging.warning(f"Error waiting for subscription: {e}")
    
    # Greet the user immediately on connection to test the audio pipeline
    await session.say("Hello, how can I help you today?", allow_interruptions=True)

if __name__ == '__main__':
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler
    ))
