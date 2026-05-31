import os
# Pin VAD/ONNX runtime threads to prevent extreme thread contention and realtime delay
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import asyncio
import logging
import json
from datetime import datetime
from bson import ObjectId

# Load local environment variables if present
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

from livekit.agents import JobContext, WorkerOptions, JobRequest, cli, llm
from livekit.agents.voice import Agent, AgentSession, UserInputTranscribedEvent, ConversationItemAddedEvent
from livekit.agents.voice.room_io import RoomOptions
from livekit.agents.llm import ChatMessage
from livekit.plugins import openai, silero
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)

class AppointmentTools:
    def __init__(self, business_id: ObjectId):
        self.client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
        self.db = self.client["voice_agent"]
        self.appointments = self.db["appointments"]
        self.business_id = business_id

    @llm.function_tool
    async def check_availability(self, time: str) -> str:
        """Check if an appointment slot is available at the requested date and time."""
        existing = await self.appointments.find_one({"business_id": self.business_id, "time": time, "status": "scheduled"})
        if existing:
            return f"No, that slot at {time} is already booked."
        return f"Yes, the slot at {time} is available."

    @llm.function_tool
    async def book_appointment(self, name: str, phone: str, time: str, reason: str) -> str:
        """Book a new appointment. You must check availability first."""
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
        """Cancel an existing scheduled appointment using the phone number and scheduled time."""
        res = await self.appointments.update_many(
            {"business_id": self.business_id, "phone": phone, "time": time, "status": "scheduled"},
            {"$set": {"status": "cancelled"}}
        )
        if res.modified_count > 0:
            return f"Success! The appointment at {time} for phone {phone} has been cancelled."
        return f"No scheduled appointment was found for phone {phone} at {time}."


async def request_handler(request: JobRequest) -> None:
    logging.info(f"Received an incoming job request for room: {request.room.name}")
    # Handle wildcard dynamic room prefixes assigned by the SIP engine
    if request.room.name.startswith("sip_room"):
        await request.accept()
    else:
        logging.info("Rejecting request: Room prefix mismatch.")
        await request.reject()


async def entrypoint(ctx: JobContext):
    logging.info(f"Connecting worker process to room session: {ctx.room.name}")
    await ctx.connect()
    
    await asyncio.sleep(0.5)
    
    extension = "499"  # Default fallback
    
    # Wait up to 2 seconds for participant to join to prevent extension resolution race condition
    if not ctx.room.remote_participants:
        logging.info("Waiting for participant to join room to resolve extension...")
        for _ in range(20):
            await asyncio.sleep(0.1)
            if ctx.room.remote_participants:
                break
                
    logging.info(f"Checking remote participants: {list(ctx.room.remote_participants.keys())}")
    for identity, participant in ctx.room.remote_participants.items():
        logging.info(f"Participant {identity}: attributes={participant.attributes}, metadata={participant.metadata}")
        
    for identity, participant in ctx.room.remote_participants.items():
        if "sip.trunkPhoneNumber" in participant.attributes:
            extension = participant.attributes["sip.trunkPhoneNumber"]
            break
        elif "extension" in participant.attributes:
            extension = participant.attributes["extension"]
            break
        elif "sip.phoneNumber" in participant.attributes:
            extension = "499"  # Handset incoming trunk fallback
            break
        if participant.metadata:
            try:
                meta = json.loads(participant.metadata)
                if "extension" in meta:
                    extension = str(meta["extension"])
                    break
            except Exception:
                pass
                
    logging.info(f"Resolved extension routing to: {extension}")
    
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
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
            "You are 'Gravity', a professional real-time voice receptionist for Compusource. "
            "Keep answers very short, concise, and natural (one or two sentences)."
        )
        voice_model = "af_bella"
        skills = ["appointments"]
        business_id = ObjectId("6659f13ba97312fba0a91e5c")
        
    # Initialize Core Model Engines
    vad_model = silero.VAD.load()
    stt_model = openai.STT(base_url="http://127.0.0.1:8000/v1", api_key="local", model="base.en")
    llm_model = openai.LLM(base_url="http://127.0.0.1:11434/v1", model="llama3.1:8b-instruct-q4_K_M", api_key="local")
    
    tts_model = openai.TTS(
        base_url="http://127.0.0.1:10201/v1",
        api_key="not-needed",
        model="tts-1",
        voice=voice_model,
        response_format="wav"
    )
    
    from livekit.agents.llm import find_function_tools
    tools_list = []
    if "appointments" in skills:
        tools_list = find_function_tools(AppointmentTools(business_id))
        
    # Instantiating the validated Agent configuration layout
    assistant = Agent(
        vad=vad_model,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        tools=tools_list,
        instructions=agent_prompt,
        allow_interruptions=True
    )
    
    # Initialize live call log record in MongoDB
    call_record = {
        "room_name": ctx.room.name,
        "extension": extension,
        "business_id": business_id,
        "business_name": agent_name,
        "phone_number": "web-sandbox",
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "duration_seconds": 0.0,
        "status": "active",
        "transcript": []
    }
    
    for identity, p in ctx.room.remote_participants.items():
        if "sip.phoneNumber" in p.attributes:
            call_record["phone_number"] = p.attributes["sip.phoneNumber"]
            break
            
    await db["calls"].insert_one(call_record)
    logging.info(f"Initialized live call record in MongoDB for room {ctx.room.name}")
    
    # Instantiate the AgentSession container
    session = AgentSession(
        vad=vad_model,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        tools=tools_list,
        allow_interruptions=True
    )

    async def save_to_mongo(dialog):
        try:
            await db["calls"].update_many(
                {"room_name": ctx.room.name, "status": "active"},
                {"$push": {"transcript": dialog}}
            )
        except Exception as e:
            logging.error(f"Error saving turn to MongoDB: {e}")

    # Transcript bindings tracking real-time user/agent turns via the session listeners
    @session.on("user_input_transcribed")
    def on_user_input(ev: UserInputTranscribedEvent):
        if ev.is_final and ev.transcript.strip():
            dialog = {
                "role": "user",
                "text": str(ev.transcript.strip()),
                "time": datetime.utcnow().isoformat()
            }
            logging.info(f"Dialog Turn [User]: {ev.transcript}")
            asyncio.create_task(save_to_mongo(dialog))

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev: ConversationItemAddedEvent):
        if isinstance(ev.item, ChatMessage) and ev.item.role == "assistant":
            text = ev.item.text_content
            if text and text.strip():
                dialog = {
                    "role": "agent",
                    "text": str(text.strip()),
                    "time": datetime.utcnow().isoformat()
                }
                logging.info(f"Dialog Turn [Agent]: {text}")
                asyncio.create_task(save_to_mongo(dialog))

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        logging.info(f"Participant disconnected: {participant.identity}")
        if not ctx.room.remote_participants:
            logging.info("Room empty. Shutting down worker...")
            ctx.shutdown()
            
    async def close_call_record():
        try:
            end_time = datetime.utcnow()
            start_time_parsed = datetime.fromisoformat(call_record["start_time"])
            duration = (end_time - start_time_parsed).total_seconds()
            
            await db["calls"].update_many(
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

    logging.info(f"Warming up task matrices... Dynamic AI Receptionist [{agent_name}] is active!")

    room_options = RoomOptions(close_on_disconnect=False)
    await session.start(assistant, room=ctx.room, room_options=room_options)

    # Wait for the first subscriber to connect and subscribe to our audio track
    try:
        if session.room_io and session.room_io.subscribed_fut:
            logging.info("Waiting for participant track subscription...")
            await session.room_io.subscribed_fut
            logging.info("Participant subscribed! Saying greeting...")
    except Exception as e:
        logging.warning(f"Error waiting for subscription: {e}")

    # Dynamically speak the starting greeting based on the resolved business name
    session.say(f"Hello! Thank you for calling {agent_name}. How can I help you today?", allow_interruptions=True)

    await session.wait_for_inactive()


if __name__ == '__main__':
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,
        agent_name=""
    ))