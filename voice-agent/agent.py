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

logger = logging.getLogger("voice_agent")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class logging_alias:
    def __init__(self, logger):
        self._logger = logger
        self.INFO = logging.INFO
        self.WARNING = logging.WARNING
        self.ERROR = logging.ERROR
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)
    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)
    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)
    def getLogger(self, name=None):
        return self._logger

logging = logging_alias(logger)

def is_goodbye(text: str) -> bool:
    text = text.lower().strip()
    # Remove punctuation
    for c in ",.?!;:-":
        text = text.replace(c, " ")
    words = text.split()
    goodbye_words = {"goodbye", "bye", "farewell"}
    if any(w in goodbye_words for w in words):
        return True
    phrases = ["have a great day", "have a nice day", "take care"]
    if any(p in text for p in phrases):
        return True
    return False


class AppointmentTools:
    def __init__(self, business_id: ObjectId):
        self.client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
        self.db = self.client["voice_agent"]
        self.appointments = self.db["appointments"]
        self.business_id = business_id

    @llm.function_tool
    async def check_availability(self, time: str) -> str:
        """Check if an appointment slot is available. Call this ONLY when the user explicitly specifies a date or time to check availability."""
        time = (time or "").strip()
        if not time or time.lower() in ["undefined", "null", "none", "placeholder", "anytime", "some time"]:
            return "Error: A specific date and time must be specified. Please ask the user for the exact time they want to check."

        existing = await self.appointments.find_one({"business_id": self.business_id, "time": time, "status": "scheduled"})
        if existing:
            return f"No, that slot at {time} is already booked."
        return f"Yes, the slot at {time} is available."

    @llm.function_tool
    async def book_appointment(self, name: str, phone: str, time: str, reason: str) -> str:
        """Book a new appointment. Call this ONLY when the user explicitly requests to book/schedule an appointment and has provided their details. You must check availability first."""
        name = (name or "").strip()
        phone = (phone or "").strip()
        time = (time or "").strip()
        reason = (reason or "").strip()

        missing = []
        if not name or name.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("name")
        if not phone or phone.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("phone number")
        if not time or time.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("time")
        if not reason or reason.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("reason for visit")

        if missing:
            return f"Error: Cannot book. The following details are missing or invalid: {', '.join(missing)}. Please politely ask the caller for these details."

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
        """Cancel an existing scheduled appointment. Call this ONLY when the user explicitly requests to cancel using their phone number and scheduled time."""
        phone = (phone or "").strip()
        time = (time or "").strip()

        missing = []
        if not phone or phone.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("phone number")
        if not time or time.lower() in ["undefined", "null", "none", "placeholder"]:
            missing.append("time")

        if missing:
            return f"Error: Cannot cancel. The following details are missing or invalid: {', '.join(missing)}. Please ask the caller for these details."

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
    vad_model = silero.VAD.load(
        min_silence_duration=0.3,
        min_speech_duration=0.1,
        prefix_padding_duration=0.2
    )
    stt_model = openai.STT(base_url="http://127.0.0.1:8000/v1", api_key="local", model="base.en")

    # Warm up the STT model in background to avoid latency on first user speech
    try:
        logging.info("Warming up STT model...")
        async def warmup_stt():
            try:
                import numpy as np
                from livekit.agents import AudioFrame
                # 100ms of silence at 16kHz
                data = np.zeros(1600, dtype=np.int16)
                frame = AudioFrame(data.tobytes(), 16000, 1, 1600)
                await stt_model.transcribe(buffer=[frame])
                logging.info("STT warmup complete.")
            except Exception as ex:
                logging.warning(f"STT warmup failed: {ex}")
        asyncio.create_task(warmup_stt())
    except Exception as e:
        logging.warning(f"Failed to schedule STT warmup: {e}")
    llm_model = openai.LLM(
        base_url="http://127.0.0.1:11434/v1",
        model="llama3.1:8b-instruct-q4_K_M",
        api_key="local",
        temperature=0.1,
        extra_body={
            "options": {
                "num_ctx": 2048,
                "num_predict": 100
            }
        }
    )

    # Warm up the LLM model in background to load it into memory
    try:
        logging.info("Warming up LLM model...")
        async def warmup_llm():
            try:
                from livekit.agents.llm import ChatMessage
                # Send a tiny query to trigger Ollama loading the Llama model into GPU RAM
                await llm_model.chat(history=[ChatMessage(role="user", content="hello")])
                logging.info("LLM warmup complete.")
            except Exception as ex:
                logging.warning(f"LLM warmup failed: {ex}")
        asyncio.create_task(warmup_llm())
    except Exception as e:
        logging.warning(f"Failed to schedule LLM warmup: {e}")
    
    tts_model = openai.TTS(
        base_url="http://127.0.0.1:10201/v1",
        api_key="not-needed",
        model="tts-1",
        voice=voice_model,
        response_format="wav"
    )

    # Warm up the TTS model with the business-specific voice to load it into memory
    try:
        logging.info(f"Warming up TTS model with voice '{voice_model}'...")
        async def warmup_tts():
            try:
                # Synthesize a tiny dummy space to force the TTS server to load/compile the voice model
                async for _ in tts_model.synthesize(" "):
                    pass
                logging.info("TTS warmup complete.")
            except Exception as ex:
                logging.warning(f"TTS warmup failed: {ex}")
        asyncio.create_task(warmup_tts())
    except Exception as e:
        logging.warning(f"Failed to schedule TTS warmup: {e}")
    
    from livekit.agents.llm import find_function_tools
    tools_list = []
    if "appointments" in skills:
        tools_list = find_function_tools(AppointmentTools(business_id))
        
    # Latency & Turn Handling configuration
    turn_handling = {
        "endpointing": {
            "min_delay": 0.3,  # Match VAD silence detection delay of 300ms
        },
        "preemptive_generation": {
            "enabled": True,
            "preemptive_tts": True,  # Synthesize audio chunks in parallel with LLM generation
        },
        "interruption": {
            "enabled": True,
        }
    }
        
    # Instantiating the validated Agent configuration layout
    assistant = Agent(
        vad=vad_model,
        stt=stt_model,
        llm=llm_model,
        tts=tts_model,
        tools=tools_list,
        instructions=agent_prompt,
        turn_handling=turn_handling
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
        turn_handling=turn_handling
    )

    async def save_to_mongo(dialog):
        try:
            await db["calls"].update_many(
                {"room_name": ctx.room.name, "status": "active"},
                {"$push": {"transcript": dialog}}
            )
        except Exception as e:
            logging.error(f"Error saving turn to MongoDB: {e}")

    user_said_goodbye = False

    # Transcript bindings tracking real-time user/agent turns via the session listeners
    @session.on("user_input_transcribed")
    def on_user_input(ev: UserInputTranscribedEvent):
        nonlocal user_said_goodbye
        if ev.is_final and ev.transcript.strip():
            text = ev.transcript.strip()
            dialog = {
                "role": "user",
                "text": str(text),
                "time": datetime.utcnow().isoformat()
            }
            logging.info(f"Dialog Turn [User]: {text}")
            asyncio.create_task(save_to_mongo(dialog))

            if is_goodbye(text):
                logging.info("User goodbye phrase detected.")
                user_said_goodbye = True

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
                
                # Auto-hangup when the agent speaks a goodbye phrase or user has already said goodbye
                if is_goodbye(text) or user_said_goodbye:
                    async def hangup_after_speech():
                        speech = session.current_speech
                        if speech:
                            try:
                                logging.info("Waiting for agent to finish speaking goodbye...")
                                await speech.wait_for_playout()
                            except Exception as e:
                                logging.warning(f"Error waiting for speech playout: {e}")
                        await asyncio.sleep(0.5)  # Let the final audio play out
                        logging.info("Goodbye sequence completed. Terminating room session.")
                        
                        try:
                            from livekit.api import LiveKitAPI
                            from livekit.protocol.room import DeleteRoomRequest
                            
                            url = os.environ.get("LIVEKIT_URL", "http://127.0.0.1:7800").replace("ws://", "http://").replace("wss://", "https://")
                            key = os.environ.get("LIVEKIT_API_KEY", "devkey1778495864")
                            secret = os.environ.get("LIVEKIT_API_SECRET", "devsecret1778495864")
                            
                            logging.info(f"Force deleting room {ctx.room.name} via Room Service API...")
                            async with LiveKitAPI(url=url, api_key=key, api_secret=secret) as lkapi:
                                await lkapi.room.delete_room(DeleteRoomRequest(room=ctx.room.name))
                            logging.info("Room deleted successfully.")
                        except Exception as ex:
                            logging.error(f"Failed to delete room via API: {ex}")
                        
                        ctx.shutdown()
                    asyncio.create_task(hangup_after_speech())

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