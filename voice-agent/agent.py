import os
# Pin VAD/ONNX runtime threads to prevent extreme thread contention and realtime delay
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import asyncio
import logging
import json
import httpx
from datetime import datetime
from bson import ObjectId
import re

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
    # Normalize German umlauts and Spanish accents
    text = text.replace("ö", "oe").replace("ä", "ae").replace("ü", "ue").replace("ß", "ss")
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    
    words = text.split()
    if not words:
        return False
        
    goodbye_words = {
        # English
        "goodbye", "bye", "farewell",
        # German
        "tschüss", "tschuess", "wiederhören", "wiederhoeren", "wiedersehen", "ciao",
        # Spanish
        "adios", "chao"
    }
    
    # Check if any goodbye word is present
    has_goodbye_word = any(w in goodbye_words for w in words)
    
    phrases = [
        # English
        "have a great day", "have a nice day", "take care",
        # German
        "schönen tag", "schoenen tag",
        # Spanish
        "hasta luego", "nos vemos", "buen día", "buen dia"
    ]
    has_goodbye_phrase = any(p in text for p in phrases)
    
    if not (has_goodbye_word or has_goodbye_phrase):
        return False
        
    # If it is a very short sentence (e.g. 1-3 words), it is highly likely to be a goodbye
    if len(words) <= 3:
        return True
        
    # Request-oriented intent keywords
    request_keywords = {
        "need", "help", "support", "transfer", "update", "question", 
        "book", "appointment", "cancel", "message", "check", "schedule", "billing"
    }
    
    # If it contains request keywords, it's likely a query (e.g. mistranscribed greeting or starting filler)
    if any(k in words for k in request_keywords):
        # Only count as goodbye if the goodbye word is the very last word of the sentence
        if words[-1] in goodbye_words:
            return True
        return False
        
    # If it doesn't contain request keywords, check if it ends with a goodbye word
    if words[-1] in goodbye_words:
        return True
        
    # Or if one of the goodbye phrases is at the end of the text
    for p in phrases:
        if text.endswith(p):
            return True
            
    # Default to False for long sentences unless explicitly ending in goodbye
    return False


def extract_greeting(prompt_str: str, business_name: str) -> str:
    normalized = prompt_str.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    
    # Try double quotes first
    match = re.search(r'say\s*:\s*\n*\s*"([^"]+)"', normalized, re.IGNORECASE)
    if match:
        greeting = match.group(1).strip()
        greeting = greeting.replace("[User Name]", business_name).replace("[Business Name]", business_name)
        return greeting
        
    # Try single quotes fallback
    match = re.search(r"say\s*:\s*\n*\s*'([^']+)'", normalized, re.IGNORECASE)
    if match:
        greeting = match.group(1).strip()
        greeting = greeting.replace("[User Name]", business_name).replace("[Business Name]", business_name)
        return greeting
        
    return f"Hello! Thank you for calling {business_name}. How can I help you today?"


def is_identity_question(text: str) -> bool:
    text = text.lower().strip()
    # Remove punctuation
    for c in ",.?!;:-'\"":
        text = text.replace(c, " ")
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    patterns = [
        r"\bwho\s+(are\s+you|is\s+this)\b",
        r"\bwho\s+am\s+i\s+(speaking|talking)\s+to\b",
        r"\bwhat\s+is\s+your\s+name\b",
        r"\bwhat\s+s\s+your\s+name\b",
        r"\bare\s+you\s+a\s+(real\s+person|robot|ai|bot|human)\b",
        r"\bwho\s+are\s+you\s+representing\b",
        r"\bidentify\s+yourself\b",
    ]
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False


class FilteredAgent(Agent):
    def tts_node(self, text, model_settings):
        async def filtered_text_generator():
            paren_level = 0
            bracket_level = 0
            brace_level = 0
            async for chunk in text:
                cleaned_chunk = ""
                for char in chunk:
                    if char == '(':
                        paren_level += 1
                    elif char == ')':
                        paren_level = max(0, paren_level - 1)
                    elif char == '[':
                        bracket_level += 1
                    elif char == ']':
                        bracket_level = max(0, bracket_level - 1)
                    elif char == '{':
                        brace_level += 1
                    elif char == '}':
                        brace_level = max(0, brace_level - 1)
                    else:
                        if paren_level == 0 and bracket_level == 0 and brace_level == 0:
                            cleaned_chunk += char
                if cleaned_chunk:
                    yield cleaned_chunk
                
        return super().tts_node(filtered_text_generator(), model_settings)

    def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings,
    ):
        # Merge consecutive user/assistant messages to prevent Ollama from failing tool-calling
        cleaned_items = []
        for item in chat_ctx.items:
            if (
                cleaned_items
                and isinstance(item, llm.ChatMessage)
                and item.role in ("user", "assistant")
                and isinstance(cleaned_items[-1], llm.ChatMessage)
                and cleaned_items[-1].role == item.role
            ):
                prev_item = cleaned_items[-1]
                prev_text = prev_item.text_content or ""
                curr_text = item.text_content or ""
                merged_text = prev_text + "\n" + curr_text
                # Use model_copy to preserve all Pydantic fields/private attributes
                cleaned_items[-1] = prev_item.model_copy(update={"content": [merged_text]})
            else:
                cleaned_items.append(item)
                
        cleaned_ctx = llm.ChatContext(cleaned_items)
        
        # Check if the last user message was an identity question
        last_user_msg = None
        for item in reversed(cleaned_items):
            if isinstance(item, llm.ChatMessage) and item.role == "user":
                last_user_msg = item.text_content
                break
        
        active_tools = tools
        if last_user_msg and is_identity_question(last_user_msg):
            logging.info(f"Identity question detected: '{last_user_msg}'. Temporarily clearing tools to prevent hallucinated tool calls.")
            active_tools = []
        
        # Log LLM inputs to debug raw JSON leaks or context issues
        logging.info("--- LLM Node Input ChatContext ---")
        for idx, item in enumerate(cleaned_items):
            if isinstance(item, llm.ChatMessage):
                logging.info(f"[{idx}] Role: {item.role}, Content: {item.text_content}")
            else:
                logging.info(f"[{idx}] Type: {item.type}, ID: {item.id}, Data: {item}")
        logging.info(f"Tools available: {[t.info.name for t in active_tools]}")
        logging.info("----------------------------------")
        
        return super().llm_node(cleaned_ctx, active_tools, model_settings)


class AppointmentTools:
    def __init__(self, business_id: ObjectId, room_name: str, room, session_holder: dict = None):
        self.client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
        self.db = self.client["voice_agent"]
        self.appointments = self.db["appointments"]
        self.business_id = business_id
        self.room_name = room_name
        self.room = room
        self.session_holder = session_holder or {}

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

    @llm.function_tool
    async def transfer_call(self, extension: str, confirmed: bool = False) -> str:
        """Transfer the call to an internal extension on the PBX. Call this ONLY when the user explicitly asks to be transferred, put on hold, or connected to another person, department, or technical support.
        
        Args:
            extension: The extension number or department to transfer to.
            confirmed: Set to True if the caller has explicitly confirmed they want to be transferred. If they have not explicitly confirmed yet, this must be False.
        """
        extension = (extension or "").strip()
        if not extension:
            return "Error: Please specify the extension number to transfer to."
        
        # Validate that the extension is a numeric string (digits only)
        clean_ext = re.sub(r'[^0-9]', '', extension)
        if not clean_ext or len(clean_ext) < 2:
            return (
                f"Error: '{extension}' is not a valid extension. Extensions must be numeric digits (e.g., '100', '101', '502'). "
                "You are strictly forbidden from transferring to names like 'Synthia' or 'Shadikur'. Please politely explain this to the caller."
            )
        
        if not confirmed:
            return (
                "Error: You must first ask the caller for confirmation before transferring them. "
                "Please politely ask: 'Would you like me to transfer you to our Technical Support (or Billing) team?' "
                "and wait for their confirmation response. Do not perform the transfer yet."
            )
        
        # Find the SIP participant in the room and resolve host
        participant_identity = None
        sip_hostname = "199.47.47.106" # Default fallback hostname
        for identity, participant in self.room.remote_participants.items():
            if "sip.phoneNumber" in participant.attributes:
                participant_identity = identity
                sip_hostname = participant.attributes.get("sip.hostname", sip_hostname)
                break
        
        if not participant_identity:
            # Fall back to any remote participant
            if self.room.remote_participants:
                participant_identity = list(self.room.remote_participants.keys())[0]
                p = self.room.remote_participants[participant_identity]
                sip_hostname = p.attributes.get("sip.hostname", sip_hostname)
                
        if not participant_identity:
            return "Error: No active participant found in the room to transfer."
            
        # Ensure it is a valid SIP or TEL URI, appending hostname to SIP URIs if missing
        if not (extension.startswith("sip:") or extension.startswith("sips:") or extension.startswith("tel:")):
            transfer_to = f"sip:{extension}@{sip_hostname}"
        else:
            transfer_to = extension
            if (transfer_to.startswith("sip:") or transfer_to.startswith("sips:")) and "@" not in transfer_to:
                transfer_to = f"{transfer_to}@{sip_hostname}"
            
        try:
            # Wait for any currently queued/speaking assistant speech to complete playout
            session = self.session_holder.get("session")
            if session:
                speech = session.current_speech
                if speech:
                    try:
                        logging.info("Waiting for agent to finish speaking before triggering SIP transfer...")
                        await asyncio.wait_for(speech.wait_for_playout(), timeout=4.0)
                    except Exception as e:
                        logging.warning(f"Timeout/error waiting for speech playout before transfer: {e}")
            
            from livekit.api import LiveKitAPI
            from livekit.protocol.sip import TransferSIPParticipantRequest
            
            url = os.environ.get("LIVEKIT_URL", "http://127.0.0.1:7800").replace("ws://", "http://").replace("wss://", "https://")
            key = os.environ.get("LIVEKIT_API_KEY", "devkey1778495864")
            secret = os.environ.get("LIVEKIT_API_SECRET", "devsecret1778495864")
            
            logging.info(f"Transferring participant {participant_identity} in room {self.room_name} to URI {transfer_to}...")
            
            async with LiveKitAPI(url=url, api_key=key, api_secret=secret) as lkapi:
                await lkapi.sip.transfer_sip_participant(
                    TransferSIPParticipantRequest(
                        room_name=self.room_name,
                        participant_identity=participant_identity,
                        transfer_to=transfer_to
                    )
                )
            return f"Success! Transferring the call to extension {extension}. Goodbye!"
        except Exception as e:
            logging.error(f"Failed to transfer call: {e}")
            return f"Error: Failed to perform transfer: {str(e)}"

    @llm.function_tool
    async def take_message(
        self,
        caller_name: str,
        phone_number: str,
        reason_for_call: str,
        company: str = "None",
        urgency: str = "False",
        best_callback_time: str = "None"
    ) -> str:
        """Take a message for the user when they are unavailable. Call this ONLY when the caller explicitly asks to leave a message, take a message, or pass on a message/note.
        
        Args:
            caller_name: The name of the person calling.
            phone_number: The callback phone number.
            reason_for_call: The detailed message or reason they are calling.
            company: The company or organization they represent, if any.
            urgency: Whether the message is urgent (e.g. 'True' or 'False').
            best_callback_time: Best time to call back, if specified.
        """
        caller_name = (caller_name or "").strip()
        phone_number = (phone_number or "").strip()
        reason_for_call = (reason_for_call or "").strip()
        company = (company or "").strip()
        urgency = (urgency or "").strip()
        best_callback_time = (best_callback_time or "").strip()
        
        # Check for placeholders or missing values
        missing = []
        is_name_placeholder = not caller_name or any(p in caller_name.lower() for p in ["unknown", "not provided", "placeholder", "undefined", "null", "none", "your", "name", "caller", "user"])
        
        # Phone number must contain at least 5 digits and not contain placeholder words
        digits_count = sum(c.isdigit() for c in phone_number)
        is_phone_placeholder = (
            not phone_number 
            or digits_count < 5 
            or any(p in phone_number.lower() for p in ["unknown", "not", "provided", "placeholder", "undefined", "null", "none", "your", "caller", "web", "sandbox", "phone", "number"])
        )
        
        is_reason_placeholder = (
            not reason_for_call 
            or len(reason_for_call.strip()) < 5
            or any(p in reason_for_call.lower() for p in ["unknown", "not provided", "placeholder", "undefined", "null", "none", "your", "reason"])
        )
        
        if is_name_placeholder:
            missing.append("caller name")
        if is_phone_placeholder:
            missing.append("caller phone number")
        if is_reason_placeholder:
            missing.append("detailed message content")
            
        if missing:
            return (
                f"Error: Cannot record message. The following required details are missing, invalid, or placeholders: {', '.join(missing)}. "
                "Please politely ask the caller for these details (e.g., 'May I have your name and phone number please?') "
                "before calling this tool."
            )
            
        message_doc = {
            "business_id": self.business_id,
            "room_name": self.room_name,
            "caller_name": caller_name,
            "phone_number": phone_number,
            "reason_for_call": reason_for_call,
            "company": company,
            "urgency": urgency,
            "best_callback_time": best_callback_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await self.db["messages"].insert_one(message_doc)
            logging.info(f"Successfully recorded message from {caller_name} in MongoDB.")
            
            # Post the message immediately to the Teams Webhook in the background
            async def send_message_webhook_bg():
                try:
                    latest_biz = await self.db["businesses"].find_one({"_id": self.business_id})
                    if latest_biz and latest_biz.get("teams_webhook_url"):
                        webhook_url = latest_biz["teams_webhook_url"]
                        logging.info(f"Teams webhook configured for messages: {webhook_url}. Posting message in background...")
                        
                        payload = {
                            "@type": "MessageCard",
                            "@context": "http://schema.org/extensions",
                            "themeColor": "FF007F",
                            "summary": f"New Message: {caller_name}",
                            "title": "📝 New Message Taken",
                            "sections": [
                                {
                                    "activityTitle": f"Recipient: **{latest_biz.get('name', 'User')}**",
                                    "activitySubtitle": f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                                    "facts": [
                                        {"name": "Caller Name", "value": caller_name},
                                        {"name": "Company", "value": company},
                                        {"name": "Phone Number", "value": phone_number},
                                        {"name": "Urgency", "value": urgency},
                                        {"name": "Best Callback Time", "value": best_callback_time}
                                    ],
                                    "markdown": True
                                },
                                {
                                    "startGroup": True,
                                    "title": "Message Content",
                                    "text": reason_for_call
                                }
                            ]
                        }
                        
                        async with httpx.AsyncClient() as http_client:
                            resp = await http_client.post(webhook_url, json=payload, timeout=10.0)
                            if resp.status_code >= 400:
                                logging.error(f"Teams message webhook returned error {resp.status_code}: {resp.text}")
                            else:
                                logging.info("Successfully posted message to Microsoft Teams (background).")
                except Exception as w_err:
                    logging.error(f"Error in background message webhook task: {w_err}")
            
            asyncio.create_task(send_message_webhook_bg())
            
        except Exception as ex:
            logging.error(f"Error saving or sending message: {ex}")
            
        return "Success: The message has been recorded and sent to the user."




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

    # Start background line hum/static noise to simulate real human telephone connection
    try:
        from livekit import rtc
        import numpy as np
        
        noise_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
        noise_track = rtc.LocalAudioTrack.create_audio_track("background_noise", noise_source)
        await ctx.room.local_participant.publish_track(noise_track)
        
        async def generate_noise():
            sample_rate = 16000
            frame_duration = 0.02
            num_samples = int(sample_rate * frame_duration)
            amplitude = 40.0  # Soft hum
            state = 0.0
            
            try:
                while True:
                    samples = np.zeros(num_samples, dtype=np.int16)
                    for i in range(num_samples):
                        white = np.random.uniform(-1.0, 1.0)
                        state = 0.9 * state + 0.1 * white
                        samples[i] = int(state * amplitude)
                    
                    frame = rtc.AudioFrame(
                        data=samples.tobytes(),
                        sample_rate=sample_rate,
                        num_channels=1,
                        samples_per_channel=num_samples
                    )
                    await noise_source.capture_frame(frame)
                    await asyncio.sleep(frame_duration)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"Error in noise generation: {e}")
                
        noise_task = asyncio.create_task(generate_noise())
        
        async def cancel_noise():
            noise_task.cancel()
            try:
                await noise_task
            except:
                pass
        ctx.add_shutdown_callback(cancel_noise)
        logging.info("Background noise generation successfully initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize background noise: {e}")
    
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
        db_prompt = business["prompt"]
        voice_model = business.get("voice", "af_bella")
        skills = business.get("skills", ["appointments"])
        business_id = business["_id"]
        
        # Extract starting greeting from the raw prompt first
        greeting_text = extract_greeting(db_prompt, agent_name)
        
        # Clean conflicting instructions and goodbye endings from custom prompt
        sanitized_prompt = db_prompt.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        sanitized_prompt = re.sub(r'(?i)##\s+Ending\s+the\s+Call\b.*', '', sanitized_prompt, flags=re.DOTALL)
        sanitized_prompt = re.sub(r'(?i)After\s+collecting\s+the\s+message,\s*say\s*:\s*\n*\s*"[^"]*"', '', sanitized_prompt)
        sanitized_prompt = re.sub(r'(?i)After\s+collecting\s+the\s+message,\s*say\s*:\s*\n*\s*\'[^\']*\'', '', sanitized_prompt)
        sanitized_prompt = sanitized_prompt.replace('"Thank you for calling. I\'ll pass on your message."', '')
        sanitized_prompt = sanitized_prompt.replace('"Thank you. I\'ll pass your message to Shadikur."', '')
        sanitized_prompt = sanitized_prompt.replace('"Thank you. Have a nice day."', '')
        
        # Prepend critical overriding rules
        agent_prompt = (
            "CRITICAL PROTOCOL (ABSOLUTE PRECEDENCE OVER ALL OTHER RULES):\n"
            "1. You are strictly FORBIDDEN from ending the call, saying goodbye, or promising to pass on a message/note/callback "
            "unless you have first successfully executed the 'take_message' tool and it has returned a success message.\n"
            "2. If the caller asks for a callback, to be called back, or to leave a message, you MUST ask them for their name "
            "and their digit-based phone number. You must ask: 'What is the best phone number for Shadikur to call you back at?' "
            "if they ask for a callback.\n"
            "3. You are strictly FORBIDDEN from fabricating, guessing, or making up any names, phone numbers, or details that were not explicitly spoken by the caller. "
            "If the caller has not explicitly spoken their phone number, you DO NOT have it, and you MUST ask them for it first before calling any tool.\n"
            "4. If the caller asks 'Who are you?', 'What is your name?', or similar identity questions, you MUST directly answer who you are (e.g. 'I'm Shadikur's phone assistant Synthia') "
            "and do NOT call any tools or take a message in that turn.\n\n"
        ) + sanitized_prompt
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
        greeting_text = extract_greeting(agent_prompt, agent_name)
    voice_lower = voice_model.lower()
    if voice_lower.startswith("de") or voice_lower == "german":
        if greeting_text.startswith("Hello! Thank you for calling"):
            greeting_text = f"Hallo! Vielen Dank für Ihren Anruf bei {agent_name}. Wie kann ich Ihnen heute helfen?"
        agent_prompt += (
            "\n\nCRITICAL: You must speak and respond ENTIRELY in German (Deutsch). Do not use English under any circumstances. "
            "Ensure all responses are written in German. Even if the user speaks in English, translate it to German and respond in German."
        )
    elif voice_lower.startswith("es") or voice_lower in ["ef_dora", "em_alex"]:
        if greeting_text.startswith("Hello! Thank you for calling"):
            greeting_text = f"¡Hola! Gracias por llamar a {agent_name}. ¿Cómo le puedo ayudar hoy?"
        agent_prompt += (
            "\n\nCRITICAL: You must speak and respond ENTIRELY in Spanish (Español). Do not use English under any circumstances. "
            "Ensure all responses are written in Spanish. Even if the user speaks in English, translate it to Spanish and respond in Spanish."
        )
        
    # Format and Output Constraints
    agent_prompt += (
        "\n\nCRITICAL formatting and capability rules:\n"
        "1. Never output any parenthetical notes, chain-of-thought, or internal justifications (like '(Note: ...)' or similar). Do not write notes like (Note: Since the caller didn't request any booking action, no function call is made).\n"
        "2. Only output the direct words you would speak to the caller.\n"
        "3. Keep answers very short, concise, and natural (one or two sentences).\n"
        "4. You are strictly FORBIDDEN from inventing, hallucinating, or disclosing details (such as prices, services, policies, guarantees, employee names, or specific business options) that are NOT explicitly mentioned in your custom instructions. If the caller asks for information not present in your instructions, you must politely state that you do not have that information and offer to connect them or take a message.\n"
    )

    prompt_lower = agent_prompt.lower()
    if "transfer" in prompt_lower or "extension" in prompt_lower:
        agent_prompt += (
            "4. You have the ability to transfer calls to internal extensions using the transfer_call tool. "
            "If the user asks to be transferred or speak to a human, you must first ask for their confirmation "
            "(e.g. 'Would you like me to transfer you?'). You must ONLY call transfer_call (with confirmed=True) "
            "after the caller has explicitly confirmed they want to be transferred. If they have not explicitly confirmed yet, "
            "do NOT call the transfer_call tool; instead, respond directly in conversation asking for their confirmation.\n"
        )

    if "appointments" in skills:
        agent_prompt += (
            "\n\nAppointments Booking Rules:\n"
            "- You MUST call the check_availability tool to verify a slot before booking.\n"
            "- You MUST call the book_appointment tool to officially book an appointment.\n"
            "- You are strictly FORBIDDEN from booking or saying an appointment is scheduled/confirmed/booked unless you have first collected the caller's Name and Phone Number.\n"
            "- If the caller's Name or Phone Number is missing, you MUST ask the caller for them first.\n"
            "- Do NOT call the book_appointment tool if the name, phone number, or time is missing. Instead, respond directly in conversation to ask the caller for them.\n"
            "- Do NOT tell the user their appointment is booked or scheduled until the book_appointment tool has been executed and returned a success message.\n"
        )
        
    # Unconditionally include message taking rules
    agent_prompt += (
        "\n\nMessage Taking Rules:\n"
        "- You MUST call the take_message tool to record a message when the caller wants to leave a message, note, or when the user is unavailable.\n"
        "- You are strictly FORBIDDEN from calling the take_message tool or saying you will pass on the message/note/goodbye unless you have first collected the caller's Name, Phone Number, and their detailed message.\n"
        "- If the caller's Name, Phone Number, or detailed message is missing, you MUST ask the caller for them first.\n"
        "- Do NOT call the take_message tool if the caller's name, phone number, or detailed message is missing or placeholders. Instead, respond directly in conversation to ask the caller for the missing information.\n"
        "- If the caller asks you to 'call me back' or leave a call back note, you MUST ask them: 'What is the best phone number for Shadikur to call you back at?' and you are FORBIDDEN from using placeholders or saying goodbye without asking for their specific digits-based callback phone number first.\n"
        "- Do NOT tell the caller that you will pass on their message, leave a note, or say goodbye until the take_message tool has been executed and returned a success message.\n"
        "- Do NOT call the take_message tool with placeholder values like 'Unknown', 'Unknown Caller', 'Not Provided', 'Your Phone Number', 'web-sandbox', or 'None'.\n"
    )
        
    # Initialize Core Model Engines
    vad_model = silero.VAD.load(
        min_silence_duration=0.3,
        min_speech_duration=0.1,
        prefix_padding_duration=0.2
    )
    stt_lang = "en"
    if voice_lower.startswith("de") or voice_lower == "german":
        stt_lang = "de"
    elif voice_lower.startswith("es") or voice_lower in ["ef_dora", "em_alex"]:
        stt_lang = "es"
        
    stt_model = openai.STT(
        base_url="http://127.0.0.1:8000/v1",
        api_key="local",
        model="small",
        language=stt_lang
    )

    # Warm up the STT model in background to avoid latency on first user speech
    try:
        logging.info("Warming up STT model...")
        async def warmup_stt():
            try:
                import numpy as np
                from livekit.rtc import AudioFrame
                # 100ms of silence at 16kHz
                data = np.zeros(1600, dtype=np.int16)
                frame = AudioFrame(data.tobytes(), 16000, 1, 1600)
                await stt_model.recognize(buffer=[frame])
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
        _strict_tool_schema=False,
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
                from livekit.agents.llm import ChatContext
                chat_ctx = ChatContext()
                chat_ctx.add_message(role="user", content="hello")
                async with llm_model.chat(chat_ctx=chat_ctx) as stream:
                    async for chunk in stream:
                        pass
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
    
    session_holder = {}
    from livekit.agents.llm import find_function_tools
    all_tools = find_function_tools(AppointmentTools(business_id, ctx.room.name, ctx.room, session_holder))
    tools_list = []
    
    # Always include take_message tool as it is a base receptionist capability
    take_msg_tool = next((t for t in all_tools if t.info.name == "take_message"), None)
    if take_msg_tool:
        tools_list.append(take_msg_tool)
        
    if "appointments" in skills:
        # Include appointment tools
        tools_list.extend([t for t in all_tools if t.info.name in ["check_availability", "book_appointment", "cancel_appointment"]])
        
    # Expose call transfer ONLY if the custom prompt explicitly mentions transfer / extension
    if "transfer" in prompt_lower or "extension" in prompt_lower:
        transfer_tool = next((t for t in all_tools if t.info.name == "transfer_call"), None)
        if transfer_tool:
            tools_list.append(transfer_tool)
        
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
    assistant = FilteredAgent(
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
    session_holder["session"] = session

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
            
    async def send_teams_summary(webhook_url: str, transcript: list, duration_seconds: float):
        try:
            if not transcript:
                summary_text = "No conversation recorded (call was disconnected immediately)."
            else:
                # Format transcript for LLM
                formatted_transcript = ""
                for turn in transcript:
                    role_name = "Caller" if turn["role"] == "user" else "AI Assistant"
                    formatted_transcript += f"{role_name}: {turn['text']}\n"
                
                summary_prompt = (
                    "You are a helpful assistant. Below is the transcript of a voice call. "
                    "Provide a concise summary of the call (2-3 sentences), highlighting any actions taken (e.g. appointments booked or cancelled, or main user inquiry). "
                    "Do not output anything else but the summary."
                )
                
                from livekit.agents.llm import ChatContext
                chat_ctx = ChatContext()
                chat_ctx.add_message(role="system", content=summary_prompt)
                chat_ctx.add_message(role="user", content=f"Call Transcript:\n{formatted_transcript}")
                
                logging.info("Requesting call summary from Ollama...")
                chunks = []
                async with llm_model.chat(chat_ctx=chat_ctx) as stream:
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            chunks.append(chunk.delta.content)
                summary_text = "".join(chunks).strip()
            
            logging.info(f"Generated summary: {summary_text}")
            
            # Send to Microsoft Teams via Webhook (MessageCard format)
            duration_mins = int(duration_seconds // 60)
            duration_secs = int(duration_seconds % 60)
            duration_str = f"{duration_mins:02d}:{duration_secs:02d}"
            
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "00F2FE",
                "summary": f"Voice Call Summary: {agent_name}",
                "title": "📞 Voice Call Session Completed",
                "sections": [
                    {
                        "activityTitle": f"Agent Profile: **{agent_name}** (Ext {extension})",
                        "activitySubtitle": f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                        "facts": [
                            {"name": "Caller Number", "value": call_record.get("phone_number", "web-sandbox")},
                            {"name": "Duration", "value": duration_str},
                            {"name": "Status", "value": "Completed"}
                        ],
                        "markdown": True
                    },
                    {
                        "startGroup": True,
                        "title": "Call Summary",
                        "text": summary_text
                    }
                ]
            }
            
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(webhook_url, json=payload, timeout=15.0)
                if resp.status_code >= 400:
                    logging.error(f"Teams webhook returned error {resp.status_code}: {resp.text}")
                else:
                    logging.info("Successfully posted call summary to Microsoft Teams.")
        except Exception as ex:
            logging.error(f"Failed to generate or send Teams summary: {ex}")

    async def close_call_record():
        try:
            end_time = datetime.utcnow()
            start_time_parsed = datetime.fromisoformat(call_record["start_time"])
            duration = (end_time - start_time_parsed).total_seconds()
            
            # Fetch the updated transcript from the DB
            updated_call = await db["calls"].find_one({"room_name": ctx.room.name})
            transcript = updated_call.get("transcript", []) if updated_call else []
            
            await db["calls"].update_many(
                {"room_name": ctx.room.name, "status": "active"},
                {"$set": {
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "status": "completed"
                }}
            )
            logging.info(f"Archived call history in MongoDB for room {ctx.room.name}. Duration: {duration}s")
            
            # Trigger Teams Webhook if configured
            latest_biz = await db["businesses"].find_one({"_id": business_id})
            if latest_biz and latest_biz.get("teams_webhook_url"):
                webhook_url = latest_biz["teams_webhook_url"]
                logging.info(f"Teams webhook configured: {webhook_url}. Sending summary post...")
                await send_teams_summary(webhook_url, transcript, duration)
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
    session.say(greeting_text, allow_interruptions=True)

    await session.wait_for_inactive()


if __name__ == '__main__':
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,
        agent_name=""
    ))