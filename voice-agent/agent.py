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
            
        # Prevent tool call loops on validation errors (e.g. missing phone/email)
        disabled_tools = set()
        last_tool_outputs = {}
        for item in cleaned_items:
            item_type = getattr(item, "type", None)
            if item_type == "function_call_output":
                t_name = getattr(item, "name", None)
                t_out = getattr(item, "output", None)
                if t_name and t_out:
                    last_tool_outputs[t_name] = str(t_out)
                    
        # Check if the user has provided updates in the latest user message
        has_new_phone = False
        has_new_email = False
        if last_user_msg:
            if re.search(r'\d{3,}', last_user_msg):
                has_new_phone = True
            if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', last_user_msg):
                has_new_email = True
                
        # Disable tool if the last execution returned a validation error and no updates were received
        for t_name, t_out in last_tool_outputs.items():
            if "error" in t_out.lower() or "cannot" in t_out.lower():
                if t_name in ("take_message", "book_appointment", "cancel_appointment"):
                    if "phone" in t_out.lower() and not has_new_phone:
                        logging.info(f"Disabling tool '{t_name}' due to validation error (missing phone) and no updates in user response.")
                        disabled_tools.add(t_name)
                elif t_name == "create_support_ticket":
                    if "email" in t_out.lower() and not has_new_email:
                        logging.info(f"Disabling tool '{t_name}' due to validation error (missing email) and no updates in user response.")
                        disabled_tools.add(t_name)
                    elif "phone" in t_out.lower() and not has_new_phone:
                        logging.info(f"Disabling tool '{t_name}' due to validation error (missing phone) and no updates in user response.")
                        disabled_tools.add(t_name)
                # CRITICAL: If transfer_call just returned an error, disable it for this turn.
                # This prevents the LLM from retrying with a guessed/hallucinated extension number.
                elif t_name == "transfer_call":
                    logging.info("Disabling 'transfer_call' for this turn: last attempt returned an error. Preventing hallucinated retry.")
                    disabled_tools.add("transfer_call")
                        
        if disabled_tools:
            active_tools = [t for t in active_tools if t.info.name not in disabled_tools]

        # TRANSFER CONFIRMATION GATE
        # The LLM must NOT call transfer_call unless the caller has explicitly confirmed they
        # want to be transferred. Gate this at the code level to prevent hallucinated transfers
        # triggered by ambiguous questions like "Can I speak to Edel?".
        transfer_confirmation_keywords = [
            # English
            "yes", "sure", "please", "go ahead", "transfer", "connect me", "put me through",
            "yes please", "ok", "okay", "yeah",
            # Spanish
            "sí", "si", "por favor", "claro", "adelante", "transfiera", "conéctame",
            # German
            "ja", "bitte", "weiterleiten", "verbinden"
        ]
        if last_user_msg and any(t.info.name == "transfer_call" for t in active_tools):
            msg_lower = last_user_msg.lower()
            has_transfer_confirmation = any(kw in msg_lower for kw in transfer_confirmation_keywords)
            if not has_transfer_confirmation:
                logging.info(
                    f"Transfer confirmation gate: blocking 'transfer_call' — "
                    f"no explicit confirmation found in: '{last_user_msg}'"
                )
                active_tools = [t for t in active_tools if t.info.name != "transfer_call"]

        # Detect if this turn is responding to a tool result — if so, increase num_predict
        # so the LLM has enough tokens to generate a spoken follow-up (avoids silent hangup).
        last_item_is_tool_result = (
            cleaned_items
            and getattr(cleaned_items[-1], "type", None) == "function_call_output"
        )
        effective_model_settings = model_settings
        if last_item_is_tool_result:
            logging.info("Last context item is a tool result — increasing num_predict to 200 for follow-up response.")
            try:
                # model_settings may be a dataclass/object; try to copy and patch it
                import copy
                effective_model_settings = copy.copy(model_settings)
                # Patch extra_body options if accessible
                if hasattr(effective_model_settings, "extra_body") and isinstance(effective_model_settings.extra_body, dict):
                    eb = copy.deepcopy(effective_model_settings.extra_body)
                    eb.setdefault("options", {})
                    eb["options"]["num_predict"] = 200
                    effective_model_settings.extra_body = eb
            except Exception as ms_err:
                logging.warning(f"Could not patch model_settings num_predict: {ms_err}")
                effective_model_settings = model_settings
        
        # Log LLM inputs to debug raw JSON leaks or context issues
        logging.info("--- LLM Node Input ChatContext ---")
        for idx, item in enumerate(cleaned_items):
            if isinstance(item, llm.ChatMessage):
                logging.info(f"[{idx}] Role: {item.role}, Content: {item.text_content}")
            else:
                logging.info(f"[{idx}] Type: {item.type}, ID: {item.id}, Data: {item}")
        logging.info(f"Tools available: {[t.info.name for t in active_tools]}")
        logging.info("----------------------------------")
        
        return super().llm_node(cleaned_ctx, active_tools, effective_model_settings)


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
    async def transfer_call(self, extension: str) -> str:
        """Transfer the call to an internal extension on the PBX. Call this ONLY when the user explicitly requests to be transferred, and they have already confirmed they want to be transferred. You are strictly forbidden from calling this tool unless the transfer is confirmed.
        
        Args:
            extension: The extension number or department to transfer to.
        """
        extension = (extension or "").strip()
        if not extension:
            return "Error: Please specify the extension number to transfer to."
        
        # Validate that the extension is a numeric string (digits only)
        clean_ext = re.sub(r'[^0-9]', '', extension)
        if not clean_ext or len(clean_ext) < 2:
            return (
                f"Error: '{extension}' is not a valid extension. Extensions must be numeric digits (e.g., '100', '101', '502'). "
                "You are strictly forbidden from transferring to contact names or assistant names. You must transfer using numeric digits. Please politely explain this to the caller."
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

    @llm.function_tool
    async def create_support_ticket(
        self,
        subject: str,
        name: str,
        email: str,
        message: str,
        department: str = "4",
        priority: str = "2"
    ) -> str:
        """Create a support ticket in the CRM.

        STRICT RULES — you MUST follow all of these before calling this tool:
        1. The caller must have EXPLICITLY asked to 'create a ticket', 'submit a ticket',
           'report an issue' as a formal request — NOT just mentioning a problem in passing.
        2. You MUST have collected ALL of the following DIRECTLY from the caller in this
           conversation (do NOT assume, invent, or reuse values from previous turns):
           - Their full name (caller said it themselves)
           - A valid email address (caller said it themselves)
           - A clear subject/summary
           - A detailed description of the issue
        3. If ANY detail is missing, ask the caller for it first. Do NOT call this tool
           with placeholder, assumed, or hallucinated values.

        Args:
            subject: Short summary of the ticket (from caller).
            name: The caller's full name as they stated it.
            email: The caller's email address as they stated it.
            message: Detailed description of the issue (from caller).
            department: Department ID ('1'=Admin, '4'=Tech Support, '5'=Quotation). Default '4'.
            priority: Priority level ('1'=Low, '2'=Medium, '3'=High). Default '2'.
        """
        subject = (subject or "").strip()
        name = (name or "").strip()
        email = (email or "").strip()
        message = (message or "").strip()
        department = (department or "4").strip()
        priority = (priority or "2").strip()

        missing = []
        is_name_placeholder = not name or any(p in name.lower() for p in ["unknown", "not provided", "placeholder", "undefined", "null", "none", "your", "name", "caller", "user"])
        is_email_placeholder = not email or "@" not in email or "." not in email or any(p in email.lower() for p in ["unknown", "placeholder", "undefined", "null", "none", "your", "email"])
        is_subject_placeholder = not subject or any(p in subject.lower() for p in ["unknown", "placeholder", "undefined", "null", "none", "your", "subject"])
        is_message_placeholder = not message or len(message) < 5 or any(p in message.lower() for p in ["unknown", "placeholder", "undefined", "null", "none", "your", "message"])

        if is_name_placeholder:
            missing.append("your name")
        if is_email_placeholder:
            missing.append("valid email address")
        if is_subject_placeholder:
            missing.append("ticket subject")
        if is_message_placeholder:
            missing.append("detailed description of the issue")

        if missing:
            return (
                f"Error: Cannot create ticket. The following details are missing, invalid, or placeholders: {', '.join(missing)}. "
                "Please politely ask the caller for these missing details before submitting the ticket."
            )

        form_url = "https://mis.compusource.net/forms/ticket?styled=1&with_logo=1"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        
        try:
            async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
                logging.info("create_support_ticket: Fetching CSRF token from CRM...")
                resp = await client.get(form_url)
                if resp.status_code != 200:
                    return f"Error: Failed to reach the CRM ticket form (Status: {resp.status_code})."
                
                # Extract CSRF token
                match = re.search(r'name="csrf_token_name"\s+value="([a-f0-9]+)"', resp.text)
                if not match:
                    match = re.search(r'value="([a-f0-9]+)"\s+name="csrf_token_name"', resp.text)
                if not match:
                    return "Error: Failed to parse security verification token from the ticket portal."
                
                csrf_token = match.group(1)
                
                # Prepare payload
                data = {
                    "csrf_token_name": csrf_token,
                    "subject": subject,
                    "name": name,
                    "email": email,
                    "department": department,
                    "priority": priority,
                    "message": message
                }
                
                # We must send it as multipart/form-data to succeed, so we use files parameter
                files = {
                    "attachments[]": ("", b"", "application/octet-stream")
                }
                
                post_headers = {
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": form_url
                }
                
                logging.info(f"create_support_ticket: Submitting ticket POST for {name} ({email})...")
                post_resp = await client.post(form_url, data=data, files=files, headers=post_headers)
                
                if post_resp.status_code == 200:
                    try:
                        res_json = post_resp.json()
                        if res_json.get("success") is True:
                            msg = res_json.get("message", "Ticket submitted successfully.")
                            return f"Success! {msg}"
                        else:
                            err_msg = res_json.get("message", "Unknown error returned by CRM.")
                            return f"Error from ticket portal: {err_msg}"
                    except Exception:
                        if "success" in post_resp.text.lower() or "thank you" in post_resp.text.lower():
                            return "Success! Support ticket has been created in the CRM."
                        return f"Error: Received unexpected response format from CRM: {post_resp.text[:200]}"
                else:
                    return f"Error: Failed to submit ticket. HTTP status: {post_resp.status_code}"
                    
        except Exception as e:
            logging.error(f"Error creating support ticket: {e}")
            return f"Error: Failed to process ticket submission due to connection error: {str(e)}"




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
            
            # Synthetic Call Center Ambient Synthesizer
            hum_state = 0.0
            num_chatter_sources = 6
            import random
            phases = [random.uniform(0, 2*np.pi) for _ in range(num_chatter_sources)]
            freqs = [random.uniform(90.0, 220.0) for _ in range(num_chatter_sources)]
            amps = [random.uniform(2.0, 5.0) for _ in range(num_chatter_sources)]
            
            click_decay = 0.985
            click_state = 0.0
            click_freq = 1800.0
            click_phase = 0.0
            
            ring_cycle_samples = 12 * sample_rate
            ring_phase1 = 0.0
            ring_phase2 = 0.0
            
            sample_idx = 0
            try:
                while True:
                    samples = np.zeros(num_samples, dtype=np.float32)
                    
                    # Slowly drift the chatter frequencies/amplitudes once per frame (dynamic human murmur effect)
                    for j in range(num_chatter_sources):
                        freqs[j] = np.clip(freqs[j] + random.uniform(-2.0, 2.0), 80.0, 250.0)
                        amps[j] = np.clip(amps[j] + random.uniform(-0.2, 0.2), 1.0, 6.0)
                        
                    for i in range(num_samples):
                        # 1. Base hum (deep ventilation background hum)
                        white = np.random.uniform(-1.0, 1.0)
                        hum_state = 0.995 * hum_state + 0.005 * white
                        val = hum_state * 20.0
                        
                        # 2. Quiet murmurs (distant voice chatter)
                        chatter_val = 0.0
                        for j in range(num_chatter_sources):
                            phases[j] += 2 * np.pi * freqs[j] / sample_rate
                            chatter_val += np.sin(phases[j]) * amps[j]
                        val += chatter_val
                        
                        # 3. Dynamic keyboard clicks (quiet typing in the background)
                        if random.random() < 0.0008:
                            click_state = random.uniform(5.0, 12.0)
                            click_freq = random.uniform(1400.0, 2200.0)
                            
                        if click_state > 0.1:
                            click_phase += 2 * np.pi * click_freq / sample_rate
                            val += np.sin(click_phase) * click_state
                            click_state *= click_decay
                            
                        # 4. Distant telephone ring cadences (low-amplitude ring once every 12 seconds)
                        ring_idx = sample_idx % ring_cycle_samples
                        in_ring = False
                        if 0 <= ring_idx < int(1.2 * sample_rate): # First ring (1.2s duration)
                            in_ring = True
                        elif int(2.0 * sample_rate) <= ring_idx < int(3.2 * sample_rate): # Second ring (1.2s duration)
                            in_ring = True
                            
                        if in_ring:
                            ring_phase1 += 2 * np.pi * 853.0 / sample_rate
                            ring_phase2 += 2 * np.pi * 960.0 / sample_rate
                            val += (np.sin(ring_phase1) + np.sin(ring_phase2)) * 0.8
                            
                        samples[i] = val
                        sample_idx += 1
                        
                    # Normalize and convert to 16-bit PCM
                    int_samples = np.clip(samples, -32768, 32767).astype(np.int16)
                    frame = rtc.AudioFrame(
                        data=int_samples.tobytes(),
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
    
    caller_phone = "web-sandbox"
    for identity, p in ctx.room.remote_participants.items():
        if "sip.phoneNumber" in p.attributes:
            caller_phone = p.attributes["sip.phoneNumber"]
            break

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
    else:
        logging.warning(f"No registered business found for extension {extension}. Falling back to default.")
        agent_name = "Gravity"
        db_prompt = (
            "You are 'Gravity', a professional real-time voice receptionist for Compusource. "
            "Keep answers very short, concise, and natural (one or two sentences)."
        )
        voice_model = "af_bella"
        skills = ["appointments"]
        business_id = ObjectId("6659f13ba97312fba0a91e5c")
        
    # Look up caller history memory in MongoDB, filtering by business_id to support multi-tenancy
    memory_context = ""
    if caller_phone and caller_phone != "web-sandbox":
        logging.info(f"Looking up caller history for phone number: {caller_phone} under business: {agent_name} ({business_id})")
        try:
            past_calls = await db["calls"].find({
                "business_id": business_id,
                "phone_number": caller_phone,
                "status": "completed"
            }).sort("start_time", -1).to_list(length=2)
            
            if past_calls:
                memory_context = "\n\nCALLER HISTORY MEMORY (MongoDB):\n"
                memory_context += f"The caller from phone/extension '{caller_phone}' has called before. Details of recent interactions:\n"
                for pc in past_calls:
                    time_str = pc.get("start_time", "Unknown Date")
                    readable_time = time_str
                    try:
                        dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                        readable_time = dt.strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        pass
                    summary = pc.get("summary")
                    if not summary:
                        # Fallback heuristic if summary is missing
                        transcript = pc.get("transcript", [])
                        caller_name = "unknown"
                        for turn in transcript:
                            if turn["role"] == "user" and "this is" in turn["text"].lower():
                                m = re.search(r'(?i)this is\s+([A-Za-z]+)', turn["text"])
                                if m:
                                    caller_name = m.group(1)
                                    break
                        summary = f"Call transcript indicates caller name is likely '{caller_name}'."
                    memory_context += f"- Call on {readable_time}: {summary}\n"
                memory_context += (
                    "Use this memory context to greet them familiarly (e.g. 'Welcome back!') "
                    "and follow up on their previous issue if they are calling about the same thing. "
                    "Do NOT assume the caller's name is the same as any employee, contact, or user name mentioned in the instructions or history. "
                    "Never call the caller by a name unless they explicitly confirmed their identity or the history explicitly confirms their caller name. "
                    "If you need to confirm their identity, ask: 'Am I speaking with [Name]?' rather than assuming. "
                    "You do not need to ask for their name or phone number if they are already clearly identified in the memory transcript or summary. "
                    "Confirm the callback number if needed, but do not ask for it from scratch if already known."
                )
        except Exception as db_err:
            logging.error(f"Error looking up caller memory: {db_err}")

    # Extract starting greeting from the raw prompt first
    greeting_text = extract_greeting(db_prompt, agent_name)
    
    # Clean conflicting instructions and goodbye endings from custom prompt
    sanitized_prompt = db_prompt.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    sanitized_prompt = re.sub(r'(?i)##\s+Ending\s+the\s+Call\b.*', '', sanitized_prompt, flags=re.DOTALL)
    sanitized_prompt = re.sub(r'(?i)After\s+collecting\s+the\s+message,\s*say\s*:\s*\n*\s*"[^"]*"', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)After\s+collecting\s+the\s+message,\s*say\s*:\s*\n*\s*\'[^\']*\'', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)"Thank you for calling\.\s+I\'ll\s+pass\s+on\s+your\s+message\."', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)"Thank you\.\s+I\'ll\s+pass\s+your\s+message\s+to\s+[^"]+"\.', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)"Thank you\.\s+Have\s+a\s+nice\s+day\."', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)\'Thank you for calling\.\s+I\'ll\s+pass\s+on\s+your\s+message\.\'', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)\'Thank you\.\s+I\'ll\s+pass\s+your\s+message\s+to\s+[^\']+\.\'', '', sanitized_prompt)
    sanitized_prompt = re.sub(r'(?i)\'Thank you\.\s+Have\s+a\s+nice\s+day\.\'', '', sanitized_prompt)
    # Prepend critical overriding rules
    agent_prompt = (
        "CRITICAL PROTOCOL (ABSOLUTE PRECEDENCE OVER ALL OTHER RULES):\n"
        "1. You are strictly FORBIDDEN from ending the call, saying goodbye, or promising to pass on a message/note/callback "
        "unless you have first successfully executed the 'take_message' tool and it has returned a success message.\n"
        "2. If the caller asks for a callback, to be called back, or to leave a message, you MUST ask them for their name "
        "and their digit-based phone number. You must ask: 'What is the best phone number to call you back at?' "
        "if they ask for a callback.\n"
        "3. You are strictly FORBIDDEN from fabricating, guessing, or making up any names, phone numbers, or details that were not explicitly spoken by the caller. "
        "If the caller has not explicitly spoken their name or phone number, you DO NOT have it. You must NEVER assume or guess the caller's name or details from the business instructions or context.\n"
        "4. If the caller asks 'Who are you?', 'What is your name?', or similar identity questions, you MUST directly answer who you are based on your custom instructions (e.g. state your name and role as defined in your instructions) "
        "and do NOT call any tools or take a message in that turn.\n\n"
    ) + sanitized_prompt
    
    if memory_context:
        agent_prompt += memory_context
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
        "5. CRITICAL: After ANY tool call returns a result (success OR error), you MUST immediately speak a short spoken response to the caller acknowledging the outcome. You are STRICTLY FORBIDDEN from remaining silent after a tool result. For example, after a ticket is created say: 'Your support ticket has been submitted successfully. Is there anything else I can help you with?' You MUST always follow a tool result with a spoken reply.\n"
    )

    prompt_lower = agent_prompt.lower()
    # Enable transfer capability ONLY when the prompt explicitly mentions 'transfer' AND includes
    # a numeric extension destination (e.g. 'extension 101', 'transfer to 200').
    # A bare mention of 'extension' in a business name or description does NOT activate this.
    has_transfer_config = bool(
        re.search(r'transfer', prompt_lower)
        and re.search(r'(?:extension|ext\.?)\s*\d+|transfer\s+(?:to\s+)?\d+', prompt_lower)
    )
    if has_transfer_config:
        agent_prompt += (
            "4. You have the ability to transfer calls to internal extensions using the transfer_call tool. "
            "If the user asks to be transferred or speak to a human, you must first ask for their confirmation "
            "(e.g. 'Would you like me to transfer you?'). You must ONLY call the transfer_call tool "
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
        
    if "tickets" in skills:
        agent_prompt += (
            "\n\nSupport Ticket CRM Rules:\n"
            "- You have the ability to create a support ticket in the CRM using the 'create_support_ticket' tool. "
            "You are strictly FORBIDDEN from calling this tool or offering to create a ticket unless the caller explicitly requests to file a ticket, open a ticket, or submit a support request. "
            "Do NOT assume the caller wants a ticket filed just because they report a problem; instead, first ask them: 'Would you like me to open a support ticket for you?' or wait until they explicitly ask to open a ticket.\n"
            "- If they confirm they want a ticket created, you MUST collect the caller's Name, Email Address, Subject of the issue, and detailed Message/description of the problem before submitting it.\n"
            "- If any of these details (Name, Email, Subject, or Message) are missing, you MUST ask the caller for them first. Ask politely and clearly.\n"
            "- Do NOT call the create_support_ticket tool if any of those details are missing. Instead, ask the user for them in conversation.\n"
            "- Do NOT tell the caller that their ticket is created/submitted until the create_support_ticket tool has been executed and returned a success message.\n"
            "- By default, use department '4' (Tech Support) for all technical support tickets, and priority '2' (Medium), unless the caller specifies otherwise.\n"
        )
        
    # Unconditionally include message taking rules
    agent_prompt += (
        "\n\nMessage Taking Rules:\n"
        "- You MUST call the take_message tool to record a message ONLY when the caller explicitly requests to leave a message or note, and ONLY after you have successfully collected the caller's Name, Phone Number, and detailed message.\n"
        "- You are strictly FORBIDDEN from calling the take_message tool or saying you will pass on the message/note/goodbye unless you have first collected the caller's Name, Phone Number, and their detailed message.\n"
        "- If the caller's Name, Phone Number, or detailed message is missing, you MUST ask the caller for them first.\n"
        "- Do NOT call the take_message tool if the caller's name, phone number, or detailed message is missing or placeholders. Instead, respond directly in conversation to ask the caller for the missing information.\n"
        "- If the caller asks you to 'call me back' or leave a call back note, you MUST ask them: 'What is the best phone number to call you back at?' and you are FORBIDDEN from using placeholders or saying goodbye without asking for their specific digits-based callback phone number first.\n"
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
                # 6000 tokens: comfortably fits system prompt (~300 tok) + tool
                # definitions (~600 tok) + a full multi-tool conversation (~4000 tok).
                # 3072 was too small and caused context truncation mid-call.
                # 8192 added ~3s prefill overhead per turn — 6000 is the balance.
                "num_ctx": 6000,
                "num_predict": 120,
                "keep_alive": -1
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
        
    if "tickets" in skills:
        # Include support ticket tools
        ticket_tool = next((t for t in all_tools if t.info.name == "create_support_ticket"), None)
        if ticket_tool:
            tools_list.append(ticket_tool)
        
    # Expose call transfer ONLY if the prompt explicitly defines a transfer destination (numeric extension).
    # This prevents the tool from being available on agents where transfer is not configured,
    # which would cause the LLM to hallucinate extension numbers.
    if has_transfer_config:
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
        "phone_number": caller_phone,
        "start_time": datetime.utcnow().isoformat(),
        "end_time": None,
        "duration_seconds": 0.0,
        "status": "active",
        "transcript": []
    }
            
    call_record_result = await db["calls"].insert_one(call_record)
    call_id = call_record_result.inserted_id
    logging.info(f"Initialized live call record in MongoDB for room {ctx.room.name} (id: {call_id}, business: {agent_name})")
    
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
            await db["calls"].update_one(
                {"_id": call_id, "business_id": business_id},
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
            
    async def send_teams_summary(webhook_url: str, summary_text: str, duration_seconds: float):
        try:
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
            logging.error(f"Failed to send Teams summary: {ex}")

    async def close_call_record():
        try:
            end_time = datetime.utcnow()
            start_time_parsed = datetime.fromisoformat(call_record["start_time"])
            duration = (end_time - start_time_parsed).total_seconds()
            
            # Fetch the updated transcript from the DB — pinned by exact call _id and business_id
            updated_call = await db["calls"].find_one({"_id": call_id, "business_id": business_id})
            transcript = updated_call.get("transcript", []) if updated_call else []
            
            summary_text = None
            if transcript:
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
                
                logging.info("Requesting call summary from Ollama for database archiving...")
                chunks = []
                async with llm_model.chat(chat_ctx=chat_ctx) as stream:
                    async for chunk in stream:
                        if chunk.delta and chunk.delta.content:
                            chunks.append(chunk.delta.content)
                summary_text = "".join(chunks).strip()
            else:
                summary_text = "No conversation recorded (call was disconnected immediately)."
            
            update_fields = {
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "status": "completed"
            }
            if summary_text:
                update_fields["summary"] = summary_text
                
            await db["calls"].update_one(
                {"_id": call_id, "business_id": business_id},
                {"$set": update_fields}
            )
            logging.info(f"Archived call for [{agent_name}] room {ctx.room.name} (id: {call_id}). Duration: {duration}s")
            
            # Trigger Teams Webhook if configured
            latest_biz = await db["businesses"].find_one({"_id": business_id})
            if latest_biz and latest_biz.get("teams_webhook_url") and summary_text:
                webhook_url = latest_biz["teams_webhook_url"]
                logging.info(f"Teams webhook configured: {webhook_url}. Sending summary post...")
                await send_teams_summary(webhook_url, summary_text, duration)
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