import os
import sys
import subprocess
import json
import tempfile
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync_sip")

def load_keys():
    # Load keys.yaml for dynamic credential mapping
    if os.path.exists("keys.yaml"):
        try:
            with open("keys.yaml") as f:
                content = f.read().strip()
                if content and ":" in content:
                    key, secret = content.split(":", 1)
                    return key.strip(), secret.strip()
        except Exception as e:
            logger.error(f"Error loading keys.yaml: {e}")
    return "devkey1778495864", "devsecret1778495864"

async def sync():
    key, secret = load_keys()
    env = os.environ.copy()
    env["LIVEKIT_URL"] = "http://127.0.0.1:7800"
    env["LIVEKIT_API_KEY"] = key
    env["LIVEKIT_API_SECRET"] = secret

    # 1. Fetch active trunks from LiveKit
    res = subprocess.run(["lk", "sip", "inbound", "list"], env=env, capture_output=True, text=True)
    existing_trunks = {} # maps extension -> trunk_id
    for line in res.stdout.splitlines():
        normalized_line = line.replace("│", "|")
        if "ST_" in normalized_line:
            parts = [p.strip() for p in normalized_line.split("|") if p.strip()]
            if len(parts) >= 3:
                trunk_id = parts[0]
                numbers_part = parts[2]
                for num in numbers_part.split(","):
                    existing_trunks[num.strip()] = trunk_id

    logger.info(f"Existing LiveKit SIP Trunks: {existing_trunks}")

    # 1.5 Fetch active dispatch rules from LiveKit
    res = subprocess.run(["lk", "sip", "dispatch", "list"], env=env, capture_output=True, text=True)
    existing_rules = set() # set of rule names
    for line in res.stdout.splitlines():
        normalized_line = line.replace("│", "|")
        if "SDR_" in normalized_line:
            parts = [p.strip() for p in normalized_line.split("|") if p.strip()]
            if len(parts) >= 2:
                # Typically index 1 is the rule Name
                existing_rules.add(parts[1])

    logger.info(f"Existing LiveKit SIP Dispatch Rules: {existing_rules}")

    # 2. Fetch businesses from MongoDB
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
    db = client["voice_agent"]
    businesses = await db["businesses"].find().to_list(None)

    for business in businesses:
        ext = business["extension"]
        name = business["name"]
        
        trunk_id = existing_trunks.get(ext)
        if not trunk_id:
            logger.info(f"SIP Trunk for {name} (extension {ext}) is missing. Registering...")
            trunk_data = {
                "trunk": {
                    "name": f"Trunk-{ext}",
                    "numbers": [ext],
                    "allowedAddresses": ["199.47.47.106"]
                }
            }
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(trunk_data, f)
                temp_trunk_path = f.name
            
            try:
                res_trunk = subprocess.run(["lk", "sip", "inbound", "create", temp_trunk_path], env=env, capture_output=True, text=True, check=True)
                for line in res_trunk.stdout.splitlines():
                    if "SIPTrunkID:" in line or "SipTrunkID" in line:
                        trunk_id = line.split(":")[-1].strip()
                os.unlink(temp_trunk_path)
            except Exception as e:
                logger.error(f"Error creating trunk for {name} ({ext}): {e}")
                continue
        else:
            logger.info(f"SIP Trunk for {name} (extension {ext}) already exists: {trunk_id}")

        if not trunk_id:
            logger.error(f"Cannot proceed without trunk_id for extension {ext}")
            continue

        rule_name = f"Route-{ext}"
        if rule_name not in existing_rules:
            logger.info(f"SIP Dispatch Rule {rule_name} is missing. Registering...")
            try:
                res_disp = subprocess.run([
                    "lk", "sip", "dispatch", "create",
                    "--name", rule_name,
                    "--trunks", trunk_id,
                    "--individual", "sip_room",
                    "--randomize"
                ], env=env, capture_output=True, text=True, check=True)
                
                dispatch_id = None
                for line in res_disp.stdout.splitlines():
                    if "SIPDispatchRuleID:" in line or "SipDispatchRuleID" in line:
                        dispatch_id = line.split(":")[-1].strip()
                        
                logger.info(f"Registered SIP Dispatch Rule for {name} ({ext}): rule={dispatch_id}")
            except subprocess.CalledProcessError as spe:
                logger.error(f"Error registering dispatch rule for {name} ({ext}): {spe}\nStdout: {spe.stdout}\nStderr: {spe.stderr}")
            except Exception as e:
                logger.error(f"Error registering dispatch rule for {name} ({ext}): {e}")
        else:
            logger.info(f"SIP Dispatch Rule {rule_name} already exists.")

if __name__ == "__main__":
    asyncio.run(sync())
