import time
import requests
import jwt

# Core credentials matching your deployment
API_KEY = "devkey1778495864"
API_SECRET = "devsecret1778495864"
LIVEKIT_URL = "http://127.0.0.1:7800"

def create_lk_token():
    payload = {
        "iss": API_KEY,
        "nbf": int(time.time()),
        "exp": int(time.time()) + 600,
        "video": {"admin": True},
        "sip": {"admin": True}
    }
    return jwt.encode(payload, API_SECRET, algorithm="HS256")

headers = {
    "Authorization": f"Bearer {create_lk_token()}",
    "Content-Type": "application/json"
}

# 1. Define Trunk Properties
trunk_data = {
    "trunk": {
        "name": "FusionPBX",
        "numbers": ["499"],
        "allowed_addresses": ["199.47.47.106"]
    }
}

# 2. Define Dispatch Properties
dispatch_data = {
    "name": "Route-To-Agent",
    "rule": {
        "dispatchRuleIndividual": {
            "roomPrefix": "sip_room"
        }
    },
    "roomConfig": {
        "agents": [
            {
                "agentName": ""
            }
        ]
    }
}

print("Pushing Trunk Profile...")
r1 = requests.post(f"{LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPInboundTrunk", json=trunk_data, headers=headers)
print("Response:", r1.status_code, r1.text)

print("\nPushing Dispatch Route Rule...")
r2 = requests.post(f"{LIVEKIT_URL}/twirp/livekit.SIP/CreateSIPDispatchRule", json=dispatch_data, headers=headers)
print("Response:", r2.status_code, r2.text)
