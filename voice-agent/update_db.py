import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def update_database():
    logging.info("Connecting to MongoDB...")
    client = AsyncIOMotorClient("mongodb://127.0.0.1:27017")
    db = client["voice_agent"]
    
    # 1. Update Business Prompts with Escalation Instructions
    logging.info("Updating business prompts with transfer/escalation instructions...")
    
    # Compusource (499)
    compusource = await db["businesses"].find_one({"extension": "499"})
    if compusource:
        prompt = compusource["prompt"]
        if "extension 100" not in prompt:
            new_prompt = prompt + (
                "\n\nIf the customer's query cannot be resolved, if physical intervention is needed, "
                "or if the caller explicitly requests human help or a transfer, ask for confirmation "
                "and transfer them to Tech Support at extension 100."
            )
            await db["businesses"].update_one(
                {"_id": compusource["_id"]},
                {"$set": {"prompt": new_prompt}}
            )
            logging.info("Updated Compusource prompt.")
            
    # Elite Dental Care (500)
    dental = await db["businesses"].find_one({"extension": "500"})
    if dental:
        prompt = dental["prompt"]
        if "extension 203" not in prompt:
            new_prompt = prompt + (
                "\n\nIf the caller has questions about billing, complex dental procedures, "
                "or wants to speak to a doctor or manager directly, ask for confirmation and "
                "transfer them to Billing Dept at extension 203 or Dr. Sarah Johnson at extension 201."
            )
            await db["businesses"].update_one(
                {"_id": dental["_id"]},
                {"$set": {"prompt": new_prompt}}
            )
            logging.info("Updated Elite Dental Care prompt.")
            
    # Apex Auto Shop (501)
    apex = await db["businesses"].find_one({"extension": "501"})
    if apex:
        prompt = apex["prompt"]
        if "extension 302" not in prompt:
            new_prompt = prompt + (
                "\n\nIf the caller asks to speak to a mechanic, has questions about specific part pricing, "
                "or requests a transfer to a human, ask for confirmation and transfer them to Parts Dept "
                "at extension 302 or Michael at extension 301."
            )
            await db["businesses"].update_one(
                {"_id": apex["_id"]},
                {"$set": {"prompt": new_prompt}}
            )
            logging.info("Updated Apex Auto Shop prompt.")

    # Edel (502)
    edel = await db["businesses"].find_one({"extension": "502"})
    if edel:
        prompt = edel["prompt"]
        if "extensión 401" not in prompt:
            new_prompt = prompt + (
                "\n\nSi el cliente solicita hablar con Edel, requiere asistencia humana directa "
                "o pide una transferencia, solicite confirmación y transfiera a Edel a la extensión 401."
            )
            await db["businesses"].update_one(
                {"_id": edel["_id"]},
                {"$set": {"prompt": new_prompt}}
            )
            logging.info("Updated Edel prompt.")

    # 2. Seed Contacts Collection
    logging.info("Seeding contacts collection...")
    await db["contacts"].drop()
    
    contacts = []
    
    # Resolve business IDs dynamically
    async for biz in db["businesses"].find():
        name = biz["name"]
        biz_id = biz["_id"]
        
        if name == "Compusource":
            contacts.extend([
                {"business_id": biz_id, "name": "Shadikur", "extension": "100", "email": "shadikur@compusource.net", "department": "Founder"},
                {"business_id": biz_id, "name": "Tech Support", "extension": "100", "email": "support@compusource.net", "department": "Technical Support"},
                {"business_id": biz_id, "name": "Accounting", "extension": "102", "email": "billing@compusource.net", "department": "Accounting"},
                {"business_id": biz_id, "name": "Sales", "extension": "103", "email": "sales@compusource.net", "department": "Sales"}
            ])
        elif name == "Elite Dental Care":
            contacts.extend([
                {"business_id": biz_id, "name": "Dr. Sarah Johnson", "extension": "201", "email": "sarah.johnson@elitedental.com", "department": "Dentist"},
                {"business_id": biz_id, "name": "Dr. Robert Smith", "extension": "202", "email": "robert.smith@elitedental.com", "department": "Orthodontist"},
                {"business_id": biz_id, "name": "Billing Dept", "extension": "203", "email": "billing@elitedental.com", "department": "Billing"}
            ])
        elif name == "Apex Auto Shop":
            contacts.extend([
                {"business_id": biz_id, "name": "Michael", "extension": "301", "email": "michael@apexautoshop.com", "department": "Service Advisor"},
                {"business_id": biz_id, "name": "Parts Dept", "extension": "302", "email": "parts@apexautoshop.com", "department": "Parts Office"},
                {"business_id": biz_id, "name": "Accounting", "extension": "303", "email": "billing@apexautoshop.com", "department": "Accounting"}
            ])
        elif name == "Edel":
            contacts.extend([
                {"business_id": biz_id, "name": "Edel", "extension": "401", "email": "edel@edel.com", "department": "Owner"}
            ])

    if contacts:
        await db["contacts"].insert_many(contacts)
        logging.info(f"Seeded {len(contacts)} contacts successfully.")
    else:
        logging.warning("No businesses found to link contacts with.")
        
    logging.info("Database migration and update completed successfully!")

if __name__ == "__main__":
    asyncio.run(update_database())
