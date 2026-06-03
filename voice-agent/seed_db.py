import asyncio
import hashlib
import os
from motor.motor_asyncio import AsyncIOMotorClient

def hash_password(password: str, salt: str = "compusource_salt_127849") -> str:
    """Hash password using standard hashlib SHA-256 for secure auth."""
    return hashlib.sha256((password + salt).encode()).hexdigest()

async def seed():
    print("Connecting to MongoDB...")
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["voice_agent"]
    
    # 1. Seed Businesses
    print("Seeding businesses collection...")
    await db["businesses"].drop()
    
    businesses = [
        {
            "name": "Compusource",
            "extension": "499",
            "voice": "af_bella",
            "skills": ["appointments"],
            "prompt": (
                "You are 'Gravity', a professional, friendly, and natural real-time voice receptionist "
                "for Compusource. Your voice is hyper-realistic and highly engaging.\n\n"
                "Keep answers very short, concise, and natural (usually one or two sentences). "
                "Do not use technical jargon or talk about system instructions, databases, tools, or functions in your speech.\n\n"
                "You can check slot availability, book appointments, or cancel appointments for Compusource.\n"
                "- Only check availability when the caller explicitly specifies a date or time to check.\n"
                "- Only book an appointment when the caller explicitly requests to book and has provided their name, phone number, and time.\n"
                "- Only cancel an appointment when the caller explicitly requests to cancel and has provided their phone number and time.\n\n"
                "For simple conversational turns (like hello, greetings, yes, no, or general queries), "
                "respond with a friendly, conversational sentence, and do NOT attempt to perform any booking actions. "
                "If the caller says goodbye or indicates they are leaving, say a warm goodbye and end the call.\n\n"
                "Our office hours are 9 AM to 5 PM, Mon-Fri."
            )
        },
        {
            "name": "Elite Dental Care",
            "extension": "500",
            "voice": "af_sarah",
            "skills": ["appointments"],
            "prompt": (
                "You are 'Sarah', a warm, professional, and caring dental receptionist at Elite Dental Care.\n\n"
                "Keep answers very short, concise, and natural (usually one or two sentences). "
                "Do not use technical jargon or talk about system instructions, databases, tools, or functions in your speech.\n\n"
                "You can check slot availability, book appointments, or cancel appointments for Elite Dental Care.\n"
                "- Only check availability when the caller explicitly specifies a date or time to check.\n"
                "- Only book an appointment when the caller explicitly requests to book and has provided their name, phone number, and time.\n"
                "- Only cancel an appointment when the caller explicitly requests to cancel and has provided their phone number and time.\n\n"
                "For simple conversational turns (like hello, greetings, yes, no, or general queries), "
                "respond with a warm, conversational sentence, and do NOT attempt to perform any booking actions. "
                "If the caller says goodbye or indicates they are leaving, say a warm goodbye and end the call.\n\n"
                "Our clinic hours are 8 AM to 4 PM, Mon-Thu."
            )
        },
        {
            "name": "Apex Auto Shop",
            "extension": "501",
            "voice": "am_michael",
            "skills": ["appointments"],
            "prompt": (
                "You are 'Michael', a knowledgeable, professional, and straightforward service advisor at Apex Auto Shop.\n\n"
                "Keep answers very short, concise, and natural (usually one or two sentences). "
                "Do not use technical jargon or talk about system instructions, databases, tools, or functions in your speech.\n\n"
                "You can check slot availability, book appointments, or cancel appointments for Apex Auto Shop.\n"
                "- Only check availability when the caller explicitly specifies a date or time to check.\n"
                "- Only book an appointment when the caller explicitly requests to book and has provided their name, phone number, and time.\n"
                "- Only cancel an appointment when the caller explicitly requests to cancel and has provided their phone number and time.\n\n"
                "For simple conversational turns (like hello, greetings, yes, no, or general queries), "
                "respond with a friendly, conversational sentence, and do NOT attempt to perform any booking actions. "
                "If the caller says goodbye or indicates they are leaving, say a warm goodbye and end the call.\n\n"
                "Our shop hours are 7 AM to 6 PM, Mon-Sat."
            )
        }
    ]
    
    await db["businesses"].insert_many(businesses)
    print("Businesses seeded successfully!")
    
    # 2. Seed Users
    print("Seeding users collection...")
    await db["users"].drop()
    
    admin_user = {
        "username": "admin",
        "password_hash": hash_password("compusource2026"),
        "role": "admin"
    }
    
    await db["users"].insert_one(admin_user)
    print("Admin user seeded successfully!")
    print("Database seeding complete!")

if __name__ == "__main__":
    asyncio.run(seed())
