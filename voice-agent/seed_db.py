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
                "You can check slot availability, book appointments, or cancel appointments for Compusource. "
                "Always check slot availability before booking. Our office hours are 9 AM to 5 PM, Mon-Fri."
            )
        },
        {
            "name": "Elite Dental Care",
            "extension": "500",
            "voice": "af_sarah",
            "skills": ["appointments"],
            "prompt": (
                "You are 'Sarah', a warm and caring dental receptionist at Elite Dental Care.\n\n"
                "Keep answers very short, concise, and professional (usually one or two sentences). "
                "You can check dental appointment availability, book cleanings or checkups, or cancel bookings. "
                "Always check slot availability before booking. Our clinic hours are 8 AM to 4 PM, Mon-Thu."
            )
        },
        {
            "name": "Apex Auto Shop",
            "extension": "501",
            "voice": "am_michael",
            "skills": ["appointments"],
            "prompt": (
                "You are 'Michael', a knowledgeable and straightforward service advisor at Apex Auto Shop.\n\n"
                "Keep answers very short, concise, and helpful (usually one or two sentences). "
                "You can check vehicle service slot availability, book repair/maintenance appointments, or cancel bookings. "
                "Always check slot availability before booking. Our shop hours are 7 AM to 6 PM, Mon-Sat."
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
