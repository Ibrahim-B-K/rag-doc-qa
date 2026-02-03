import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file")
    exit()

print(f"🔑 Checking EMBEDDING models for API Key: {api_key[:5]}...")

genai.configure(api_key=api_key)

try:
    print("\n✅ AVAILABLE EMBEDDING MODELS (Copy one of these exactly):")
    print("-" * 50)
    for m in genai.list_models():
        # This time we look for 'embedContent' capability
        if 'embedContent' in m.supported_generation_methods:
            print(f"👉 {m.name}")
    print("-" * 50)

except Exception as e:
    print(f"\n❌ Error listing models: {e}")