import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

print("Listing available models for your API key:")
try:
    for model in client.models.list():
        print(f"Model Name: {model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
