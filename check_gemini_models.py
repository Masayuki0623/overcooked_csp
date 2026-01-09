import google.generativeai as genai
import os

def read_key_from_file(filename):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    return None

api_key = os.getenv("GOOGLE_API_KEY") or read_key_from_file("google_api_key.txt")
if not api_key:
    print("GOOGLE_API_KEY is not set and google_api_key.txt not found.")
else:
    try:
        genai.configure(api_key=api_key)
        print("Listing available models...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error: {e}")
