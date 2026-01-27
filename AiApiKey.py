import os
from dotenv import load_dotenv

def get_api_key():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("api_key Work")
    else:
        print("api_key not Work")