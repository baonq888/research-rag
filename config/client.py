import os
from dotenv import load_dotenv
from together import Together

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=api_key)