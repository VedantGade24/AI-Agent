from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This is the personality of your AI
system_prompt = """
You are Jarvis, a smart personal AI assistant for Vedant.
You are helpful, friendly and a little funny.
You help Vedant with coding, studies, internship preparation and Gen AI learning.
Always address him by his name Vedant.
Keep responses short and clear unless asked for detail.
"""

# Start conversation with system prompt
conversation_history = [
    {"role": "system", "content": system_prompt}
]

print("Jarvis AI Assistant is ready!")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Vedant: ")
    
    if user_input.lower() == "quit":
        print("Jarvis: Goodbye Vedant! Keep building! 🚀")
        break
    
    conversation_history.append({
        "role": "user",
        "content": user_input
    })
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_history
    )
    
    ai_reply = response.choices[0].message.content
    
    conversation_history.append({
        "role": "assistant",
        "content": ai_reply
    })
    
    print(f"\nJarvis: {ai_reply}\n")