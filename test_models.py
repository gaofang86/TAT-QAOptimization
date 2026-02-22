from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")
print("API KEY EXISTS:", bool(api_key))

client = Groq(api_key=api_key)

print("Testing model...")

try:
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0
    )
    print("SUCCESS")
    print(completion.choices[0].message.content)

except Exception as e:
    print("ERROR:", e)


