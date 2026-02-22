from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_model(prompt, model_name="openai/gpt-oss-120b", temperature=0):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )

    return {
        "text": response.choices[0].message.content,
        "usage": response.usage
    }
