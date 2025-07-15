from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

# API 키 로드 및 클라이언트 생성
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_quote_with_gpt(user_input: str, emotion: str) -> str:
    prompt = f"""
    You are a kind and thoughtful assistant.
    The user has shared this message:

    "{user_input}"

    This message was analyzed as: "{emotion}" emotion.

    Based on the message and their emotional state, generate a short comforting and motivational quote. 
    It should sound natural, warm, and encouraging. Avoid general or cliché phrases.
    Use less than 30 words.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPT Error:", e)
        return "힘든 하루였네요. 당신의 마음이 편해지길 바라요."
