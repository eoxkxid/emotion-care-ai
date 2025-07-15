from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import random

# GPT 위로 문구 생성 함수 추가
from .utils.gpt_quote import generate_quote_with_gpt

app = FastAPI()

# CORS 실행 (Streamlit와 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 입력 형식 정의
class TextInput(BaseModel):
    text: str

# 데이터 로드 및 초기화
content_df = pd.read_csv("data/content.csv")

# 컨텐츠 추천 함수 정의
def get_random_recommendation(emotion_tag: str):
    filtered_content = content_df[content_df["emotion"] == emotion_tag]

    movies = filtered_content[filtered_content["type"] == "movie"]["title"].tolist()
    music = filtered_content[filtered_content["type"] == "music"]["title"].tolist()

    recommended_movie = random.choice(movies) if movies else "추천 영화가 없습니다."
    recommended_music = random.choice(music) if music else "추천 음악이 없습니다."

    return recommended_movie, recommended_music

@app.post("/recommend")
async def recommend_emotion_care(item: TextInput):
    user_text = item.text.lower()

    # 임시 감정 분석 -> 이후 실제 모델로 교체 필요
    negative_keywords = ["슬프", "힘들", "우울", "짜증"]
    emotion = "부정" if any(word in user_text for word in negative_keywords) else "긍정"

    # 콘텐츠 추천
    movie, music = get_random_recommendation(emotion)

    # GPT 기반 위로 문구 생성 (상황 맥락 반영)
    quote = generate_quote_with_gpt(user_input=user_text, emotion=emotion)

    return {
        "emotion": emotion,
        "movie": movie,
        "music": music,
        "quote": quote
    }