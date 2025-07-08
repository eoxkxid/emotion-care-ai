from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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

@app.post("/recommend")
async def recommend_emotion_care(item: TextInput):
    user_text = item.text

    # 간단한 감정 분석 (예시)
    if any(word in user_text.lower() for word in ["슬프", "힘들", "우울", "짜증"]):
        emotion = "부정"
        movie = "인사이드 아웃"
        music = "잔잔한 피아노곡"
        quote = "지나가는 감정이에요. 당신은 충분히 잘하고 있어요."
    else:
        emotion = "긍정"
        movie = "코코"
        music = "기분 좋은 재즈"
        quote = "오늘처럼 웃을 수 있는 날이 자주 오길 바라요."

    return {
        "emotion": emotion,
        "movie": movie,
        "music": music,
        "quote": quote
    }    