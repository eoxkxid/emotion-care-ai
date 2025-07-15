import streamlit as st
import requests

st.title("EmotionCare AI")
st.write("당신의 하루는 어땠나요?")

user_input = st.text_area("지금 기분을 자유롭게 적어주세요.")

if st.button("감정 전달"):
    if not user_input.strip():
        st.warning("문장을 입력해 주세요.")
    else:
        # FastAPI 서버로 요청 보내기
        response = requests.post(
            "http://localhost:8000/recommend",
            json={"text": user_input}
        )

        if response.status_code == 200:
            data = response.json()

            # 결과 출력
            st.subheader("분석 결과")
            st.markdown(f"**감정:** {data['emotion']}")

            st.subheader("🎬 추천 영화")
            st.markdown(f"- {data['movie']}")

            st.subheader("🎵 추천 음악")
            st.markdown(f"- {data['music']}")

            st.subheader("💬 위로 문구")
            st.markdown(f"> {data['quote']}")

        else:
            st.error("서버에서 오류가 발생했어요. API가 실행 중인지 확인해보세요.")