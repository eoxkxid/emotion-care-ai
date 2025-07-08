#전처리
with open ('test.txt',encoding='utf-8') as f:
    text = f.read()

import re
clean_text = re.sub(r"[^가-힣a-zA-Z0-9.,!?\s]","", text.lower())  # 한글과 공백만 남기기

# 토큰화
chars = sorted(list(set(clean_text)))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}