import torch
import torch.nn as nn
import re

# ----------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------

texts = []
labels = []

label_map = {"부정": 0, "긍정": 1}

with open('test.txt', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"잘못된 형식 라인 건너뜀: {line}")
            continue
        label_str, text = parts
        if label_str not in label_map:
            print(f"알 수 없는 라벨 건너뜀: {label_str}")
            continue
        labels.append(label_map[label_str])
        texts.append(text)

if len(texts) == 0:
    raise ValueError("데이터가 없습니다! test.txt 파일과 형식을 다시 확인하세요.")

def clean_text(t):
    return re.sub(r"[^가-힣a-zA-Z0-9.,!?\s]", "", t.lower())

texts = [clean_text(t) for t in texts]

# ----------------------------
# 2. 토큰화 및 인덱스 매핑
# ----------------------------

all_text = " ".join(texts)
chars = sorted(list(set(all_text)))
stoi = {c: i+1 for i, c in enumerate(chars)}  # 0은 패딩용
itos = {i: c for c, i in stoi.items()}

def text_to_ids(text):
    return [stoi.get(c, 0) for c in text]

text_ids_list = [text_to_ids(t) for t in texts]

max_len = max(len(ids) for ids in text_ids_list)

def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))

input_ids = torch.tensor([pad_sequence(ids, max_len) for ids in text_ids_list])
labels = torch.tensor(labels)

# ----------------------------
# 3. 감정 분류 모델 정의
# ----------------------------

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, block_size, num_classes):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        )
        self.fc_out = nn.Linear(embedding_dim, num_classes)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[:, 0, :]  # 첫 토큰 임베딩
        return self.fc_out(x)

# ----------------------------
# 4. 모델 초기화 및 학습 준비
# ----------------------------

num_classes = 2
model = EmotionClassifier(
    vocab_size=len(stoi) + 1,
    embedding_dim=128,
    num_heads=8,
    num_layers=2,
    block_size=max_len,
    num_classes=num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 5. 학습 루프 (간단)
# ----------------------------

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# 6. 평가 (학습 데이터로)
# ----------------------------

model.eval()
with torch.no_grad():
    logits = model(input_ids)
    preds = torch.argmax(logits, dim=1)
    print("예측값:", preds.tolist())
    print("정답값:", labels.tolist())

# ----------------------------
# 7. 사용자 입력 문장 분류 함수
# ----------------------------

def classify_text(model, text, stoi, max_len):
    model.eval()
    text = re.sub(r"[^가-힣a-zA-Z0-9.,!?\s]", "", text.lower())
    ids = [stoi.get(c, 0) for c in text]
    if len(ids) > max_len:
        ids = ids[:max_len]
    else:
        ids += [0] * (max_len - len(ids))
    input_tensor = torch.tensor([ids])
    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).item()
    return "긍정" if pred == 1 else "부정"

# ----------------------------
# 8. 테스트: 사용자 입력 문장 분류
# ----------------------------

while True:
    user_input = input("\n문장을 입력하세요 (종료하려면 '종료' 입력): ")
    if user_input == "종료":
        break
    result = classify_text(model, user_input, stoi, max_len)
    print(f"예측 감정: {result}")