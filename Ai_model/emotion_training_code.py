import torch
import torch.nn as nn
import re
import os
import json

# ----------------------------
# 1. 데이터 로드 및 전처리
# ----------------------------

label_map = {"부정": 0, "긍정": 1}

def clean_text(t):
    return re.sub(r"[^가-힣a-zA-Z0-9.,!?\s]", "", t.lower())

def load_training_data(file_path):
    data = []
    if not os.path.exists(file_path):
        return data
    with open(file_path, encoding='utf-8') as f:
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
            data.append((clean_text(text), label_map[label_str]))
    return data

# 전체 데이터 불러오기
training_data = load_training_data('training_emotion_text.txt')
training_data += load_training_data('feedback_emotion_data.txt')

if len(training_data) == 0:
    raise ValueError("데이터가 없습니다! 학습용 파일들을 확인하세요.")

texts, labels = zip(*training_data)

# ----------------------------
# 2. 문자 집합 생성 / 불러오기 및 저장
# ----------------------------

vocab_path = "vocab.json"

def build_vocab(texts):
    all_text = " ".join(texts)
    chars = sorted(list(set(all_text)))
    stoi = {c: i+1 for i, c in enumerate(chars)}  # 0은 패딩용
    return stoi

def save_vocab(stoi, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)

def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        stoi = json.load(f)
    return {k: int(v) for k, v in stoi.items()}

if not os.path.exists(vocab_path):
    print("vocab.json이 없어 문자 집합을 새로 생성합니다.")
    stoi = build_vocab(texts)
    save_vocab(stoi, vocab_path)
    print(f"문자 집합 크기: {len(stoi)}")
else:
    stoi = load_vocab(vocab_path)
    print("vocab.json에서 문자 집합을 불러왔습니다.")
    print(f"문자 집합 크기: {len(stoi)}")

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
# 3. 모델 정의
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

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x[:, 0, :]
        return self.fc_out(x)

# ----------------------------
# 4. 모델 생성 및 학습 / 로드
# ----------------------------

model_path = "emotion_model.pth"

vocab_size = len(stoi) + 1

model = EmotionClassifier(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=8,
    num_layers=2,
    block_size=max_len,
    num_classes=2
)

def load_model_safely(model, path, vocab_size):
    if not os.path.exists(path):
        return False

    try:
        state_dict = torch.load(path)
        # 저장된 임베딩 크기 확인
        saved_vocab_size = state_dict['token_embedding.weight'].size(0)
        if saved_vocab_size != vocab_size:
            print(f"임베딩 크기 불일치! 저장된: {saved_vocab_size}, 현재: {vocab_size}")
            print("새로 학습을 시작합니다.")
            return False
        model.load_state_dict(state_dict)
        print("저장된 모델을 정상적으로 불러왔습니다.")
        return True
    except Exception as e:
        print("모델 로드 중 오류 발생:", e)
        print("새로 학습을 시작합니다.")
        return False

loaded = load_model_safely(model, model_path, vocab_size)

if not loaded:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), model_path)
    print("처음부터 학습 완료 후 모델 저장")

# ----------------------------
# 5. 분류 함수
# ----------------------------

def classify_text(model, text, stoi, max_len):
    model.eval()
    text = clean_text(text)
    ids = [stoi.get(c, 0) for c in text]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    input_tensor = torch.tensor([ids])
    with torch.no_grad():
        logits = model(input_tensor)
        pred = torch.argmax(logits, dim=1).item()
    return "긍정" if pred == 1 else "부정"

def preprocess(text, stoi, max_len):
    text = clean_text(text)
    ids = [stoi.get(c, 0) for c in text]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return torch.tensor([ids])

def label_to_index(label):
    return torch.tensor([label_map[label]])

# ----------------------------
# 6. 사용자 입력 및 피드백 수집
# ----------------------------

feedback_data = []

while True:
    user_input = input("\n문장을 입력하세요 (종료하려면 '종료' 입력): ")
    if user_input == "종료":
        break
    result = classify_text(model, user_input, stoi, max_len)
    print(f"예측 감정: {result}")

    correct_answer = input("정답(긍정/부정): ")
    if correct_answer not in ["긍정", "부정"]:
        print("잘못된 정답 형식입니다. '긍정' 또는 '부정'으로 입력해주세요.")
        continue

    print(f"입력한 문장: {user_input}")
    print(f"예측 감정: {result}, 정답: {correct_answer}")
    if result == correct_answer:
        print("정답입니다!")
    else:
        print("오답입니다.")

    feedback_data.append((clean_text(user_input), correct_answer))

# ----------------------------
# 7. 피드백 학습 및 저장 (중복 방지)
# ----------------------------

if feedback_data:
    print(f"\n수집된 피드백 {len(feedback_data)}건 중 중복 제거 중...")

    existing_feedback_sentences = set()
    if os.path.exists("feedback_emotion_data.txt"):
        with open("feedback_emotion_data.txt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    _, sentence = parts
                    existing_feedback_sentences.add(sentence.strip())

    unique_feedback = [(s, l) for (s, l) in feedback_data if s not in existing_feedback_sentences]

    print(f"중복 제거 후 {len(unique_feedback)}건만 재학습에 사용됩니다.")

    if unique_feedback:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        model.train()

        for sentence, label in unique_feedback:
            x = preprocess(sentence, stoi, max_len)
            y = label_to_index(label)

            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        torch.save(model.state_dict(), model_path)
        print("피드백 기반 재학습 완료 및 모델 저장.")

        with open("feedback_emotion_data.txt", "a", encoding="utf-8") as f:
            for sentence, label in unique_feedback:
                f.write(f"\n{label}\t{sentence}\n")
        print("새 피드백이 feedback_emotion_data.txt 파일에 저장되었습니다.")
    else:
        print("새로운 피드백이 없어 재학습을 건너뜁니다.")