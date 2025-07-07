import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random

# 1. 텍스트 토큰화
text = "나는 오늘 너무 우울한 하루를 보냈어.. 하지만 내일은 더 나은 날이 될 거야."
tokens = text.split()

# 2. Vocabulary 만들기
vocab = list(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

# 3. Skip-Gram 데이터 만들기
def generate_skipgram(tokens, window_size=1):
    pairs = []
    for i, center in enumerate(tokens):
        for j in range(i - window_size, i + window_size + 1):
            if i != j and 0 <= j < len(tokens):
                pairs.append((tokens[i], tokens[j]))
    return pairs

pairs = generate_skipgram(tokens, window_size=1)
train_data = [(word2idx[a], word2idx[b]) for a, b in pairs]
print("train_data:", train_data)

# 4. Word2Vec 모델 정의
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words):
        center_embeds = self.embeddings(center_words)     # (1, D)
        context_embeds = self.embeddings(context_words)   # (1, D)
        score = torch.sum(center_embeds * context_embeds)  # 점곱 → 스칼라
        return score

# 5. 모델 생성 및 학습 준비
embedding_dim = 10
model = Word2Vec(vocab_size, embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

# 6. 학습 루프
for epoch in range(100):
    total_loss = 0
    for center, context in tqdm(train_data, desc=f"Epoch {epoch+1}"):
        center_tensor = torch.tensor([center])
        context_tensor = torch.tensor([context])
        label = torch.tensor(1.0)

        # 부정 샘플링 (랜덤 단어)
        negative_idx = random.choice([i for i in range(vocab_size) if i != context])
        negative_tensor = torch.tensor([negative_idx])
        negative_label = torch.tensor(0.0)

        optimizer.zero_grad()

        pos_score = model(center_tensor, context_tensor)
        neg_score = model(center_tensor, negative_tensor)

        loss = loss_fn(pos_score, label) + loss_fn(neg_score, negative_label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# 7. 특정 단어의 임베딩 확인
word = "나는"
idx = word2idx[word]
vec = model.embeddings(torch.tensor([idx]))
print(f"\nWord: {word}, Vector: {vec.detach().numpy()}")

# 8. 가장 유사한 단어 출력 함수
def most_similar(word, top_n=3):
    idx = word2idx[word]
    target_vec = model.embeddings(torch.tensor([idx]))     # (1, D)
    all_embeddings = model.embeddings.weight               # (V, D)

    similarities = torch.nn.functional.cosine_similarity(target_vec, all_embeddings)  # (V,)
    values, indices = torch.topk(similarities, top_n + 1)  # +1은 자기 자신 포함

    print(f"\nTop {top_n} words similar to '{word}':")
    for i, v in zip(indices[1:], values[1:]):  # 자기 자신 제외
        print(f"{idx2word[i.item()]}: {v.item():.3f}")

# 9. 유사 단어 보기
most_similar("하루를")
