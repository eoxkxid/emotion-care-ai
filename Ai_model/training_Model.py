import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_stize, embedding_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        self.transformer_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        )
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.block_size = block_size

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # Shape: (1, T)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.fc_out(x)