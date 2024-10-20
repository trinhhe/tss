import pytorch_lightning as pl
import torch
from mamba_ssm import Mamba
from torch import nn, optim
from torch.nn import functional as F

## based on https://2084.substack.com/p/2084-marcrandbot-speech-synthesis


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout=0.2) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class Block(nn.Module):
    def __init__(self, n_embed) -> None:
        super().__init__()
        # 3 * expand * d_model^2 parameters
        self.sa_head = Mamba(
            d_model=n_embed,  # d_model dim
            d_state=16,  # SSM state dim
            d_conv=4,  # Local conv width
            expand=2,  # block expansion factor
        )
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MambaAudioModel(pl.LightningModule):
    def __init__(
        self, vocab_size=1024, n_embed=384, block_size=2000, n_layers=6
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.ffn = FeedForward(n_embed)
        print("layers", n_layers)
        self.blocks = nn.Sequential(*[Block(n_embed) for _ in range(n_layers)])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C_e)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb  # (B, T, Q, C_e)
        x = self.blocks(x)  # (B, T, Q, C_e)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            logits = logits.view(B, T, C)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self.forward(batch)
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=1e-3)
        return opt
