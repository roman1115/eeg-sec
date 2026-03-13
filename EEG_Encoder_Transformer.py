import torch
import torch.nn as nn
import torch.nn.functional as F
from AlignmentHead import AlignmentHead


class EEGFeatureExtractor(nn.Module):
    """
    Transformer-only baseline
    - No CNN
    - No TCN
    - No Subject Adversarial
    """
    def __init__(
        self,
        chans=30,
        hidden_dim=256,
        embed_dim=256,
        n_emotions=4,
        dropout=0.2,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()

        # === Linear projection (no CNN) ===
        self.input_proj = nn.Sequential(
            nn.Linear(chans, hidden_dim),
            nn.Dropout(dropout)
        )

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.ln = nn.LayerNorm(hidden_dim)

        # === Attention Pooling ===
        self.att_pool = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 1, 1)
        )

        # === Projection ===
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh()
        )

        # === Alignment head ===
        self.align = AlignmentHead(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            n_emotions=n_emotions,
            dropout=dropout
        )

    def forward(self, eeg):
        """
        eeg: [B, chans, time]
        """
        # [B, T, chans]
        x = eeg.transpose(1, 2)

        # Linear projection
        x = self.input_proj(x)            # [B, T, hidden_dim]

        # Transformer
        x = self.transformer(x)           # [B, T, hidden_dim]
        x = self.ln(x)

        # Attention Pooling
        x = x.transpose(1, 2)             # [B, hidden_dim, T]
        att = F.softmax(self.att_pool(x), dim=-1)
        x = torch.sum(att * x, dim=-1, keepdim=True)  # [B, hidden_dim, 1]

        # Projection
        x = self.proj(x)                  # [B, hidden_dim, 1]

        # Alignment
        out, logits = self.align(x)

        return out, logits
