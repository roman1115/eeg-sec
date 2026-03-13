import torch
import torch.nn as nn
import torch.nn.functional as F
from AlignmentHead import AlignmentHead


class EEGFeatureExtractor(nn.Module):
    """
    CNN + BiLSTM baseline
    - Temporal modeling via BiLSTM
    - No TCN
    - No Transformer
    - No Subject Adversarial
    """
    def __init__(
        self,
        chans=30,
        pca_dim=64,
        hidden_dim=256,
        embed_dim=256,
        n_emotions=4,
        dropout=0.2,
        lstm_hidden=128,
        lstm_layers=1,
    ):
        super().__init__()

        # === Spatial projection ===
        self.spatial_proj = nn.Sequential(
            nn.Linear(chans, pca_dim),
            nn.Dropout(dropout)
        )

        # === CNN feature extractor ===
        self.cnn = nn.Sequential(
            nn.Conv1d(pca_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2)
        )

        # === BiLSTM for temporal modeling ===
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True
        )

        # === Attention Pooling ===
        self.att_pool = nn.Sequential(
            nn.Conv1d(256, 64, 1),
            nn.Tanh(),
            nn.Conv1d(64, 1, 1)
        )

        # === Projection ===
        self.proj = nn.Sequential(
            nn.Conv1d(256, hidden_dim, 1),
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

        # Spatial projection
        x = self.spatial_proj(x)          # [B, T, pca_dim]
        x = x.transpose(1, 2)             # [B, pca_dim, T]

        # CNN
        x = self.cnn(x)                   # [B, 256, T']

        # BiLSTM expects [B, T', C]
        x = x.transpose(1, 2)             # [B, T', 256]
        x, _ = self.bilstm(x)             # [B, T', 256]

        # Attention Pooling
        x = x.transpose(1, 2)             # [B, 256, T']
        att = F.softmax(self.att_pool(x), dim=-1)
        x = torch.sum(att * x, dim=-1, keepdim=True)  # [B, 256, 1]

        # Projection
        x = self.proj(x)                  # [B, hidden_dim, 1]

        # Alignment
        out, logits = self.align(x)

        return out, logits
