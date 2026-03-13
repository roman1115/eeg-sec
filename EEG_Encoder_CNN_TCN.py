import torch
import torch.nn as nn
import torch.nn.functional as F
from AlignmentHead import AlignmentHead
# ↑ 如果你 TemporalConvBlock 就在同一个文件，直接 import 或复制即可

class TemporalConvBlock(nn.Module):
    """轻量级时序建模模块 (Depthwise Separable Conv + Residual)"""
    def __init__(self, dim, kernel_size=5, dropout=0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        self.pwconv = nn.Conv1d(dim, dim, 1)
        self.norm = nn.BatchNorm1d(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x + residual
class EEGFeatureExtractor(nn.Module):
    """
    CNN + TCN baseline
    - Temporal modeling via lightweight TCN (TemporalConvBlock)
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
        tcn_kernel=5,
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

        # === TCN (lightweight temporal modeling) ===
        self.tcn = TemporalConvBlock(
            dim=256,
            kernel_size=tcn_kernel,
            dropout=dropout
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

        # TCN
        x = self.tcn(x)                   # [B, 256, T']

        # Attention Pooling
        att = F.softmax(self.att_pool(x), dim=-1)
        x = torch.sum(att * x, dim=-1, keepdim=True)  # [B, 256, 1]

        # Projection
        x = self.proj(x)                  # [B, hidden_dim, 1]

        # Alignment
        out, logits = self.align(x)

        return out, logits
