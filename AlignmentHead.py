# ✅ Final AlignmentHead (input: [B, 256, T], output: x_processed [B,256,1] + logits)
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class AlignmentHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 256,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        n_emotions: int = 4,  # ✅ 设置为你指定的5类
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_emotions = n_emotions
        self.dropout = dropout

        # ✅ Feature projection (保持和你原来一致)
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.proj_emb = nn.Linear(hidden_dim, embed_dim)

        # ✅ Classification head (保持原风格)
        self.cls_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.cls_fc2 = nn.Linear(hidden_dim, n_emotions)
        self.cls_dropout = nn.Dropout(dropout)

        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: ✅ Expected shape [B, 256, T]  (T can be 1 or >1)
        Returns:
            x_processed: ✅ [B, 256, 1] (same format as original input, goes to downstream model)
            logits:      ✅ [B, n_emotions] (only for loss)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, 256, T], got {x.shape}")

        # ✅ Pool over time dimension T → [B, 256]
        pooled = x.mean(dim=2)  # T=1时不会改变值

        # ✅ Shared projection (保持原有 MLP结构)
        h = self.act(self.proj1(pooled))  # [B, hidden_dim]

        # ✅ Embedding projection (再 reshape 回 [B,256,1] 用于下游)
        emb = self.proj_emb(h)  # [B, 256]
        x_processed = emb.unsqueeze(-1)  # ✅ [B,256,1] —— 这就是传给下游的

        # ✅ Classification head
        cls_h = self.act(self.cls_fc1(h))
        cls_h = self.cls_dropout(cls_h)
        logits = self.cls_fc2(cls_h)  # [B, n_emotions]

        return x_processed, logits


if __name__ == "__main__":
    # ✅ Quick test
    B, D, T = 16, 256, 1
    model = AlignmentHead(in_dim=D, hidden_dim=256, embed_dim=256, n_emotions=4)
    x = torch.randn(B, D, T)
    x_out, logits = model(x)
    print("Input:", x.shape, "| x_processed:", x_out.shape, "| logits:", logits.shape)
