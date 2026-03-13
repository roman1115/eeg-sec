import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3,3), stride=(1,1,1), padding=(1,1,1)):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
        self.bn   = nn.BatchNorm3d(out_ch)
        self.act  = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SER_Embedding(nn.Module):
    """
    输入: mel 或 mel+delta (随你)
          shape = (B, C, n_mels, T)
    输出: utterance-level emotion embedding
    特别适用于 Grad-TTS style loss
    """
    def __init__(self, mel_channels=80, input_channels=1, hidden=256):
        super().__init__()

        # 3D CNN expects input shape (B, 1, C, n_mels, T)
        # 所以把 mel 放入 "depth" 维
        self.input_channels = input_channels

        self.conv1 = Conv3DBlock(1, 16)
        self.pool1 = nn.MaxPool3d((1,2,2))

        self.conv2 = Conv3DBlock(16, 32)
        self.pool2 = nn.MaxPool3d((1,2,2))

        self.conv3 = Conv3DBlock(32, 64)
        self.pool3 = nn.MaxPool3d((1,2,2))

        # BLSTM
        # After three MaxPool3d((1,2,2)) layers the mel dimension (M) is downsampled by 2^3.
        # Compute the resulting per-time flattened feature size and use it as LSTM input_size.
        import math
        downsample_factor = 2 ** 3  # three pool layers each halve mel/time dims
        reduced_mel = math.ceil(mel_channels / downsample_factor)
        lstm_input_size = 64 * input_channels * reduced_mel
        self.blstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.proj = nn.Linear(hidden*2, hidden*2)
        self.att_vec = nn.Parameter(torch.randn(hidden*2))

    def forward(self, mel):
        """
        mel: (B, C, n_mels, T)
        C = 1 或 3
        """
        B, C, M, T = mel.shape

        # reshape -> (B, 1, C, M, T)
        x = mel.unsqueeze(1)

        # 3D-CNN
        x = self.conv1(x); feat1 = x; x = self.pool1(x)
        x = self.conv2(x); feat2 = x; x = self.pool2(x)
        x = self.conv3(x); feat3 = x; x = self.pool3(x)

        # flatten for BLSTM
        B, ch, depth, m2, t2 = x.shape

        # (B, t2, ch*depth*m2)
        x = x.permute(0,4,1,2,3).contiguous().view(B, t2, -1)

        out, _ = self.blstm(x)  # (B, t2, 2H)

        # Attention pooling
        u = torch.tanh(self.proj(out))
        att = torch.matmul(u, self.att_vec)  # (B, t2)
        att = torch.softmax(att, dim=1).unsqueeze(-1)

        emb = torch.sum(out * att, dim=1)  # (B, 2H)
        # print('SER embedding shape:', emb.shape)

        # return emb, {"feat1": feat1, "feat2": feat2, "feat3": feat3}
        return emb
