import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from models import SynthesizerTrn
from text.symbols import symbols
from EEG_Encoder_CNN_Transformer import EEGFeatureExtractor
import utils
import commons
from data_utils import (
    TextAudioSpeakerEmotionEEGLoader,
    TextAudioSpeakerEmotionEEGCollate
)

# =============================
# Metrics
# =============================
def compute_metrics(E, U):
    E = F.normalize(E, dim=1)
    U = F.normalize(U, dim=1)
    pcs = F.cosine_similarity(E, U, dim=1).mean().item()
    nmse = F.mse_loss(E, U).item()
    return pcs, nmse


# =============================
# Feature extraction (ROBUST)
# - Match training behavior: slice spec to fixed seg_frames
# - Fix spec_lengths if it is NOT spectrogram-frame lengths
# =============================
@torch.no_grad()
def extract_features(net_g, align_module, eeg_encoder, loader, device, hps, max_frames=None):
    net_g.eval()
    align_module.eval()
    eeg_encoder.eval()

    eeg_feats = []
    audio_feats = []

    seg_frames = hps.train.segment_size // hps.data.hop_length  # 和训练一致
    hop = hps.data.hop_length

    for batch in loader:
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg = batch

        spec = spec.to(device)                 # [B, 513, T_pad]
        spec_lengths = spec_lengths.to(device) # [B]
        y_lengths = y_lengths.to(device)       # [B] (waveform length, samples)
        sid = sid.to(device)
        eeg = eeg.to(device)

        # EEG shape fix (与你训练一致)
        eeg = eeg.squeeze(1) if eeg.dim() == 4 else eeg
        eeg = eeg.squeeze(-1)  # [B, Chans, Time]

        B, _, T_pad = spec.shape

        # ------------------------------------------------------------
        # 1) 纠错 spec_lengths：如果它明显不像“谱帧长度”，用 y_lengths 推
        #    典型异常：spec pad 到 430，但 spec_lengths.max() 只有 64（其实是 text length）
        # ------------------------------------------------------------
        # 用 waveform 长度推导谱帧长度（粗略但足够用于 mask）
        spec_len_from_y = torch.clamp(y_lengths // hop, min=1, max=T_pad)

        # 如果 spec_lengths 最大值明显小于当前 spec 的时间长度（阈值可稍微宽松）
        if int(spec_lengths.max().item()) < max(2, T_pad // 2):
            spec_lengths = spec_len_from_y

        # ------------------------------------------------------------
        # 2) 和训练一致：把 spec 切成固定 seg_frames
        #    注意：切完后也要把 spec_lengths clamp 到 seg_frames
        # ------------------------------------------------------------
        # 生成 indices：保证 slice_segments 不越界
        # 如果 T_pad < seg_frames，commons.slice_segments 可能不适配，这里先做 padding/截断策略
        if T_pad >= seg_frames:
            indices = torch.arange(0, T_pad, seg_frames, device=device)
            spec = commons.slice_segments(spec, indices, seg_frames)  # [B, 513, seg_frames]
        else:
            # T_pad 太短：直接截断/补零到 seg_frames
            pad = seg_frames - T_pad
            spec = F.pad(spec, (0, pad))  # pad time dim to seg_frames

        # 切片后长度对齐
        spec_lengths = torch.clamp(spec_lengths, min=1, max=seg_frames)

        # ------------------------------------------------------------
        # 3) 可选 max_frames（通常不需要了，因为已经固定 seg_frames）
        # ------------------------------------------------------------
        if max_frames is not None and max_frames > 0:
            T_new = min(spec.size(2), int(max_frames))
            spec = spec[:, :, :T_new]
            spec_lengths = torch.clamp(spec_lengths, max=T_new)

        # -------- audio emotion feature --------
        _, _, _, eu, *_ = net_g.enc_q(spec, spec_lengths, g=sid, temp=1.0)
        eu_proj, _ = align_module(eu)          # [B,256,1]

        # -------- eeg emotion feature --------
        eeeg_proj, _ = eeg_encoder(eeg)        # [B,256,1]

        audio_feats.append(eu_proj.squeeze(-1).cpu())  # [B,256]
        eeg_feats.append(eeeg_proj.squeeze(-1).cpu())  # [B,256]

    E = torch.cat(eeg_feats, dim=0)    # [N,256]
    U = torch.cat(audio_feats, dim=0)  # [N,256]
    return E, U


# =============================
# Main
# =============================
def inference_stage2_metrics(
    stage1_ckpt,
    stage2_ckpt,
    save_dir="./logs/Stage2_CNN+Transformer",
    split="val",
    device="cuda",
    max_frames=None
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # -------- Load hps --------
    hps = utils.get_hparams()

    # -------- Load Stage1 --------
    ckpt1 = torch.load(stage1_ckpt, map_location=device)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    net_g.load_state_dict(ckpt1.get("net_g", ckpt1), strict=False)
    align = net_g.align

    for p in net_g.parameters():
        p.requires_grad = False
    for p in align.parameters():
        p.requires_grad = False

    net_g.eval()
    align.eval()
    print("✅ Stage1 loaded")

    # -------- Load Stage2 --------
    eeg_encoder = EEGFeatureExtractor().to(device)
    ckpt2 = torch.load(stage2_ckpt, map_location=device)
    state = ckpt2["eeg_encoder"] if isinstance(ckpt2, dict) and "eeg_encoder" in ckpt2 else ckpt2
    eeg_encoder.load_state_dict(state, strict=True)
    eeg_encoder.eval()
    print("✅ Stage2 loaded")

    # -------- DataLoader --------
    filelist = hps.data.validation_files if split == "val" else hps.data.training_files

    loader = DataLoader(
        TextAudioSpeakerEmotionEEGLoader(filelist, hps.data),
        batch_size=hps.train.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=TextAudioSpeakerEmotionEEGCollate()
    )

    # -------- Extract + Metrics --------
    E, U = extract_features(net_g, align, eeg_encoder, loader, device, hps, max_frames=max_frames)
    pcs, nmse = compute_metrics(E, U)

    print("\n====== Cross-modal Alignment Metrics ======")
    print(f"Paired Cosine Similarity (↑): {pcs:.4f}")
    print(f"Normalized MSE (↓):          {nmse:.6f}")

    out_json = Path(save_dir) / f"metrics_{split}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"PCS": pcs, "nMSE": nmse, "max_frames": max_frames}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved metrics to: {out_json}")


if __name__ == "__main__":
    inference_stage2_metrics(
        stage1_ckpt="./logs/EAV_Fine-tuning_100_n=4/G_266000.pth",
        stage2_ckpt="./logs/Stage2_CNN+Transformer/stage2_epoch_750.pt",
        save_dir="./logs/Stage2_CNN+Transformer",
        split="val",
        device="cuda",
        max_frames=None
    )
