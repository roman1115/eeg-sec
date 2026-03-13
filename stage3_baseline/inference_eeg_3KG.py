#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG-guided Emotional Voice Conversion inference
+ Emotion-dependent Prosodic Feature Shift evaluation
+ Emotion Embedding Distance (SER-based)

filelist format:
wav|speaker|emotion|text|eeg
"""

import argparse
from pathlib import Path
import pickle
import scipy.io
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
from collections import defaultdict
from sklearn.metrics import silhouette_score


import librosa

import utils
from text.symbols import symbols
from model_stage3_Transformer import SynthesizerTrn
from mel_processing import spectrogram_torch, mel_spectrogram_torch

# SER
from split_ser_encoder import SER_Embedding


EMO_MAP = {
    0: "Neutral",
    1: "Angry",
    2: "Happy",
    3: "Sad"
}

# ==================================================
# 1. filelist
# ==================================================
def parse_filelist(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            wav, spk, emo, text, eeg = line.split("|")
            samples.append({
                "wav": wav,
                "speaker": spk,
                "emotion": int(emo),
                "text": text,
                "eeg": eeg,
            })
    return samples


def group_by_speaker(samples):
    groups = {}
    for s in samples:
        groups.setdefault(s["speaker"], []).append(s)
    return groups


# ==================================================
# 2. EEG loader
# ==================================================
def load_eeg(eeg_path):
    if eeg_path.endswith(".pkl"):
        with open(eeg_path, "rb") as f:
            eeg = pickle.load(f)
    elif eeg_path.endswith(".mat"):
        mat = scipy.io.loadmat(eeg_path)
        eeg = mat["eeg_trial"].T
    else:
        raise ValueError("Unsupported EEG format")

    if eeg.ndim == 2:
        eeg = eeg[:, :, None]

    return torch.from_numpy(eeg.astype(np.float32))


# ==================================================
# 3. Prosodic feature extraction
# ==================================================
def extract_prosody(wav, sr):
    wav = wav.astype(np.float32)

    f0 = librosa.yin(wav, fmin=50, fmax=600, sr=sr)
    f0 = f0[f0 > 0]
    mean_f0 = float(np.mean(f0)) if len(f0) > 0 else 0.0

    energy = float(np.mean(wav ** 2))
    duration = len(wav) / sr

    return {
        "f0": mean_f0,
        "energy": energy,
        "duration": duration,
    }


# ==================================================
# 4. SER utilities
# ==================================================
def ser_predict(ser_model, wav_path, hps, device):
    wav, _ = utils.load_wav_to_torch(str(wav_path))
    wav = wav / hps.data.max_wav_value
    wav = wav.unsqueeze(0).to(device)

    mel = mel_spectrogram_torch(
        wav,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
        center=False,
    )

    with torch.no_grad():
        _, logits = ser_model(mel)
        pred = torch.argmax(logits, dim=-1).item()

    return pred


def ser_embedding(ser_model, wav_path, hps, device):
    wav, _ = utils.load_wav_to_torch(str(wav_path))
    wav = wav / hps.data.max_wav_value
    wav = wav.unsqueeze(0).to(device)

    mel = mel_spectrogram_torch(
        wav,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
        center=False,
    )

    with torch.no_grad():
        emb, _ = ser_model(mel)
        emb = F.normalize(emb, dim=-1)

    return emb.squeeze(0)  # [D]


# ==================================================
# 5. Single VC + Evaluation
# ==================================================
def run_single_conversion(
    net_g, ser_model, emotion_prototypes, hps,
    src, trg,
    wav_root, out_dir,
    device, stats
):
    src_wav_path = Path(wav_root) / src["wav"]
    wav_src, _ = utils.load_wav_to_torch(str(src_wav_path))
    wav_src = wav_src / hps.data.max_wav_value

    pros_src = extract_prosody(wav_src.numpy(), hps.data.sampling_rate)

    spec = spectrogram_torch(
        wav_src.unsqueeze(0),
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    ).to(device)

    spec_len = torch.LongTensor([spec.shape[2]]).to(device)

    eeg = load_eeg(trg["eeg"]).unsqueeze(0).to(device).squeeze(-1)

    sid_src = torch.LongTensor([int(src["speaker"])]).to(device)
    sid_trg = torch.LongTensor([int(trg["speaker"])]).to(device)

    with torch.no_grad():
        audio, _, _ = net_g.voice_conversion(
            y=spec,
            y_lengths=spec_len,
            y1=None,
            y1_lengths=None,
            eeg=eeg,
            sid_src=sid_src,
            sid_trg=sid_trg,
        )

    audio = audio.squeeze().cpu().numpy()

    spk_dir = Path(out_dir) / f"spk_{src['speaker']}"
    src_dir = spk_dir / "source"
    conv_dir = spk_dir / "converted"
    src_dir.mkdir(parents=True, exist_ok=True)
    conv_dir.mkdir(parents=True, exist_ok=True)

    src_save_path = src_dir / Path(src["wav"]).name
    if not src_save_path.exists():
        write(str(src_save_path), hps.data.sampling_rate,
              (wav_src.numpy() * 32767).astype("int16"))

    out_path = conv_dir / f"{Path(src['wav']).stem}_to_{Path(trg['wav']).stem}.wav"
    write(str(out_path), hps.data.sampling_rate,
          (audio * 32767).astype("int16"))

    pros_conv = extract_prosody(audio, hps.data.sampling_rate)

    emo = trg["emotion"]
    stats["prosody"][emo]["df0"].append(pros_conv["f0"] - pros_src["f0"])
    stats["prosody"][emo]["denergy"].append(pros_conv["energy"] - pros_src["energy"])
    stats["prosody"][emo]["dduration"].append(pros_conv["duration"] - pros_src["duration"])
    stats["count"] += 1

    if ser_model is not None:
        pred = ser_predict(ser_model, out_path, hps, device)
        stats["ser"]["total"] += 1
        stats["ser"]["confusion"][emo][pred] += 1
        stats["ser"]["per_emo"][emo]["total"] += 1
        if pred == emo:
            stats["ser"]["correct"] += 1
            stats["ser"]["per_emo"][emo]["correct"] += 1

        emb_conv = ser_embedding(ser_model, out_path, hps, device)
        proto = emotion_prototypes[emo]

        # embedding distance
        dist = 1.0 - torch.sum(emb_conv * proto).item()
        stats["embed_dist"][emo].append(dist)

        # 保存用于聚类分析
        stats["all_embeddings"].append(emb_conv.cpu().numpy())
        stats["all_emotions"].append(emo)

        # Direction consistency
        emb_src = ser_embedding(ser_model, src_wav_path, hps, device)

        direction_conv = emb_conv - emb_src
        direction_target = proto - emb_src

        cos_sim = F.cosine_similarity(
            direction_conv.unsqueeze(0),
            direction_target.unsqueeze(0)
        ).item()

        stats["direction_cos"][emo].append(cos_sim)



# ==================================================
# 6. Group inference
# ==================================================
def run_group_inference(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hps = utils.get_hparams_from_file(args.config)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    net_g.eval()
    utils.load_checkpoint(args.checkpoint, net_g, None)

    ser_model = None
    if args.ser_ckpt:
        ser_model = SER_Embedding(num_classes=4).to(device)
        ser_model.load_state_dict(torch.load(args.ser_ckpt, map_location=device))
        ser_model.eval()

    samples = parse_filelist(args.filelist)
    groups = group_by_speaker(samples)

    emotion_prototypes = {}
    if ser_model is not None:
        tmp = defaultdict(list)
        for s in samples:
            emb = ser_embedding(ser_model, Path(args.wav_root) / s["wav"], hps, device)
            tmp[s["emotion"]].append(emb)
        for emo in tmp:
            emotion_prototypes[emo] = torch.stack(tmp[emo]).mean(dim=0)

    stats = {
    "count": 0,
    "prosody": defaultdict(lambda: {"df0": [], "denergy": [], "dduration": []}),
    "ser": {
        "total": 0,
        "correct": 0,
        "confusion": defaultdict(lambda: defaultdict(int)),
        "per_emo": defaultdict(lambda: {"total": 0, "correct": 0})
    },
    "embed_dist": defaultdict(list),

    # 新增
    "all_embeddings": [],
    "all_emotions": [],
    "direction_cos": defaultdict(list)
}

    for spk, items in groups.items():
        srcs = [x for x in items if x["emotion"] == 0]
        trgs = [x for x in items if x["emotion"] != 0]
        if not srcs or not trgs:
            continue
        if args.use_first_src:
            srcs = srcs[:1]

        for src in srcs:
            for trg in trgs:
                run_single_conversion(
                    net_g, ser_model, emotion_prototypes, hps,
                    src, trg,
                    args.wav_root, args.out_dir,
                    device, stats
                )

    
    print("\n========== Objective Metrics ==========")

    # ---------- Prosody ----------
    print("\n====== Emotion-dependent Prosodic Shift ======")
    for emo in sorted(stats["prosody"]):
        df0 = np.array(stats["prosody"][emo]["df0"])
        de = np.array(stats["prosody"][emo]["denergy"])
        dd = np.array(stats["prosody"][emo]["dduration"])

        if len(df0) == 0:
            continue

        print(f"\nEmotion {EMO_MAP[emo]}:")
        print(f"  ΔF0     : mean={df0.mean():.2f}, std={df0.std():.2f}")
        print(f"  ΔEnergy : mean={de.mean():.6f}, std={de.std():.6f}")
        print(f"  ΔDur    : mean={dd.mean():.3f}s, std={dd.std():.3f}s")

    # ---------- SER ----------
    if ser_model is not None and stats["ser"]["total"] > 0:
        print("\n====== SER-based Emotion Accuracy ======")

        for emo in sorted(stats["ser"]["per_emo"]):
            total = stats["ser"]["per_emo"][emo]["total"]
            correct = stats["ser"]["per_emo"][emo]["correct"]
            acc = correct / max(1, total)

            print(f"Neutral → {EMO_MAP[emo]}: "
                  f"{acc*100:.2f}% ({correct}/{total})")

        overall = stats["ser"]["correct"] / stats["ser"]["total"]
        print(f"\nOverall SER Accuracy: {overall*100:.2f}% "
              f"({stats['ser']['correct']}/{stats['ser']['total']})")

    # ---------- Emotion Embedding Distance ----------
    print("\n====== Emotion Embedding Distance (↓) ======")
    for emo in sorted(stats["embed_dist"]):
        d = np.array(stats["embed_dist"][emo])
        if len(d) == 0:
            continue
        print(f"{EMO_MAP[emo]} : mean={d.mean():.4f}, std={d.std():.4f}")
    if len(stats["all_embeddings"]) > 10:
        X = np.array(stats["all_embeddings"])
        y = np.array(stats["all_emotions"])

        try:
            sil = silhouette_score(X, y, metric="cosine")
            print("\n====== Emotion Cluster Separability (↑) ======")
            print(f"Silhouette Score: {sil:.4f}")
        except Exception as e:
            print("\n[Warning] Silhouette score failed:", e)
    print("\n====== Emotion Direction Consistency (↑) ======")
    for emo in sorted(stats["direction_cos"]):
        arr = np.array(stats["direction_cos"][emo])
        if len(arr) == 0:
            continue
        print(f"{EMO_MAP[emo]} : mean={arr.mean():.4f}, std={arr.std():.4f}")




# ==================================================
# 7. CLI
# ==================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", required=True)
    parser.add_argument("--wav-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ser-ckpt", default=None)
    parser.add_argument("--out-dir", default="./eeg_vc_out")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-first-src", action="store_true")
    args = parser.parse_args()

    run_group_inference(args)


if __name__ == "__main__":
    main()
