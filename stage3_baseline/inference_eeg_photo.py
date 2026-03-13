#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG-guided Emotional Voice Conversion
+ SER embedding clustering visualization (t-SNE only)

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

import librosa
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
# filelist
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
# EEG loader
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
# SER embedding
# ==================================================
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

    return emb.squeeze(0)


# ==================================================
# t-SNE plot
# ==================================================
def plot_emotion_clusters(embeddings, emotions, save_path):
    X = np.array(embeddings)
    y = np.array(emotions)

    print(f"[Cluster] total samples: {len(X)}")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=0
    )

    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for emo in np.unique(y):
        idx = y == emo
        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=18,
            alpha=0.75,
            label=EMO_MAP[emo]
        )

    plt.legend()
    plt.title("Emotion Clusters of Converted Speech (SER Embedding)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Cluster] saved to {save_path}")


# ==================================================
# Single conversion
# ==================================================
def run_single_conversion(
    net_g, ser_model, hps,
    src, trg,
    wav_root, out_dir,
    device, stats
):
    src_wav_path = Path(wav_root) / src["wav"]
    wav_src, _ = utils.load_wav_to_torch(str(src_wav_path))
    wav_src = wav_src / hps.data.max_wav_value

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
    conv_dir = spk_dir / "converted"
    conv_dir.mkdir(parents=True, exist_ok=True)

    out_path = conv_dir / f"{Path(src['wav']).stem}_to_{Path(trg['wav']).stem}.wav"
    write(str(out_path), hps.data.sampling_rate,
          (audio * 32767).astype("int16"))

    if ser_model is not None:
        emb_conv = ser_embedding(ser_model, out_path, hps, device)
        stats["all_embeddings"].append(emb_conv.cpu().numpy())
        stats["all_emotions"].append(trg["emotion"])


# ==================================================
# Group inference
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

    stats = {
        "all_embeddings": [],
        "all_emotions": []
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
                    net_g, ser_model, hps,
                    src, trg,
                    args.wav_root, args.out_dir,
                    device, stats
                )

    if len(stats["all_embeddings"]) > 10:
        fig_path = Path(args.out_dir) / "emotion_cluster_tsne.png"
        plot_emotion_clusters(stats["all_embeddings"], stats["all_emotions"], fig_path)
    else:
        print("Not enough samples for clustering.")


# ==================================================
# CLI
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
