import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import logging
from sklearn.manifold import TSNE

from models import SynthesizerTrn
from text.symbols import symbols
from EEG_Encoder_CNN_Transformer import EEGFeatureExtractor
import utils
from data_utils import (
    TextAudioSpeakerEmotionEEGLoader,
    TextAudioSpeakerEmotionEEGCollate
)
import commons
import tempfile

tempfile.tempdir = "./tmp"
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# =============================
# EEG-only t-SNE 可视化
# =============================
def plot_tsne_eeg_only(tsne_data, save_dir, epoch):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    os.makedirs(save_dir, exist_ok=True)

    eeeg = tsne_data["eeeg_features"]
    labels = tsne_data["labels"]

    if eeeg.ndim != 2:
        raise ValueError("EEG features must be [N, D]")
    if labels.ndim != 1:
        raise ValueError("Labels must be [N]")
    if len(labels) != eeeg.shape[0]:
        raise ValueError("Mismatch between EEG features and labels")

    N, D = eeeg.shape

    # -------- Standardize --------
    eeeg = StandardScaler().fit_transform(eeeg)

    # -------- PCA --------
    pca_dim = min(50, D, max(1, N - 1))
    eeeg = PCA(n_components=pca_dim).fit_transform(eeeg)

    # -------- t-SNE --------
    perplexity = min(30, max(5, (N - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate=200,
        init="random",
        random_state=42
    )
    proj = tsne.fit_transform(eeeg)

    # -------- Plot --------
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'm', 'y', 'c']
    for i, lab in enumerate(np.unique(labels)):
        idx = labels == lab
        plt.scatter(
            proj[idx, 0],
            proj[idx, 1],
            s=18,
            alpha=0.75,
            c=colors[i % len(colors)],
            label=f"Emotion {lab}",
            edgecolors="none"
        )

    plt.title(f"EEG Feature t-SNE (Epoch {epoch})")
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"tsne_eeg_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()


# =============================
# Stage2 Evaluation
# =============================
def eval_stage2(net_g, align_module, eeg_encoder, eval_loader, device):
    net_g.eval()
    eeg_encoder.eval()

    total_loss, total_align, total_cls = 0, 0, 0
    eeeg_list, labels_list = [], []

    with torch.no_grad():
        for batch in eval_loader:
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg = batch
            x, spec, eeg = x.to(device), spec.to(device), eeg.to(device)
            x_lengths, spec_lengths = x_lengths.to(device), spec_lengths.to(device)
            sid, eid = sid.to(device), eid.to(device)

            eeg = eeg.squeeze(1) if eeg.dim() == 4 else eeg
            eeg = eeg.squeeze(-1)

            indices = torch.arange(
                0, spec.size(2),
                utils.get_hparams().train.segment_size //
                utils.get_hparams().data.hop_length,
                device=device
            )
            spec = commons.slice_segments(
                spec,
                indices,
                utils.get_hparams().train.segment_size //
                utils.get_hparams().data.hop_length
            )

            _, _, _, eu, *_ = net_g.enc_q(spec, spec_lengths, g=sid, temp=1.0)
            eu_proj, _ = align_module(eu)
            eeeg_proj, eegc = eeg_encoder(eeg)

            loss_align = F.mse_loss(eeeg_proj, eu_proj)
            loss_cls = F.cross_entropy(eegc, eid)
            loss = 0.9 * loss_align + 0.1 * loss_cls

            total_loss += loss.item()
            total_align += loss_align.item()
            total_cls += loss_cls.item()

            eeeg_list.append(eeeg_proj.squeeze(-1).cpu().numpy())
            labels_list.append(eid.cpu().numpy())

    tsne_data = {
        "eeeg_features": np.concatenate(eeeg_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0)
    }

    n = len(eval_loader)
    print(f"[Eval] Total:{total_loss/n:.4f} Align:{total_align/n:.4f} Cls:{total_cls/n:.4f}")
    return total_loss / n, total_align / n, total_cls / n, tsne_data


# =============================
# Stage2 Training
# =============================
def train_stage2(
    stage1_ckpt,
    save_dir="./checkpoints/stage2",
    epochs=1000,
    lr=2e-4,
    device="cuda",
    eval_interval=2,
    save_epoch=1
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # -------- Load Stage1 --------
    ckpt = torch.load(stage1_ckpt, map_location=device)
    hps = utils.get_hparams()

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).to(device)

    net_g.load_state_dict(ckpt.get("net_g", ckpt), strict=False)
    align = net_g.align

    for p in net_g.parameters():
        p.requires_grad = False
    for p in align.parameters():
        p.requires_grad = False

    net_g.eval()
    align.eval()
    print("✅ Stage1 loaded")

    # -------- EEG Encoder --------
    eeg_encoder = EEGFeatureExtractor().to(device)
    eeg_encoder.align.load_state_dict(align.state_dict())

    optimizer = optim.AdamW(eeg_encoder.parameters(), lr=lr, weight_decay=1e-4)

    train_loader = DataLoader(
        TextAudioSpeakerEmotionEEGLoader(hps.data.training_files, hps.data),
        batch_size=hps.train.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=TextAudioSpeakerEmotionEEGCollate()
    )

    eval_loader = DataLoader(
        TextAudioSpeakerEmotionEEGLoader(hps.data.validation_files, hps.data),
        batch_size=hps.train.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=TextAudioSpeakerEmotionEEGCollate()
    )

    best_loss = 1e9

    for epoch in range(epochs):
        eeg_encoder.train()
        total_loss = 0

        for batch in train_loader:
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg = batch
            spec, eeg = spec.to(device), eeg.to(device)
            spec_lengths = spec_lengths.to(device)
            sid, eid = sid.to(device), eid.to(device)

            eeg = eeg.squeeze(1) if eeg.dim() == 4 else eeg
            eeg = eeg.squeeze(-1)

            indices = torch.arange(
                0, spec.size(2),
                hps.train.segment_size // hps.data.hop_length,
                device=device
            )
            spec = commons.slice_segments(
                spec,
                indices,
                hps.train.segment_size // hps.data.hop_length
            )

            with torch.no_grad():
                _, _, _, eu, *_ = net_g.enc_q(spec, spec_lengths, g=sid, temp=1.0)
                eu_proj, _ = align(eu)

            eeeg_proj, eegc = eeg_encoder(eeg)

            loss_align = F.mse_loss(eeeg_proj, eu_proj)
            loss_cls = F.cross_entropy(eegc, eid)
            loss = 0.9 * loss_align + 0.1 * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch}] Train Loss: {total_loss/len(train_loader):.4f}")

        # -------- Eval --------
        if epoch % eval_interval == 0:
            eval_loss, _, _, tsne_data = eval_stage2(
                net_g, align, eeg_encoder, eval_loader, device
            )
            plot_tsne_eeg_only(tsne_data, save_dir, epoch)

            if eval_loss < best_loss:
                best_loss = eval_loss
                torch.save(
                    {"eeg_encoder": eeg_encoder.state_dict(), "epoch": epoch},
                    Path(save_dir) / "stage2_best.pt"
                )

        if epoch % save_epoch == 0:
            torch.save(
                {"eeg_encoder": eeg_encoder.state_dict(), "epoch": epoch},
                Path(save_dir) / f"stage2_epoch_{epoch}.pt"
            )

    print("✅ Stage2 training finished")


# =============================
# Main
# =============================
if __name__ == "__main__":
    train_stage2(
        stage1_ckpt="./logs/EAV_Fine-tuning_100_n=4/G_266000.pth",
        save_dir="./logs/Stage2_CNN+Transformer",
        epochs=1000,
        lr=2e-4,
        eval_interval=2,
        save_epoch=1
    )
