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
from models import SynthesizerTrn  # 阶段一模型
from text.symbols import symbols
from EEG_encoder import EEGFeatureExtractor
import utils
from data_utils import (
    TextAudioSpeakerEmotionEEGLoader,
    TextAudioSpeakerEmotionEEGCollate
)
import commons
import tempfile
tempfile.tempdir = "./tmp"
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# -----------------------------
# 工具函数
# -----------------------------
def cosine_loss(a, b):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return 1.0 - (a * b).sum(dim=-1).mean()

def plot_tsne_side_by_side(tsne_data, save_dir, epoch):
    """
    Robust PCA -> t-SNE plotting for EEG (left) and Audio (right) side-by-side.
    Expects tsne_data keys:
      - "eeeg_features": np.array [N, D]  (EEG, already numpy)
      - "eu_features":   np.array [N, D]  (Audio, already numpy)
      - "labels":        np.array [N]     (int labels 0..C-1)

    Saves: save_dir/tsne_side_by_side_epoch_{epoch}.png
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    os.makedirs(save_dir, exist_ok=True)

    eeeg = tsne_data["eeeg_features"]
    eu = tsne_data["eu_features"]
    labels = tsne_data["labels"]

    # Basic sanity
    if eeeg.ndim != 2 or eu.ndim != 2:
        raise ValueError("eeeg_features/eu_features must be 2D arrays [N, D].")
    if labels.ndim != 1:
        raise ValueError("labels must be 1D array [N].")
    if len(labels) != eeeg.shape[0] or len(labels) != eu.shape[0]:
        raise ValueError("labels length must match number of samples in features.")

    N = eeeg.shape[0]
    feature_dim = eeeg.shape[1]

    # ---------- Standardize ----------
    scaler_eeg = StandardScaler()
    scaler_audio = StandardScaler()
    eeeg_norm = scaler_eeg.fit_transform(eeeg)
    eu_norm = scaler_audio.fit_transform(eu)

    # ---------- PCA (to at most 50 dims, but also <= n_samples-1 and <= feature_dim) ----------
    max_pca = 50
    pca_dim = min(max_pca, feature_dim, max(1, N - 1))
    if pca_dim < 1:
        pca_dim = 1

    pca_eeg = PCA(n_components=pca_dim)
    pca_audio = PCA(n_components=pca_dim)
    eeeg_pca = pca_eeg.fit_transform(eeeg_norm)
    eu_pca = pca_audio.fit_transform(eu_norm)

    # ---------- TSNE ----------
    # choose perplexity safely: typical range [5,50], <= (N-1)/3
    max_perp = max(5, min(50, (N - 1) // 3))
    perplexity = min(30, max_perp)
    # fallback if N is very small
    if perplexity < 1:
        perplexity = 1

    tsne_params = dict(n_components=2, perplexity=float(perplexity), learning_rate=200, init='random', random_state=42)
    tsne_eeg = TSNE(**tsne_params)
    tsne_audio = TSNE(**tsne_params)

    eeeg_proj = tsne_eeg.fit_transform(eeeg_pca)
    eu_proj = tsne_audio.fit_transform(eu_pca)

    # ---------- Plot side-by-side ----------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['r', 'g', 'b', 'm', 'y', 'c']  # support more than 4 if needed
    unique_labels = np.unique(labels)
    # left: EEG
    ax = axes[0]
    for i, lab in enumerate(unique_labels):
        idxs = np.where(labels == lab)[0]
        if idxs.size == 0:
            continue
        color = colors[int(i % len(colors))]
        ax.scatter(eeeg_proj[idxs, 0], eeeg_proj[idxs, 1],
                   c=color, s=15, alpha=0.75, label=f'EEG-{lab}', edgecolors='none')
    ax.set_title(f"EEG Features t-SNE Epoch {epoch}")
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')

    # right: Audio
    ax = axes[1]
    for i, lab in enumerate(unique_labels):
        idxs = np.where(labels == lab)[0]
        if idxs.size == 0:
            continue
        color = colors[int(i % len(colors))]
        ax.scatter(eu_proj[idxs, 0], eu_proj[idxs, 1],
                   c=color, s=15, marker='s', alpha=0.75, label=f'Audio-{lab}', edgecolors='none')
    ax.set_title(f"Audio Features t-SNE Epoch {epoch}")
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"tsne_side_by_side_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)



# -----------------------------
# 阶段二评估
# -----------------------------
def eval_stage2(net_g, align_module, eeg_encoder, eval_loader, device):
    net_g.eval()
    eeg_encoder.eval()
    total_loss, total_align, total_eegcls = 0, 0, 0
    eeeg_features_list, eu_features_list, labels_list = [], [], []

    with torch.no_grad():
        for batch in eval_loader:
            x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg = batch
            x = x.to(device)
            spec = spec.to(device)
            eeg = eeg.to(device)
            x_lengths = x_lengths.to(device)
            spec_lengths = spec_lengths.to(device)
            sid = sid.to(device)
            eid = eid.to(device)

            eeg = eeg.squeeze(1) if eeg.dim() == 4 else eeg
            eeg = eeg.squeeze(-1)

            indices = torch.arange(0, spec.size(2), utils.get_hparams().train.segment_size // utils.get_hparams().data.hop_length, device=device)
            spec = commons.slice_segments(spec, indices, utils.get_hparams().train.segment_size // utils.get_hparams().data.hop_length)

            z, m, logs, eu, *_ = net_g.enc_q(spec, spec_lengths, g=sid, temp=1.0)
            eu_proj, ec = align_module(eu)
            eeeg_proj, eegc = eeg_encoder(eeg)

            loss_align = F.mse_loss(eeeg_proj, eu_proj)
            #teacher_prob = F.softmax(ec, dim=-1)
            #student_logprob = F.log_softmax(eegc, dim=-1)
            #loss_cls = F.kl_div(student_logprob, teacher_prob, reduction="batchmean")
            #loss = 0.9*loss_align + 0.1*loss_cls
            loss_eegcls = F.cross_entropy(eegc, eid)
            loss = 0.9*loss_align + 0.1*loss_eegcls

            total_loss += loss.item()
            total_align += loss_align.item()
            total_eegcls += loss_eegcls.item()

            eeeg_features_list.append(eeeg_proj.squeeze(-1).cpu().numpy())
            eu_features_list.append(eu_proj.squeeze(-1).cpu().numpy())
            labels_list.append(eid.cpu().numpy())

    tsne_data = {
        "eeeg_features": np.concatenate(eeeg_features_list, axis=0),
        "eu_features": np.concatenate(eu_features_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0)
    }

    avg_loss = total_loss / len(eval_loader)
    avg_align = total_align / len(eval_loader)
    avg_eegcls = total_eegcls / len(eval_loader)
    print(f"eval  {{ Total:{avg_loss:.4f} Align:{avg_align:.4f} eeg_cls:{avg_eegcls:.4f} }}")
    return avg_loss, avg_align, avg_eegcls, tsne_data


# -----------------------------
# 阶段二训练
# -----------------------------
def train_stage2(stage1_ckpt: str, save_dir: str = "./checkpoints/stage2", epochs: int = 1000,
                 lr: float = 2e-4, device: str = "cuda", eval_interval: int = 2, save_epoch: int =5):

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    # ---------- Loss log files ----------
    train_log_path = os.path.join(save_dir, "train_losses.txt")
    eval_log_path = os.path.join(save_dir, "eval_losses.txt")

    # ---------- 1. 加载阶段一模型 ----------
    ckpt = torch.load(stage1_ckpt, map_location=device)
    hps = utils.get_hparams()
    net_g = SynthesizerTrn(len(symbols),
                            hps.data.filter_length // 2 + 1,
                            hps.train.segment_size // hps.data.hop_length,
                            **hps.model).to(device)

    net_g_state = ckpt.get("net_g", ckpt)
    net_g.load_state_dict(net_g_state, strict=False)

    align = getattr(net_g, "align", None)
    if align is None:
        raise RuntimeError("未检测到 align 模块，请确认阶段一模型正确。")

    for p in net_g.parameters():
        p.requires_grad = False
    for p in align.parameters():
        p.requires_grad = False
    net_g.eval()
    align.eval()
    print("✅ Stage1 模型加载完毕")

    # ---------- 2. 初始化 EEGEncoder ----------
    eeg_encoder = EEGFeatureExtractor().to(device)
    with torch.no_grad():
        eeg_encoder.align.load_state_dict(net_g.align.state_dict())
    optimizer = optim.AdamW(eeg_encoder.parameters(), lr=lr, weight_decay=1e-4)

    train_dataset = TextAudioSpeakerEmotionEEGLoader(hps.data.training_files, hps.data)
    collate_fn = TextAudioSpeakerEmotionEEGCollate()
    train_loader = DataLoader(train_dataset, batch_size=hps.train.batch_size,
                              shuffle=True, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    eval_dataset = TextAudioSpeakerEmotionEEGLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, batch_size=hps.train.batch_size,
                             shuffle=False, pin_memory=True, collate_fn=collate_fn, drop_last=True)

    # ---------- 3. 训练循环 ----------
    train_losses, val_losses = [], []
    best_eval_loss = 1e9
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,6))

    for epoch in range(epochs):
        eeg_encoder.train()
        total_loss, total_align, total_eegcls = 0,0,0
        num_batches = 0

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, sid, eid, eeg) in enumerate(train_loader):
            x = x.to(device)
            spec = spec.to(device)
            eeg = eeg.to(device)
            x_lengths = x_lengths.to(device)
            spec_lengths = spec_lengths.to(device)
            sid = sid.to(device)
            eid = eid.to(device)

            eeg = eeg.squeeze(1) if eeg.dim() == 4 else eeg
            eeg = eeg.squeeze(-1)

            indices = torch.arange(0, spec.size(2), hps.train.segment_size // hps.data.hop_length, device=device)
            spec = commons.slice_segments(spec, indices, hps.train.segment_size // hps.data.hop_length)

            with torch.no_grad():
                z, m, logs, eu, *_ = net_g.enc_q(spec, spec_lengths, g=sid, temp=1.0)
                eu_proj, ec = align(eu)

            eeeg_proj, eegc = eeg_encoder(eeg)

            loss_align = F.mse_loss(eeeg_proj, eu_proj)
            loss_eegcls = F.cross_entropy(eegc, eid)
            #teacher_prob = F.softmax(ec, dim=-1)
            #student_logprob = F.log_softmax(eegc, dim=-1)
            #loss_cls = F.kl_div(student_logprob, teacher_prob, reduction="batchmean")
            #loss = 0.9*loss_align + 0.1*loss_cls
            loss = 0.9*loss_align + 0.1*loss_eegcls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_align += loss_align.item()
            total_eegcls += loss_eegcls.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_align = total_align / num_batches
        avg_eegcls = total_eegcls / num_batches
        print(f"[Epoch {epoch+1}/{epochs}] Total:{avg_loss:.4f} Align:{avg_align:.4f} eeg_cls:{avg_eegcls:.4f}")
        train_losses.append(avg_loss)
        # ---------- save training loss ----------
        with open(train_log_path, "a") as f:
            f.write(f"{epoch+1} {avg_loss:.6f} {avg_align:.6f} {avg_eegcls:.6f}\n")


        # ---------- 4. 验证与 t-SNE ----------
        if epoch % eval_interval == 0 or epoch == epochs-1:
            eval_loss, eval_align, eval_eegcls, tsne_data = eval_stage2(net_g, align, eeg_encoder, eval_loader, device)
            
            val_losses.append(eval_loss)
            plot_tsne_side_by_side(tsne_data, save_dir, epoch)
            # ---------- save eval loss ----------
            with open(eval_log_path, "a") as f:
                f.write(f"{epoch+1} {eval_loss:.6f} {eval_align:.6f} {eval_eegcls:.6f}\n")

            if eval_loss < best_eval_loss:
                torch.save({"eeg_encoder": eeg_encoder.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                           Path(save_dir)/f"stage2_best.pt")
                best_eval_loss = eval_loss

        # ---------- 5. 绘制训练曲线 ----------
        ax.clear()
        ax.plot(train_losses, label='Training Loss', color='blue')
        ax.plot(val_losses, label='Validation Loss', color='orange')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        plt.pause(0.1)
        plt.savefig(os.path.join(save_dir, "training_validation_loss.png"))

        # ---------- 6. 保存模型 ----------
        if epoch % save_epoch == 0 and epoch > 0:
            torch.save({"eeg_encoder": eeg_encoder.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                       Path(save_dir)/f"stage2_epoch_{epoch}.pt")

    torch.save({"eeg_encoder": eeg_encoder.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs-1},
               Path(save_dir)/"stage2_last.pt")
    plt.ioff()
    plt.close()
    print("✅ 阶段二训练完成，模型保存至：", save_dir)


# -----------------------------
# 主函数
# -----------------------------
if __name__ == "__main__":
    train_stage2(
        stage1_ckpt="./logs/EAV_Fine-tuning_100_n=4/G_266000.pth",
        save_dir="./logs/compare_stage2/Stage2_CNN_TCN_Transformer",
        epochs=1000,
        lr=2e-4,
        eval_interval=2,
        save_epoch=1
    )
