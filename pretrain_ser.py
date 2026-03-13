#!/usr/bin/env python3
"""Minimal example to pretrain SER_Embedding and save checkpoint.

This script uses random data by default as a placeholder. Replace the
`RandomSERDataset` with your real dataset that yields (mel, label) where
`mel` is a tensor of shape [C, n_mels, T] (C usually 1) and `label` is an int.

Example usage:
  python scripts/pretrain_ser.py --epochs 20 --save-path checkpts/ser_embedding.pt

"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import os
import sys
# progress bar
from tqdm.auto import tqdm
# Ensure repository root is on sys.path so local modules (e.g. data.py) can be imported
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from data import MelLabelDataset

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram

from ser_encoder import SER_Embedding



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # replace RandomSERDataset with your real dataset
    train_dataset = MelLabelDataset(filelist_path=args.filelist, 
                             cmudict_path="resources/cmu_dictionary",
                             add_blank=True,
                             n_fft=1024, n_mels=80, sample_rate=22050,
                             hop_length=256, win_length=1024, f_min=0., f_max=8000)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # validation loader (optional)
    val_loader = None
    if args.val_filelist is not None:
        val_dataset = MelLabelDataset(filelist_path=args.val_filelist,
                                      cmudict_path="resources/cmu_dictionary",
                                      add_blank=True,
                                      n_fft=1024, n_mels=80, sample_rate=22050,
                                      hop_length=256, win_length=1024, f_min=0., f_max=8000)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ser = SER_Embedding(mel_channels=80, input_channels=1, hidden=args.hidden)
    classifier = nn.Linear(args.hidden * 2, args.num_classes)

    # move models to device and enable multi-GPU if available
    ser = ser.to(device)
    classifier = classifier.to(device)

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_dataparallel = n_gpus > 1
    if use_dataparallel:
        print(f"Using DataParallel on {n_gpus} GPUs")
        ser = nn.DataParallel(ser)
        classifier = nn.DataParallel(classifier)

    criterion = nn.CrossEntropyLoss()
    # create optimizer after potential DataParallel wrapping so .parameters() is correct
    optimizer = optim.Adam(list(ser.parameters()) + list(classifier.parameters()), lr=args.lr)

    best_val_loss = None
    ser.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_acc = 0.0
        # training loop with progress bar
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            mel = batch['y'].to(device)
            label = batch['x'].to(device)
            # labels may be shape [B,1] -> flatten to [B]
            if label.ndim > 1:
                label = label.view(-1)
            # add channel dim if missing: expected [B, C, n_mels, T]
            if mel.ndim == 3:
                mel = mel.unsqueeze(1)

            emb = ser(mel)  # emb: [B, 2*hidden]
            logits = classifier(emb)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * mel.size(0)
            preds = logits.argmax(dim=1)
            total_acc += (preds == label).sum().item()

        avg_loss = total_loss / len(train_dataset)
        avg_acc = total_acc / len(train_dataset)
        print(f'Epoch {epoch} train_loss={avg_loss:.4f} train_acc={avg_acc:.4f}')

        # run validation if available
        if val_loader is not None:
            ser.eval()
            classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_count = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val", leave=False):
                    mel = batch['y'].to(device)
                    label = batch['x'].to(device)
                    if label.ndim > 1:
                        label = label.view(-1)
                    if mel.ndim == 3:
                        mel = mel.unsqueeze(1)
                    emb = ser(mel)
                    logits = classifier(emb)
                    loss = criterion(logits, label)
                    bs = mel.size(0)
                    val_loss += loss.item() * bs
                    preds = logits.argmax(dim=1)
                    val_correct += (preds == label).sum().item()
                    val_count += bs

            val_loss = val_loss / val_count if val_count > 0 else 0.0
            val_acc = val_correct / val_count if val_count > 0 else 0.0
            print(f'Epoch {epoch} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

            # save best by val loss
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = Path(args.save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                # if DataParallel used, save underlying module
                to_save = ser.module.state_dict() if isinstance(ser, nn.DataParallel) else ser.state_dict()
                torch.save(to_save, str(save_path))
                print('Saved BEST SER checkpoint to', save_path)

            ser.train()
            classifier.train()

    # if no val_loader was provided, save final checkpoint
    if val_loader is None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = ser.module.state_dict() if isinstance(ser, nn.DataParallel) else ser.state_dict()
        torch.save(to_save, str(save_path))
        print('Saved SER checkpoint to', save_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--num-classes', type=int, default=5)
    p.add_argument('--save-path', type=str, default='checkpts/ser_embedding.pt')
    p.add_argument('--filelist', type=str, default=None, help='path to filelist with lines like "audio.wav|label"')
    p.add_argument('--val-filelist', type=str, default=None, help='path to validation filelist (optional)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
