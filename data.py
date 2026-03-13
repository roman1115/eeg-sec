# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import random
import numpy as np

import torch
import torchaudio as ta

from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import parse_filelist, intersperse
from model.utils import fix_len_compatibility
from model import EEGFeatureExtractor
from params import seed as random_seed
from torch.utils.data import DataLoader
from scipy.io import loadmat 

import sys
sys.path.insert(0, 'hifi-gan')
from meldataset import mel_spectrogram


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, text = filepath_and_text[0], filepath_and_text[1]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (text, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.IntTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        text, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': text}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []

        for i, item in enumerate(batch):
            y_, x_ = item['y'], item['x']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}


class TextMelSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker = line[0], line[1], line[2]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        return (text, mel, speaker)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


class TextMelSpeakerBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        y_lengths, x_lengths = [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_ = item['y'], item['x'], item['spk']
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk}


class TextMelSpeakerEEGBatchCollate(object):
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item['y'].shape[-1] for item in batch])
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item['x'].shape[-1] for item in batch])
        n_feats = batch[0]['y'].shape[-2]

        # We expect per-sample EEG to already be a global embedding: [1, emb] or [emb]
        first_eeg = batch[0]['eeg']
        if isinstance(first_eeg, torch.Tensor):
            if first_eeg.ndim == 2 and first_eeg.shape[0] == 1:
                emb_dim = first_eeg.shape[1]
            elif first_eeg.ndim == 1:
                emb_dim = first_eeg.shape[0]
            else:
                raise ValueError(f"Expected eeg embedding shape [1,emb] or [emb], got {tuple(first_eeg.shape)}")
        else:
            arr = np.asarray(first_eeg)
            if arr.ndim == 2 and arr.shape[0] == 1:
                emb_dim = arr.shape[1]
            elif arr.ndim == 1:
                emb_dim = arr.shape[0]
            else:
                raise ValueError(f"Expected eeg embedding array shape [1,emb] or [emb], got {arr.shape}")

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        eeg = torch.zeros((B, emb_dim), dtype=torch.float32)
        y_lengths, x_lengths, eeg_lengths = [], [], []
        spk = []

        for i, item in enumerate(batch):
            y_, x_, spk_, eeg_ = item['y'], item['x'], item['spk'], item['eeg']
            # lengths
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])

            # copy y and x
            y[i, :, :y_.shape[-1]] = y_
            x[i, :x_.shape[-1]] = x_

            # normalize eeg embedding to 1D vector of length emb_dim
            if isinstance(eeg_, torch.Tensor):
                if eeg_.ndim == 2 and eeg_.shape[0] == 1:
                    vec = eeg_.squeeze(0)
                elif eeg_.ndim == 1:
                    vec = eeg_
                else:
                    raise ValueError(f"Unexpected eeg shape for embedding: {tuple(eeg_.shape)}")
            else:
                vec = torch.from_numpy(np.asarray(eeg_)).float().squeeze()

            # pad or truncate if necessary
            if vec.numel() < emb_dim:
                v = torch.zeros(emb_dim, dtype=torch.float32)
                v[:vec.numel()] = vec
                vec = v
            elif vec.numel() > emb_dim:
                vec = vec[:emb_dim]

            eeg[i] = vec
            eeg_lengths.append(vec.numel())
            spk.append(spk_)

        y_lengths = torch.LongTensor(y_lengths)
        x_lengths = torch.LongTensor(x_lengths)
        eeg_lengths = torch.LongTensor(eeg_lengths)
        spk = torch.cat(spk, dim=0)
        return {'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths, 'spk': spk, 'eeg': eeg, 'eeg_lengths': eeg_lengths}



class TextMelSpeakerEEGDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        super().__init__()
        self.filelist = parse_filelist(filelist_path, split_char='|')
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.add_blank = add_blank
        random.seed(random_seed)
        random.shuffle(self.filelist)

    def get_triplet(self, line):
        filepath, text, speaker, eeg_filepath = line[0], line[4], line[2], line[3]
        text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        speaker = self.get_speaker(speaker)
        eeg = self.get_eeg(eeg_filepath)
        return (text, mel, speaker, eeg)
    

    def get_eeg(self, filepath):
        # print(f"Loading EEG file: {filepath}")
        eeg_data = loadmat(filepath)  # 加载 MATLAB 文件
        # print(f"EEG data keys: {eeg_data.keys()}")  # 打印字典中的键
        if 'eeg_trial' not in eeg_data:  # 检查是否包含 'eeg_trial' 键
            raise KeyError(f"'eeg_trial' key not found in EEG file: {filepath}") 

        # 简化处理：按已知格式 (time, channels) -> 转为 (1, channels, time)
        eeg_np = np.asarray(eeg_data['eeg_trial'])
        eeg_t = torch.from_numpy(eeg_np.T).float().unsqueeze(0)  # [1, channels, time]

        # 实例化并调用特征提取器（eval + no_grad）
        extractor = EEGFeatureExtractor()
        extractor.eval()
        with torch.no_grad():
            eeg_tensor = extractor(eeg_t)

        return eeg_tensor
    
    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_text(self, text, add_blank=True):
        text_norm = text_to_sequence(text, dictionary=self.cmudict)
        if self.add_blank:
            text_norm = intersperse(text_norm, len(symbols))  # add a blank token, whose id number is len(symbols)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_speaker(self, speaker):
        speaker = torch.LongTensor([int(speaker)])
        return speaker

    def __getitem__(self, index):
        text, mel, speaker, eeg = self.get_triplet(self.filelist[index])
        item = {'y': mel, 'x': text, 'spk': speaker, 'eeg': eeg}
        return item

    def __len__(self):
        return len(self.filelist)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch


####################################
# 测试TextMelSpeakerEEGDataset代码段
####################################
# filelist_path="train_list_22k_text.txt"
# cmudict_path="resources/cmu_dictionary"

# test_dataset = TextMelSpeakerEEGDataset(
#    filelist_path=filelist_path,
#    cmudict_path=cmudict_path,
#    add_blank=True,
#    n_fft=1024,
#    n_mels=80,
#    sample_rate=22050,
#    hop_length=256,
#    win_length=1024,
#    f_min=0.,
#    f_max=8000
# )
# test_batch=test_dataset.sample_test_batch(1)
# for i, sample in enumerate(test_batch):
#    print(f"Sample {i+1}:")
#    print(f"Text: {sample['x']}")
#    print(f"Mel-Spectrogram: {sample['y'].shape}")
#    print(f"Speaker ID: {sample['spk']}")
#    print(f"EEG Data: {sample['eeg'].shape}")

############################################    
# 测试TextMelSpeakerEEGBatchCollate代码段
############################################
# filelist_path = "train_list_22k_text.txt"
# cmudict_path = "resources/cmu_dictionary"

# ds = TextMelSpeakerEEGDataset(
#     filelist_path=filelist_path,
#     cmudict_path=cmudict_path,
#     add_blank=True,
#     n_fft=1024,
#     n_mels=80,
#     sample_rate=22050,
#     hop_length=256,
#     win_length=1024,
#     f_min=0.,
#     f_max=8000
# )

# collate = TextMelSpeakerEEGBatchCollate()
# # 选择小一点的 batch，避免一次性加载太多 EEG 导致慢
# loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate, num_workers=0)

# batch = next(iter(loader))

# def print_info(name, v):
#     if isinstance(v, torch.Tensor):
#         print(f"{name}: shape={tuple(v.shape)}, dtype={v.dtype}, min={v.min().item():.4f}, max={v.max().item():.4f}")
#     else:
#         print(f"{name}: type={type(v)}, value={v}")

# print_info('x', batch['x'])
# print_info('x_lengths', batch['x_lengths'])
# print_info('y', batch['y'])
# print_info('y_lengths', batch['y_lengths'])
# print_info('spk', batch['spk'])
# print_info('eeg', batch['eeg'])
# print_info('eeg_lengths', batch['eeg_lengths'])

class MelLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filelist_path, cmudict_path, add_blank=True,
                 n_fft=1024, n_mels=80, sample_rate=22050,
                 hop_length=256, win_length=1024, f_min=0., f_max=8000):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.cmudict = cmudict.CMUDict(cmudict_path)
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        random.seed(random_seed)
        random.shuffle(self.filepaths_and_text)

    def get_pair(self, filepath_and_text):
        filepath, label = filepath_and_text[0], filepath_and_text[2]
        label = self.get_label(label, add_blank=self.add_blank)
        mel = self.get_mel(filepath)
        return (label, mel)

    def get_mel(self, filepath):
        audio, sr = ta.load(filepath)
        assert sr == self.sample_rate
        mel = mel_spectrogram(audio, self.n_fft, self.n_mels, self.sample_rate, self.hop_length,
                              self.win_length, self.f_min, self.f_max, center=False).squeeze()
        return mel

    def get_label(self, label, add_blank=True):
        label = torch.LongTensor([int(label)])
        return label

    def __getitem__(self, index):
        label, mel = self.get_pair(self.filepaths_and_text[index])
        item = {'y': mel, 'x': label}
        return item

    def __len__(self):
        return len(self.filepaths_and_text)

    def sample_test_batch(self, size):
        idx = np.random.choice(range(len(self)), size=size, replace=False)
        test_batch = []
        for index in idx:
            test_batch.append(self.__getitem__(index))
        return test_batch

# ########################################################
# # 测试 MelLabelDataset 代码段
# ########################################################
# filelist_path = "resources/filelists/EAV/train_list_22k_text.txt"

# test_dataset = MelLabelDataset(
#    filelist_path=filelist_path,
#    cmudict_path="resources/cmu_dictionary",
#    add_blank=True,
#    n_fft=1024,
#    n_mels=80,
#    sample_rate=22050,
#    hop_length=256,
#    win_length=1024,
#    f_min=0.,
#    f_max=8000   
# )
# test_batch=test_dataset.sample_test_batch(4)
# for i, sample in enumerate(test_batch):
#    print(f"Sample {i+1}:")
#    print(f"Label: {sample['x']}")
#    print(f"Mel-Spectrogram: {sample['y'].shape}")