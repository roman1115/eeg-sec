import pickle
import time
import os
import random
import math
import numpy as np
import torch
import torch.utils.data
import torch.distributed as dist
import scipy.io
import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
from scipy.signal import butter, sosfilt


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        for audiopath, text in self.audiopaths_and_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_and_text_new.append([audiopath, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            print("self.cleaned_text:",self.cleaned_text)
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


"""Multi speaker version"""
class TextAudioSpeakerEmotionLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, eid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, eid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_audio_text_speaker_emotion_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, eid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3]
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_id(sid)
        eid = self.get_id(eid)
                     
        return (text, spec, wav, sid, eid)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_id(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_emotion_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerEmotionCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        eid = torch.LongTensor(len(batch))    

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            eid[i] = row[4]

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid

def make_weights_for_balanced_classes(train_dataset):
    data = train_dataset
    classes = []
    if False:
        for item in train_dataset:
            classes += list(item[-1].cpu().numpy())
        classes = set(classes)
        print("classes:",classes)
        count = {c:0 for c in classes}                                                    
        for item in data:
            s= list(item[-1].cpu().numpy())
            count[s[0]] += 1                 
        weight_per_class = {c:0.0 for c in classes}                                 
        N = float(sum(count))                                                   
        for c in classes: 
            weight_per_class[c] = N/float(count[c])
        print("weight_per_class:",weight_per_class)
    else:
        weight_per_class = {100000: 37.16519930160853, 1: 322.7258064516129, 2: 322.7258064516129, 3: 322.7258064516129, 4: 322.7258064516129, 5: 322.7258064516129, 6: 322.7258064516129, 0: 322.7258064516129, 7: 322.7258064516129, 9: 322.7258064516129, 8: 322.7258064516129}
    
    weight = [0] * len(data)                                              
    for idx, val in enumerate(data):
        s = list(val[-1].cpu().numpy())
        weight[idx] = weight_per_class[s[0]] 
    return weight

class TextAudioSpeakerEmotionEEGLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, emotion_id, text, eeg pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files (与第二个文件保持一致)
        4) loads EEG data from pickle or mat files
    """
    def __init__(self, audiopaths_sid_text_eeg, hparams):
        # 使用第二个文件的数据加载方式
        self.audiopaths_sid_text_eeg = load_filepaths_and_text(audiopaths_sid_text_eeg)
        
        # 验证数据格式
        for item in self.audiopaths_sid_text_eeg:
            if len(item) != 5:
                raise ValueError(f"数据格式错误，期望5个字段，但得到{len(item)}个字段: {item}")
        
        # 使用第二个文件的音频处理参数
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length = hparams.hop_length
        self.win_length = hparams.win_length

        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        # EEG相关参数（根据第二个文件的音频参数调整）
        self.eeg_sampling_rate = self.sampling_rate  # 使用音频采样率
        # 不再需要max_mel_length，因为第二个文件有自己的长度处理逻辑

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text_eeg)
        self._filter()

    def _filter(self):
        """与第二个文件完全一致的过滤逻辑"""
        audiopaths_sid_text_eeg_new = []
        lengths = []
        for audiopath, sid, eid, text, eeg_path in self.audiopaths_sid_text_eeg:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_eeg_new.append([audiopath, sid, eid, text, eeg_path])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text_eeg = audiopaths_sid_text_eeg_new
        self.lengths = lengths
        import numpy as np


    def bandpass_eeg(eeg_tensor, low=5, high=80, fs=500):
        """
        eeg_tensor: torch.Tensor, shape (C, T) 或 (B, C, T)
        保持与 EEGNet 完全一致的带通滤波方式：
        Butterworth(5阶) + SOS + sosfilt（单向，有相位延迟）
        """

        # 设计滤波器（与原始 EEGNet 完全一致）
        sos = butter(5, [low, high], btype='band', fs=fs, output='sos')

        # 确保在 CPU 上滤波（sosfilt 在 CPU 上执行）
        eeg = eeg_tensor.detach().cpu().numpy()

        if eeg.ndim == 2:     # (C, T)
            C, T = eeg.shape
            for c in range(C):
                eeg[c] = sosfilt(sos, eeg[c])
            return torch.from_numpy(eeg).to(eeg_tensor.device)

        elif eeg.ndim == 3:   # (B, C, T)
            B, C, T = eeg.shape
            for b in range(B):
                for c in range(C):
                    eeg[b, c] = sosfilt(sos, eeg[b, c])
            return torch.from_numpy(eeg).to(eeg_tensor.device)

        else:
            raise ValueError("EEG tensor must be (C,T) or (B,C,T)")


    def get_audio_text_speaker_emotion_eeg_pair(self, audiopath_sid_text_eeg):
        # 分离各个字段
        audiopath, sid, eid, text, eeg_path = audiopath_sid_text_eeg
        
        # 使用第二个文件的音频处理方式
        text = self.get_text(text)
        spec, wav = self.get_audio(audiopath)  # 与第二个文件完全一致
        sid = self.get_id(sid)
        eid = self.get_id(eid)
        
        # 加载EEG数据
        eeg = self.get_eeg(eeg_path)
        eeg = self.bandpass_eeg(eeg, low=5, high=80, fs=self.eeg_sampling_rate)
        return (text, spec, wav, sid, eid, eeg)

    def get_audio(self, filename):
        """与第二个文件完全一致的音频处理方式"""
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_eeg(self, eeg_path):
        """加载EEG数据（保持第一个文档的逻辑）"""
        if eeg_path.endswith(".pkl"):
            with open(eeg_path, "rb") as f:
                eeg_data = pickle.load(f)
        elif eeg_path.endswith(".mat"):
            mat = scipy.io.loadmat(eeg_path)
            eeg_data = mat["eeg_trial"]
            eeg_data = np.transpose(eeg_data, (1, 0))

        # 确保EEG数据是3维的
        if eeg_data.ndim == 2:
            eeg_data = np.expand_dims(eeg_data, -1)
        
        eeg_tensor = torch.from_numpy(eeg_data.astype(np.float32))
        return eeg_tensor

    def get_text(self, text):
        """与第二个文件一致的文本处理"""
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def get_id(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_emotion_eeg_pair(self.audiopaths_sid_text_eeg[index])

    def __len__(self):
        return len(self.audiopaths_sid_text_eeg)


class TextAudioSpeakerEmotionEEGCollate():
    """ Zero-pads model inputs and targets including EEG data
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio, speaker identities and EEG data
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, eid, eeg]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        eid = torch.LongTensor(len(batch))

        # EEG数据填充
        eeg_shape = batch[0][5].shape
        eeg_padded = torch.zeros((len(batch),) + eeg_shape).float()

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]
            eid[i] = row[4]

            eeg = row[5]
            eeg_padded[i] = eeg

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid, eeg_padded, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, eid, eeg_padded

class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
        
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
