#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import librosa
import torch
from torch.utils.data import Dataset
import torchaudio

import numpy as np
import pandas as pd
import os.path
from glob import glob
import soundfile as sf


#dataset classes
class StronglyLabeledDataset(Dataset):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        #refer: https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * encoder.sr
        self.return_name = return_name

        #construct clip dictionary with filename = {path, events} where events = {label, onset and offset}
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]), "events": []}
            if not np.isnan(row["onset"]):
                clips[row["filename"]]["events"].append({"event_label": row["event_label"],
                                                         "onset": row["onset"],
                                                         "offset": row["offset"]})
        self.clips = clips #dictionary for each clip
        self.clip_list = list(clips.keys()) # list of all clip names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)

        # get labels
        events = clip["events"]
        if not len(events): #label size = [frames, nclass]
            label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        else:
            label = self.encoder.encode_strong_df(pd.DataFrame(events))
            label = torch.from_numpy(label).float()
        label = label.transpose(0, 1)

        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class WeaklyLabeledDataset(Dataset):
    def __init__(self, tsv_read, dataset_dir, return_name, encoder):
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #construct clip dictionary with file name, path, label, onset and offset
        clips = {}
        for _, row in tsv_read.iterrows():
            if row["filename"] not in clips.keys():
                clips[row["filename"]] = {"path": os.path.join(dataset_dir, row["filename"]),
                                          "events": row["event_labels"].split(",")}
        #dictionary for each clip
        self.clips = clips
        self.clip_list = list(clips.keys()) # all file names

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        path = clip["path"]

        #get labels
        events = clip["events"]
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels))
        if len(events):
            label_encoded = self.encoder.encode_weak(events)      # label size: [n_class]
            label[0, :] = torch.from_numpy(label_encoded).float() # label size: [n_frames, n_class]
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_dir, return_name, encoder):
        self.encoder = encoder
        self.pad_to = encoder.audio_len * self.encoder.sr
        self.return_name = return_name

        #list of clip directories
        self.clips = glob(os.path.join(dataset_dir, '*.wav'))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        path = self.clips[idx]
        filename = os.path.split(path)[-1]

        #produce empty label
        label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels)).float()
        label = label.transpose(0, 1)

        # get wav
        wav, pad_mask = waveform_modification(path, self.pad_to, self.encoder)
        # return
        out_args = [wav, label, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, path])
        return out_args


def waveform_modification(filepath, pad_to, encoder):
    wav, sr = sf.read(filepath)
    wav = to_mono(wav)
    if sr != encoder.sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=encoder.sr)
    wav, pad_mask = pad_wav(wav, pad_to, encoder)
    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    return wav, pad_mask


def normalize_wav(wav):
    return wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)


def to_mono(wav, rand_ch=False):
    if wav.ndim > 1:
        if rand_ch:
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            wav = np.mean(wav, axis=-1)
    return wav


def pad_wav(wav, pad_to, encoder):
    if len(wav) < pad_to:
        pad_from = len(wav)
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        wav = wav[:pad_to]
        pad_from = pad_to
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx # size = n_frame, [0, 0, 0, 0, 0, ..., 0, 1, ..., 1]
    return wav, pad_mask


def setmelspectrogram(feature_cfg):
    return torchaudio.transforms.MelSpectrogram(sample_rate=feature_cfg["sample_rate"],
                                                n_fft=feature_cfg["n_window"],
                                                win_length=feature_cfg["n_window"],
                                                hop_length=feature_cfg["hop_length"],
                                                f_min=feature_cfg["f_min"],
                                                f_max=feature_cfg["f_max"],
                                                n_mels=feature_cfg["n_mels"],
                                                window_fn=torch.hamming_window,
                                                wkwargs={"periodic": False},
                                                power=1) # 1:energy, 2:power

