import os
from copy import deepcopy
import random
import numpy as np
import soundfile as sf
import resampy
import torch
import torchaudio.transforms
from utils.settings import get_configs, get_labeldict, get_encoder
from utils.utils import Scaler, take_log
# from utils.dataset import waveform_modification
from utils.model import CRNN



class FDYSED:
    def __init__(self, config='configs/config.yaml', stud_path='models/best_student.pt', tch_path='models/best_teacher.pt'):
        configs, server_cfg, train_cfg, feature_cfg = get_configs(config)
        self.cfg = configs

        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=feature_cfg["sample_rate"],
            n_fft=feature_cfg["n_window"],
            win_length=feature_cfg["n_window"],
            hop_length=feature_cfg["hop_length"],
            f_min=feature_cfg["f_min"],
            f_max=feature_cfg["f_max"],
            n_mels=feature_cfg["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1
        )

        self.in_hop_secs = feature_cfg["hop_length"] / feature_cfg["sample_rate"]

        self.scaler = Scaler(
            statistic=configs["scaler"]["statistic"], 
            normtype=configs["scaler"]["normtype"], 
            dims=configs["scaler"]["dims"])

        self.labels = labels = get_labeldict()
        self.encoder = get_encoder(labels, feature_cfg, feature_cfg["audio_max_len"])

        self.net = CRNN(**configs["CRNN"]).to(device).eval()
        self.ema_net = CRNN(**configs["CRNN"]).to(device).eval()
        self.net.training=False
        self.ema_net.training=False
        stud_path and self.net.load_state_dict(torch.load(stud_path, map_location='cpu'))
        tch_path and self.ema_net.load_state_dict(torch.load(tch_path, map_location='cpu'))

    def load_audio_wav(self, path):
        '''Given an audio path, load and prepare the audio samples.'''
        wav, file_sr = sf.read(path)
        wav = wav[:,0] if wav.ndim > 1 else wav
        sr = self.cfg['feature']["sample_rate"]
        if sr != file_sr:
            wav = resampy.resample(wav, file_sr, sr)
        wav = torch.from_numpy(wav).float()
        wav = wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)
        return wav        

    def load_inputs(self, path):
        '''Convert an audio path into the desired model inputs. '''
        wav = self.load_audio_wav(path)
        mels = self.melspec(wav[None])
        logmels = self.scaler(take_log(mels))
        return logmels

    def predict(self, logmels):
        '''Run the model on prepared model inputs.'''
        with torch.no_grad():
            strong_pred_stud, weak_pred_stud = self.net(logmels)
            strong_pred_tch, weak_pred_tch = self.ema_net(logmels)
        return [
            (strong_pred_stud, weak_pred_stud),
            (strong_pred_tch, weak_pred_tch),
        ]

    def __call__(self, path):
        logmels = self.load_inputs(path)
        return self.predict(logmels)



def set_seed(seed, device):
    '''Set the random seed for the million things you need to change.'''
    if seed:
        torch.random.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


def print_labels(preds, labels):
    '''Print out the model predictions.'''
    labels = labels or {}
    labels = dict(zip(labels.values(), labels.keys()))
    for i, p in sorted(enumerate(preds), key=lambda x: x[1], reverse=True):
        print(p, '\t', labels.get(i, i))

def desc(*xs):
    '''Debug a tensor/array/variable'''
    for x in xs:
        try:
            print(x.shape, x.dtype, x.min(), x.max())
        except AttributeError:
            l = ''
            try:
                l = len(x)
            except Exception:
                pass
            print(type(x).__name__, l)
        

def plot_results(path, logmels, stud_probs, tch_probs, labels, spec_hop_secs=1):
    '''Plot the model inputs / outputs.'''
    inv_labels = dict(zip(labels.values(), labels.keys()))

    out_hop_secs = logmels.shape[1] / stud_probs.shape[1] * spec_hop_secs

    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    plt.subplot(311)
    plt.imshow(logmels, aspect='auto')
    N = logmels.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*spec_hop_secs])

    plt.ylabel('mel spec')

    plt.subplot(312)
    plt.title('Student')
    plt.imshow(stud_probs, aspect='auto')

    N = stud_probs.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*out_hop_secs])

    N = stud_probs.shape[0]-1
    yticks = np.linspace(0, N, N+1, dtype=int)
    plt.yticks(yticks, np.array([inv_labels.get(i, i) for i in yticks]))

    plt.subplot(313)
    plt.title('Teacher')
    plt.imshow(tch_probs, aspect='auto')

    N = tch_probs.shape[1]-1
    yticks = np.linspace(0, N, 10, dtype=int)
    plt.xticks(yticks, [f'{x:.1f}' for x in yticks*out_hop_secs])

    N = tch_probs.shape[0]-1
    yticks = np.linspace(0, N, N+1, dtype=int)
    plt.yticks(yticks, np.array([inv_labels.get(i, i) for i in yticks]))

    plt.ylabel('time')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(path.split('/')[-1])
    plt.savefig(f'{path}.png')
    plt.close()





def main(*paths, **kw):
    model = FDYSED(**kw)
    print(model.net)
    for path in paths:
        print('-'*20)
        print(path)

        logmels = model.load_inputs(path)

        [
            (strong_pred_stud, weak_pred_stud),
            (strong_pred_tch, weak_pred_tch),
        ] = model.predict(logmels)

        print('Student:')
        print_labels(weak_pred_stud[0], model.labels)
        print()
        print('Teacher:')
        print_labels(weak_pred_tch[0], model.labels)
        plot_results(path, logmels[0].numpy(), strong_pred_stud[0], strong_pred_tch[0], model.labels, model.in_hop_secs)


if __name__ == '__main__':
    import fire
    fire.Fire(main)