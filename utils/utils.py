#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Sampler
import os
import math
import scipy
from pathlib import Path

from utils.evaluation_measures import compute_sed_eval_metrics
from utils.dataset import *


class Encoder:
    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples/2/self.frame_hop)*2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))           #버림 -> 해당 time frame에 걸쳐있으면 true
                offset = int(np.ceil(self._time_to_frame(row["offset"])))  #올림 -> 해당 time frame에 걸쳐있으면 true
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))


def decode_pred_batch(outputs, weak_preds, filenames, encoder, thresholds, median_filter, decode_weak, pad_idx=None):
    pred_dfs = {}
    for threshold in thresholds:
        pred_dfs[threshold] = pd.DataFrame()
    for batch_idx in range(outputs.shape[0]): #outputs size = [bs, n_class, frames]
        for c_th in thresholds:
            output = outputs[batch_idx]       #outputs size = [n_class, frames]
            if pad_idx is not None:
                true_len = int(output.shape[-1] * pad_idx[batch_idx].item)
                output = output[:true_len]
            output = output.transpose(0, 1).detach().cpu().numpy() #output size = [frames, n_class]
            if decode_weak: # if decode_weak = 1 or 2
                for class_idx in range(weak_preds.size(1)):
                    if weak_preds[batch_idx, class_idx] < c_th:
                        output[:, class_idx] = 0
                    elif decode_weak > 1: # use only weak predictions (weakSED)
                        output[:, class_idx] = 1
            if decode_weak < 2: # weak prediction masking
                output = output > c_th
                for mf_idx in range(len(median_filter)):
                    output[:, mf_idx] = scipy.ndimage.filters.median_filter(output[:, mf_idx], (median_filter[mf_idx]))
            pred = encoder.decode_strong(output)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[batch_idx]).stem + ".wav"
            pred_dfs[c_th] = pred_dfs[c_th].append(pred, ignore_index=True)
    return pred_dfs


class ConcatDatasetBatchSampler(Sampler):
    def __init__(self, samplers, batch_sizes, epoch=0):
        self.batch_sizes = batch_sizes
        self.samplers = samplers
        self.offsets = [0] + np.cumsum([len(x) for x in self.samplers]).tolist()[:-1]

        self.epoch = epoch
        self.set_epoch(self.epoch)

    def _iter_one_dataset(self, c_batch_size, c_sampler, c_offset):
        batch = []
        for idx in c_sampler:
            batch.append(c_offset + idx)
            if len(batch) == c_batch_size:
                yield batch

    def set_epoch(self, epoch):
        if hasattr(self.samplers[0], "epoch"):
            for s in self.samplers:
                s.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(i) for i in self.samplers]
        tot_batch = []
        for b_num in range(len(self)):
            for samp_idx in range(len(self.samplers)):
                c_batch = []
                while len(c_batch) < self.batch_sizes[samp_idx]:
                    c_batch.append(self.offsets[samp_idx] + next(iterators[samp_idx]))
                tot_batch.extend(c_batch)
            yield tot_batch
            tot_batch = []

    def __len__(self):
        min_len = float("inf")
        for idx, sampler in enumerate(self.samplers):
            c_len = (len(sampler)) // self.batch_sizes[idx]
            min_len = min(c_len, min_len)
        return min_len


class ExponentialWarmup(object):
    def __init__(self, optimizer, max_lr, rampup_length, exponent=-5.0):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        return self.max_lr * self._get_scaling_factor()

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

    # def load_state_dict(self, state_dict):
    #     self.__dict__.update(state_dict)
    #
    # def state_dict(self):
    #     return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        else:
            current = np.clip(self.step_num, 0.0, self.rampup_length)
            phase = 1.0 - current / self.rampup_length
            return float(np.exp(self.exponent * phase * phase))


def update_ema(net, ema_net, step, ema_factor):
    # update EMA model
    alpha = min(1 - 1 / step, ema_factor)
    for ema_params, params in zip(ema_net.parameters(), net.parameters()):
        ema_params.data.mul_(alpha).add_(params.data, alpha=1 - alpha)
    return ema_net


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """ Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


class Scaler(nn.Module):
    def __init__(self, statistic="instance", normtype="minmax", dims=(0, 2), eps=1e-8):
        super(Scaler, self).__init__()
        self.statistic = statistic
        self.normtype = normtype
        self.dims = dims
        self.eps = eps

    def load_state_dict(self, state_dict, strict=True):
        if self.statistic == "dataset":
            super(Scaler, self).load_state_dict(state_dict, strict)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.statistic == "dataset":
            super(Scaler, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                      unexpected_keys, error_msgs)

    def forward(self, input):
        if self.statistic == "dataset":
            if self.normtype == "mean":
                return input - self.mean
            elif self.normtype == "standard":
                std = torch.sqrt(self.mean_squared - self.mean ** 2)
                return (input - self.mean) / (std + self.eps)
            else:
                raise NotImplementedError

        elif self.statistic =="instance":
            if self.normtype == "mean":
                return input - torch.mean(input, self.dims, keepdim=True)
            elif self.normtype == "standard":
                return (input - torch.mean(input, self.dims, keepdim=True)) / (
                        torch.std(input, self.dims, keepdim=True) + self.eps)
            elif self.normtype == "minmax":
                return (input - torch.amin(input, dim=self.dims, keepdim=True)) / (
                    torch.amax(input, dim=self.dims, keepdim=True)
                    - torch.amin(input, dim=self.dims, keepdim=True) + self.eps)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError


class AsymmetricalFocalLoss(nn.Module):
    def __init__(self, gamma=0, zeta=0):
        super(AsymmetricalFocalLoss, self).__init__()
        self.gamma = gamma   # balancing between classes
        self.zeta = zeta     # balancing between active/inactive frames

    def forward(self, pred, target):
        losses = - (((1 - pred) ** self.gamma) * target * torch.clamp_min(torch.log(pred), -100) +
                    (pred ** self.zeta) * (1 - target) * torch.clamp_min(torch.log(1 - pred), -100))
        return torch.mean(losses)


def take_log(feature):
    amp2db = torchaudio.transforms.AmplitudeToDB(stype="amplitude")
    amp2db.amin = 1e-5
    return amp2db(feature).clamp(min=-50, max=80)


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params += param
    return total_params
