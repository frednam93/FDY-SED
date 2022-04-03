import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import logging
import pickle
import yaml
import shutil
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

from utils.utils import *
from utils.model import *
from utils.dataset import *


def get_configs(config_dir, server_config_dir="./configs/config_server.yaml"):
    #get hyperparameters from yaml
    with open(config_dir, "r") as f:
        configs = yaml.safe_load(f)
    with open(server_config_dir, "r") as f:
        server_cfg = yaml.safe_load(f)

    train_cfg = configs["training"]
    feature_cfg = configs["feature"]
    train_cfg["batch_sizes"] = server_cfg["batch_size"]
    train_cfg["net_pooling"] = feature_cfg["net_subsample"]
    return configs, server_cfg, train_cfg, feature_cfg


def get_save_directories(configs, train_cfg, iteration, args):
    general_cfg = configs["generals"]
    save_folder = general_cfg["save_folder"]
    savepsds = general_cfg["savepsds"]

    # set save folder
    if save_folder.count("new_exp") > 0:
        save_folder = save_folder + '_gpu=' + str(args.gpu)
        configs["generals"]["save_folder"] = save_folder
    if not train_cfg["test_only"]:
        if iteration is not None:
            save_folder = save_folder + '_iter:' + str(iteration)
        print("save directory : " + save_folder)
        if os.path.isdir(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)  # saving folder
        with open(os.path.join(save_folder, 'config.yaml'), 'w') as f:
            yaml.dump(configs, f)  # save yaml in the saving folder

    #set best paths
    stud_best_path = os.path.join(save_folder, "best_student.pt")
    tch_best_path = os.path.join(save_folder, "best_teacher.pt")
    train_cfg["best_paths"] = [stud_best_path, tch_best_path]

    # psds folder
    if savepsds:
        stud_psds_folder = os.path.join(save_folder, "psds_student")
        tch_psds_folder = os.path.join(save_folder, "psds_teacher")
        psds_folders = [stud_psds_folder, tch_psds_folder]
    else:
        psds_folders = [None, None]
    train_cfg["psds_folders"] = psds_folders
    return configs, train_cfg


def get_logger(save_folder):
    logger = logging.getLogger()
    formatter = logging.Formatter('%(message)s')
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(save_folder, "log.txt"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def get_labeldict():
    return OrderedDict({"Alarm_bell_ringing": 0,
                        "Blender": 1,
                        "Cat": 2,
                        "Dishes": 3,
                        "Dog": 4,
                        "Electric_shaver_toothbrush": 5,
                        "Frying": 6,
                        "Running_water": 7,
                        "Speech": 8,
                        "Vacuum_cleaner": 9})


def get_encoder(LabelDict, feature_cfg, audio_len):
    return Encoder(list(LabelDict.keys()),
                   audio_len=audio_len,
                   frame_len=feature_cfg["frame_length"],
                   frame_hop=feature_cfg["hop_length"],
                   net_pooling=feature_cfg["net_subsample"],
                   sr=feature_cfg["sr"])


def get_mt_datasets(configs, server_cfg, train_cfg):
    general_cfg = configs["generals"]
    encoder = train_cfg["encoder"]
    dataset_cfg = configs["dataset"]
    batch_size_val = server_cfg["batch_size_val"]
    num_workers = server_cfg["num_workers"]
    batch_sizes = server_cfg["batch_size"]
    synthdataset_cfg = configs["synth_dataset"]

    synth_train_tsv = synthdataset_cfg["synth_train_tsv"]
    synth_train_df = pd.read_csv(synth_train_tsv, sep="\t")
    weak_dir = dataset_cfg["weak_folder"]
    weak_df = pd.read_csv(dataset_cfg["weak_tsv"], sep="\t")
    weak_train_df = weak_df.sample(frac=train_cfg["weak_split"], random_state=train_cfg["seed"])
    weak_valid_df = weak_df.drop(weak_train_df.index).reset_index(drop=True)
    weak_train_df = weak_train_df.reset_index(drop=True)
    synth_valid_dir = synthdataset_cfg["synth_val_folder"]
    synth_valid_tsv = synthdataset_cfg["synth_val_tsv"]
    synth_valid_df = pd.read_csv(synth_valid_tsv, sep="\t")
    synth_valid_dur = synthdataset_cfg["synth_val_dur"]

    synth_train_dataset = StronglyLabeledDataset(synth_train_df, synthdataset_cfg["synth_train_folder"], False, encoder)
    weak_train_dataset = WeaklyLabeledDataset(weak_train_df, weak_dir, False, encoder)
    unlabeled_dataset = UnlabeledDataset(dataset_cfg["unlabeled_folder"], False, encoder)
    synth_vaild_dataset = StronglyLabeledDataset(synth_valid_df, synth_valid_dir, True, encoder)
    weak_valid_dataset = WeaklyLabeledDataset(weak_valid_df, weak_dir, True, encoder)
    if not general_cfg["test_on_public_eval"]:
        test_tsv = dataset_cfg["test_tsv"]
        test_df = pd.read_csv(test_tsv, sep="\t")
        test_dur = dataset_cfg["test_dur"]
        test_dataset = StronglyLabeledDataset(test_df, dataset_cfg["test_folder"], True, encoder)
    else:
        test_tsv = dataset_cfg["pubeval_tsv"]
        test_df = pd.read_csv(test_tsv, sep="\t")
        test_dur = dataset_cfg["pubeval_dur"]
        test_dataset = StronglyLabeledDataset(test_df, dataset_cfg["pubeval_folder"], True, encoder)
    # use portion of datasets for debugging
    if train_cfg["div_dataset"]:
        div_ratio = train_cfg["div_ratio"]
        synth_train_dataset = torch.utils.data.Subset(synth_train_dataset,
                                                      torch.arange(int(len(synth_train_dataset) / div_ratio)))
        weak_train_dataset = torch.utils.data.Subset(weak_train_dataset,
                                                     torch.arange(int(len(weak_train_dataset) / div_ratio)))
        unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset,
                                                    torch.arange(int(len(unlabeled_dataset) / div_ratio)))
        synth_vaild_dataset = torch.utils.data.Subset(synth_vaild_dataset,
                                                      torch.arange(int(len(synth_vaild_dataset) / div_ratio)))
        weak_valid_dataset = torch.utils.data.Subset(weak_valid_dataset,
                                                     torch.arange(int(len(weak_valid_dataset) / div_ratio)))
        test_dataset = torch.utils.data.Subset(test_dataset, torch.arange(int(len(test_dataset) / div_ratio)))
    # build dataloaders
    # get train dataset
    train_data = [synth_train_dataset, weak_train_dataset, unlabeled_dataset]
    train_dataset = torch.utils.data.ConcatDataset(train_data)
    train_samplers = [torch.utils.data.RandomSampler(x) for x in train_data]
    train_batch_sampler = ConcatDatasetBatchSampler(train_samplers, batch_sizes)
    train_cfg["trainloader"] = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=num_workers)
    # get valid dataset
    valid_dataset = torch.utils.data.ConcatDataset([synth_vaild_dataset, weak_valid_dataset])
    train_cfg["validloader"] = DataLoader(valid_dataset, batch_size=batch_size_val, num_workers=num_workers)
    # get test dataset
    train_cfg["testloader"] = DataLoader(test_dataset, batch_size=batch_size_val, num_workers=num_workers)
    train_cfg["train_tsvs"] = [synth_train_df, synth_train_tsv]
    train_cfg["valid_tsvs"] = [synth_valid_dir, synth_valid_tsv, synth_valid_dur, weak_dir]
    train_cfg["test_tsvs"] = [test_tsv, test_dur]
    return train_cfg


def get_models(configs, train_cfg, multigpu):
    net = CRNN(**configs["CRNN"])
    # ema network
    ema_net = deepcopy(net)
    for param in ema_net.parameters():
        param.detach_()

    if multigpu and (train_cfg["n_gpu"] > 1):
        net = nn.DataParallel(net)
        ema_net = nn.DataParallel(ema_net)

    net = net.to(train_cfg["device"])
    ema_net = ema_net.to(train_cfg["device"])
    return net, ema_net


def get_scaler(scaler_cfg):
    return Scaler(statistic=scaler_cfg["statistic"], normtype=scaler_cfg["normtype"], dims=scaler_cfg["dims"])


def get_f1calcs(n_class, device):
    stud_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
    tch_f1calc = pl.metrics.classification.F1(n_class, average="macro", multilabel=True, compute_on_step=False)
    return stud_f1calc.to(device), tch_f1calc.to(device)


def get_printings():
    printing_epoch = '[Epc %d] tt: %0.3f, cl_st: %0.3f, cl_wk: %0.3f, cn_st: %0.3f, cn_wk: %0.3f, ' \
                          'st_vl: %0.3f, t_vl: %0.3f, t: %ds'

    printing_test = "      test result is out!" \
                    "\n      [student] psds1: %.4f, psds2: %.4f"\
                    "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "\
                    "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"\
                    "\n      [teacher] psds1: %.4f, psds2: %.4f"\
                    "\n                event_macro_f1: %.3f, event_micro_f1: %.3f, "\
                    "\n                segment_macro_f1: %.3f, segment_micro_f1: %.3f, intersection_f1: %.3f"
    return printing_epoch, printing_test


class History:
    def __init__(self):
        self.history = {"train_total_loss": [], "train_class_strong_loss": [], "train_class_weak_loss": [],
                        "train_cons_strong_loss": [], "train_cons_weak_loss": [], "stud_val_metric": [],
                        "tch_val_metric": []}

    def update(self, train_return, val_return):
        total, class_str, class_wk, cons_str, cons_wk = train_return

        stud_val_metric, tch_val_metric = val_return

        self.history['train_total_loss'].append(total)
        self.history['train_class_strong_loss'].append(class_str)
        self.history['train_class_weak_loss'].append(class_wk)
        self.history['train_cons_strong_loss'].append(cons_str)
        self.history['train_cons_weak_loss'].append(cons_wk)
        self.history['stud_val_metric'].append(stud_val_metric)
        self.history['tch_val_metric'].append(tch_val_metric)
        return stud_val_metric, tch_val_metric

    def save(self, save_dir):
        with open(save_dir, 'wb') as handle:
            pickle.dump(self.history, handle, protocol=pickle.HIGHEST_PROTOCOL)


class BestModels:
    # Class to keep track of the best student/teacher models and save them after training
    def __init__(self):
        self.stud_best_val_metric = 0.0
        self.tch_best_val_metric = 0.0
        self.stud_best_state_dict = None
        self.tch_best_state_dict = None

    def update(self, train_cfg, logger, val_metrics):
        stud_update = False
        tch_update = False
        if val_metrics[0] > self.stud_best_val_metric:
            self.stud_best_val_metric = val_metrics[0]
            self.stud_best_state_dict = train_cfg["net"].state_dict()
            stud_update = True
            # lr_reduc = 0
        if val_metrics[1] > self.tch_best_val_metric:
            self.tch_best_val_metric = val_metrics[1]
            self.tch_best_state_dict = train_cfg["ema_net"].state_dict()
            tch_update = True
            # lr_reduc = 0

        if train_cfg["epoch"] > int(train_cfg["n_epochs"] * 0.5):
            if stud_update:
                if tch_update:
                    logger.info("     best student & teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
                else:
                    logger.info("     best student model updated at epoch %d!" % (train_cfg["epoch"] + 1))
            elif tch_update:
                logger.info("     best teacher model updated at epoch %d!" % (train_cfg["epoch"] + 1))
        return logger

    def get_bests(self, best_paths):
        torch.save(self.stud_best_state_dict, best_paths[0])
        torch.save(self.tch_best_state_dict, best_paths[1])
        return self.stud_best_val_metric, self.tch_best_val_metric


def get_ensemble_models(train_cfg):
    ensemble_folder = train_cfg["ensemble_dir"]
    stud_nets_saved = glob(ensemble_folder + '*/best_student.pt')
    tch_nets_saved = glob(ensemble_folder + '*/best_teacher.pt')

    train_cfg["stud_nets"] = []
    train_cfg["tch_nets"] = []
    for i in range(len(stud_nets_saved)):
        net_temp = deepcopy(train_cfg["net"])
        net_temp = net_temp.to(train_cfg["device"])
        net_temp.load_state_dict(torch.load(stud_nets_saved[i], map_location=train_cfg["device"]))
        train_cfg["stud_nets"].append(net_temp)
    for i in range(len(tch_nets_saved)):
        net_temp = deepcopy(train_cfg["net"])
        net_temp = net_temp.to(train_cfg["device"])
        net_temp.load_state_dict(torch.load(tch_nets_saved[i], map_location=train_cfg["device"]))
        train_cfg["tch_nets"].append(net_temp)

    return train_cfg

