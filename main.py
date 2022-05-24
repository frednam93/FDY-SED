#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
#Paper describing this code is on https://arxiv.org/abs/2107.03649
import torch
import torch.optim as optim
import torch.nn as nn

import os.path
import warnings
import argparse
import numpy as np
import pandas as ps
from time import time
from datetime import datetime
from tqdm import tqdm

from utils.dataset import *
from utils.utils import *
from utils.settings import *
from utils.data_aug import *
from utils.evaluation_measures import compute_per_intersection_macro_f1, compute_psds_from_operating_points


def main(iteration=None):
    print("="*50 + "start!!!!" + "="*50)
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--gpu', default=0, type=int, help='selection of gpu when you run separate trainings on single server')
    parser.add_argument('--multigpu', default=False, type=bool)
    args = parser.parse_args()

    #set configurations
    configs, server_cfg, train_cfg, feature_cfg = get_configs(config_dir="./configs/config.yaml")

    #declare test_only/debugging mode
    if train_cfg["test_only"]:
        print(" "*40 + "<"*10 + "test only" + ">"*10)
    if train_cfg["debug"]:
        train_cfg["div_dataset"] = True
        train_cfg["n_epochs"] = 1
        print("!" * 10 + "   DEBUGGING MODE   " + "!" * 10)


    #set save directories
    configs, train_cfg = get_save_directories(configs, train_cfg, iteration, args)

    #set logger
    logger = get_logger(configs["generals"]["save_folder"])

    #torch information
    logger.info("date & time of start is : " + str(datetime.now()).split('.')[0])
    logger.info("torch version is: " + str(torch.__version__))
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    train_cfg["n_gpu"] = torch.cuda.device_count()
    logger.info("number of GPUs: " + str(train_cfg["n_gpu"]))
    train_cfg["device"] = device
    logger.info("device: " + str(device))

    #seed
    torch.random.manual_seed(train_cfg["seed"])
    if device == 'cuda':
        torch.cuda.manual_seed_all(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    #do not show warning
    if not configs["generals"]["warn"]:
        warnings.filterwarnings("ignore")

    #class label dictionary
    LabelDict = get_labeldict()

    #set encoder
    train_cfg["encoder"] = get_encoder(LabelDict, feature_cfg, feature_cfg["audio_max_len"])

    #set Dataloaders
    train_cfg = get_mt_datasets(configs, server_cfg, train_cfg)

    #set network
    train_cfg["net"], train_cfg["ema_net"] = get_models(configs, train_cfg, args.multigpu)
    logger.info("Total Trainable Params: %.3f M" % (count_parameters(train_cfg["net"]) * 1e-6)) #print number of learnable parameters in the model

    #set feature
    train_cfg["feat_ext"] = setmelspectrogram(feature_cfg).to(device)

    #set scaler
    train_cfg["scaler"] = get_scaler(configs["scaler"])

    #set f1 calculators
    train_cfg["f1calcs"] = get_f1calcs(len(LabelDict), device)

    #loss function, optimizer, scheduler
    if train_cfg["afl_loss"] is None:
        train_cfg["criterion_class"] = nn.BCELoss().cuda()
    else:
        train_cfg["criterion_class"] = AsymmetricalFocalLoss(train_cfg["afl_loss"][0], train_cfg["afl_loss"][1])
    train_cfg["criterion_cons"] = nn.MSELoss().cuda()
    train_cfg["optimizer"] = optim.Adam(train_cfg["net"].parameters(), 1e-3, betas=(0.9, 0.999))
    warmup_steps = train_cfg["n_epochs_warmup"] * len(train_cfg["trainloader"])
    train_cfg["scheduler"] = ExponentialWarmup(train_cfg["optimizer"], configs["opt"]["lr"], warmup_steps)
    printing_epoch, printing_test = get_printings()

    ##############################                TRAIN/VALIDATION                ##############################
    if not train_cfg["test_only"]:
        logger.info('   training starts!')
        start_time = time()
        history = History()
        bestmodels = BestModels()
        for train_cfg["epoch"] in range(train_cfg["n_epochs"]):
            epoch_time = time()
            #training
            train_return = train(train_cfg)
            val_return = validation(train_cfg)
            #save best model when best validation metrics occur
            val_metrics = history.update(train_return, val_return)
            logger.info(printing_epoch % ((train_cfg["epoch"] + 1,) + train_return + val_return +
                                          (time() - epoch_time,)))
            logger = bestmodels.update(train_cfg, logger, val_metrics)

        #save model parameters & history dictionary
        logger.info("        best student/teacher val_metrics: %.3f / %.3f" % bestmodels.get_bests(train_cfg["best_paths"]))
        history.save(os.path.join(configs["generals"]["save_folder"], "history.pickle"))
        logger.info("   training took %.2f mins" % ((time()-start_time)/60))

    ##############################                        TEST                        ##############################
    logger.info("   test starts!")

    # test on best model
    train_cfg["net"].load_state_dict(torch.load(train_cfg["best_paths"][0]))
    train_cfg["ema_net"].load_state_dict(torch.load(train_cfg["best_paths"][1]))
    test_returns = test(train_cfg)
    logger.info(printing_test % test_returns)

    logger.info("date & time of end is : " + str(datetime.now()).split('.')[0])
    logging.shutdown()
    print("<"*30 + "DONE!" + ">"*30)


########################################################################################################################
#                                                        TRAIN                                                         #
########################################################################################################################
def train(train_cfg):
    train_cfg["net"].train()
    train_cfg["ema_net"].train()
    total_loss, class_strong_loss, class_weak_loss, cons_strong_loss, cons_weak_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    strong_bs, weak_bs, _ = train_cfg["batch_sizes"]
    n_train = len(train_cfg["trainloader"])
    tk0 = tqdm(train_cfg["trainloader"], total=n_train, leave=False, desc="training processing")
    for _, (wavs, labels, _, _) in enumerate(tk0, 0):
        wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
        mels = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]

        ### get mask for strongly/weakly labeled data
        batch_num = mels.size(0)
        mask_strong = torch.zeros(batch_num).to(mels).bool()
        mask_strong[:strong_bs] = 1                     # mask_strong size = [bs]
        mask_weak = torch.zeros(batch_num).to(mels).bool()
        mask_weak[strong_bs:(strong_bs + weak_bs)] = 1  # mask_weak size = [bs]
        #collapse weak label for weakly labeled data
        labels_weak = (torch.sum(labels[mask_weak], -1) > 0).float() # labels_weak size = [bs, n_class] (weak data only)

        ### apply data augmentations
        # frame_shift
        mels, labels = frame_shift(mels, labels, train_cfg["net_pooling"])
        # mix-up
        if train_cfg["mixup_type"] is not None and train_cfg["mixup_rate"] > torch.rand(1).item():
            # weak data mixup                      # strong_masked feature size = [bs_strong, freq, frames]
            mels[mask_weak], labels_weak = mixup(mels[mask_weak], labels_weak,
                                                     mixup_label_type=train_cfg["mixup_type"])
            # strong data mixup                    # weak_masked feature size = [bs_weak, freq, frames]
            mels[mask_strong], labels[mask_strong] = mixup(mels[mask_strong], labels[mask_strong],
                                                               mixup_label_type=train_cfg["mixup_type"])
        # time masking
        mels[mask_strong], labels[mask_strong] = time_mask(mels[mask_strong], labels[mask_strong],
                                                               train_cfg["net_pooling"],
                                                               mask_ratios=train_cfg["time_mask_ratios"])
        # other data augmentations that does not affect labels
        mels_stud, mels_tch = feature_transformation(mels, **train_cfg["transform"])

        # take log
        logmels_stud = train_cfg["scaler"](take_log(mels_stud))
        logmels_tch = train_cfg["scaler"](take_log(mels_tch))
        # model predictions
        train_cfg["optimizer"].zero_grad()                              # strong prediction size = [bs, n_class, frames]
        strong_pred_stud, weak_pred_stud = train_cfg["net"](logmels_stud)     # weak prediction size = [bs, n_class]
        with torch.no_grad():
            strong_pred_tch, weak_pred_tch = train_cfg["ema_net"](logmels_tch)

        ### loss functions
        # classification losses                    # strong masked label size = [bs_strong, n_class, frames]
        loss_class_strong = train_cfg["criterion_class"](strong_pred_stud[mask_strong],
                                                         labels[mask_strong])
        loss_class_weak = train_cfg["criterion_class"](weak_pred_stud[mask_weak], labels_weak)

        # consistency losses
        loss_cons_strong = train_cfg["criterion_cons"](strong_pred_stud, strong_pred_tch.detach())
        loss_cons_weak = train_cfg["criterion_cons"](weak_pred_stud, weak_pred_tch.detach())

        # total loss
        w_cons = train_cfg["w_cons_max"] * train_cfg["scheduler"]._get_scaling_factor()
        if not train_cfg["trainweak_only"]:
            loss_total = loss_class_strong + train_cfg["w_weak"] * loss_class_weak + \
                         w_cons * (loss_cons_strong + train_cfg["w_weak_cons"] * loss_cons_weak)
        else:
            loss_total = train_cfg["w_weak"] * loss_class_weak + w_cons * train_cfg["w_weak_cons"] * loss_cons_weak
        loss_total.backward()
        train_cfg["optimizer"].step()
        train_cfg["scheduler"].step()

        # update EMA model
        train_cfg["ema_net"] = update_ema(train_cfg["net"], train_cfg["ema_net"], train_cfg["scheduler"].step_num,
                                          train_cfg["ema_factor"])

        total_loss += loss_total.item()
        class_strong_loss += loss_class_strong.item()
        class_weak_loss += loss_class_weak.item()
        cons_strong_loss += loss_cons_strong.item()
        cons_weak_loss = loss_cons_weak.item()

    total_loss /= n_train
    class_strong_loss /= n_train
    class_weak_loss /= n_train
    cons_strong_loss /= n_train
    cons_weak_loss /= n_train
    return total_loss, class_strong_loss, class_weak_loss, cons_strong_loss, cons_weak_loss


########################################################################################################################
#                                                      VALIDATION                                                      #
########################################################################################################################
def validation(train_cfg):
    encoder = train_cfg["encoder"]
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    n_valid = len(train_cfg["validloader"])
    for f1calc in train_cfg["f1calcs"]:
        f1calc.reset()
    val_stud_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    val_tch_buffer = {k: pd.DataFrame() for k in train_cfg["val_thresholds"]}
    synth_valid_dir, synth_valid_tsv, synth_valid_dur, weak_dir = train_cfg["valid_tsvs"]
    rand_plot_idx = torch.randint(high=2500, size=(1, 1)).item()
    with torch.no_grad():
        tk1 = tqdm(train_cfg["validloader"], total=n_valid, leave=False, desc="validation processing")
        for _, (wavs, labels, _, indexes, filenames, paths) in enumerate(tk1, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            mels = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]
            logmels = train_cfg["scaler"](take_log(mels))

            strong_pred_stud, weak_pred_stud = train_cfg["net"](logmels)
            strong_pred_tch, weak_pred_tch = train_cfg["ema_net"](logmels)

            mask_weak = (torch.tensor([str(Path(x).parent) == str(Path(weak_dir)) for x in paths])
                         .to(logmels).bool())
            mask_strong = (torch.tensor([str(Path(x).parent) == str(Path(synth_valid_dir)) for x in paths])
                           .to(logmels).bool())


            if torch.any(mask_weak):
                labels_weak = (torch.sum(labels[mask_weak], -1) > 0).float()  # labels_weak size = [bs, n_class]
                #accumulate f1score for weak labels
                train_cfg["f1calcs"][0](weak_pred_stud[mask_weak], labels_weak)
                train_cfg["f1calcs"][1](weak_pred_tch[mask_weak], labels_weak)

            if torch.any(mask_strong):
                #decoded_stud/tch_strong for intersection f1 score
                paths_strong = [x for x in paths if Path(x).parent == Path(synth_valid_dir)]
                stud_pred_dfs = decode_pred_batch(strong_pred_stud[mask_strong], weak_pred_stud[mask_strong],
                                                  paths_strong, encoder, list(val_stud_buffer.keys()),
                                                  train_cfg["median_window"], train_cfg["decode_weak_valid"])
                tch_pred_dfs = decode_pred_batch(strong_pred_tch[mask_strong], weak_pred_tch[mask_strong],
                                                 paths_strong, encoder, list(val_tch_buffer.keys()),
                                                 train_cfg["median_window"], train_cfg["decode_weak_valid"])
                for th in val_stud_buffer.keys():
                    val_stud_buffer[th] = val_stud_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
                for th in val_tch_buffer.keys():
                    val_tch_buffer[th] = val_tch_buffer[th].append(tch_pred_dfs[th], ignore_index=True)

    stud_weak_f1 = train_cfg["f1calcs"][0].compute()
    tch_weak_f1 = train_cfg["f1calcs"][1].compute()
    stud_intersection_f1 = compute_per_intersection_macro_f1(val_stud_buffer, synth_valid_tsv, synth_valid_dur)
    tch_intersection_f1 = compute_per_intersection_macro_f1(val_tch_buffer, synth_valid_tsv, synth_valid_dur)
    if not train_cfg["trainweak_only"]:
        stud_val_metric = stud_weak_f1.item() + stud_intersection_f1
        tch_val_metric = tch_weak_f1.item() + tch_intersection_f1
        return stud_val_metric, tch_val_metric
    else:
        return stud_weak_f1.item(), tch_weak_f1.item()


########################################################################################################################
#                                                         TEST                                                         #
########################################################################################################################
def test(train_cfg):
    encoder = train_cfg["encoder"]
    psds_folders = train_cfg["psds_folders"]
    thresholds = np.arange(1 / (train_cfg["n_test_thresholds"] * 2), 1, 1 / train_cfg["n_test_thresholds"])
    train_cfg["net"].eval()
    train_cfg["ema_net"].eval()
    test_tsv, test_dur = train_cfg["test_tsvs"]
    with torch.no_grad():
        stud_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
        tch_test_psds_buffer = {k: pd.DataFrame() for k in thresholds}
        stud_test_f1_buffer = pd.DataFrame()
        tch_test_f1_buffer = pd.DataFrame()
        tk2 = tqdm(train_cfg["testloader"], total=len(train_cfg["testloader"]), leave=False, desc="test processing")
        for _, (wavs, labels, _, indexes, filenames, paths) in enumerate(tk2, 0):
            wavs, labels = wavs.to(train_cfg["device"]), labels.to(train_cfg["device"]) # labels size = [bs, n_class, frames]
            mels = train_cfg["feat_ext"](wavs)  # features size = [bs, freqs, frames]
            logmels = train_cfg["scaler"](take_log(mels))

            stud_preds, weak_stud_preds = train_cfg["net"](logmels)
            tch_preds, weak_tch_preds = train_cfg["ema_net"](logmels)

            stud_pred_dfs = decode_pred_batch(stud_preds, weak_stud_preds, paths, encoder,
                                              list(stud_test_psds_buffer.keys()), train_cfg["median_window"],
                                              train_cfg["decode_weak_test"])
            tch_pred_dfs = decode_pred_batch(tch_preds, weak_tch_preds, paths, encoder,
                                             list(tch_test_psds_buffer.keys()), train_cfg["median_window"],
                                             train_cfg["decode_weak_test"])
            for th in stud_test_psds_buffer.keys():
                stud_test_psds_buffer[th] = stud_test_psds_buffer[th].append(stud_pred_dfs[th], ignore_index=True)
            for th in tch_test_psds_buffer.keys():
                tch_test_psds_buffer[th] = tch_test_psds_buffer[th].append(tch_pred_dfs[th], ignore_index=True)
            stud_pred_df_halfpoint = decode_pred_batch(stud_preds, weak_stud_preds, paths, encoder, [0.5],
                                                       train_cfg["median_window"], train_cfg["decode_weak_test"])
            tch_pred_df_halfpoint = decode_pred_batch(tch_preds, weak_tch_preds, paths, encoder, [0.5],
                                                      train_cfg["median_window"], train_cfg["decode_weak_test"])
            stud_test_f1_buffer = stud_test_f1_buffer.append(stud_pred_df_halfpoint[0.5], ignore_index=True)
            tch_test_f1_buffer = tch_test_f1_buffer.append(tch_pred_df_halfpoint[0.5], ignore_index=True)


    # calculate psds
    psds1_kwargs = {"dtc_threshold": 0.7, "gtc_threshold": 0.7, "alpha_ct": 0, "alpha_st": 1}
    psds2_kwargs = {"dtc_threshold": 0.1, "gtc_threshold": 0.1, "cttc_threshold": 0.3, "alpha_ct": 0.5, "alpha_st": 1}
    stud_psds1 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[0],
                                                    **psds1_kwargs)
    stud_psds2 = compute_psds_from_operating_points(stud_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[0],
                                                    **psds2_kwargs)
    tch_psds1 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[1],
                                                   **psds1_kwargs)
    tch_psds2 = compute_psds_from_operating_points(tch_test_psds_buffer, test_tsv, test_dur, save_dir=psds_folders[1],
                                                   **psds2_kwargs)
    s_evt_ma_f1, s_evt_mi_f1, s_seg_ma_f1, s_seg_mi_f1 = log_sedeval_metrics(stud_test_f1_buffer,
                                                                             test_tsv, psds_folders[0])
    s_inter_f1 = compute_per_intersection_macro_f1({"0.5": stud_test_f1_buffer}, test_tsv, test_dur)
    t_evt_ma_f1, t_evt_mi_f1, t_seg_ma_f1, t_seg_mi_f1 = log_sedeval_metrics(tch_test_f1_buffer,
                                                                             test_tsv, psds_folders[1])
    t_inter_f1 = compute_per_intersection_macro_f1({"0.5": tch_test_f1_buffer}, test_tsv, test_dur)
    return stud_psds1, stud_psds2, s_evt_ma_f1, s_evt_mi_f1, s_seg_ma_f1, s_seg_mi_f1, s_inter_f1, \
           tch_psds1, tch_psds2, t_evt_ma_f1, t_evt_mi_f1, t_seg_ma_f1, t_seg_mi_f1, t_inter_f1


if __name__ == "__main__":
    n_repeat = 16
    for iter in range(n_repeat):
        main(iter)
        # main()
