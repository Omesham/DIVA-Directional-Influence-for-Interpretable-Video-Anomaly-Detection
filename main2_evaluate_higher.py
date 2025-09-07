import argparse
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from codes import bbox, signal, vad
from codes.utils import load_config, task
from collections import defaultdict

normal_prompts = [
    "talking on a cellphone", "looking at a cellphone", "group of people walking",
    "reading a book", "strolling casually while checking a phone", "a group chatting casually",
    "sitting quietly on a bench", "a group sitting on a bench or ledge by the walkway",
    "multiple people relaxing on benches near the walkway", "sitting", "chatting"
]

anomalous_prompts = [
    "riding a bicycle", "skateboarding", "a motor scooter driving",
    "a motorized cart being driven through the pedestrian walkway",
    "a motorized tricycle driving through the pedestrian zone", "a physical fight",
    "two individuals chasing each other", "kicking and attacking someone",
    "snatching or stealing a backpack and sprinting or running away",
    "running down while carrying a stolen bag", "running"
]

prompts = normal_prompts + anomalous_prompts

def gaussian_video_d(d, sigma=3):
    ret = {}
    for k in d:
        ret[k] = {}
        for vname in d[k]:
            ret[k][vname] = gaussian_filter1d(d[k][vname], sigma)
    return ret

def calc_AUROC_d(dataset_name, d):
    lengths = np.load(f'/mmfs1/scratch/jacks.local/ojanigala/Anomaly-Detection-CKNN-PyTorch/meta/test_lengths_{dataset_name}.npy')
    labels = np.load(f'/mmfs1/scratch/jacks.local/ojanigala/Anomaly-Detection-CKNN-PyTorch/meta/frame_labels_{dataset_name}.npy')
    vnames = vad.get_vnames(dataset_name, mode='test')
    lengths = np.concatenate([[0], lengths])

    ret = {}
    for k in d:
        ret[k] = {}
        for i_s, i_e, vname in zip(lengths[:-1], lengths[1:], vnames):
            arr = d[k][vname]
            y_true = np.concatenate([[0], labels[i_s: i_e], [1]])
            y_pred = np.concatenate([[0], arr, [1000000]])
            AUROC = roc_auc_score(y_true, y_pred)
            ret[k][vname] = AUROC * 100
    return ret

def main(args, config):
    dataset_name = args.dataset_name
    uvadmode = args.mode
    cf_sig = config.signals

    with task('Load features'):
        d = {'te_scorebbox': {}, }
        d['te_scorebbox']['mot'] = signal.MotSignal(dataset_name, cf_sig.mot, uvadmode).get()
        d['te_scorebbox']['app'] = signal.AppSignal(dataset_name, cf_sig.app, uvadmode).get()

    with task():
        obj = bbox.VideosFrameBBs.load(dataset_name, mode='test')
        keys_use = [k for k in config.signals if getattr(config.signals[k], 'use', False)]
        for key in keys_use:
            obj.add_signal(key, d['te_scorebbox'][key])

    d_scores_save = {}
    d_scores = obj.get_framesignal_maximum()
    d_scores_save.update(d_scores)

    if config.postprocess.sigma > 0:
        d_scores = gaussian_video_d(d_scores, config.postprocess.sigma)

    d_scores_save['all'] = d_scores['all']

    tcav_data = np.load("outputs/xai/tcavoutputs/UntitledFolder/tcav_per_frame_for_plot.npy", allow_pickle=True).item()

    di_per_video = defaultdict(list)

    lengths_cum = np.load(f"/mmfs1/scratch/jacks.local/ojanigala/Anomaly-Detection-CKNN-PyTorch/meta/test_lengths_{dataset_name}.npy")
    vnames = vad.get_vnames(dataset_name, mode="test")
    lengths = np.diff(np.concatenate([[0], lengths_cum]))

    default_di_score = 6.5129

    for (video, idx), fr in tcav_data.items():
        inc_vals = fr.get("top10_di_inc_values", [default_di_score])
        dec_vals = fr.get("top10_di_dec_values", [default_di_score])
        # di_score = np.sum(np.abs(inc_vals)) - np.sum(np.abs(dec_vals))
        # di_score = np.sum(np.abs(inc_vals))
        di_score = np.sum(np.abs(dec_vals))

        

        di_per_video[video].append(di_score)

    for v in di_per_video:
        di_per_video[v] = gaussian_filter1d(np.array(di_per_video[v]), sigma=2)

    fused_scores = {}
    alpha = 0.7  # weight for CKNN

    for v in vnames:
        cknn_arr = np.array(d_scores['all'][v])
        num_frames = len(cknn_arr)

        di_arr = np.array(di_per_video.get(v, [default_di_score] * num_frames))

        if len(di_arr) < num_frames:
            di_arr = np.concatenate([di_arr, np.full(num_frames - len(di_arr), default_di_score)])
        elif len(di_arr) > num_frames:
            di_arr = di_arr[:num_frames]

        cknn_norm = (cknn_arr - cknn_arr.min()) / (cknn_arr.max() - cknn_arr.min() + 1e-8)
        di_norm = (di_arr - di_arr.min()) / (di_arr.max() - di_arr.min() + 1e-8)

        fused = alpha * cknn_norm + (1 - alpha) * di_norm
        fused_scores[v] = fused

    d_scores_save['fused'] = fused_scores

    d_AUROC = calc_AUROC_d(dataset_name, d_scores_save)

    print("==============================")
    print(f"AUROC CKNN:  {np.mean(list(d_AUROC['all'].values())):.2f}")
    print(f"AUROC FUSED: {np.mean(list(d_AUROC['fused'].values())):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_name", default='ped2', choices=['shanghaitech', 'avenue', 'ped2'])
    parser.add_argument("--mode", default='partial', choices=['partial', 'merge'])
    parser.add_argument("--quiet", action='store_true', default=True)
    parser.add_argument("--override", default='{}', type=str)

    args_ = parser.parse_args()
    config_ = load_config(args_.config, args_)

    main(args_, config_)
