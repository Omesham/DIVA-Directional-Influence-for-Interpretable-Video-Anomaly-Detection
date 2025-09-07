import argparse
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from codes import bbox, signal, vad
from codes.utils import load_config, task
from collections import defaultdict

# Set reproducibility seeds
random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.use_deterministic_algorithms(True)

# Apply Gaussian smoothing to every video signal in a nested dict
def gaussian_video_d(d, sigma=3):
    ret = {}
    for k in d:
        ret[k] = {}
        for vname in d[k]:
            ret[k][vname] = gaussian_filter1d(d[k][vname], sigma)
    return ret

# Compute per-video AUROC scores from anomaly predictions and ground truth labels
def calc_AUROC_d(dataset_name, d):
    lengths = np.load(f'/Anomaly-Detection-CKNN-XAI/meta/test_lengths_{dataset_name}.npy')
    labels = np.load(f'/Anomaly-Detection-CKNN-XAI/meta/frame_labels_{dataset_name}.npy')
    vnames = vad.get_vnames(dataset_name, mode='test')
    lengths = np.concatenate([[0], lengths])  # cumulative lengths

    ret = {}
    for k in d:
        ret[k] = {}
        for i_s, i_e, vname in zip(lengths[:-1], lengths[1:], vnames):
            arr = d[k][vname]
            # Add dummy 0 and 1 values to prevent AUROC edge cases
            y_true = np.concatenate([[0], labels[i_s: i_e], [1]])
            y_pred = np.concatenate([[0], arr, [1000000]])
            AUROC = roc_auc_score(y_true, y_pred)
            ret[k][vname] = AUROC * 100
    return ret

def main(args, config):
    dataset_name = args.dataset_name
    uvadmode = args.mode
    cf_sig = config.signals

    # Load appearance and motion scores
    with task('Load features'):
        d = {'te_scorebbox': {}, }
        d['te_scorebbox']['mot'] = signal.MotSignal(dataset_name, cf_sig.mot, uvadmode).get()
        d['te_scorebbox']['app'] = signal.AppSignal(dataset_name, cf_sig.app, uvadmode).get()

    # Attach signals to bounding boxes
    with task():
        obj = bbox.VideosFrameBBs.load(dataset_name, mode='test')
        keys_use = [k for k in config.signals if getattr(config.signals[k], 'use', False)]
        for key in keys_use:
            obj.add_signal(key, d['te_scorebbox'][key])

    d_scores_save = {}

    # Get maximum score per frame over all bounding boxes
    d_scores = obj.get_framesignal_maximum()
    d_scores_save.update(d_scores)

    # Optional post-processing with Gaussian smoothing
    if config.postprocess.sigma > 0:
        d_scores = gaussian_video_d(d_scores, config.postprocess.sigma)

    d_scores_save['all'] = d_scores['all']

    # Load per-frame DI (Directional Influence) values from TCAV
    tcav_data = np.load("outputs/xai/di_per_frame_for_plot.npy", allow_pickle=True).item()

    di_per_video = defaultdict(list)

    # Load test sequence information
    lengths_cum = np.load(f"/Anomaly-Detection-CKNN-XAI/meta/test_lengths_{dataset_name}.npy")
    vnames = vad.get_vnames(dataset_name, mode="test")
    lengths = np.diff(np.concatenate([[0], lengths_cum]))

    default_di_score = 0.0  # fallback DI score for frames with missing data

    # Accumulate sum of absolute DI scores (for increasing concepts) per frame
    for (video, idx), fr in tcav_data.items():
        inc_vals = fr.get("top10_di_inc_values", [default_di_score])        
        di_score = np.sum(np.abs(inc_vals))           
        di_per_video[video].append(di_score)

    # Smooth DI scores per video
    for v in di_per_video:
        di_per_video[v] = gaussian_filter1d(np.array(di_per_video[v]), sigma=5)

    fused_scores = {}
    alpha = 0.7  # weighting factor: 70% CKNN + 30% DI

    for v in vnames:
        cknn_arr = np.array(d_scores['all'][v])
        num_frames = len(cknn_arr)

        # Retrieve DI scores (or fill missing frames with default)
        di_arr = np.array(di_per_video.get(v, [default_di_score] * num_frames))

        # Pad or trim DI scores to match frame length
        if len(di_arr) < num_frames:
            di_arr = np.concatenate([di_arr, np.full(num_frames - len(di_arr), default_di_score)])
        elif len(di_arr) > num_frames:
            di_arr = di_arr[:num_frames]

        # Normalize both scores to [0, 1]
        cknn_norm = (cknn_arr - cknn_arr.min()) / (cknn_arr.max() - cknn_arr.min() + 1e-8)
        di_norm = (di_arr - di_arr.min()) / (di_arr.max() - di_arr.min() + 1e-8)

        # Fuse scores using convex combination
        fused = alpha * cknn_norm + (1 - alpha) * di_norm
        fused_scores[v] = fused

    # Save the fused scores for evaluation
    d_scores_save['fused'] = fused_scores

    # Evaluate AUROC for original CKNN and fused scores
    d_AUROC = calc_AUROC_d(dataset_name, d_scores_save)

    print("==============================")
    print(f"AUROC CKNN:  {np.mean(list(d_AUROC['all'].values())):.2f}")
    print(f"AUROC FUSED: {np.mean(list(d_AUROC['fused'].values())):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_name", default='ped2', choices=['shanghaitech', 'ped2'])
    parser.add_argument("--mode", default='partial', choices=['partial', 'merge'])
    parser.add_argument("--quiet", action='store_true', default=True)
    parser.add_argument("--override", default='{}', type=str)

    args_ = parser.parse_args()
    config_ = load_config(args_.config, args_)

    main(args_, config_)
