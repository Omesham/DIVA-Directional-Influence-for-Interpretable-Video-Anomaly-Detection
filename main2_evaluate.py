import argparse
import random
import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from codes.xai import get_explainer
from codes import featurebank
from codes.grader import KNNGrader
import matplotlib.pyplot as plt
from codes import bbox, signal, vad
from codes.utils import load_config, task
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib import gridspec
from matplotlib import rcParams
# --- Global font settings ---
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.weight'] = 'bold'

# --- Global look & size ---
FIGSIZE = (30, 16)  # bigger figures everywhere

rcParams.update({
    "figure.figsize": FIGSIZE,
    "figure.dpi": 300,
    "savefig.dpi": 500,
    "axes.titlesize": 50,
    "axes.titleweight": "bold",
    "axes.labelsize": 50,
    "axes.labelweight": "bold",
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "axes.linewidth": 2.2,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "legend.fontsize": 50,
    "legend.title_fontsize": 50,
})

def boldify(ax):
    """Force bold on everything in this axes (title, labels, ticks)."""
    ax.title.set_weight("bold")
    ax.xaxis.label.set_weight("bold")
    ax.yaxis.label.set_weight("bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.use_deterministic_algorithms(True)

base_dir = os.getcwd()



# normal_prompts = [
#     "pedestrians walking together on a campus walkway",
#     "several students strolling side by side along the sidewalk",
#     "a crowd of people walking at a normal pace on the campus path"
# ]

# anomalous_prompts = [   
#     "a person riding a bicycle through the pedestrian path",
#     "a skateboarder gliding along the sidewalk",
#     "a person driving a maintenance cart on the pedestrian path"   
# ]
normal_prompts = [   
    "group of people walking",
    "a group chatting casually while sitted",
    "sitting quietly on a bench",
    "a group sitting on a bench or ledge by the walkway"
]

#Refined Anomalous Prompts
anomalous_prompts = [
    "a person riding a bicycle",
    "a person skateboarding",
    "a tricycle driving through the pedestrian walkway",
    "two individuals chasing each other"
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
    lengths = np.load(f'/mmfs1/scratch/ojanigala/Anomaly-Detection-CKNN-PyTorch/meta/test_lengths_{dataset_name}.npy')
    labels = np.load(f'/mmfs1/scratch/ojanigala/Anomaly-Detection-CKNN-PyTorch/meta/frame_labels_{dataset_name}.npy')
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



def get_prompt_color(prompt):
    if prompt in normal_prompts:
        return '#2ca02c'  # green for normal
    elif prompt in anomalous_prompts:
        return '#d62728'  # red for anomalous
    else:
        return '#1f77b4'  # blue if not found (just in case)


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

    if getattr(config, "xai", None) and config.xai.use:
        tr_feats = featurebank.get(dataset_name, "app", "train", uvadmode)
        if cf_sig.app.cleanse_scorename:
            cleanse = np.load(
                f"{base_dir}/features/{dataset_name}/cleansescores/"
                f"{uvadmode}_{cf_sig.app.cleanse_scorename}_flat.npy", allow_pickle=True)
            mask = cleanse <= np.percentile(cleanse, cf_sig.app.percentile_cleanse)
            tr_feats = tr_feats[mask]

        local_grader = KNNGrader(tr_feats, K=cf_sig.app.NN, key="app")
        te_feats = featurebank.get(dataset_name, "app", "test", uvadmode)
        te_scores = d['te_scorebbox']['app']

        lengths_cum = np.load(f"{base_dir}/meta/test_lengths_{dataset_name}.npy")
        vnames = vad.get_vnames(dataset_name, mode="test")
        
        # ✅ Convert cumulative lengths to per-video lengths
        lengths = np.diff(np.concatenate([[0], lengths_cum]))
        
        frame_labels = np.load(f"{base_dir}/meta/frame_labels_{dataset_name}.npy")
        
        # Build frame_to_video mapping correctly
        frame_to_video = [v for v, n in zip(vnames, lengths) for _ in range(n)]
        
        # Build first_idx_of mapping (starting index for each video)
        first_idx_of = {}
        start = 0
        for v, n in zip(vnames, lengths):
            first_idx_of[v] = start
            start += n

       
        explainer = get_explainer(
            method="tcav",
            grader=local_grader,
            train_feats=tr_feats,
            prompts=prompts,
            text_emb_path="clip_text_emb.npy",
            alpha=0.2
        )

        per_frame_tcav = {}
        tcav_plot_full = {}
        os.makedirs("outputs/xai", exist_ok=True)

        for frame_idx, (scores, feats) in enumerate(zip(te_scores, te_feats)):
            if not len(scores):
                continue

            video = frame_to_video[frame_idx]
            local_idx = frame_idx - first_idx_of[video]
            # frame_vec = np.mean(feats, axis=0, keepdims=True)
            # gt_label = int(frame_labels[frame_idx])

            # result = explainer.explain(frame_vec, gt_label=gt_label)[0]
            patch_idx  = int(np.argmax(scores))          # scores is the per-patch CKNN list
            patch_vec  = feats[patch_idx : patch_idx+1]  # shape (1, D)  -> keeps API happy
           
            gt_label = int(frame_labels[first_idx_of[video] + local_idx])  


            result     = explainer.explain(patch_vec, gt_label=gt_label)[0]
            result["score"] = float(max(scores))

            # === Plot Top-10 Similarity Scores ===
            sims = result["cosine_before"]
            topk_sim = np.argsort(sims)[-10:][::-1]
            labels_sim = [prompts[i] for i in topk_sim]
            values_sim = np.array(sims)[topk_sim]
            colors_sim = [get_prompt_color(p) for p in labels_sim]

            # fig1, ax1 = plt.subplots(figsize=(12, 8))
            # ax1.barh(labels_sim, values_sim, color=colors_sim)
            # ax1.invert_yaxis()
            # #ax1.set_title(f"{video} frame {local_idx} Top-10 Prompt Similarities")
            # fig1.tight_layout()
            
            # # --- Save high-res figure for LaTeX ---
            # fig1.savefig(
            #     f"outputs/xai/{video}_frame_{local_idx:03d}_sims.png",
            #     dpi=500,              # High resolution for print
            #     bbox_inches="tight"   # Avoid cutting off labels
            # )
            # plt.close(fig1)

            # === Plot Top-10 Directional Influence (TCAV) ===
            di = np.array(result["normalized_directional_influence"])        
            # spread values *inside this frame* so tallest bars stand out
            di = (di - di.mean()) / (di.std())    # ← add this line
            # Save full normalized DI vector
            result["di_normalized_full"] = di.tolist()
            result["all_prompts_order"] = prompts
            #di = -di  
            top_inc_idx = np.argsort(di)[-10:][::-1]
            labels_inc = [prompts[i] for i in top_inc_idx]
            values_inc = di[top_inc_idx]
            colors_inc = [get_prompt_color(p) for p in labels_inc]

            
            # --- TCAV increase ---
            fig_inc, ax_inc = plt.subplots(figsize=FIGSIZE)
            ax_inc.barh(labels_inc, values_inc, color=colors_inc)
            ax_inc.set_xlabel("Directional Influence")
            ax_inc.set_ylabel("Concepts")
            ax_inc.invert_yaxis()
            boldify(ax_inc)
            fig_inc.tight_layout()
            fig_inc.savefig(f"outputs/xai/{video}_frame_{local_idx:03d}_tcav_inc.pdf",
                            dpi=500, bbox_inches="tight")
            plt.close(fig_inc)


            
            # # Top-10 decreasing prompts
            top_dec_idx = np.argsort(di)[:10]
            labels_dec = [prompts[i] for i in top_dec_idx]
            values_dec = di[top_dec_idx]
            colors_dec = [get_prompt_color(p) for p in labels_dec]
            
            # --- TCAV decrease ---
            fig_dec, ax_dec = plt.subplots(figsize=FIGSIZE)
            ax_dec.barh(labels_dec, values_dec, color=colors_dec)
            ax_dec.set_xlabel("Directional Influence")
            ax_dec.set_ylabel("Concepts")
            ax_dec.invert_yaxis()
            boldify(ax_dec)
            fig_dec.tight_layout()
            fig_dec.savefig(f"outputs/xai/{video}_frame_{local_idx:03d}_tcav_dec.pdf",
                            dpi=500, bbox_inches="tight")
            plt.close(fig_dec)

            # === Save full plot-relevant dict ===
            tcav_plot_full[(video, local_idx)] = {
                "frame_idx": local_idx,
                "video_name": video,
                "gt_label": gt_label,
        
                "all_prompts_order": prompts,
                "di_normalized_full": di.tolist(),
        
                "top10_di_inc_prompts": labels_inc,
                "top10_di_inc_values": values_inc.tolist(),
        
                "top10_di_dec_prompts": labels_dec,
                "top10_di_dec_values": values_dec.tolist(),
            }

            # === Save full explanation dict ===
            result.update({
            "frame_idx": frame_idx,
            "video_name": video,
        
            # Cosine similarity
            "top10_sim_indices": topk_sim.tolist(),
            "top10_sim_prompts": labels_sim,
            "top10_sim_values": values_sim.tolist(),
            "top10_sim_colors": colors_sim,
        
            # Directional Influence ↑ (increase)
            "top10_di_inc_indices": top_inc_idx.tolist(),
            "top10_di_inc_prompts": labels_inc,
            "top10_di_inc_values": values_inc.tolist(),
            "top10_di_inc_colors": colors_inc,
        
            # Directional Influence ↓ (decrease)
            "top10_di_dec_indices": top_dec_idx.tolist(),
            "top10_di_dec_prompts": labels_dec,
            "top10_di_dec_values": values_dec.tolist(),
            "top10_di_dec_colors": colors_dec,
            })

            per_frame_tcav[(video, frame_idx)] = result

        # np.save("outputs/xai/tcav_per_frame.npy", per_frame_tcav)
        np.save("outputs/xai/tcav_per_frame_shan.npy", per_frame_tcav)           # ← the original full one
        np.save("outputs/xai/tcav_per_frame_for_plot_shan.npy", tcav_plot_full)   # ← the clean plot-specific one


    d['te_score'] = d_scores_save
    d_AUROC = calc_AUROC_d(dataset_name, d_scores)
    d['AUROC'] = d_AUROC['all']
    AUROC = np.mean(list(d_AUROC['all'].values()))
    print(f'AUROC {args.dataset_name} ({args.mode}): {AUROC:.1f}', end='')
    if args.quiet:
        print()
    else:
        print(' ', args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--dataset_name", default='ped2', choices=['shanghaitech', 'avenue', 'ped2'])
    parser.add_argument("--mode", default='partial', choices=['partial', 'merge'])

    parser.add_argument("--quiet", action='store_true', default=True)
    parser.add_argument("--override", default='{}', type=str)
    args_ = parser.parse_args()
    config_ = load_config(args_.config, args_)
    if not args_.quiet:
        print(args_)

    main(args_, config_)