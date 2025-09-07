from .base import BaseExplainer
import numpy as np
import torch
import torch.nn.functional as F



class TCAVExplainer(BaseExplainer):
    def __init__(self, grader, train_feats, prompts, text_emb_path, alpha=0.2):
        super().__init__(grader, train_feats)
        self.prompts = prompts
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # unit text embeddings for cosine + step direction
        self.text_emb_unit = F.normalize(
            torch.from_numpy(np.load(text_emb_path).astype(np.float32)).to(self.device), dim=1
        )

    def explain(self, X_raw, gt_label=None):
        results = []
        for f_np in X_raw:
            # RAW feature for CKNN
            f_raw = torch.from_numpy(f_np).float().to(self.device)
            f_len = f_raw.norm() + 1e-8

            # UNIT copies for cosine + direction
            f_unit = f_raw / f_len
            cos_before = F.cosine_similarity(f_unit.unsqueeze(0), self.text_emb_unit).cpu().numpy()

            # Perturb in UNIT space ? map back to RAW norm
            delta_unit = self.text_emb_unit - f_unit.unsqueeze(0)          # [num_prompts, d]
            f_unit_pert = F.normalize(f_unit.unsqueeze(0) + self.alpha * delta_unit, dim=1)
            f_pert_raw  = f_unit_pert * f_len                 # match bank scale

            # CKNN scoring expects numpy
            f_np_orig       = f_raw.detach().cpu().numpy()
            f_np_perturbed  = f_pert_raw.detach().cpu().numpy()

            s0 = self._score_batch(f_np_orig[None, :])[0]                   # scalar
            s1 = self._score_batch(f_np_perturbed)                          # [num_prompts]
            delta_scores = s1 - s0

            # DI per-unit step toward concept
            concept_dist = torch.norm(delta_unit, dim=1).cpu().numpy()  # ||c_unit - f_unit||
            normed_di = delta_scores / (self.alpha * concept_dist + 1e-8)

            cos_after = F.cosine_similarity(f_unit_pert, self.text_emb_unit).cpu().numpy()

            topk_idx    = np.argsort(-normed_di)
            bottomk_idx = np.argsort( normed_di)

            results.append({
                "original_score": float(s0),
                "gt_label": int(gt_label) if gt_label is not None else None,
                "delta_scores": delta_scores.tolist(),
                "normalized_directional_influence": normed_di.tolist(),
                "cosine_before": cos_before.tolist(),
                "cosine_after": cos_after.tolist(),
                "top_di_inc_prompts": [self.prompts[i] for i in topk_idx],
                "top_di_inc_values": [float(normed_di[i]) for i in topk_idx],
                "top_di_dec_prompts": [self.prompts[i] for i in bottomk_idx],
                "top_di_dec_values": [float(normed_di[i]) for i in bottomk_idx],
            })
        return results

