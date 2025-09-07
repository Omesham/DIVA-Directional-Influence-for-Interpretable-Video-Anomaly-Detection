from codes.xai.lime_explainer import LimeExplainer
from codes.xai.shap_explainer import ShapExplainer
from codes.xai.glime_explainer import GlimeExplainer
from codes.xai.counterfactual_explainer import CounterfactualExplainer
from codes.xai.tcav_explainer import TCAVExplainer

def get_explainer(method, grader, train_feats, **kw):
    if method == "lime":
        return LimeExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            num_features  = kw.get("num_features", 10),
        )

    if method == "shap":
        return ShapExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            nsamples      = kw.get("nsamples", 200),
        )

    if method == "glime":
        return GlimeExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            num_features  = kw.get("num_features", 10),
            sampler       = kw.get("sampler", "binomial"),
            kernel_width  = kw.get("kernel_width", 0.75),
            random_state  = kw.get("random_state", 0),  # ✅ Added for determinism
        )

    if method == "counterfactual":
        return CounterfactualExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            delta         = kw.get("delta", 0.02),
            max_iter      = kw.get("max_iter", 200),
            threshold     = kw.get("threshold", 0.5),
        )



    if method == "tcav":
        return TCAVExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            alpha         = kw.get("alpha", 0.2),
        )

    raise ValueError(f"Unknown XAI method: {method}")
