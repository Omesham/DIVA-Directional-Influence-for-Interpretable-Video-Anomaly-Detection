from codes.xai.tcav_explainer import TCAVExplainer

def get_explainer(method, grader, train_feats, **kw):  
    if method == "tcav":
        return TCAVExplainer(
            grader, train_feats,
            prompts       = kw["prompts"],
            text_emb_path = kw["text_emb_path"],
            alpha         = kw.get("alpha", 0.2),
        )

    raise ValueError(f"Unknown XAI method: {method}")
