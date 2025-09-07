#!/usr/bin/env python3
# save_clip_text_emb.py
import clip
import torch
import numpy as np

normal_prompts = [
    "pedestrians walking together on a campus walkway",
    "several students strolling side by side along the sidewalk",
    "a crowd of people walking at a normal pace on the campus path"
]

anomalous_prompts = [   
    "a person riding a bicycle through the pedestrian path",
    "a skateboarder gliding along the sidewalk",
    "a person driving a maintenance cart on the pedestrian path"   
]

 
# normal_prompts = [   
#     "group of people walking",
#     "a group chatting casually while sitted",
#     "sitting quietly on a bench",
#     "a group sitting on a bench or ledge by the walkway"
# ]

# #Refined Anomalous Prompts
# anomalous_prompts = [
#     "a person riding a bicycle",
#     "a person skateboarding",
#     "a tricycle driving through the pedestrian walkway",
#     "two individuals chasing each other"
# ]

prompts = normal_prompts + anomalous_prompts  

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    # 1. Tokenize & encode
    with torch.no_grad():
        tokens   = clip.tokenize(prompts).to(device)           # [42, token_len]
        text_emb = model.encode_text(tokens).float().cpu().numpy()  # [42, D]

    # 2. Normalize
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)

    # 3. Save
    np.save("clip_text_emb.npy", text_emb)
    print("✅ Saved text embeddings to clip_text_emb.npy")

if __name__ == "__main__":
    main()