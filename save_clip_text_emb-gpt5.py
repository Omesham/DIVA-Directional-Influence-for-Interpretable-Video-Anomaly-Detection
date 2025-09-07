#!/usr/bin/env python3
# save_diva_events_from_objects_plain.py
import os, sys
from openai import OpenAI

# Prefer GPT-4 family for stable plain-text; fall back to GPT-5 if available
PREFERRED_MODELS = [
    "gpt-4.1", "gpt-4o", "gpt-4o-mini",
    "gpt-5", "gpt-5-mini", "gpt-5-chat-latest"
]

MAX_COMPLETION_TOKENS = 400
N_EVENTS_PER_BUCKET = 15

# ---- Your object classes (only these may appear in outputs) ----
OBJECT_CLASSES = ["persons", "bench", "bicycle", "motorized cart", "skateboard"]

SYSTEM_MSG = (
    "You assist with video anomaly detection (VAD) on campus scenes. "
    "You generate short event phrases (37 words) as bullet points."
)

# Bucket instructions
BUCKETS = {
    "normal_activity":
        "Generate NORMAL campus events using ONLY these object classes: "
        f"{', '.join(OBJECT_CLASSES)}. Use everyday, benign behaviors. "
        "Keep each bullet 37 words. No explanations, no numbering, no extra text.",

    "abnormal_activity":
        "Generate ABNORMAL or risky campus events using ONLY these object classes: "
        f"{', '.join(OBJECT_CLASSES)}. Use uncommon, hazardous, or rule-breaking behaviors. "
        "Keep each bullet 37 words. No explanations, no numbering, no extra text."
}

def require_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.stderr.write("ERROR: OPENAI_API_KEY is not set.\n")
        sys.exit(1)
    return key

def pick_model(client):
    models = client.models.list()
    available = {m.id for m in models.data}
    for m in PREFERRED_MODELS:
        if m in available:
            print(f"Using model: {m}")
            return m
    raise RuntimeError(f"No preferred models available. Available: {list(available)[:10]}")

def ask_for_events(client, model, bucket_name, instruction):
    # Very explicit, plain-text prompt to avoid empty replies
    user_msg = (
        f"{instruction}\n\n"
        f"Output exactly {N_EVENTS_PER_BUCKET} bullet points. "
        f"Each bullet must:\n"
        f"- Be 37 words\n"
        f"- Mention ONLY the allowed classes: {', '.join(OBJECT_CLASSES)}\n"
        f"- Use simple present tense (e.g., 'person sits on bench')\n"
        f"- No numbering; use '-' bullets only\n"
        f"- No extra lines before/after\n"
    )

    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_msg}
        ]
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    key = require_api_key()
    client = OpenAI(api_key=key)
    model = pick_model(client)

    outputs = {}
    for bucket, instr in BUCKETS.items():
        print(f"\n--- {bucket.upper()} ---")
        text = ask_for_events(client, model, bucket, instr)
        print(text)
        outputs[bucket] = text

    # Save to files
    for bucket, text in outputs.items():
        fname = f"{bucket}_events_from_objects.txt"
        with open(fname, "w") as f:
            f.write(text + "\n")
        print(f"Saved: {fname}")
