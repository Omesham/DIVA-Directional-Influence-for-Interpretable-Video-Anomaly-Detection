# DIVA: Directional Influence for Interpretable Video Anomaly Detection

DIVA (Directional Influence for Video Anomaly Detection) is a gradient-free TCAV adaptation for black-box VAD models (e.g., KNN, clustering, KDE). It perturbs CLIP embeddings along concept directions to reveal how interpretable concepts influence anomaly scores—no model access needed.

---

## 🔍 Cleansed k-Nearest Neighbor (CKNN)

This repository includes the official PyTorch implementation of **CKNN**, introduced in **CIKM 2024**, as a base VAD method for fusing with DIVA explanations.

> **Citation**  
```bibtex
@article{yi2024cknn,
  title={CKNN: Cleansed k-Nearest Neighbor for Unsupervised Video Anomaly Detection},
  author={Yi, Jihun and Yoon, Sungroh},
  journal={arXiv preprint arXiv:2408.03014},
  year={2024}
}
```

---

## Installation

### Step 1. Install Libraries

- Python version: `3.7`
- Install all required dependencies:

```bash
pip install -r requirements.txt
```
---

## 📦 Data Preparation

### Step 2. Download Required Files

Download the following required files from cknn authors' google drive:

- **[Download Precomputed Features](https://drive.google.com/file/d/1FT97l_fN6rvvXYRvEnq4SKoIOyP8RNOK/view)**
- **[Download Meta Data](https://drive.google.com/file/d/1BmoY_BQnXxMnS8etydHMaqXg13c3uJ7l/view)**
- **[Download Patches (6 files)](https://drive.google.com/drive/folders/1PK7-0K-it4Ldt-uSYNtCbj1-TKzafYBi)**

### Step 3. Extract and Organize

Extract all files so the directory looks like this:

```
CKNN/
├── features/
│   ├── ped2/{train,test}/
│   └── shanghaitech/{train,test}/
├── meta/
│   ├── frame_labels_*.npy
│   ├── test_lengths_*.npy
│   ├── test_bboxes_*.npy
│   └── train_bboxes_*.npy
├── patches/
│   ├── ped2/{train,test}/
│   └── shanghaitech/{train,test}/
```

---

## CKNN + DIVA Pipeline

### Step 4. Generate Pseudo Anomaly Scores (CKNN)

```bash
./run1_pseudo_anomaly_scores.sh ped2
```

> Replace `ped2` with `shanghaitech` if shanghaitech.

---

### Step 5. Generate Textual Prompts
Run to generate unique object classes list:

```bash
python codes/generate_unique_object_class_list.py
```

This will generate unique object classes (e.g., bicycle, person) to be passed to Gpt-5 LLM.

Run the prompt generator using GPT-5:

```bash
python codes/generate_text_prompts_gpt5.py
```

This will generate concept prompts (e.g., sitting, running, fighting) used for directional influence.

---

### Step 6. Encode Prompts with CLIP

Use CLIP to convert the text prompts into embeddings:

```bash
python codes/save_clip_text_emb.py
```

This saves text embeddings needed for the directional influence calculations.

---

### Step 7. Compute Directional Influence Scores

Run the DIVA directional influence computation:

```bash
python create_di_scores.py --config configs/ped2.yaml --dataset_name ped2
```

This computes per-frame DI scores using the CLIP-aligned concept vectors.

---

### Step 8. Fuse CKNN + DIVA Scores and Evaluate

Finally, fuse the scores and evaluate AUROC:

```bash
!./run_main_diva_xai.sh shanghaitech
```

This computes:
- CKNN-only AUROC with DI explanations
- Fused CKNN + DI AUROC

---

## Folder Structure

```
codes/
├── bbox.py
├── cleanse.py
├── featurebank.py
├── grader.py
├── main.py
├── signal.py
├── utils.py
├── save_clip_text_emb.py  
├── generate_unique_object_class_list.py
├── generate_text_prompts_gpt5.py
├── xai/
│   ├── __init__.py
│   ├── base.py
│   └── directional-influence.py
create_di_scores.py
main2_evaluate.py
configs/
meta/
patches/
features/
outputs/
```

---

## Acknowledgements

- CKNN baseline from **Yi & Yoon, 2024 (CIKM)**.
- DIVA explanation module inspired by **TCAV** and adapted for CLIP.
- CLIP embedding via `clip-anytorch` and `open_clip_torch`.

---
