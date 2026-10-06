
# DAKF: Domain-Adaptive Knowledge Fusion for Medical Image Captioning

A parameter-efficient fine-tuning pipeline for medical vision-language models, evaluated on the Radiology_mini benchmark. This repository contains the full training, evaluation, and ablation code for four PEFT strategies — LoRA, AdaLoRA, KPL-METER, and DAKF — with reproducible results, confidence intervals, and significance testing.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Method](#method)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Configuration Reference](#configuration-reference)
- [Hardware](#hardware)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

Medical image captioning requires models that produce clinically accurate, structurally fluent reports. Full fine-tuning of large vision-language models (VLMs) is prohibitively expensive on consumer hardware. This project benchmarks four parameter-efficient fine-tuning (PEFT) methods for this task:

1. **Standard LoRA** — Low-rank adaptation baseline
2. **AdaLoRA** — Adaptive rank allocation with orthogonality regularization
3. **KPL-METER** — Knowledge-injected alignment using UMLS concept annotations
4. **DAKF** — Domain-Adaptive Knowledge Fusion (proposed method)

The proposed **DAKF** pipeline combines three techniques:
- **Retrieval-augmented exemplar prompting** using top-k semantically similar training captions
- **Dual-pass generation** (draft → clinical refinement)
- **UMLS-grounded query formulation** for retrieval

All experiments run on a single NVIDIA T4 (16 GB) GPU with QLoRA quantization.

---

## Key Results

Evaluation on 100 held-out samples from Radiology_mini:

| Method | BLEU-1 | BLEU-3 | BLEU-4 | ROUGE-L | Entity Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard LoRA | 0.1461 | 0.0458 | 0.0295 | 0.2029 | **0.7950** |
| AdaLoRA | 0.1004 | 0.0264 | 0.0164 | 0.1371 | 0.6950 |
| KPL-METER | 0.1367 | 0.0499 | 0.0351 | 0.2089 | 0.7450 |
| **DAKF (Ours)** | **0.1597** | **0.0728** | **0.0608** | 0.1973 | 0.7400 |

**Headline result:** DAKF achieves a **106% improvement in BLEU-4** over the Standard LoRA baseline (0.0608 vs. 0.0295), demonstrating that retrieval-augmented dual-pass generation substantially improves caption quality for medical VLMs.

**Trade-off note:** Standard LoRA retains higher Entity Accuracy (0.7950 vs. 0.7400), reflecting its tendency to reproduce reference terminology verbatim rather than paraphrasing.

---

## Method

### DAKF Pipeline

```text
┌─────────────────────────────┐
│ Input Medical Image         │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ UMLS Query Formulation      │
│ (from cui annotations)      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Retrieval: top-k similar    │
│ training captions           │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Pass 1: Retrieval-augmented │
│ draft generation            │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Pass 2: Clinical refinement │
│ with anatomical detail      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ Final Radiology Report      │
└─────────────────────────────┘
```

### Training Pipeline

| Stage | Data | Steps | LR | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| 1 | Retrieval-augmented KPL | 35 | 2e-4 | Knowledge-grounded alignment |
| 2 | Dual-turn clinical CoT | 15 | 8e-5 | Findings → Impression format |
| 3 | Plain captions | 10 | 3e-5 | Contrastive re-anchoring |
| 4 | Retrieval-enriched DAKF | 15 | 1e-5 | Final adaptation |

- **Base model:** `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`
- **LoRA configuration:** rank 32, alpha 32, RSLoRA enabled, dropout 0.05, targeting `q_proj`, `k_proj`, `v_proj`, `o_proj`.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── run_training.py
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_dakf_pipeline.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── metrics.py
│   ├── retrieval.py
│   ├── dakf_inference.py
│   └── data_conversion.py
├── configs/
│   ├── lora.yaml
│   ├── adalora.yaml
│   ├── kpl_meter.yaml
│   └── dakf.yaml
├── results/
│   ├── results_lora.json
│   ├── results_ada.json
│   ├── results_kpl_only.json
│   ├── results_dakf.json
│   ├── per_sample_all.json
│   └── final_all_results.json
├── figures/
│   ├── final_comparison.png
│   └── qualitative_best.png
└── checkpoints/
    ├── lora_baseline/
    ├── adalora_baseline/
    ├── kpl_only/
    ├── final_ours/
    └── final_dakf/
```

---

## Requirements

- **Python:** 3.10+
- **PyTorch:** 2.1+
- **CUDA:** 12.1+ (for GPU training)
- **GPU VRAM:** 16 GB minimum (tested on NVIDIA T4)

### Python Packages (`requirements.txt`)
```text
unsloth
bitsandbytes
accelerate
xformers
peft
trl
triton
cut_cross_entropy
transformers
datasets
sentence-transformers
faiss-cpu
rouge-score
nltk
scipy
matplotlib
```

---

## Installation

```bash
git clone https://github.com/sourishdey2005/dakf-medical-captioning.git
cd dakf-medical-captioning

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Dataset

This project uses the `Radiology_mini` dataset from Unsloth, a subset of ROCOv2 radiology with expert captions.

```python
from datasets import load_dataset

dataset = load_dataset("unsloth/Radiology_mini", split="train")
```

The dataset contains:
- `image`: Medical image (X-ray, CT, ultrasound)
- `image_id`: Unique identifier
- `caption`: Expert-written radiology description
- `cui`: List of UMLS Concept Unique Identifiers

- **Training subset:** 500 samples (configurable via `MAX_SAMPLES`)
- **Evaluation:** 100 held-out samples (configurable via `n_samples`)

---

## Usage

### 1. Data Preparation
```bash
python run_training.py prepare
```
*Generates `alignment_plain.json`, `alignment_kpl.json`, `cot_data.json`, and `knowledge_cache.json`.*

### 2. Baseline Training
```bash
python run_training.py lora
python run_training.py adalora
```

### 3. DAKF Pipeline
```bash
python run_training.py dakf
```

### 4. Evaluation
```bash
python run_training.py eval_all
python run_training.py eval_dakf
```
*Results are written to the `results/` directory.*

---

## Reproducibility

All experiments use `seed=3407` for deterministic training.

### Determinism Notes
- Set `CUDA_VISIBLE_DEVICES=0` before importing `torch` to force single-GPU mode.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation.
- Set `TOKENIZERS_PARALLELISM=false` to avoid tokenizer warnings.

### Expected Training Times (NVIDIA T4, 16 GB)

| Task | Duration |
| :--- | :--- |
| Data preparation | ~3 min |
| LoRA baseline (60 steps) | ~7 min |
| AdaLoRA baseline (60 steps) | ~8 min |
| KPL-METER (60 steps) | ~7 min |
| DAKF (4 stages, 75 steps) | ~12 min |
| Full evaluation (n=100, 4 models) | ~20 min |

---

## Configuration Reference

### Training Hyperparameters

| Parameter | LoRA | AdaLoRA | KPL-METER | DAKF |
| :--- | :---: | :---: | :---: | :---: |
| LoRA rank (r) | 16 | 16 → 8 | 16 | 32 |
| LoRA alpha | 16 | 16 | 16 | 32 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 | 1e-5 (stage 4) |
| Batch size (per device) | 2 | 2 | 2 | 1 |
| Gradient accumulation | 4 | 4 | 4 | 8 |
| Warmup steps | 5 | 5 | 5 | 5 |
| Scheduler | linear | linear | linear | cosine |
| Max sequence length | 1024 | 1024 | 1024 | 1024 |

### AdaLoRA Schedule (`adalora.yaml`)
```yaml
init_r: 16
target_r: 8
tinit: 5
tfinal: 40
deltaT: 5
total_step: 60
orth_reg_weight: 0.0
```
*Note: AdaLoRA's orthogonality regularization is disabled due to an incompatibility with Unsloth's fused loss kernel. The AdaLoRA forward pass is patched to bypass loss post-processing while retaining rank allocation.*

---

## Hardware

- **Development:** Kaggle Notebooks, NVIDIA Tesla T4 (16 GB)
- **GPU mode:** Single GPU (`CUDA_VISIBLE_DEVICES=0` to avoid DataParallel device-mismatch errors)
- **Precision:** FP16 with 4-bit quantization (nf4, double quant)
- **Gradient checkpointing:** Enabled via Unsloth

---

## Citation

If you use this code or the DAKF method in your research, please cite:

```bibtex
@software{dakf2026,
  author = {Sourish Dey},
  title = {DAKF: Domain-Adaptive Knowledge Fusion for Medical Image Captioning},
  year = {2026},
  url = {https://github.com/sourishdey2005/dakf-medical-captioning}
}
```

If you use the `Radiology_mini` dataset:

```bibtex
@misc{radiology_mini,
  author = {Unsloth},
  title = {Radiology_mini: Sampled ROCOv2 radiology dataset},
  year = {2024},
  url = {https://huggingface.co/datasets/unsloth/Radiology_mini}
}
```

---

## License

Released under the Apache 2.0 License. See `LICENSE` for details.

The base model (`Qwen2-VL-2B-Instruct`) and the `Radiology_mini` dataset are subject to their own licenses. Users must comply with the Qwen and Hugging Face terms of use.

---

## Acknowledgments

- **Unsloth** for the 2x faster fine-tuning framework and QLoRA integrations
- **Hugging Face** for `transformers`, `datasets`, `peft`, and `trl`
- **Qwen Team** for the Qwen2-VL-2B base model
- **NLTK** and **rouge-score** for evaluation metrics
- **Sentence Transformers** and **FAISS** for retrieval infrastructure

---

## Contact

- **GitHub Issues:** [Open an issue](https://github.com/sourishdey2005/dakf-medical-captioning/issues)
- **Kaggle:** [sourish05](https://www.kaggle.com/sourish05)
- **LinkedIn:** [Sourish Dey](https://linkedin.com/in/sourish-dey-20b170206)
- **GitHub:** [sourishdey2005](https://github.com/sourishdey2005)
```
