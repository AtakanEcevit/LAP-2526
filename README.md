<div align="center">

# 🔐 Biometric Few-Shot Verification Framework

**Siamese Networks × Prototypical Networks × Multi-Modal Biometrics**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Version](https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge)]()

*A production-grade deep metric learning framework that verifies identities using as few as **1–5 samples** across signatures, faces, and fingerprints.*

---

[Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Cloud Training](#-cloud-training-google-colab) · [Evaluation](#-evaluation--metrics) · [Contributing](#-contributing)

</div>

---

## 🎯 The Problem

Traditional biometric systems need **thousands** of images per person to learn identity patterns. In real-world security scenarios, you often have only **1 to 5** enrollment samples.

> **Our Solution:** Deep Metric Learning models that learn *how to compare* rather than *who to memorize* — enabling accurate verification from minimal data across completely different biometric types.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 Dual Architecture
- **Siamese Networks** — Pairwise comparison with Contrastive Loss
- **Prototypical Networks** — Prototype-based few-shot classification
- Shared ResNet-18 backbone with L2-normalized embeddings

</td>
<td width="50%">

### 🔬 Multi-Modal Support
- ✍️ **Signatures** — CEDAR dataset (genuine vs. forgery)
- 👤 **Faces** — AT&T/ORL + LFW datasets
- 🖐️ **Fingerprints** — SOCOFing dataset (real vs. altered)

</td>
</tr>
<tr>
<td width="50%">

### 📊 NIST-Standard Evaluation
- Equal Error Rate (EER), FAR, FRR, AUC, d-prime
- Automated ROC, DET, Score Distribution plots
- t-SNE embedding visualizations
- Multi k-shot evaluation (k = 1, 3, 5, 10)

</td>
<td width="50%">

### ⚡ Hardware Flexibility
- 🟢 **NVIDIA CUDA** — Native GPU acceleration
- 🔵 **AMD DirectML** — Windows GPU support
- ⚪ **CPU** — Automatic fallback
- ☁️ **Google Colab** — One-click cloud training

</td>
</tr>
</table>

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Biometric Image                   │
│              (Signature / Face / Fingerprint)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PREPROCESSING PIPELINE                          │
│  Signatures → Otsu Binarization + Inversion                    │
│  Faces      → Histogram Equalization                           │
│  Fingerprints → CLAHE Enhancement                              │
│  All        → Albumentations Augmentation                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RESNET-18 BACKBONE                            │
│    1-ch Grayscale → 512-d → 256 → ReLU → Dropout → 128-d      │
│                      (L2 Normalized)                            │
└──────────┬───────────────────────────────────────┬──────────────┘
           │                                       │
           ▼                                       ▼
┌─────────────────────────┐         ┌─────────────────────────────┐
│    SIAMESE NETWORK      │         │   PROTOTYPICAL NETWORK      │
│                         │         │                             │
│  Img_A ──┐              │         │  Support Set → Prototypes   │
│          ├─→ |emb diff| │         │  Query ──→ Distance to each │
│  Img_B ──┘   → Score    │         │           → Classification  │
│                         │         │                             │
│  Loss: Contrastive      │         │  Loss: Prototypical (NLL)   │
└─────────────────────────┘         └─────────────────────────────┘
           │                                       │
           └───────────────┬───────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION ENGINE                            │
│    EER · FAR · FRR · AUC · d-prime · ROC · DET · t-SNE         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
LAP/
│
├── 📂 configs/                    # Experiment configurations (YAML)
│   ├── siamese_signature.yaml
│   ├── siamese_face.yaml
│   ├── siamese_fingerprint.yaml
│   ├── proto_signature.yaml
│   ├── proto_face.yaml
│   └── proto_fingerprint.yaml
│
├── 📂 data/                       # Data pipeline
│   ├── base_loader.py             # Abstract dataset with in-memory caching
│   ├── signature_loader.py        # CEDAR + BHSig260 loaders
│   ├── face_loader.py             # AT&T/ORL + LFW loaders
│   ├── fingerprint_loader.py      # SOCOFing loader
│   ├── samplers.py                # PairSampler + EpisodeSampler
│   └── augmentations.py           # Per-modality augmentation pipelines
│
├── 📂 models/                     # Network architectures
│   ├── backbone.py                # ResNet-18 + LightCNN encoders
│   ├── siamese.py                 # Siamese Network
│   └── prototypical.py            # Prototypical Network
│
├── 📂 losses/                     # Loss functions
│   └── losses.py                  # Contrastive, Prototypical, Triplet, BCE
│
├── 📂 training/                   # Training engine
│   └── trainer.py                 # Unified training loop with early stopping
│
├── 📂 evaluation/                 # Evaluation & visualization
│   ├── metrics.py                 # EER, FAR, FRR, AUC, d-prime
│   ├── visualize.py               # ROC, DET, t-SNE, score distributions
│   └── benchmark.py               # Cross-config benchmark runner
│
├── 📂 results/                    # 🚫 Git-ignored (checkpoints + figures)
├── 📂 data/raw/                   # 🚫 Git-ignored (dataset images)
│
├── train.py                       # 🏋️ Local training entry point
├── evaluate.py                    # 📊 Evaluation entry point
├── colab_train.py                 # ☁️ All-in-one Colab training script
├── download_datasets.py           # 📥 Automated dataset downloader
├── utils.py                       # 🔧 Device detection (DirectML/CUDA/CPU)
├── requirements.txt               # 📦 Python dependencies
└── PROJECT_DOCUMENTATION.md       # 📖 Full engineering documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1️⃣ Clone & Install

```bash
git clone https://github.com/AtakanEcevit/LAP-2526.git
cd LAP-2526

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 2️⃣ Download Datasets

```bash
python download_datasets.py
```

This automatically downloads and extracts:

| Modality | Dataset | Subjects | Images | Type |
|:--------:|:-------:|:--------:|:------:|:----:|
| ✍️ Signature | CEDAR | 55 | 2,640 | Genuine + Forgery |
| 👤 Face | AT&T/ORL | 40 | 400 | Multi-pose |
| 👤 Face | LFW | 5,749 | 13,000+ | In-the-wild |
| 🖐️ Fingerprint | SOCOFing | 600 | 6,000 | Real + Altered |

### 3️⃣ Train a Model

```bash
python train.py --config configs/siamese_signature.yaml
```

### 4️⃣ Evaluate

```bash
python evaluate.py \
  --config configs/siamese_signature.yaml \
  --checkpoint results/siamese_signature_cedar/checkpoints/best.pth
```

> 📁 Outputs (ROC plots, metrics, t-SNE maps) are saved to `results/*/figures/`

---

## ☁️ Cloud Training (Google Colab)

For significantly faster training using free NVIDIA A100/T4 GPUs:

<details>
<summary><b>📋 Step-by-step Colab Instructions</b></summary>

1. **Prepare data locally:**
   ```bash
   # Creates data_raw.zip preserving folder structure
   python create_zip.py
   ```

2. **Upload to Google Drive:**
   - Upload `data_raw.zip` and `colab_train.py` to your Drive root.

3. **In a Colab notebook** (Runtime → GPU):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !cp /content/drive/MyDrive/data_raw.zip .
   !cp /content/drive/MyDrive/colab_train.py .
   !python colab_train.py
   ```

4. **Download results:**
   ```python
   !cp /content/results_best.zip /content/drive/MyDrive/
   ```

> 💡 The script auto-discovers dataset paths, trains all 6 models sequentially, and packages the best checkpoints.

</details>

---

## 📊 Evaluation & Metrics

The framework produces comprehensive biometric verification metrics:

| Metric | Description | Ideal Value |
|:------:|:-----------:|:-----------:|
| **EER** | Equal Error Rate — where FAR = FRR | → 0% |
| **FAR** | False Acceptance Rate (security) | → 0% |
| **FRR** | False Rejection Rate (convenience) | → 0% |
| **AUC** | Area Under ROC Curve | → 1.0 |
| **d-prime** | Distribution separation measure | → ∞ |

### Generated Visualizations

Each evaluation run automatically produces:

- 📈 **ROC Curves** — Per k-shot (1, 3, 5, 10)
- 📉 **DET Curves** — NIST-standard detection error tradeoff
- 📊 **Score Distributions** — Genuine vs. impostor overlap
- 🗺️ **t-SNE Maps** — 2D embedding cluster visualization

---

## ⚙️ Configuration

All experiments are driven by YAML configs in `configs/`:

```yaml
model:
  type: siamese           # 'siamese' or 'prototypical'
  backbone: resnet         # 'resnet' or 'light'
  embedding_dim: 128
  pretrained: true

dataset:
  modality: signature      # 'signature', 'face', or 'fingerprint'
  name: cedar
  root_dir: data/raw/signatures/CEDAR

training:
  epochs: 50
  batch_size: 32
  lr: 0.0001
  patience: 15             # Early stopping

evaluation:
  k_shots: [1, 3, 5, 10]
```

### Available Configs

| Config | Model | Modality | Dataset |
|:------:|:-----:|:--------:|:-------:|
| `siamese_signature.yaml` | Siamese | Signature | CEDAR |
| `proto_signature.yaml` | Prototypical | Signature | CEDAR |
| `siamese_face.yaml` | Siamese | Face | AT&T |
| `proto_face.yaml` | Prototypical | Face | AT&T |
| `siamese_fingerprint.yaml` | Siamese | Fingerprint | SOCOFing |
| `proto_fingerprint.yaml` | Prototypical | Fingerprint | SOCOFing |

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

> ⚠️ Please ensure `evaluate.py` completes without errors on your changes before submitting a PR.

---

## 📄 Documentation

| Document | Description |
|:--------:|:-----------:|
| [`PROJECT_DOCUMENTATION.md`](PROJECT_DOCUMENTATION.md) | Full engineering reference |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Non-technical stakeholder overview |
| [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | Version history & changelog |

---

<div align="center">

**Built with** ❤️ **using PyTorch**

`v1.0.0`

</div>
