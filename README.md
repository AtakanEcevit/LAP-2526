# Biometric Few-Shot Verification Framework
**v1.0.0**

A comprehensive Deep Metric Learning framework for recognizing individuals with extremely limited data (few-shot). This repository implements **Siamese Networks** and **Prototypical Networks** to detect forgeries and verify identities across three distinct biometric modalities: **Signatures, Faces, and Fingerprints**.

## 🌟 Capabilities

Unlike standard classification models that require thousands of images per person, this framework learns a generalized "distance metric". It embeds images into an L2-normalized 128-dimensional hypersphere, measuring similarity via Euclidean and Cosine distances.

*   **Multi-Modal:** Out-of-the-box support for CEDAR (Signatures), AT&T/LFW (Faces), and SOCOFing (Fingerprints).
*   **Few-Shot:** Dynamically works with k=1, 3, 5, or 10 enrollments/support images.
*   **Cloud & Hardware Optimized:** Runs on CPU, DirectML (AMD GPUs), and Native CUDA (NVIDIA A100/T4). Includes an automated Google Colab integration script (`colab_train.py`) for rapid cloud iteration.
*   **NIST Standard Evaluation:** Automatically computes Equal Error Rate (EER), FAR, FRR, AUC, d-prime, along with visual ROC, DET, Score Distributions, and t-SNE maps.

## 📦 Project Structure

```
LAP/
├── configs/              # YAML experiment configurations
├── data/                 # Caching data loaders, samplers, and Albumentations
├── evaluation/           # Metrics calculation and graphical visualizers
├── losses/               # Contrastive & Prototypical loss functions
├── models/               # ResNet Backbones and network architectures
├── results/              # Output directory for checkpoints and .png figures
├── colab_train.py        # All-in-one automated Cloud/Colab training script
├── evaluate.py           # Evaluation entry point (checkpoint loading & local inference)
├── run_eval.py           # Batch evaluation script
├── train.py              # Local training entry point
└── PROJECT_DOCUMENTATION.md # Comprehensive engineering doc
```

## 🚀 Quick Start (Local)

**1. Set up the environment:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Prepare Data:**
Extract your raw datasets into `data/raw/` preserving the folder structure (e.g., `data/raw/signatures/CEDAR/full_org/`).

**3. Train locally (CPU/DirectML):**
```bash
python train.py --config configs/siamese_signature.yaml
```

**4. Evaluate Checkpoints:**
```bash
python evaluate.py --config configs/siamese_signature.yaml --checkpoint results/siamese_signature_cedar/checkpoints/best.pth
```
Outputs (metrics and plots) will be saved to `results/siamese_signature_cedar/figures/`.

## ☁️ Quick Start (Google Colab - Recommended)
If training locally is slow (e.g., on AMD DirectML), use the cloud script:
1. Zip your `data/raw/` directory into `data_raw.zip`.
2. Upload `data_raw.zip` and `colab_train.py` to your Google Drive.
3. In a Colab Notebook (A100/T4 selected):
```python
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/data_raw.zip .
!cp /content/drive/MyDrive/colab_train.py .
!python colab_train.py
```
This automatically extracts data, discovers paths, trains all 6 modalities, and zips the best checkpoints for download.

## 📊 Evaluation & Metrics
The pipeline computes:
*   **EER (Equal Error Rate):** The critical biometric threshold where False Acceptances equal False Rejections.
*   **FAR/FRR:** Real-world security vs convenience curves.
*   **d-prime ($d'$):** Statistical separation capacity of the trained embeddings.
Plots are automatically placed in `results/*/figures/kshot_N/`.

## ⚙️ Configuration
Hyperparameters are strictly managed via YAML files in `/configs`.
```yaml
model:
  type: siamese # or prototypical
  backbone: resnet
  embedding_dim: 128
training:
  epochs: 50
  batch_size: 32
  lr: 0.0001
```

## 🤝 Contributing
Contributions are welcome. Please ensure that the `evaluate.py` pipeline completes without errors on your updated checkpoints before issuing a pull request.

---
*Version 1.0.0*
