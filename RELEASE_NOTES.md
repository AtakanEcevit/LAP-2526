# Release Notes: v1.0.0 - "Genesis Release"

We are thrilled to announce the official `v1.0.0` release of the **Biometric Few-Shot Verification** framework. This release marks the culmination of extensive architectural design, robust data engineering, and cross-platform hardware optimization to deliver a state-of-the-art solution for biometric forgery detection under extreme low-data constraints.

## 🌟 Highlights & Core Value
Traditional biometric systems fail when presented with only 1 to 5 enrollments per person. This framework introduces **Deep Metric Learning** (Siamese and Prototypical Networks) that learn localized distance metrics rather than specific identities. 

This creates a highly robust, "write-once, verify-anywhere" system applicable across multiple biometric modalities (Signatures, Faces, and Fingerprints) using a unified, L2-normalized ResNet backbone.

## 🚀 Key Features

*   **Multi-Modal Unified Architecture:**
    *   One unified deep metric learning pipeline supporting **CEDAR Signatures**, **AT&T/LFW Faces**, and **SOCOFing Fingerprints**.
    *   Plug-and-play architecture—easily extendable to Iris, Vein, or Palm print modalities.
*   **Dual Few-Shot Paradigms:**
    *   **Siamese Networks:** Optimized with Contrastive Loss for 1-to-1 verification scoring.
    *   **Prototypical Networks:** Optimized with dynamic episodic metric learning for N-way K-shot clustering.
*   **Enterprise-Grade Evaluation Suite:**
    *   Automated extraction of NIST-standard biometric parameters: **Accuracy, Equal Error Rate (EER), FAR, FRR, AUC, and d-prime ($d'$).**
    *   Built-in plotting for **ROC curves, DET curves, Score Distributions**, and qualitative **t-SNE** 2D embedding visualizations.
*   **Robust Data Engineering Pipelines:**
    *   Abstracted PyTorch data loaders with intelligent in-memory caching for massive I/O speedups.
    *   Automated, modality-specific preprocessing on-the-fly (Otsu binarization for signatures, CLAHE enhancement for fingerprints, Histogram Equalization for faces).
    *   Dynamic data augmentation pipelines via `albumentations`.

## ⚡ Improvements & Performance Enhancements

*   **Colab / Cloud Scaling Ready (`colab_train.py`):** The entire pipeline has been streamlined into an automated execution script tailored for Google Colab. The script automatically auto-discovers dataset structures (bypassing OS-level ZIP flattening issues) and leverages A100/T4 CUDA acceleration, dropping end-to-end multi-modal training from days to under an hour.
*   **Cross-Platform Hardware Execution:** Dynamically supports pure CPU arrays, PyTorch DirectML (for AMD architectures), and Native CUDA. 

## 🛠️ Fixes & Stability

*   **DirectML Backend Stability:** Engineered critical workarounds for standard PyTorch `BatchNorm2d` implementations that sporadically crash the Microsoft DirectML `backward()` pass due to encoding and locale mismatches. 
*   **Dataset Ingestion Safety:** Heavily guarded I/O pipelines that gracefully ignore corrupted images and mismatched filename schemas common in academic datasets.

---
**Prepared for deployment, research ablation studies, and further scale.** 
*Enjoy the release!*
