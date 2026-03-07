# Release Notes: v1.1.0 - "Input Pipeline Hardening"

We are proud to release `v1.1.0` of the **Biometric Few-Shot Verification** framework. This major update upgrades the framework from experimental research scripts into a production-ready deployable system, complete with a REST API, an interactive Web UI, and mission-critical data validation guardrails.

## 🌐 Production API & Diagnostic Web UI
Going beyond model training wrappers, the codebase now ships with a fully operational production inference system:
*   **FastAPI Backend (`inference/`):** Blazing fast REST endpoints (`/enroll`, `/verify`, `/compare`) maintaining models in-memory for instant <50ms verification scoring. Integrates persistent enrollment storage to track distinct User IDs and their aggregated prototypes.
*   **Interactive Frontend (`ui/`):** A sleek, dark-themed glassmorphism interface supporting drag-and-drop image testing, dynamic score threshold visualization bars, and persistent enrollment management tables.

## 🛡️ Input Validation & Cross-Modality Protection
In initial releases, the model lacked contextual awareness of its input, leading to situations where it would happily process completely out-of-distribution modalities (such as encoding a face through the signature backbone) and outputting an uninformed score metric.

This release introduces an aggressive, sub-5ms heuristic preprocessing layer spanning the API, UI, and CLI.

*   **Hard Rejections:** 
    *   The engine automatically detects and raises explicit errors for corrupt byte-streams, purely blank/solid images (0.0 Laplacian variance), or inputs smaller than an irreducible 32x32 pixel matrix.
*   **Soft Modality Warnings:**
    *   Using rapid OpenCV array operations (Zero ML overhead), the system estimates the statistical probability of a modality match.
    *   Edge Density analysis immediately flags a face supplied when a high-density fingerprint was requested.
    *   Texture (Laplacian variance) and aspect ratio bounds softly warn the user of cross-modality mismatches without rigidly failing edge-case genuine samples.
*   **UI Integration:**
    *   All API responses now include a `validation: {passed, confidence, warnings[]}` payload.
    *   The local web application now features a dedicated amber-styled warning banner array to bubble up metric anomalies to the user before displaying the verdict.

---

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
