# Release Notes: v1.4.0 — "Selective Training & CLI Overrides"

This release adds command-line filtering and hyper-parameter overrides to `colab_train.py`, allowing selective model training without editing YAML configs.

*   **Selective Training:** New `--modality` and `--model` flags let you train a single model (e.g., `--modality signature --model siamese`) instead of all 6 sequentially.
*   **CLI Hyper-Parameter Overrides:** New flags `--epochs`, `--lr`, `--batch_size`, `--patience`, and `--loss` override YAML config values directly from the command line — ideal for quick experiments on Colab.
*   **Backward Compatible:** Running `colab_train.py` with no arguments still trains all models with their default configs.

---

# Release Notes: v1.3.0 — "Trainer Enhancements & Inference Thresholds"

This release includes enhancements to the training loop, expanded evaluation configurations, and calibrated distance thresholds for inference.

*   **Trainer Enhancements:** Minor fixes and improvements to the core validation and checkpointing pipelines.
*   **Calibrated Inference Thresholds:** System thresholds have been pre-calibrated for all modalities to improve out-of-the-box system accuracy.
*   **Evaluation Fixes:** Cleaned up reporting in the testing suite and expanded metric tracking.
*   **Version Alignments:** Ensured consistency across config schemas and documentation.

---

# Release Notes: v1.2.0 — "Training Pipeline Overhaul"

This release fundamentally re-architects the training loop, introduces proper validation-based model selection, and consolidates duplicated data logic into single-source-of-truth modules.

## 🔄 Validation-Based Training
The training loop now performs a proper train/validation split before each run and selects the best checkpoint by **validation loss** instead of training loss.

*   **Subject-Level Splits:** `split_subjects()` partitions identities — not images — into train/val/test buckets, preventing data leakage and ensuring fair evaluation.
*   **Early Stopping on Val Loss:** Patience-based early stopping now monitors validation loss rather than training loss, producing more generalizable models.
*   **Per-Epoch Validation Pass:** After each training epoch, a no-gradient validation sweep runs over the val split (with augmentation disabled) to track overfitting in real time.

## ⚡ DataLoader-Parallel Batch Iteration
Training has been moved from sequential per-pair / per-episode image loading to PyTorch `DataLoader`-based parallel iteration.

*   **`SiamesePairDataset`** (`data/pair_dataset.py`): Wraps pre-sampled pairs so image loading, preprocessing, and augmentation happen in parallel DataLoader workers instead of the main thread.
*   **`PrototypicalEpisodeDataset`** (`data/episode_dataset.py`): Flattens all episode images into a single DataLoader pass, then reconstructs per-episode support/query boundaries from the flat tensor output.
*   **Configurable Workers:** `num_workers` and `prefetch_factor` are configurable via YAML; Windows auto-defaults to 0 workers (spawn overhead), Linux/Colab defaults to 2.

## 🏭 Centralized Data Modules
Duplicated dataset creation, preprocessing, and configuration logic has been consolidated.

*   **`dataset_factory.py`**: Single `get_dataset(config)` function replaces four identical copies that were in `train.py`, `evaluate.py`, `calibrate_thresholds.py`, and `colab_train.py`.
*   **`data/preprocessing.py`**: Canonical image sizes and modality-specific preprocessing functions (`preprocess_signature`, `preprocess_face`, `preprocess_fingerprint`) shared by both data loaders and the inference pipeline.

## ☁️ Colab Script Refactoring
*   **840 lines removed** from `colab_train.py` — the script now re-uses the same `Trainer` class and `dataset_factory` as the local pipeline instead of duplicating everything inline.

## 🧪 New Test Coverage
*   `test_data_loading.py` — dataset construction, DataLoader integration, and sampler correctness
*   `test_dataloader_training.py` — end-to-end training loop smoke tests for both Siamese and Prototypical paths

## ⚙️ Config Improvements
*   Prototypical YAML configs now expose `n_way`, `k_shot`, and `q_query` parameters at the training level.
*   All configs include a consolidated `results_dir` key.

---

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
