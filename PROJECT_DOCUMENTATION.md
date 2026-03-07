# Biometric Few-Shot Verification System

## End-to-End Project Documentation



This document serves as the comprehensive guide to the **Biometric Few-Shot Verification** project. It outlines the architecture, data pipelines, model designs, training methodologies, and evaluation processes developed to perform person verification using extremely limited data (few-shot learning) across multiple biometric modalities (Signatures, Faces, and Fingerprints).



This system is designed to be modular, scalable, and research-ready, allowing a developer to train and evaluate models across entirely different datasets using a unified approach.



---



## 1. Project Overview & Objectives



**The Problem:** Traditional biometric classification systems require thousands of images per class (person) to train effectively. In real-world security scenarios, you often only have 1 to 5 enrollments (images) of a person. 

**The Solution:** We implemented **Deep Metric Learning** approaches—specifically Siamese Networks and Prototypical Networks—which learn a generalized "distance metric" instead of memorizing specific people. They learn how to tell if two images belong to the same person, regardless of whether the model has seen that specific person during training.



**Core Modalities Supported:**

1. **Signatures:** CEDAR dataset (Genuine vs. Forgery)

2. **Faces:** AT&T (ORL) dataset and Labeled Faces in the Wild (LFW)

3. **Fingerprints:** SOCOFing dataset (Real vs. Altered)



---



## 2. System Architecture & Components



The codebase is organized into modular components. Below is the high-level flow from raw data to final evaluation.



### A. Data Pipeline (`data/`)

The foundation of the project is a robust data ingestion and preprocessing pipeline. Academic datasets are notoriously messy, so we abstract this into unified PyTorch `Dataset` objects.



1. **`base_loader.py` (BiometricDataset)**

   - **Responsibility:** An abstract base class that defines the blueprint for all data loaders. It handles `__len__`, indexing genuine/forgery splits, applying Albumentations transforms, and crucially, an **in-memory caching system**.

   - **Why it matters:** Reading thousands of small images from a hard drive during training is a massive bottleneck. The cache reads and preprocessing into RAM once, accelerating training by orders of magnitude.



2. **Modality-Specific Loaders (`signature_loader.py`, `face_loader.py`, `fingerprint_loader.py`)**

   - **Responsibility:** Each loader inherits from `BiometricDataset` and implements `_load_data()` (parsing specific folder structures and filenames) and `_preprocess()`.

   - **Validation Phase (New in v1.1.0):**
     - Before processing, a heuristic scanner explicitly rejects un-decodable, tiny, or blank inputs.
     - Modality-mismatches (e.g. submitting a face to the signature pipeline) are flagged with low confidence and soft warnings (based on aspect ratio, edge density, texture variance) before inference occurs.
   - **Pre-processing Specifics:**

     - *Signatures:* Converted to Grayscale -> Otsu Binarization (to separate ink from background) -> Conditional Inversion (if mean pixel < 127, inverts to ensure light background / dark ink) -> Resized to 155x220.

     - *Faces:* Histogram Equalization (normalizes lighting) -> Resized to 105x105.

     - *Fingerprints:* CLAHE (Contrast Limited Adaptive Histogram Equalization, enhances ridge/valley details locally) -> Resized to 96x96.



3. **Augmentations (`augmentations.py`)**

   - **Responsibility:** Artificially inflates the training set using the `Albumentations` library.

   - **Specifics:** Signatures get elastic transformations (simulating hand jitter), shift/scale/rotate, and Gaussian noise. Faces get horizontal flips, rotations, and brightness/contrast shifts. Fingerprints get conservative rotations, elastic transforms, and Gaussian noise. All modalities get random brightness/contrast shifts and Gaussian blurring.



4. **Samplers (`samplers.py`)**

   - Neural networks in metric learning don't just take a single image; they take sets.

   - **`PairSampler`:** For Siamese networks. Generates batches consisting of 50% Genuine pairs (same person) and 50% Impostor pairs (different people, or genuine vs forgery).

   - **`EpisodeSampler`:** For Prototypical networks. Generates "N-way, K-shot" episodes (e.g., 5 random people, 5 support images each, and Q query images to test against).

5. **Dataset Factory (`dataset_factory.py`)**

   - **Responsibility:** A single `get_dataset(config)` entry point that creates the correct loader from a YAML config dict. Replaces duplicated factory logic that was in `train.py`, `evaluate.py`, `calibrate_thresholds.py`, and `colab_train.py`.

6. **Shared Preprocessing (`preprocessing.py`)**

   - **Responsibility:** Canonical image sizes (`IMAGE_SIZES`) and modality-specific preprocessing functions (`preprocess_signature`, `preprocess_face`, `preprocess_fingerprint`) shared by both data loaders and the inference pipeline.

7. **DataLoader Wrappers (`pair_dataset.py`, `episode_dataset.py`)**

   - **`SiamesePairDataset`:** Wraps pre-sampled pairs into a PyTorch `Dataset` so image loading, preprocessing, and augmentation run in parallel DataLoader workers instead of the main training thread.
   - **`PrototypicalEpisodeDataset`:** Wraps pre-sampled episode paths for parallel loading. All episodes are flattened into a single DataLoader pass, then per-episode support/query boundaries are reconstructed from the flat tensor output.



### B. Model Architectures (`models/`)

We use a unified backbone strategy. The underlying feature extractor (encoder) is shared regardless of the biometric modality.



1. **The Backbone (`backbone.py`)**

   - **`ResNetEncoder`:** Re-engineers standard PyTorch ResNet-18 models. We replace the input layer from 3-channel (RGB) to 1-channel (Grayscale). If pretrained weights are used, the original RGB weights are averaged into a single channel. The final fully connected classification layer is stripped off and replaced with an embedding pipeline (`Linear(512→256)` → `ReLU` → `Dropout(0.3)` → `Linear(256→128)` → **L2 Normalization**).

   - **`LightCNNEncoder`:** A lightweight 4-block CNN alternative (Conv→BN→ReLU→MaxPool ×4) for faster experimentation and smoke tests. Outputs L2-normalized 128-d embeddings from a `Linear(256→128)` projection.

   - **L2 Normalization:** Forces all embeddings onto a hypersphere, ensuring distance calculations rely purely on cosine/euclidean angles rather than the magnitude of activations.



2. **The Siamese Network (`siamese.py`)**

   - **How it works:** Takes two images simultaneously. Extracts embeddings for both. Calculates the absolute difference `|emb1 - emb2|` and passes it through a small classifier to output a similarity score (0.0 to 1.0).

   - **Use Case:** "1-to-1 Verification". (Does Image A match Image B?)



3. **The Prototypical Network (`prototypical.py`)**

   - **How it works:** Takes a "Support Set" (known enrollments) and a "Query Set" (new images). It encodes all support images and averages them into a single "Prototype" vector representing that person. It then calculates the Euclidean distance or Cosine similarity between the query image and the prototype.

   - **Use Case:** "Few-shot Verification". (Given 5 known enrollments, does this new query match?)



### C. Losses (`losses/losses.py`)

1. **Contrastive Loss:** Used for Siamese networks. Pushes embeddings of the same person together (distance → 0) and pulls embeddings of different people apart up to a certain `margin` (distance → margin).

2. **Triplet Loss:** An alternative to Contrastive Loss that operates on (anchor, positive, negative) triplets. Enforces that `d(anchor, positive) < d(anchor, negative) + margin`.

3. **Prototypical Loss:** Computes log-softmax over negative distances to prototypes, effectively acting as a dynamic cross-entropy loss over the generated classes in the current episode.

4. **Binary Cross-Entropy Loss:** An alternative for Siamese networks that uses the classifier head's sigmoid similarity score directly, treating verification as binary classification.



---



## 3. The Execution Flow



### A. Training & Hardware Constraints

**Training Architecture (v1.2.0):**

The `Trainer` class in `training/trainer.py` drives both Siamese and Prototypical training via a unified loop:

- **Validation-Based Model Selection:** Before training, `split_subjects()` partitions identities (not images) into train/val/test sets. Each epoch runs a training pass (with augmentation) followed by a no-gradient validation pass (without augmentation). The best checkpoint is selected by **validation loss**, and early stopping monitors val loss with configurable patience.

- **DataLoader-Parallel Iteration:** Image loading, preprocessing, and augmentation run in parallel via PyTorch `DataLoader` workers. For Siamese training, pre-sampled pairs are wrapped in `SiamesePairDataset`; for Prototypical training, all episode images are flattened into `PrototypicalEpisodeDataset` for a single efficient DataLoader pass.

- **Configurable Workers:** `num_workers` and `prefetch_factor` are set via YAML config. Windows auto-defaults to 0 workers (due to `spawn` overhead); Linux/Colab defaults to 2.

**The DirectML Challenge:**

The user hardware (AMD RX 9070 XT) required Microsoft's `torch-directml` for GPU acceleration on Windows. We encountered a deep framework bug where standard PyTorch `BatchNorm2d` layers caused the backend to crash with a `UnicodeDecodeError` during the backward pass (traced to Turkish locale `cp1254` formatting expectations). DirectML remained prohibitively slow (hours per epoch) even with workarounds.

**The Colab Solution (`colab_train.py`):**

The Colab script has been refactored (840 lines removed) to re-use the same `Trainer` class and `dataset_factory` as the local pipeline. It auto-discovers dataset paths from extracted ZIP files and runs sequential training across all 6 configurations on Colab CUDA GPUs (A100/T4).



### B. Evaluation Pipeline (`evaluate.py`)

After training, the saved checkpoints (`best.pth`) are downloaded to the local machine and passed into the evaluation pipeline. The script loops over different `k-shot` scenarios (1, 3, 5, 10 enrollments available).



**Metrics Calculated (`metrics.py`):**

1. **Accuracy:** Basic correct/incorrect thresholding.

2. **Equal Error Rate (EER):** The core metric of biometrics. The point where False Rejections strictly equal False Acceptances. Lower is better.

3. **FAR & FRR:** False Acceptance Rate (security metric) and False Rejection Rate (convenience metric).

4. **AUC (Area Under ROC Curve):** Overall distinguishing power of the model.

5. **d-prime ($d'$):** A statistical measure of the separation between the genuine score distribution and the impostor score distribution.



**Visualizations (`visualize.py`):**

1. **ROC Curves:** True Positive Rate vs False Positive Rate.

2. **DET Curves:** Detection Error Tradeoff (Log-Log scale of FAR vs FRR). The standard in NIST biometric evaluations.

3. **Score Distributions:** Histograms showing the overlap between genuine and impostor scores.

4. **t-SNE Maps:** 2D projections of the 128-dimensional embeddings to visually prove the network clusters identities together.



---



## 4. Dependencies & Configurations

The project relies on `.yaml` configuration files inside `configs/`. This avoids hardcoding variables and allows rapid experimentation. 

- *Configs control:* Backbone choice, embedding dimensions, learning rates, margins, batch sizes, dataset paths, and early-stopping patience.



**Requirements:**

- `torch>=2.1.0`, `torchvision>=0.16.0`, `torch-directml>=0.2.0` (CUDA or DirectML depending on the hardware target).

- `albumentations>=1.3.0` and `opencv-python>=4.8.0` (for image loading/augmentation).

- `scikit-learn>=1.3.0`, `matplotlib>=3.7.0`, `seaborn>=0.12.0` (for evaluation metrics and plotting).

- `tqdm>=4.65.0`, `pyyaml>=6.0` (for UX and configuration).

- `tensorboard>=2.14.0` (for optional training logging).

- `numpy>=1.24.0`, `Pillow>=10.0.0` (core array/image processing).



---



## 5. Summary

By orchestrating **Data Loaders -> PyTorch Models -> Custom Training Loops -> Metric Analysis -> Mathematical Visualization**, this repository represents a complete pipeline. It demonstrates the ability to normalize highly disparate input signals (a face vs ink on paper), map them into a shared geometric space via Deep Metric Learning, and mathematically verify identity with as little as a single reference image.



