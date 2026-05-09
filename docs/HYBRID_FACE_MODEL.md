# Hybrid Face Model Artifact

FaceVerify Campus supports an optional `hybrid` face model. The model artifact is
not committed because `results/` is intentionally git-ignored.

Expected local artifact path:

```text
results/hybrid_face/checkpoints/best.pth
```

The current local artifact was copied from:

```text
C:\Users\USER\Downloads\best_hybrid_model.pth\best_hybrid_model.pth
```

Integration notes:

- The checkpoint is a PyTorch dictionary with `model_state` and `val_threshold`.
- The adapter loads the InceptionResnet-style backbone weights and ignores
  training classifier heads for verification.
- The hybrid path uses RGB `160x160` preprocessing and returns L2-normalized
  512-dimensional embeddings.
- Higher score means stronger match. The initial threshold is loaded from
  `val_threshold`, approximately `0.300`.
