"""Diagnostic script to analyze model scoring distributions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from inference.engine import VerificationEngine
from inference.preprocessing import preprocess_image

# Load both models
print("Loading Siamese model...")
siamese = VerificationEngine()
siamese.load("signature", "siamese")

print("Loading Prototypical model...")
proto = VerificationEngine()
proto.load("signature", "prototypical")

# Test pairs
genuine_same = [
    ("data/raw/signatures/CEDAR/full_org/original_1_1.png",
     "data/raw/signatures/CEDAR/full_org/original_1_2.png"),
    ("data/raw/signatures/CEDAR/full_org/original_1_3.png",
     "data/raw/signatures/CEDAR/full_org/original_1_4.png"),
    ("data/raw/signatures/CEDAR/full_org/original_2_1.png",
     "data/raw/signatures/CEDAR/full_org/original_2_2.png"),
]
genuine_vs_forgery = [
    ("data/raw/signatures/CEDAR/full_org/original_1_1.png",
     "data/raw/signatures/CEDAR/full_forg/forgeries_1_1.png"),
    ("data/raw/signatures/CEDAR/full_org/original_2_1.png",
     "data/raw/signatures/CEDAR/full_forg/forgeries_2_1.png"),
]
different_writers = [
    ("data/raw/signatures/CEDAR/full_org/original_1_1.png",
     "data/raw/signatures/CEDAR/full_org/original_2_1.png"),
    ("data/raw/signatures/CEDAR/full_forg/forgeries_2_1.png",
     "data/raw/signatures/CEDAR/full_forg/forgeries_1_1.png"),
    ("data/raw/signatures/CEDAR/full_org/original_1_1.png",
     "data/raw/signatures/CEDAR/full_org/original_5_1.png"),
]


def analyse(engine, name, p1, p2):
    t1 = preprocess_image(p1, "signature").to(engine.device)
    t2 = preprocess_image(p2, "signature").to(engine.device)
    with torch.no_grad():
        if name == "Siamese":
            out = engine.model(t1, t2)
            sim = out["similarity"].item()
            dist = out["distance"].item()
            return sim, dist
        else:
            emb1 = engine.model.get_embedding(t1)
            emb2 = engine.model.get_embedding(t2)
            dist = torch.sqrt(((emb1 - emb2) ** 2).sum(dim=1) + 1e-8).item()
            score = 1.0 / (1.0 + dist)
            return score, dist


print("\n" + "=" * 70)
print("  SIAMESE MODEL - Score Distributions")
print("=" * 70)

print("\n--- Same writer (should be HIGH similarity, LOW distance) ---")
for p1, p2 in genuine_same:
    sim, dist = analyse(siamese, "Siamese", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  sim={sim:.6f}  dist={dist:.4f}")

print("\n--- Genuine vs forgery (should be LOW similarity, HIGH distance) ---")
for p1, p2 in genuine_vs_forgery:
    sim, dist = analyse(siamese, "Siamese", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  sim={sim:.6f}  dist={dist:.4f}")

print("\n--- Different writers (should be LOW similarity, HIGH distance) ---")
for p1, p2 in different_writers:
    sim, dist = analyse(siamese, "Siamese", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  sim={sim:.6f}  dist={dist:.4f}")


print("\n" + "=" * 70)
print("  PROTOTYPICAL MODEL - Score Distributions")
print("=" * 70)

print("\n--- Same writer ---")
for p1, p2 in genuine_same:
    score, dist = analyse(proto, "Proto", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  score={score:.6f}  dist={dist:.4f}")

print("\n--- Genuine vs forgery ---")
for p1, p2 in genuine_vs_forgery:
    score, dist = analyse(proto, "Proto", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  score={score:.6f}  dist={dist:.4f}")

print("\n--- Different writers ---")
for p1, p2 in different_writers:
    score, dist = analyse(proto, "Proto", p1, p2)
    print(f"  {os.path.basename(p1):>20s} vs {os.path.basename(p2):<20s}  score={score:.6f}  dist={dist:.4f}")

print("\n" + "=" * 70)
print("  CONCLUSION")
print("=" * 70)
