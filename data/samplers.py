"""
Pair and Episode samplers for Siamese and Prototypical training.
"""

import random
import numpy as np
import torch
from torch.utils.data import Sampler


class PairSampler:
    """
    Generates positive and negative pairs for Siamese Network training.
    
    Positive pair: two images from the same subject (both genuine)
    Negative pair: one genuine + one forgery from same subject,
                   OR two images from different subjects
    """

    def __init__(self, dataset_data, batch_size=32, neg_ratio=0.5):
        """
        Args:
            dataset_data: dict mapping subject_id -> {'genuine': [...], 'forgery': [...]}
            batch_size: number of pairs per batch
            neg_ratio: ratio of negative pairs (0.5 = balanced)
        """
        self.data = dataset_data
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.subjects = list(dataset_data.keys())

    def sample_batch(self):
        """
        Returns:
            pairs: list of (path1, path2, label) tuples
                   label = 1 for same person, 0 for different/forgery
        """
        pairs = []
        n_neg = int(self.batch_size * self.neg_ratio)
        n_pos = self.batch_size - n_neg

        # Positive pairs: same subject, both genuine
        for _ in range(n_pos):
            subj = random.choice(self.subjects)
            genuine = self.data[subj]['genuine']
            if len(genuine) >= 2:
                a, b = random.sample(genuine, 2)
                pairs.append((a, b, 1))
            elif len(genuine) == 1:
                # Duplicate if only 1 sample (will be augmented differently)
                pairs.append((genuine[0], genuine[0], 1))

        # Negative pairs
        for _ in range(n_neg):
            strategy = random.random()

            if strategy < 0.5:
                # Strategy 1: genuine vs forgery (same subject)
                subj = random.choice(self.subjects)
                genuine = self.data[subj]['genuine']
                forgery = self.data[subj].get('forgery', [])
                if genuine and forgery:
                    pairs.append((random.choice(genuine), random.choice(forgery), 0))
                    continue

            # Strategy 2: different subjects (both genuine)
            s1, s2 = random.sample(self.subjects, 2)
            g1 = self.data[s1]['genuine']
            g2 = self.data[s2]['genuine']
            if g1 and g2:
                pairs.append((random.choice(g1), random.choice(g2), 0))

        random.shuffle(pairs)
        return pairs

    def __iter__(self):
        while True:
            yield self.sample_batch()


class EpisodeSampler:
    """
    Generates N-way K-shot episodes for Prototypical Network training.
    
    Each episode consists of:
        - Support set: N classes × K samples each
        - Query set: N classes × Q samples each
    """

    def __init__(self, dataset_data, n_way=5, k_shot=5, q_query=5):
        """
        Args:
            dataset_data: dict mapping subject_id -> {'genuine': [...], 'forgery': [...]}
            n_way: number of classes per episode
            k_shot: number of support samples per class
            q_query: number of query samples per class
        """
        self.data = dataset_data
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

        # Filter subjects with enough samples
        min_samples = k_shot + q_query
        self.valid_subjects = [
            s for s in dataset_data.keys()
            if len(dataset_data[s]['genuine']) >= min_samples
        ]

        if len(self.valid_subjects) < n_way:
            raise ValueError(
                f"Not enough valid subjects ({len(self.valid_subjects)}) "
                f"for {n_way}-way episodes. Each subject needs >= "
                f"{min_samples} genuine samples."
            )

        print(f"[EpisodeSampler] {len(self.valid_subjects)} valid subjects "
              f"for {n_way}-way {k_shot}-shot episodes")

    def sample_episode(self):
        """
        Returns:
            support_paths: list of (path, class_idx) for support set
            query_paths:   list of (path, class_idx) for query set
        """
        # Select N random classes
        episode_classes = random.sample(self.valid_subjects, self.n_way)

        support_paths = []
        query_paths = []

        for class_idx, subj in enumerate(episode_classes):
            # Sample K + Q images from genuine
            genuine = list(self.data[subj]['genuine'])
            selected = random.sample(genuine, self.k_shot + self.q_query)

            # Split into support and query
            support = selected[:self.k_shot]
            query = selected[self.k_shot:]

            for path in support:
                support_paths.append((path, class_idx))
            for path in query:
                query_paths.append((path, class_idx))

        return support_paths, query_paths

    def sample_verification_episode(self):
        """
        Generate a binary verification episode (genuine vs forgery).
        
        Returns:
            support_paths: list of (path, label) — K genuine samples from one subject
            query_paths:   list of (path, label) — mix of genuine and forgery
                           label: 1 = genuine, 0 = forgery
        """
        subj = random.choice(self.valid_subjects)
        genuine = list(self.data[subj]['genuine'])
        forgery = list(self.data[subj].get('forgery', []))

        # Support: K genuine samples
        support = random.sample(genuine, min(self.k_shot, len(genuine)))
        support_paths = [(p, 1) for p in support]

        # Query: mix of remaining genuine + forgery
        remaining_genuine = [g for g in genuine if g not in support]
        query_paths = [(p, 1) for p in remaining_genuine[:self.q_query]]

        if forgery:
            query_forgery = random.sample(forgery, min(self.q_query, len(forgery)))
            query_paths.extend([(p, 0) for p in query_forgery])

        random.shuffle(query_paths)
        return support_paths, query_paths

    def __iter__(self):
        while True:
            yield self.sample_episode()
