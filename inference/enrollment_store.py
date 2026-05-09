"""
Enrollment store — persists user identity → embedding vectors.

Uses a thread-safe JSON file backend suitable for demos and
small-scale deployments. Can be swapped for a database later
without changing the public interface.
"""

import json
import os
import threading
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from inference.config import DEFAULT_ENROLLMENT_PATH, VALID_MODALITIES


class EnrollmentStore:
    """
    Manages enrolled user identities and their embedding vectors.

    Storage format (JSON):
        {
            "users": {
                "john_doe": {
                    "modality": "signature",
                    "model_type": "siamese",
            "embeddings": [[0.12, -0.34, ...], ...],
                    "enrolled_at": "2026-03-07T15:00:00",
                    "sample_count": 5
                }
            }
        }
    """

    def __init__(self, store_path: str = None):
        """
        Args:
            store_path: Path to the JSON enrollment file.
                        Defaults to data/enrollments.json.
        """
        self.store_path = store_path or DEFAULT_ENROLLMENT_PATH
        self._lock = threading.Lock()
        self._data = self._load()

    # ── Public API ───────────────────────────────────────────────────────

    def enroll(
        self,
        user_id: str,
        modality: str,
        model_type: str,
        embedding: np.ndarray,
    ) -> dict:
        """
        Add an embedding sample for a user.

        If the user already exists, the new embedding is appended.
        If the user is new, a fresh entry is created.

        Args:
            user_id:    Unique identifier (e.g. "john_doe")
            modality:   "signature", "face", or "fingerprint"
            model_type: "siamese" or "prototypical"
            embedding:  numpy array of shape (embedding_dim,)

        Returns:
            dict with user_id and total sample_count.

        Raises:
            ValueError: if the user exists with a different modality/model_type.
        """
        self._validate_user_id(user_id)
        if modality not in VALID_MODALITIES:
            raise ValueError(f"Invalid modality: {modality}")

        emb_list = embedding.tolist()

        with self._lock:
            users = self._data.setdefault("users", {})

            if user_id in users:
                existing = users[user_id]
                # Guard: modality and model_type must match
                if existing["modality"] != modality:
                    raise ValueError(
                        f"User '{user_id}' is enrolled under modality "
                        f"'{existing['modality']}', cannot add '{modality}' data."
                    )
                if existing["model_type"] != model_type:
                    raise ValueError(
                        f"User '{user_id}' is enrolled with model_type "
                        f"'{existing['model_type']}', cannot mix with '{model_type}'."
                    )
                existing["embeddings"].append(emb_list)
                existing["sample_count"] = len(existing["embeddings"])
            else:
                users[user_id] = {
                    "modality": modality,
                    "model_type": model_type,
                    "embeddings": [emb_list],
                    "enrolled_at": datetime.now().isoformat(),
                    "sample_count": 1,
                }

            self._save()

        return {
            "user_id": user_id,
            "sample_count": users[user_id]["sample_count"],
        }

    def get_user(self, user_id: str) -> Optional[dict]:
        """
        Get a user's enrollment record.

        Returns:
            dict with modality, model_type, embeddings, etc. or None.
        """
        with self._lock:
            return self._data.get("users", {}).get(user_id)

    def get_prototype(self, user_id: str) -> np.ndarray:
        """
        Compute the prototype (mean embedding) for an enrolled user.

        Returns:
            numpy array of shape (embedding_dim,).

        Raises:
            KeyError: if user is not enrolled.
        """
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(f"User '{user_id}' is not enrolled.")

        embeddings = np.array(user["embeddings"])
        return embeddings.mean(axis=0)

    def get_embeddings(self, user_id: str) -> np.ndarray:
        """
        Get all raw enrolled embeddings for a user.

        Returns:
            numpy array of shape (N, embedding_dim).

        Raises:
            KeyError: if user is not enrolled.
        """
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(f"User '{user_id}' is not enrolled.")

        return np.array(user["embeddings"])

    def delete_user(self, user_id: str) -> bool:
        """
        Remove a user's enrollment.

        Returns:
            True if the user existed and was removed, False otherwise.
        """
        with self._lock:
            users = self._data.get("users", {})
            if user_id in users:
                del users[user_id]
                self._save()
                return True
            return False

    def list_users(self) -> List[dict]:
        """
        List all enrolled users with summary info.

        Returns:
            list of dicts with user_id, modality, model_type, sample_count.
        """
        with self._lock:
            users = self._data.get("users", {})
            return [
                {
                    "user_id": uid,
                    "modality": info["modality"],
                    "model_type": info["model_type"],
                    "sample_count": info["sample_count"],
                    "enrolled_at": info.get("enrolled_at", "unknown"),
                }
                for uid, info in users.items()
            ]

    def clear(self):
        """Remove all enrollments."""
        with self._lock:
            self._data = {"users": {}}
            self._save()

    # ── Private ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        """Load the enrollment file, or return empty structure."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # Corrupted file — start fresh
                return {"users": {}}
        return {"users": {}}

    def _save(self):
        """Atomically write the enrollment data to disk."""
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)

        # Write to temp file, then rename for crash safety
        dir_name = os.path.dirname(self.store_path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            # Atomic rename (on Windows this may not be truly atomic,
            # but it's the best we can do without external deps)
            if os.path.exists(self.store_path):
                os.remove(self.store_path)
            os.rename(tmp_path, self.store_path)
        except Exception:
            # Clean up temp file on failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @staticmethod
    def _validate_user_id(user_id: str):
        """Validate user_id format."""
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string.")
        if len(user_id) > 64:
            raise ValueError("user_id must be 64 characters or fewer.")
        # Allow alphanumeric, underscores, hyphens, dots
        import re
        if not re.match(r"^[a-zA-Z0-9._-]+$", user_id):
            raise ValueError(
                "user_id may only contain letters, numbers, "
                "underscores, hyphens, and dots."
            )
