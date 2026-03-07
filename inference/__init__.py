"""
Inference package for biometric few-shot verification.

Provides model loading, preprocessing, enrollment storage,
and verification logic for production use.
"""

from inference.engine import VerificationEngine
from inference.preprocessing import preprocess_image
from inference.enrollment_store import EnrollmentStore

__all__ = ["VerificationEngine", "preprocess_image", "EnrollmentStore"]
