"""
Campus exam verification demo store.

This module keeps university-facing demo data separate from the lower-level
biometric enrollment store. It is intentionally JSON-backed so the showcase can
run locally without a database, while preserving domain concepts that can later
move to real persistence.
"""

import csv
import io
import json
import os
import tempfile
import threading
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

from inference.config import MODEL_REGISTRY, VALID_MODEL_TYPES


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAMPUS_PATH = os.path.join(PROJECT_ROOT, "data", "campus_demo.json")

STATUS_NOT_STARTED = "Not Started"
STATUS_ENROLLMENT_NEEDED = "Enrollment Needed"
STATUS_PENDING = "Pending Verification"
STATUS_VERIFIED = "Verified"
STATUS_MANUAL_REVIEW = "Manual Review"
STATUS_REJECTED = "Rejected"
STATUS_APPROVED = "Approved by Proctor"
STATUS_FALLBACK = "Fallback Requested"

DECISION_VERIFIED = "verified"
DECISION_MANUAL_REVIEW = "manual_review"
DECISION_REJECTED = "rejected"

DEFAULT_FACE_MODEL = "hybrid"
DEFAULT_FACE_THRESHOLD = MODEL_REGISTRY.get(
    ("face", DEFAULT_FACE_MODEL), {}
).get("threshold", 0.65)
MANUAL_REVIEW_BAND = 0.08


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def default_demo_data() -> dict:
    students = [
        ("NB-2026-1042", "Aylin Kaya", "aylin.kaya@northbridge.edu"),
        ("NB-2026-1043", "Marcus Chen", "marcus.chen@northbridge.edu"),
        ("NB-2026-1044", "Sofia Martinez", "sofia.martinez@northbridge.edu"),
        ("NB-2026-1045", "Noah Williams", "noah.williams@northbridge.edu"),
        ("NB-2026-1046", "Elif Demir", "elif.demir@northbridge.edu"),
        ("NB-2026-1047", "Priya Nair", "priya.nair@northbridge.edu"),
        ("NB-2026-1048", "Jonas Becker", "jonas.becker@northbridge.edu"),
        ("NB-2026-1049", "Mina Park", "mina.park@northbridge.edu"),
        ("NB-2026-1050", "Omar Haddad", "omar.haddad@northbridge.edu"),
        ("NB-2026-1051", "Lena Hoffmann", "lena.hoffmann@northbridge.edu"),
        ("NB-2026-1052", "Daniel Evans", "daniel.evans@northbridge.edu"),
        ("NB-2026-1053", "Zeynep Arslan", "zeynep.arslan@northbridge.edu"),
        ("NB-2026-1054", "Iris Novak", "iris.novak@northbridge.edu"),
        ("NB-2026-1055", "Samir Patel", "samir.patel@northbridge.edu"),
        ("NB-2026-1056", "Hannah Reed", "hannah.reed@northbridge.edu"),
        ("NB-2026-1057", "Kenji Sato", "kenji.sato@northbridge.edu"),
        ("NB-2026-1058", "Layla Hassan", "layla.hassan@northbridge.edu"),
        ("NB-2026-1059", "Theo Laurent", "theo.laurent@northbridge.edu"),
        ("NB-2026-1060", "Maya Thompson", "maya.thompson@northbridge.edu"),
        ("NB-2026-1061", "Emre Yilmaz", "emre.yilmaz@northbridge.edu"),
        ("NB-2026-1062", "Nora Silva", "nora.silva@northbridge.edu"),
        ("NB-2026-1063", "Adam Kowalski", "adam.kowalski@northbridge.edu"),
        ("NB-2026-1064", "Sara Ahmed", "sara.ahmed@northbridge.edu"),
        ("NB-2026-1065", "Leo Fischer", "leo.fischer@northbridge.edu"),
        ("NB-2026-1066", "Isabella Rossi", "isabella.rossi@northbridge.edu"),
    ]
    student_map = {
        sid: {
            "student_id": sid,
            "name": name,
            "email": email,
            "course_ids": ["CS204-2026S"],
            "enrollment_status": "not_enrolled",
            "sample_count": 0,
            "reference_preview": None,
            "enrolled_at": None,
        }
        for sid, name, email in students
    }
    return {
        "courses": {
            "CS204-2026S": {
                "course_id": "CS204-2026S",
                "name": "CS 204 - Data Structures",
                "instructor": "Dr. Elena Morris",
                "term": "Spring 2026",
            }
        },
        "exams": {
            "CS204-MIDTERM-1": {
                "exam_id": "CS204-MIDTERM-1",
                "course_id": "CS204-2026S",
                "name": "Midterm 1",
                "start_time": "2026-05-12T10:00:00",
                "end_time": "2026-05-12T11:30:00",
                "threshold": DEFAULT_FACE_THRESHOLD,
                "model_type": DEFAULT_FACE_MODEL,
                "verification_required": True,
            }
        },
        "students": student_map,
        "attempts": [],
        "review_actions": [],
        "audit_log": [],
        "updated_at": now_iso(),
    }


class CampusStore:
    """Thread-safe JSON store for FaceVerify Campus demo data."""

    def __init__(self, store_path: str = None):
        self.store_path = store_path or DEFAULT_CAMPUS_PATH
        self._lock = threading.Lock()
        self._data = self._load()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_unlocked()

    def reset_demo(self) -> dict:
        with self._lock:
            self._data = default_demo_data()
            self._audit_unlocked(
                "demo_reset",
                "system",
                "Demo data reset to Northbridge University defaults.",
            )
            self._save_unlocked()
            return self._snapshot_unlocked()

    def student_ids(self) -> List[str]:
        with self._lock:
            return list(self._data.get("students", {}).keys())

    def get_student(self, student_id: str) -> Optional[dict]:
        with self._lock:
            student = self._data.get("students", {}).get(student_id)
            return deepcopy(student) if student else None

    def get_exam(self, exam_id: str) -> Optional[dict]:
        with self._lock:
            exam = self._data.get("exams", {}).get(exam_id)
            return deepcopy(exam) if exam else None

    def list_courses(self) -> List[dict]:
        with self._lock:
            return list(deepcopy(self._data.get("courses", {})).values())

    def list_exams(self) -> List[dict]:
        with self._lock:
            return list(deepcopy(self._data.get("exams", {})).values())

    def list_students(self, course_id: str = None) -> List[dict]:
        with self._lock:
            students = list(deepcopy(self._data.get("students", {})).values())
        if course_id:
            students = [s for s in students if course_id in s.get("course_ids", [])]
        return sorted(students, key=lambda item: item["student_id"])

    def create_course(
        self,
        course_id: str,
        name: str,
        instructor: str,
        term: str,
    ) -> dict:
        self._validate_id(course_id, "course_id")
        if not name.strip():
            raise ValueError("Course name is required.")
        with self._lock:
            self._data.setdefault("courses", {})[course_id] = {
                "course_id": course_id,
                "name": name.strip(),
                "instructor": instructor.strip() or "Unassigned",
                "term": term.strip() or "Demo Term",
            }
            self._audit_unlocked("course_saved", "admin", f"Course {course_id} saved.")
            self._save_unlocked()
            return deepcopy(self._data["courses"][course_id])

    def create_exam(
        self,
        exam_id: str,
        course_id: str,
        name: str,
        start_time: str,
        end_time: str,
        threshold: float,
        model_type: str,
    ) -> dict:
        self._validate_id(exam_id, "exam_id")
        if course_id not in self._data.get("courses", {}):
            raise ValueError(f"Unknown course_id '{course_id}'.")
        if model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of: {sorted(VALID_MODEL_TYPES)}."
            )
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0.")
        if not name.strip():
            raise ValueError("Exam name is required.")
        with self._lock:
            self._data.setdefault("exams", {})[exam_id] = {
                "exam_id": exam_id,
                "course_id": course_id,
                "name": name.strip(),
                "start_time": start_time.strip(),
                "end_time": end_time.strip(),
                "threshold": round(float(threshold), 6),
                "model_type": model_type,
                "verification_required": True,
            }
            self._audit_unlocked("exam_saved", "admin", f"Exam {exam_id} saved.")
            self._save_unlocked()
            return deepcopy(self._data["exams"][exam_id])

    def import_roster(self, course_id: str, csv_text: str) -> dict:
        if course_id not in self._data.get("courses", {}):
            raise ValueError(f"Unknown course_id '{course_id}'.")
        reader = csv.DictReader(io.StringIO(csv_text))
        required = {"student_id", "name", "email"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV must include student_id,name,email columns.")

        imported = []
        rejected = []
        with self._lock:
            students = self._data.setdefault("students", {})
            for line_number, row in enumerate(reader, start=2):
                student_id = (row.get("student_id") or "").strip()
                name = (row.get("name") or "").strip()
                email = (row.get("email") or "").strip()
                if not student_id or not name or not email:
                    rejected.append({
                        "line": line_number,
                        "reason": "Missing student_id, name, or email.",
                    })
                    continue
                try:
                    self._validate_id(student_id, "student_id")
                except ValueError as exc:
                    rejected.append({"line": line_number, "reason": str(exc)})
                    continue

                existing = students.get(student_id, {})
                course_ids = set(existing.get("course_ids", []))
                course_ids.add(course_id)
                students[student_id] = {
                    "student_id": student_id,
                    "name": name,
                    "email": email,
                    "course_ids": sorted(course_ids),
                    "enrollment_status": existing.get(
                        "enrollment_status", "not_enrolled"
                    ),
                    "sample_count": existing.get("sample_count", 0),
                    "reference_preview": existing.get("reference_preview"),
                    "enrolled_at": existing.get("enrolled_at"),
                    "face_source": existing.get("face_source"),
                    "face_identity": existing.get("face_identity"),
                    "face_enrollment_images": existing.get("face_enrollment_images", []),
                    "face_query_image": existing.get("face_query_image"),
                    "face_dataset_dir": existing.get("face_dataset_dir"),
                    "face_model_type": existing.get("face_model_type"),
                }
                imported.append(student_id)
            self._audit_unlocked(
                "roster_imported",
                "admin",
                f"Imported {len(imported)} student(s) into {course_id}.",
            )
            self._save_unlocked()
        return {"imported": imported, "rejected": rejected}

    def record_enrollment(
        self,
        student_id: str,
        sample_count: int,
        reference_preview: str = None,
    ) -> dict:
        with self._lock:
            students = self._data.get("students", {})
            if student_id not in students:
                raise KeyError(f"Unknown student_id '{student_id}'.")
            students[student_id]["enrollment_status"] = "enrolled"
            students[student_id]["sample_count"] = sample_count
            students[student_id]["enrolled_at"] = now_iso()
            if reference_preview:
                students[student_id]["reference_preview"] = reference_preview
            self._audit_unlocked(
                "student_enrolled",
                student_id,
                f"{student_id} enrolled with {sample_count} face sample(s).",
            )
            self._save_unlocked()
            return deepcopy(students[student_id])

    def record_flux_enrollment(
        self,
        student_id: str,
        sample_count: int,
        model_type: str,
        face_identity: str,
        enrollment_images: List[str],
        query_image: str,
        dataset_dir: str = None,
        reference_preview: str = None,
    ) -> dict:
        """Attach a FLUXSynID identity to a campus student after embedding enrollment."""
        with self._lock:
            students = self._data.get("students", {})
            if student_id not in students:
                raise KeyError(f"Unknown student_id '{student_id}'.")
            students[student_id].update({
                "enrollment_status": "enrolled",
                "sample_count": sample_count,
                "enrolled_at": now_iso(),
                "face_source": "flux_synid",
                "face_identity": face_identity,
                "face_enrollment_images": enrollment_images,
                "face_query_image": query_image,
                "face_dataset_dir": dataset_dir,
                "face_model_type": model_type,
            })
            if reference_preview:
                students[student_id]["reference_preview"] = reference_preview
            self._audit_unlocked(
                "student_preuploaded",
                student_id,
                f"{student_id} preuploaded from FLUXSynID identity {face_identity}.",
            )
            self._save_unlocked()
            return deepcopy(students[student_id])

    def flux_summary(self) -> dict:
        with self._lock:
            students = list(self._data.get("students", {}).values())
        preuploaded = [
            student for student in students
            if student.get("face_source") == "flux_synid"
        ]
        return {
            "preuploaded_students": len(preuploaded),
            "student_ids": sorted(student["student_id"] for student in preuploaded),
        }

    def record_attempt(
        self,
        exam_id: str,
        student_id: str,
        score: float,
        threshold: float,
        model_type: str,
        validation: dict = None,
        query_preview: str = None,
        attempt_source: str = "upload",
        scenario: str = None,
    ) -> dict:
        with self._lock:
            exam = self._data.get("exams", {}).get(exam_id)
            student = self._data.get("students", {}).get(student_id)
            if not exam:
                raise KeyError(f"Unknown exam_id '{exam_id}'.")
            if not student:
                raise KeyError(f"Unknown student_id '{student_id}'.")

            decision = self._decision_from_score(score, threshold)
            status = self._status_from_decision(decision)
            attempt = {
                "attempt_id": make_id("att"),
                "exam_id": exam_id,
                "course_id": exam["course_id"],
                "student_id": student_id,
                "student_name": student["name"],
                "score": round(float(score), 6),
                "threshold": round(float(threshold), 6),
                "decision": decision,
                "status": status,
                "final_status": status,
                "model_type": model_type,
                "timestamp": now_iso(),
                "warnings": self._warnings_from_validation(validation),
                "validation": validation or {},
                "query_preview": query_preview,
                "attempt_source": attempt_source,
                "scenario": scenario,
                "review": None,
            }
            self._data.setdefault("attempts", []).append(attempt)
            self._audit_unlocked(
                "verification_attempted",
                student_id,
                f"{student_id} produced {status} for {exam_id}.",
                attempt_id=attempt["attempt_id"],
            )
            self._save_unlocked()
            return deepcopy(attempt)

    def review_attempt(
        self,
        attempt_id: str,
        reviewer: str,
        action: str,
        reason: str,
    ) -> dict:
        action_map = {
            "approve": STATUS_APPROVED,
            "deny": STATUS_REJECTED,
            "fallback": STATUS_FALLBACK,
        }
        if action not in action_map:
            raise ValueError("action must be approve, deny, or fallback.")
        with self._lock:
            attempt = self._find_attempt_unlocked(attempt_id)
            if not attempt:
                raise KeyError(f"Unknown attempt_id '{attempt_id}'.")
            review = {
                "review_id": make_id("rev"),
                "attempt_id": attempt_id,
                "reviewer": reviewer.strip() or "Proctor",
                "action": action,
                "reason": reason.strip() or "No reason provided.",
                "timestamp": now_iso(),
                "final_status": action_map[action],
            }
            attempt["review"] = review
            attempt["final_status"] = review["final_status"]
            self._data.setdefault("review_actions", []).append(review)
            self._audit_unlocked(
                "manual_review_completed",
                review["reviewer"],
                f"{action} applied to attempt {attempt_id}.",
                attempt_id=attempt_id,
            )
            self._save_unlocked()
            return deepcopy(attempt)

    def list_attempts(self, exam_id: str = None, student_id: str = None) -> List[dict]:
        with self._lock:
            attempts = deepcopy(self._data.get("attempts", []))
        if exam_id:
            attempts = [a for a in attempts if a.get("exam_id") == exam_id]
        if student_id:
            attempts = [a for a in attempts if a.get("student_id") == student_id]
        return sorted(attempts, key=lambda item: item["timestamp"], reverse=True)

    def get_attempt(self, attempt_id: str) -> Optional[dict]:
        with self._lock:
            attempt = self._find_attempt_unlocked(attempt_id)
            return deepcopy(attempt) if attempt else None

    def exam_roster(self, exam_id: str) -> dict:
        with self._lock:
            exam = deepcopy(self._data.get("exams", {}).get(exam_id))
            if not exam:
                raise KeyError(f"Unknown exam_id '{exam_id}'.")
            course = deepcopy(self._data.get("courses", {}).get(exam["course_id"]))
            students = [
                deepcopy(s)
                for s in self._data.get("students", {}).values()
                if exam["course_id"] in s.get("course_ids", [])
            ]
            attempts = deepcopy(self._data.get("attempts", []))

        latest_by_student = {}
        for attempt in sorted(attempts, key=lambda item: item["timestamp"]):
            if attempt.get("exam_id") == exam_id:
                latest_by_student[attempt["student_id"]] = attempt

        roster = []
        for student in sorted(students, key=lambda item: item["student_id"]):
            latest = latest_by_student.get(student["student_id"])
            if student.get("enrollment_status") != "enrolled":
                status = STATUS_ENROLLMENT_NEEDED
            elif not latest:
                status = STATUS_NOT_STARTED
            else:
                status = latest.get("final_status") or latest.get("status")
            roster.append({
                **student,
                "exam_status": status,
                "latest_attempt": latest,
            })
        return {"exam": exam, "course": course, "roster": roster}

    def audit_log(self) -> List[dict]:
        with self._lock:
            return deepcopy(self._data.get("audit_log", []))

    def audit_csv(self) -> str:
        rows = self.audit_log()
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "timestamp",
                "event_type",
                "actor",
                "message",
                "attempt_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "timestamp": row.get("timestamp", ""),
                "event_type": row.get("event_type", ""),
                "actor": row.get("actor", ""),
                "message": row.get("message", ""),
                "attempt_id": row.get("attempt_id", ""),
            })
        return output.getvalue()

    def _load(self) -> dict:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if all(k in data for k in ("courses", "exams", "students")):
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return default_demo_data()

    def _save_unlocked(self):
        self._data["updated_at"] = now_iso()
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.store_path), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            if os.path.exists(self.store_path):
                os.remove(self.store_path)
            os.rename(tmp_path, self.store_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def _snapshot_unlocked(self) -> dict:
        data = deepcopy(self._data)
        data["courses"] = list(data.get("courses", {}).values())
        data["exams"] = list(data.get("exams", {}).values())
        data["students"] = sorted(
            list(data.get("students", {}).values()),
            key=lambda item: item["student_id"],
        )
        data["attempts"] = sorted(
            data.get("attempts", []),
            key=lambda item: item["timestamp"],
            reverse=True,
        )
        return data

    def _audit_unlocked(
        self,
        event_type: str,
        actor: str,
        message: str,
        attempt_id: str = "",
    ):
        self._data.setdefault("audit_log", []).append({
            "timestamp": now_iso(),
            "event_type": event_type,
            "actor": actor,
            "message": message,
            "attempt_id": attempt_id,
        })

    @staticmethod
    def _decision_from_score(score: float, threshold: float) -> str:
        score = float(score)
        threshold = float(threshold)
        if score >= threshold:
            return DECISION_VERIFIED
        if score >= max(0.0, threshold - MANUAL_REVIEW_BAND):
            return DECISION_MANUAL_REVIEW
        return DECISION_REJECTED

    @staticmethod
    def _status_from_decision(decision: str) -> str:
        if decision == DECISION_VERIFIED:
            return STATUS_VERIFIED
        if decision == DECISION_MANUAL_REVIEW:
            return STATUS_MANUAL_REVIEW
        return STATUS_REJECTED

    @staticmethod
    def _warnings_from_validation(validation: dict = None) -> List[str]:
        if not validation:
            return []
        warnings = validation.get("warnings")
        if isinstance(warnings, list):
            return warnings
        return []

    def _find_attempt_unlocked(self, attempt_id: str) -> Optional[dict]:
        for attempt in self._data.get("attempts", []):
            if attempt.get("attempt_id") == attempt_id:
                return attempt
        return None

    @staticmethod
    def _validate_id(value: str, label: str):
        value = value or ""
        if not value.strip():
            raise ValueError(f"{label} is required.")
        if len(value) > 80:
            raise ValueError(f"{label} must be 80 characters or fewer.")
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        if any(ch not in allowed for ch in value):
            raise ValueError(
                f"{label} may only contain letters, numbers, dots, hyphens, and underscores."
            )
