"""
FLUXSynID discovery helpers for the campus dashboard.

The raw FLUXSynID dataset stays outside the repository. This module only
discovers eligible identities and returns file paths that the API can enroll
through the normal face verification pipeline.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FLUX_ROOT = Path.home() / "Downloads" / "FLUXSynID"
DEFAULT_FLUX_COUNT = 25
DEFAULT_FLUX_SEED = 42
DEFAULT_FLUX_AGE_BUCKETS = {"15-19", "20-29"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class FluxPreuploadError(ValueError):
    """Raised when FLUXSynID input cannot support preupload."""


@dataclass(frozen=True)
class FluxCampusIdentity:
    identity: str
    doc_image: Path
    live_a_image: Path
    live_e_image: Path
    live_p_image: Path
    age_bucket: Optional[str] = None

    @property
    def enrollment_images(self) -> Tuple[Path, Path, Path]:
        return (self.doc_image, self.live_a_image, self.live_e_image)

    @property
    def query_image(self) -> Path:
        return self.live_p_image


def normalize_flux_dataset_dir(dataset_dir: Optional[str | Path] = None) -> Path:
    """Resolve a user-supplied FLUXSynID path to the identity-folder root."""
    root = Path(dataset_dir).expanduser() if dataset_dir else DEFAULT_FLUX_ROOT
    if not root.exists():
        raise FluxPreuploadError(f"FLUXSynID dataset path does not exist: {root}")
    if not root.is_dir():
        raise FluxPreuploadError(f"FLUXSynID dataset path is not a directory: {root}")

    for candidate in _dataset_candidates(root):
        if _looks_like_identity_root(candidate):
            return candidate

    raise FluxPreuploadError(
        "Could not find FLUXSynID identity folders under "
        f"{root}. Expected folders containing *_f_doc.jpg and live variants."
    )


def discover_flux_identities(
    dataset_dir: str | Path,
    *,
    age_buckets: Optional[Sequence[str]] = DEFAULT_FLUX_AGE_BUCKETS,
) -> Dict[str, FluxCampusIdentity]:
    """Return eligible FLUXSynID identities keyed by identity id."""
    root = normalize_flux_dataset_dir(dataset_dir)
    allowed_ages = set(age_buckets) if age_buckets is not None else None
    identities: Dict[str, FluxCampusIdentity] = {}

    for identity_dir in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        item = _read_identity(identity_dir)
        if item is None:
            continue
        if allowed_ages is not None and item.age_bucket not in allowed_ages:
            continue
        identities[item.identity] = item

    if not identities:
        age_note = (
            f" for age buckets {sorted(allowed_ages)}"
            if allowed_ages is not None
            else ""
        )
        raise FluxPreuploadError(f"No eligible FLUXSynID identities found{age_note}.")
    return identities


def select_flux_identities(
    identities: Dict[str, FluxCampusIdentity],
    *,
    count: int = DEFAULT_FLUX_COUNT,
    seed: int = DEFAULT_FLUX_SEED,
) -> List[FluxCampusIdentity]:
    """Select a deterministic subset of discovered FLUXSynID identities."""
    if count < 1:
        raise FluxPreuploadError("count must be at least 1.")
    keys = sorted(identities)
    rng = random.Random(seed)
    rng.shuffle(keys)
    return [identities[key] for key in sorted(keys[:count])]


def select_flux_identities_from_dataset(
    dataset_dir: str | Path,
    *,
    count: int = DEFAULT_FLUX_COUNT,
    seed: int = DEFAULT_FLUX_SEED,
    age_buckets: Optional[Sequence[str]] = DEFAULT_FLUX_AGE_BUCKETS,
) -> List[FluxCampusIdentity]:
    """Deterministically select identities without parsing every metadata file."""
    if count < 1:
        raise FluxPreuploadError("count must be at least 1.")
    root = normalize_flux_dataset_dir(dataset_dir)
    allowed_ages = set(age_buckets) if age_buckets is not None else None
    candidates = [path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected: List[FluxCampusIdentity] = []
    for candidate in candidates:
        item = _read_identity(candidate)
        if item is None:
            continue
        if allowed_ages is not None and item.age_bucket not in allowed_ages:
            continue
        selected.append(item)
        if len(selected) >= count:
            break

    if not selected:
        raise FluxPreuploadError("No eligible FLUXSynID identities found.")
    return sorted(selected, key=lambda item: item.identity)


def count_flux_identities(
    dataset_dir: Optional[str | Path] = None,
    *,
    age_buckets: Optional[Sequence[str]] = DEFAULT_FLUX_AGE_BUCKETS,
) -> int:
    return len(discover_flux_identities(dataset_dir or DEFAULT_FLUX_ROOT, age_buckets=age_buckets))


def count_flux_identity_dirs(dataset_dir: Optional[str | Path] = None) -> int:
    """Count role-complete identity folders without opening metadata JSON files."""
    root = normalize_flux_dataset_dir(dataset_dir)
    count = 0
    for identity_dir in root.iterdir():
        if not identity_dir.is_dir():
            continue
        identity = identity_dir.name
        if (
            (identity_dir / f"{identity}_f_doc.jpg").exists()
            and (identity_dir / f"{identity}_f_live_0_a_d1.jpg").exists()
            and (identity_dir / f"{identity}_f_live_0_e_d1.jpg").exists()
            and (identity_dir / f"{identity}_f_live_0_p_d1.jpg").exists()
        ):
            count += 1
    return count


def _dataset_candidates(root: Path) -> Iterable[Path]:
    seen = set()
    candidates = [
        root,
        root / "FLUXSynID",
        root / "FLUXSynID" / "FLUXSynID",
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.exists() and candidate.is_dir():
            seen.add(resolved)
            yield candidate


def _looks_like_identity_root(path: Path) -> bool:
    checked = 0
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        checked += 1
        if _read_identity(child) is not None:
            return True
        if checked >= 80:
            break
    return False


def _read_identity(identity_dir: Path) -> Optional[FluxCampusIdentity]:
    if not identity_dir.is_dir():
        return None
    images = [
        path
        for path in sorted(identity_dir.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    doc = _first_role(images, "doc")
    live_a = _first_role(images, "a")
    live_e = _first_role(images, "e")
    live_p = _first_role(images, "p")
    if not all((doc, live_a, live_e, live_p)):
        return None
    return FluxCampusIdentity(
        identity=identity_dir.name,
        doc_image=doc,
        live_a_image=live_a,
        live_e_image=live_e,
        live_p_image=live_p,
        age_bucket=_read_age_bucket(identity_dir),
    )


def _first_role(images: Sequence[Path], role: str) -> Optional[Path]:
    for image in images:
        stem = image.stem.lower()
        if role == "doc" and "_doc" in stem:
            return image
        if role != "doc" and "_live_" in stem and f"_{role}_" in stem:
            return image
    return None


def _read_age_bucket(identity_dir: Path) -> Optional[str]:
    json_path = identity_dir / f"{identity_dir.name}_f.json"
    if not json_path.exists():
        matches = sorted(identity_dir.glob("*_f.json"))
        json_path = matches[0] if matches else json_path
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    attributes = data.get("attributes") or {}
    age = attributes.get("ages.txt") or attributes.get("age") or attributes.get("ages")
    return str(age) if age else None
