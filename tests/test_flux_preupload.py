import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.flux_preupload import (  # noqa: E402
    discover_flux_identities,
    normalize_flux_dataset_dir,
    select_flux_identities,
)


def test_normalize_flux_dataset_dir_accepts_parent_and_identity_root(tmp_path):
    parent = tmp_path / "FLUXSynID"
    identity_root = parent / "FLUXSynID" / "FLUXSynID"
    _write_flux_identity(identity_root, "person_001", age="20-29")

    assert normalize_flux_dataset_dir(parent) == identity_root
    assert normalize_flux_dataset_dir(identity_root) == identity_root


def test_discover_filters_to_college_age_and_requires_roles(tmp_path):
    _write_flux_identity(tmp_path, "person_001", age="20-29")
    _write_flux_identity(tmp_path, "person_002", age="60-69")
    incomplete = tmp_path / "person_003"
    _write_image(incomplete / "person_003_f_doc.jpg")

    identities = discover_flux_identities(tmp_path)

    assert list(identities) == ["person_001"]
    item = identities["person_001"]
    assert item.doc_image.name == "person_001_f_doc.jpg"
    assert item.enrollment_images[1].name == "person_001_f_live_0_a_d1.jpg"
    assert item.enrollment_images[2].name == "person_001_f_live_0_e_d1.jpg"
    assert item.query_image.name == "person_001_f_live_0_p_d1.jpg"


def test_select_flux_identities_is_deterministic(tmp_path):
    for idx in range(8):
        _write_flux_identity(tmp_path, f"person_{idx:03d}", age="20-29")
    identities = discover_flux_identities(tmp_path)

    first = [item.identity for item in select_flux_identities(identities, count=4, seed=42)]
    second = [item.identity for item in select_flux_identities(identities, count=4, seed=42)]

    assert first == second
    assert len(first) == 4


def _write_flux_identity(root, identity, age="20-29"):
    identity_dir = root / identity
    _write_image(identity_dir / f"{identity}_f_doc.jpg", seed=1)
    _write_image(identity_dir / f"{identity}_f_live_0_a_d1.jpg", seed=2)
    _write_image(identity_dir / f"{identity}_f_live_0_e_d1.jpg", seed=3)
    _write_image(identity_dir / f"{identity}_f_live_0_p_d1.jpg", seed=4)
    identity_dir.mkdir(parents=True, exist_ok=True)
    (identity_dir / f"{identity}_f.json").write_text(
        json.dumps({"attributes": {"ages.txt": age}}),
        encoding="utf-8",
    )


def _write_image(path, seed=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="JPEG")
    path.write_bytes(out.getvalue())
