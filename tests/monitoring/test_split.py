from __future__ import annotations

from pathlib import Path

import pytest

from leiturista.monitoring import split


def _make_dir(tmp_path: Path, n: int = 40) -> Path:
    for i in range(n):
        (tmp_path / f"praekelt_{i:04d}.jpg").write_bytes(b"x")
    (tmp_path / "labels.csv").write_text("ignored")
    return tmp_path


def test_split_is_deterministic_disjoint_and_ignores_other_files(tmp_path: Path) -> None:
    images = _make_dir(tmp_path)
    ref1, hold1 = split.split_reference_images(images, seed=0, n_reference=30)
    ref2, hold2 = split.split_reference_images(images, seed=0, n_reference=30)
    assert ref1 == ref2 and hold1 == hold2
    assert len(ref1) == 30 and len(hold1) == 10 and not set(ref1) & set(hold1)
    assert split.split_reference_images(images, seed=1, n_reference=30)[0] != ref1


def test_too_few_images_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        split.split_reference_images(_make_dir(tmp_path, 5))


def test_real_dataset_counts_when_present() -> None:
    if not split.PRAEKELT_DIR.exists():
        pytest.skip("data/ ausente")
    ref, hold = split.split_reference_images()
    assert (len(ref), len(hold)) == (132, 33)
    assert len(split.traffic_images("henrik")) == 168 and len(split.traffic_images("lcd")) == 50
    assert split.traffic_images("holdout") == hold
