from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path("data")
DATASET_URL = "gs://gresearch/robotics/droid_100/1.0.0"


@pytest.fixture(scope="module")
def cache_dir() -> Path:
    path = DATA_DIR / "droid_100"
    if not (path / "metadata" / "metadata.json").is_file():
        pytest.skip("droid_100 cache not found; run `uv run cs179 download` first")
    return path


@pytest.fixture(scope="module")
def py_loader(cache_dir: Path):
    from rlds import RldsObservationLoader

    return RldsObservationLoader(data_dir=cache_dir)


@pytest.fixture(scope="module")
def native_loader(cache_dir: Path):
    pytest.importorskip("cs179._native")
    from cs179._native import RldsObservationLoader

    if not hasattr(RldsObservationLoader, "__init__"):
        pytest.skip("RldsObservationLoader missing from _native; run ./scripts/build_loader.sh")
    return RldsObservationLoader(str(cache_dir))


def test_native_loader_metadata_matches_python(py_loader, native_loader) -> None:
    assert len(native_loader) == len(py_loader)
    assert native_loader.total_steps == py_loader.total_steps
    assert native_loader.control_hz == pytest.approx(py_loader.control_hz)
    assert list(native_loader.observation_keys) == list(py_loader.observation_keys)
    native_shapes = {k: tuple(v) for k, v in dict(native_loader.field_shapes).items()}
    assert native_shapes == py_loader.field_shapes


def test_native_get_demo_matches_python(py_loader, native_loader) -> None:
    for demo_id in (0, 1, len(py_loader) - 1):
        py_demo = py_loader.get_demo(demo_id)
        native_demo = native_loader.get_demo(demo_id)
        assert set(native_demo.keys()) == set(py_demo.keys())
        for key in py_demo:
            np.testing.assert_allclose(
                native_demo[key],
                py_demo[key],
                rtol=0.0,
                atol=0.0,
                err_msg=f"demo {demo_id} key {key}",
            )


def test_native_get_demo_views_when_contiguous(py_loader, native_loader) -> None:
    for demo_id in range(min(10, len(py_loader))):
        py_views = py_loader.get_demo_views(demo_id)
        native_views = native_loader.get_demo_views(demo_id)
        if py_views is None:
            assert native_views is None
            continue
        assert native_views is not None
        for key in py_views:
            np.testing.assert_allclose(native_views[key], py_views[key], atol=0.0, rtol=0.0)
