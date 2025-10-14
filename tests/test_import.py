import sys
from pathlib import Path


def test_import_package():
    """Smoke test: ensure `src` is on sys.path and `ewc` package imports."""
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import importlib

    ewc = importlib.import_module("ewc")
    assert hasattr(ewc, "__name__") and ewc.__name__ == "ewc"
