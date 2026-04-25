from datetime import datetime
from pathlib import Path

RUNS_DIR = "runs"
LATEST_RUN_FILE = "latest_run_id.txt"


def _runs_root(base_dir: Path) -> Path:
    return Path(base_dir) / RUNS_DIR


def _latest_file(base_dir: Path) -> Path:
    return Path(base_dir) / LATEST_RUN_FILE


def create_run_dir(base_dir: Path, run_id: str = "") -> tuple[str, Path]:
    """Create and return (run_id, run_dir).

    If run_id is empty, create a timestamped id.
    """
    base_dir = Path(base_dir)
    runs_root = _runs_root(base_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    resolved_run_id = run_id.strip() or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _latest_file(base_dir).write_text(f"{resolved_run_id}\n", encoding="utf-8")
    return resolved_run_id, run_dir


def latest_run_id(base_dir: Path) -> str:
    """Return latest run id from pointer file or newest run directory."""
    base_dir = Path(base_dir)

    latest_path = _latest_file(base_dir)
    if latest_path.exists():
        run_id = latest_path.read_text(encoding="utf-8").strip()
        if run_id and (_runs_root(base_dir) / run_id).exists():
            return run_id

    runs_root = _runs_root(base_dir)
    if not runs_root.exists():
        raise FileNotFoundError(f"No run directory found under {runs_root}")

    candidates = sorted(
        [p.name for p in runs_root.iterdir() if p.is_dir()]
    )
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_root}")

    return candidates[-1]


def resolve_run_dir(base_dir: Path, run_id: str = "") -> tuple[str, Path]:
    """Resolve run id and directory.

    If run_id is empty, resolve to latest run.
    """
    base_dir = Path(base_dir)
    resolved_run_id = run_id.strip() or latest_run_id(base_dir)
    run_dir = _runs_root(base_dir) / resolved_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return resolved_run_id, run_dir
