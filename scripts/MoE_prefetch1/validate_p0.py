"""Generate and validate the MoE_prefetch1 P0 reproducibility manifest."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "MoE_prefetch1" / "p0" / "baseline_manifest.json"
EXPECTED_BRANCH = "MoE_prefetch1"
BASE_COMMIT = "e5675c5b7155b01d3f095d3f9d153fd74c6a4734"

INPUT_ROOTS = (
    ROOT / "scalesim",
    ROOT / "configs" / "MoE" / "DATE1",
    ROOT / "topologies" / "MoE" / "DATE1",
)
INPUT_FILES = (
    ROOT / "requirements.txt",
    ROOT / "setup.py",
    ROOT / "run_date1_experiments.py",
    ROOT / "run_ep_moe_experiments.py",
    ROOT / "run_ep_moe_sanity.py",
    ROOT / "validate_date1_setup.py",
    ROOT / "validate_ep_moe_reports.py",
)


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_inputs() -> list[Path]:
    paths = list(INPUT_FILES)
    for root in INPUT_ROOTS:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.exists() and "__pycache__" not in path.parts
        }
    )


def main() -> int:
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    status_lines = [line for line in git("status", "--porcelain=v1").splitlines() if line]
    tracked = [line for line in status_lines if not line.startswith("??")]
    untracked = [line for line in status_lines if line.startswith("??")]
    test_sources = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").glob("test_*.py")
    )
    dependency_versions = {}
    for package in ("numpy", "pandas", "matplotlib"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None

    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "simulator_only",
        "git": {
            "branch": branch,
            "head": head,
            "p0_base_commit": BASE_COMMIT,
            "dirty": bool(status_lines),
            "tracked_change_count": len(tracked),
            "untracked_change_count": len(untracked),
            "status": status_lines,
        },
        "runtime": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "dependencies": dependency_versions,
            "command": [sys.executable, *sys.argv],
        },
        "source_tests": test_sources,
        "inputs": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in source_inputs()
        },
        "namespace": {
            "configs": "configs/MoE/MoE_prefetch1",
            "topologies": "topologies/MoE/MoE_prefetch1",
            "outputs": "outputs/MoE_prefetch1",
            "scripts": "scripts/MoE_prefetch1",
        },
    }

    errors = []
    if branch != EXPECTED_BRANCH:
        errors.append(f"expected branch {EXPECTED_BRANCH!r}, found {branch!r}")
    if not head:
        errors.append("unable to resolve Git HEAD")
    if not test_sources:
        errors.append("no discoverable tests/test_*.py source files")

    manifest["validation"] = {"passed": not errors, "errors": errors}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"P0 manifest: {OUTPUT}")
    print(
        f"branch={branch} head={head[:12]} tracked={len(tracked)} "
        f"untracked={len(untracked)} tests={len(test_sources)} inputs={len(manifest['inputs'])}"
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("P0 manifest validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
