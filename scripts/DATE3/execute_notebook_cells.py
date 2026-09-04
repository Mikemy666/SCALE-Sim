"""Execute plain Python notebook cells when nbconvert is unavailable."""

from __future__ import annotations

import contextlib
import argparse
import io
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]


def _display(*values) -> None:
    for value in values:
        if hasattr(value, "to_string"):
            print(value.to_string())
        else:
            print(repr(value))


def execute(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__", "display": _display}
    count = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        count += 1
        stdout = io.StringIO()
        stderr = io.StringIO()
        source = "".join(cell.get("source", []))
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)
        except Exception as error:
            cell["execution_count"] = count
            cell["outputs"] = [{
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": [f"cell {index}: {type(error).__name__}: {error}"],
            }]
            path.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"{path} cell {index} failed") from error
        outputs = []
        if stdout.getvalue():
            outputs.append({
                "output_type": "stream", "name": "stdout",
                "text": stdout.getvalue().splitlines(keepends=True),
            })
        if stderr.getvalue():
            outputs.append({
                "output_type": "stream", "name": "stderr",
                "text": stderr.getvalue().splitlines(keepends=True),
            })
        cell["execution_count"] = count
        cell["outputs"] = outputs
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"executed {count} code cells: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", default=["exp1", "exp2"])
    args = parser.parse_args()
    for name in args.notebooks:
        filename = name if name.endswith(".ipynb") else f"{name}.ipynb"
        execute(ROOT / "fig" / filename)


if __name__ == "__main__":
    main()
