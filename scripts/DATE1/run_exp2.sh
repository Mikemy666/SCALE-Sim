#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python3 run_date1_experiments.py --exp exp2 "$@"

for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then
        exit 0
    fi
done

python3 select_exp2_static_configs.py
