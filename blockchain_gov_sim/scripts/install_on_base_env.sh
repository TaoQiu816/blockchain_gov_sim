#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m pip install --upgrade pip
python -m pip install -r base_env_delta_requirements.txt

echo "Base conda environment has been patched for blockchain_gov_sim." >&2
