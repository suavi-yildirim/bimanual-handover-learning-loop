#!/usr/bin/env bash
# Sets up openpi for the FR3 Duo pi0.5 fine-tune (C2_pi05_Training_Process.md, section 4).
# Idempotent: clones ~/openpi if missing, installs deps, links our policy + converter, applies the config patch.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
OPENPI="${OPENPI:-$HOME/openpi}"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

if [ ! -x "$UV" ]; then
    wget -qO- https://astral.sh/uv/install.sh | sh
    UV="$HOME/.local/bin/uv"
fi
if [ ! -d "$OPENPI" ]; then
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
fi

cd "$OPENPI"
GIT_LFS_SKIP_SMUDGE=1 "$UV" sync
GIT_LFS_SKIP_SMUDGE=1 "$UV" pip install -e .

mkdir -p examples/fr3duo
ln -sf "$HERE/convert_fr3duo_to_openpi.py" examples/fr3duo/convert_fr3duo_to_openpi.py
ln -sf "$HERE/fr3duo_policy.py" src/openpi/policies/fr3duo_policy.py

if grep -q "pi05_fr3duo_handover_lora" src/openpi/training/config.py; then
    echo "[setup] config patch already applied"
else
    git apply "$HERE/openpi_fr3duo_config.patch"
    echo "[setup] config patch applied"
fi

grep -rn "CODEBASE_VERSION =" .venv/lib/python3.*/site-packages/lerobot/common/datasets/lerobot_dataset.py
.venv/bin/python -c "import jax; print('[setup] jax devices:', jax.devices())"
