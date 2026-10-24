# source scripts/env.sh  -> CUDA libs for JAX on WSL + headless rendering
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$(ls -d $_ROOT/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')/usr/lib/wsl/lib"
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
