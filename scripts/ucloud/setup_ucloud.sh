#!/usr/bin/env bash
# One-time setup of a UCloud job (Linux). Idempotent: run it at the start of every job (~5-10 min, the venv lives on the job's local disk).
#
#   cd /work/<your-mounted-folder>            # the persistent UCloud folder: it keeps myoassist/, the project and ALL results
#   bash terrain-memory-prosthesis/scripts/ucloud/setup_ucloud.sh
#
# Optional environment variables:
#   WS=...         persistent workspace folder (default: the current directory)
#   VENV=...       job-local virtualenv (default: $HOME/venv-prosthesis)
#   REPO_URL=...   git URL of the project, used only if $WS/terrain-memory-prosthesis does not exist yet
#   MYO_URL=...    MyoAssist git URL (default: github.com/neumovelab/myoassist)
set -euo pipefail

WS="$(cd "${WS:-$PWD}" && pwd)"
VENV="${VENV:-$HOME/venv-prosthesis}"
MYO_URL="${MYO_URL:-https://github.com/neumovelab/myoassist.git}"
MYO_COMMIT="b1baf69da59dc6728515d0977cbc8fbf00b9225a"   # the commit everything was tested with
REPO_URL="${REPO_URL:-}"
export PATH="$HOME/.local/bin:$PATH"

echo "== workspace: $WS | venv: $VENV | cpus: $(nproc)"

# 1. uv (fast installer; plain pip cannot resolve MyoSuite's pins)
if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --user -q uv 2>/dev/null || python3 -m pip install --user -q --break-system-packages uv
fi

# 2. MyoAssist at the pinned commit (stays in the persistent folder: its rl_train/results/ is where trainings write)
if [ ! -d "$WS/myoassist" ]; then
  git clone -q "$MYO_URL" "$WS/myoassist"
fi
git -C "$WS/myoassist" checkout -q "$MYO_COMMIT" || echo "WARNING: could not check out $MYO_COMMIT (local changes?) - continuing with the current state"

# 3. the project
if [ ! -d "$WS/terrain-memory-prosthesis" ]; then
  [ -n "$REPO_URL" ] || { echo "ERROR: $WS/terrain-memory-prosthesis not found. Upload it, or run with REPO_URL=<git url>"; exit 1; }
  git clone -q "$REPO_URL" "$WS/terrain-memory-prosthesis"
fi

# 4. venv (Python 3.12) + CPU-only torch (the nets are tiny; the GPU build is a multi-GB download for nothing)
[ -d "$VENV" ] || uv venv -q --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu || echo "NOTE: CPU torch index unreachable, the default torch will be installed instead"
(cd "$WS/myoassist" && uv pip install -q -e .)
uv pip install -q -e "$WS/terrain-memory-prosthesis[dev]"

# 5. headless MuJoCo GL backend: egl -> osmesa -> default, first one that imports
GL=""
for cand in egl osmesa ""; do
  if MUJOCO_GL="$cand" python -c "import mujoco" 2>/dev/null; then GL="$cand"; break; fi
done
echo "== MUJOCO_GL='${GL}'"

# 6. environment file for later shells of this job
cat > "$WS/.ucloud_env.sh" <<ENV
source "$VENV/bin/activate"
export WS="$WS"
export MYOASSIST_DIR="$WS/myoassist"
export MUJOCO_GL="$GL"
export PATH="\$HOME/.local/bin:\$PATH"
ENV

# 7. sanity
(cd "$WS/terrain-memory-prosthesis" && python -m pytest -q 2>&1 | tail -2)
echo
echo "== done. In every new shell of this job:  source $WS/.ucloud_env.sh"
echo "== next:  bash $WS/terrain-memory-prosthesis/scripts/ucloud/stage1.sh measure"
