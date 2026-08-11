#!/usr/bin/env bash
# Burndeck installer -- venv, config and (optionally) a systemd unit.
# Idempotent and safe to re-run. Run from the repo root.
#
#   ./scripts/install.sh                          venv + config; prints run command
#   ./scripts/install.sh --systemd                ...and install + start the unit
#   ./scripts/install.sh --mock 8                 synthetic 8-GPU rig (no hardware)
#   ./scripts/install.sh --non-interactive --systemd    unattended (CI/seed)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

SYSTEMD=0
MOCK=""
NON_INTERACTIVE=0

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [options]
  --systemd           Render and enable deploy/burndeck.service (needs sudo).
  --mock N            Use N synthetic GPUs (BURNDECK_MOCK_GPUS=N).
  --non-interactive   Never prompt; defer auth to first-boot generation.
  -h, --help          Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --systemd) SYSTEMD=1 ;;
    --mock) MOCK="${2:?--mock needs an integer}"; shift ;;
    --non-interactive) NON_INTERACTIVE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "install: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# --- (a) prerequisites --------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "install: python3 not found" >&2; exit 1; }
py_minor="$(python3 -c 'import sys;print(sys.version_info[1])' 2>/dev/null || echo 0)"
if (( py_minor < 11 )); then
  echo "install: python 3.11+ required (found 3.${py_minor:-?})" >&2; exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "install: nvidia-smi not found -- no GPUs will be shown. Use '--mock N' for a synthetic rig." >&2
fi

# --- (b) virtualenv + dependencies -------------------------------------------
if [[ ! -x .venv/bin/python ]]; then
  echo "install: creating .venv"
  python3 -m venv .venv
fi
echo "install: installing requirements"
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

# --- (c) config.toml ----------------------------------------------------------
if [[ ! -f config.toml ]]; then
  [[ -f config.example.toml ]] || { echo "install: config.example.toml missing" >&2; exit 1; }
  cp config.example.toml config.toml
  if (( NON_INTERACTIVE )); then
    echo "install: --non-interactive: auth deferred to first-boot generation (BURNDECK_AUTH_*)."
  else
    read -rp "Dashboard username [operator]: " user </dev/tty
    user="${user:-operator}"
    pw_hash="$(./.venv/bin/python scripts/hash_password.py </dev/tty \
               | sed -nE 's/.*"([0-9a-f]+)".*/\1/p')"
    [[ -n "$pw_hash" ]] || { echo "install: no password hash produced" >&2; exit 1; }
    sed -i -E "s|^([[:space:]]*user[[:space:]]*=).*$|\1 \"$user\"|" config.toml
    sed -i -E "s|^([[:space:]]*password_sha256[[:space:]]*=).*$|\1 \"$pw_hash\"|" config.toml
  fi
  echo "install: wrote config.toml"
fi

# --- (d/e) run command or systemd unit ---------------------------------------
mock_env=""
if [[ -n "$MOCK" ]]; then mock_env="BURNDECK_MOCK_GPUS=$MOCK "; fi

if (( SYSTEMD )); then
  unit=/etc/systemd/system/burndeck.service
  tmp="$(mktemp)"
  sed -e "s|__USER__|$(whoami)|g" -e "s|__REPO__|$REPO|g" deploy/burndeck.service > "$tmp"
  [[ -n "$MOCK" ]] && sed -i "/^ExecStart=/a Environment=BURNDECK_MOCK_GPUS=$MOCK" "$tmp"
  echo "install: installing systemd unit (needs sudo)"
  sudo install -m 0644 "$tmp" "$unit"; rm -f "$tmp"
  sudo systemctl daemon-reload
  sudo systemctl enable --now burndeck
  echo "install: waiting for healthz..."
  for _ in $(seq 1 20); do
    if curl -fsS http://127.0.0.1:8801/healthz >/dev/null 2>&1; then
      echo "install: burndeck is up at http://127.0.0.1:8801"; exit 0
    fi
    sleep 1
  done
  echo "install: started but /healthz did not respond in 20s -- see: journalctl -u burndeck" >&2
  exit 1
else
  echo "install: done. Run with:"
  echo "  cd \"$REPO\" && ${mock_env}.venv/bin/python app.py"
fi
