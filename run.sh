#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ ! -x .venv/bin/python ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "Python 3 is required. Install Python 3.10 or later, then run this script again." >&2
    exit 1
  fi
fi

if ! .venv/bin/python -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
  echo "Python 3.10 or later is required." >&2
  exit 1
fi

if ! .venv/bin/python -c 'import flask, pandas, numpy, scipy, sklearn, matplotlib, requests, pptx, reportlab, PIL' >/dev/null 2>&1; then
  echo "Installing project dependencies..."
  .venv/bin/python -m pip install -r requirements.txt
fi

backend_pid=""
cleanup() {
  trap - EXIT INT TERM
  [[ -z "$backend_pid" ]] || kill "$backend_pid" 2>/dev/null || true
  [[ -z "$backend_pid" ]] || wait "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Flask serves both the API and the frontend. Pick an available port so an
# existing local server does not make startup fail with "Address already in use".
PORT="$(.venv/bin/python - <<'PY'
import os
import socket

preferred = int(os.environ.get("PORT", "5000"))
for port in [preferred, *range(preferred + 1, preferred + 21)]:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("No available port found in the configured port range.")
PY
)"
export PORT
.venv/bin/python backend/app.py &
backend_pid=$!

echo "Weather Station Behaviour Pattern Grouping is starting."
echo "App: http://localhost:${PORT}"
echo "Press Ctrl+C to stop both servers."

wait "$backend_pid"
