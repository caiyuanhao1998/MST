#!/usr/bin/env bash
set -euo pipefail
MST_DEMO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ ! -x "$MST_DEMO_ROOT/.venv/bin/python" ]]; then
  echo 'Missing .venv. Follow docs/GRADIO_DEMO.md to install the local environment.' >&2
  exit 1
fi
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_DISABLE_TELEMETRY=1
exec "$MST_DEMO_ROOT/.venv/bin/python" "$MST_DEMO_ROOT/app.py" "$@"
