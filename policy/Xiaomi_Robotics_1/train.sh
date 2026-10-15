#!/usr/bin/env bash
set -euo pipefail
echo "Xiaomi upstream provides inference only; no executable training/data recipe is included. Supply a matching trained checkpoint from an external training implementation." >&2
exit 2
