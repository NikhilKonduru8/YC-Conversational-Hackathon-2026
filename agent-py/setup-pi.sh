#!/usr/bin/env bash
# One-time setup for running Jarvis on a Raspberry Pi (Raspberry Pi OS Bookworm).
# Run from the agent-py/ folder:  bash setup-pi.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing system libraries (audio, OpenCV, I2C)..."
sudo apt-get update
sudo apt-get install -y \
  libportaudio2 \
  libgl1 libglib2.0-0 \
  i2c-tools python3-dev

echo "==> Checking for uv..."
# Put the usual install location on PATH FIRST, so an already-installed uv is
# found and we don't redownload it (the installer can hang on a slow Pi link).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "    uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Installing Python dependencies (base + Pi OLED group)..."
# A .venv copied from another machine (e.g. your Mac) has the wrong binaries and
# causes "permission denied (os error 13)". Remove it so uv rebuilds natively.
if [ -d .venv ]; then
  echo "    removing copied-in .venv (rebuilding for this machine)..."
  rm -rf .venv
fi
# If a wheel is missing for the pinned Python on your Pi, run:
#   uv python pin 3.11   (then re-run this script)
uv sync
uv sync --group pi

echo "==> Pre-downloading the openWakeWord 'hey_jarvis' model..."
uv run python -c "from openwakeword import utils; utils.download_models()"

cat <<'EOF'

==> Almost done. Two manual steps:

  1. Enable I2C for the OLED (one time):
       sudo raspi-config   ->  Interface Options  ->  I2C  ->  Enable
     then reboot, and confirm the screen is detected at 0x3C:
       i2cdetect -y 1

  2. Put your API keys in agent-py/.env.local
       cp .env.example .env.local   # if you haven't already, then edit it

  Check your hardware (mic / speaker / camera / OLED):
       uv run src/check_devices.py

  Run Jarvis:
       uv run src/jarvis.py
EOF
