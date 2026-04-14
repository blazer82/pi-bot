#!/usr/bin/env bash
# Pi-Bot setup script for Raspberry Pi 5 (Bookworm 64-bit)
# Run: chmod +x setup.sh && ./setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
OLLAMA_MODEL="gemma4:e2b-it-q4_K_M"

echo "=== Pi-Bot Setup ==="

# -----------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------
echo ""
echo "--- Installing system packages ---"
sudo apt update
sudo apt install -y espeak-ng python3-pip python3-venv cmake build-essential \
    libportaudio2 libsndfile1

# -----------------------------------------------------------------------
# 2. Python virtual environment
# -----------------------------------------------------------------------
echo ""
echo "--- Setting up Python venv ---"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"

# -----------------------------------------------------------------------
# 3. Download openWakeWord models
# -----------------------------------------------------------------------
echo ""
echo "--- Downloading openWakeWord models ---"
python3 -c "import openwakeword; openwakeword.utils.download_models()"

# -----------------------------------------------------------------------
# 4. Download whisper.cpp 'small' model
# -----------------------------------------------------------------------
echo ""
echo "--- Pre-downloading whisper 'small' model (this may take a while) ---"
python3 -c "from pywhispercpp.model import Model; Model('small')"

# -----------------------------------------------------------------------
# 5. Install ollama
# -----------------------------------------------------------------------
echo ""
echo "--- Installing ollama ---"
if command -v ollama &>/dev/null; then
    echo "ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi

# -----------------------------------------------------------------------
# 6. Pull LLM model
# -----------------------------------------------------------------------
echo ""
echo "--- Pulling $OLLAMA_MODEL (this will take a while on first run) ---"
ollama pull "$OLLAMA_MODEL"

# -----------------------------------------------------------------------
# 7. Audio device check
# -----------------------------------------------------------------------
echo ""
echo "--- Detected audio devices ---"
python3 -c "import sounddevice; print(sounddevice.query_devices())"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run Pi-Bot:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 $SCRIPT_DIR/pi_bot.py"
echo ""
echo "Make sure ollama is running (it starts automatically as a service)."
echo "If not, start it with: ollama serve"
