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
    libportaudio2 libsndfile1 \
    pipewire pipewire-pulse pipewire-alsa wireplumber alsa-utils

# -----------------------------------------------------------------------
# 1b. Audio group & user services
# -----------------------------------------------------------------------
echo ""
echo "--- Configuring audio permissions and services ---"
sudo usermod -aG audio "$USER"
sudo loginctl enable-linger "$USER"
systemctl --user enable --now pipewire wireplumber pipewire-pulse 2>/dev/null || true

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

# Set OpenMP threads to match Pi 5 core count for whisper.cpp performance
if ! grep -q 'OMP_NUM_THREADS' ~/.bashrc 2>/dev/null; then
    echo 'export OMP_NUM_THREADS=4' >> ~/.bashrc
    echo "Added OMP_NUM_THREADS=4 to ~/.bashrc"
fi
export OMP_NUM_THREADS=4

# -----------------------------------------------------------------------
# 3. Download whisper.cpp 'small' model
# -----------------------------------------------------------------------
echo ""
echo "--- Pre-downloading whisper 'small' model (this may take a while) ---"
python3 -c "from pywhispercpp.model import Model; Model('small')"

# -----------------------------------------------------------------------
# 4. Install ollama
# -----------------------------------------------------------------------
echo ""
echo "--- Installing ollama ---"
if command -v ollama &>/dev/null; then
    echo "ollama already installed: $(ollama --version)"
else
    curl -fsSL https://ollama.com/install.sh | sh
fi

# -----------------------------------------------------------------------
# 5. Pull LLM model
# -----------------------------------------------------------------------
echo ""
echo "--- Pulling $OLLAMA_MODEL (this will take a while on first run) ---"
ollama pull "$OLLAMA_MODEL"

# -----------------------------------------------------------------------
# 6. Install MBROLA voices for smoother German TTS
# -----------------------------------------------------------------------
echo ""
echo "--- Installing MBROLA German voices ---"
sudo apt install -y mbrola mbrola-de1 mbrola-de2 mbrola-de3 mbrola-de4 \
    mbrola-de5 mbrola-de6 mbrola-de7 mbrola-de8

# -----------------------------------------------------------------------
# 7. Audio device check
# -----------------------------------------------------------------------
echo ""
echo "--- Detected audio devices ---"
echo ""
echo "PipeWire devices (wpctl):"
wpctl status 2>/dev/null || echo "  PipeWire not yet active (may need a reboot after first install)"
echo ""
echo "sounddevice devices:"
python3 -c "import sounddevice; print(sounddevice.query_devices())"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "NOTE: If this is a fresh PipeWire install, reboot for audio services"
echo "to start properly: sudo reboot"
echo ""
echo "To run Pi-Bot:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 $SCRIPT_DIR/pi_bot.py"
echo ""
echo "Make sure ollama is running (it starts automatically as a service)."
echo "If not, start it with: ollama serve"
