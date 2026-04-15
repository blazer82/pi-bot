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
# 7. (Optional) Install MBROLA voices for smoother German TTS
# -----------------------------------------------------------------------
echo ""
echo "--- Optional: MBROLA voices ---"
echo "The default espeak-ng voice (-v de) is robotic. MBROLA voices sound"
echo "smoother/more natural. Install them? Available: de1(F) de2(M) de3(F)"
echo "de4(M) de5(F) de6(M) de7(F) de8(M)"
read -rp "Install MBROLA German voices? [y/N] " install_mbrola
if [[ "${install_mbrola,,}" == "y" ]]; then
    sudo apt install -y mbrola mbrola-de1 mbrola-de2 mbrola-de3 mbrola-de4 \
        mbrola-de5 mbrola-de6 mbrola-de7 mbrola-de8
    echo "Installed! To use, set language in CONFIG to e.g. 'mb-de2' (male)"
    echo "Test with: espeak-ng -v mb-de2 -s 130 -p 40 'Hallo, ich bin Pi Bot.'"
else
    echo "Skipped. You can install later with:"
    echo "  sudo apt install mbrola mbrola-de2"
fi

# -----------------------------------------------------------------------
# 8. Audio device check
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
