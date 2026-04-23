#!/usr/bin/env bash
# Pi-Bot setup script for Raspberry Pi 5 (Bookworm 64-bit)
# Run: chmod +x setup.sh && ./setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
OLLAMA_MODEL="gemma4:e4b-it-q4_K_M"

echo "=== Pi-Bot Setup ==="

# -----------------------------------------------------------------------
# 1. System packages
# -----------------------------------------------------------------------
echo ""
echo "--- Installing system packages ---"
sudo apt update
sudo apt install -y python3-pip python3-venv cmake build-essential \
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
# 3. Download quantized whisper.cpp 'small' model
# -----------------------------------------------------------------------
echo ""
WHISPER_DIR="$HOME/.local/share/pywhispercpp/models"
WHISPER_MODEL="ggml-small-q8_0.bin"
echo "--- Downloading whisper $WHISPER_MODEL ---"
mkdir -p "$WHISPER_DIR"
if [ ! -f "$WHISPER_DIR/$WHISPER_MODEL" ]; then
    curl -fsSL -o "$WHISPER_DIR/$WHISPER_MODEL" \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL"
    echo "Whisper model downloaded."
else
    echo "Whisper model already present."
fi

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
# 6. Install Piper TTS and German voice model
# -----------------------------------------------------------------------
echo ""
echo "--- Installing Piper TTS ---"
PIPER_VERSION="2023.11.14-2"
PIPER_DIR="$SCRIPT_DIR/models/piper"
mkdir -p "$PIPER_DIR"

if ! command -v piper &>/dev/null; then
    ARCH=$(uname -m)  # aarch64 on Pi 5
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${ARCH}.tar.gz"
    echo "Downloading piper from $PIPER_URL"
    curl -fsSL "$PIPER_URL" | tar -xz -C /tmp
    sudo cp /tmp/piper/piper /usr/local/bin/
    sudo cp /tmp/piper/lib*.so* /usr/local/lib/ 2>/dev/null || true
    sudo ldconfig
    rm -rf /tmp/piper
    echo "piper installed: $(piper --version)"
else
    echo "piper already installed: $(piper --version)"
fi

MODEL_FILE="$PIPER_DIR/de_DE-thorsten-medium.onnx"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading de_DE-thorsten-medium voice model..."
    curl -fsSL -o "$MODEL_FILE" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
    curl -fsSL -o "${MODEL_FILE}.json" \
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
    echo "Voice model downloaded."
else
    echo "Voice model already present."
fi

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

# -----------------------------------------------------------------------
# 8. FT81x display driver
# -----------------------------------------------------------------------
echo ""
echo "--- Installing FT81x display driver ---"
sudo apt install -y libgpiod-dev
pip install "ft81x @ git+https://github.com/blazer82/FT81x_Python_Driver.git"
echo "FT81x display driver installed."

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
