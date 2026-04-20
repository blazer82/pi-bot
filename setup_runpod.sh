#!/usr/bin/env bash
set -euo pipefail

echo "=== Pi-Bot Voice Trainer — RunPod Setup ==="
echo

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "Warning: no GPU detected. Training will fail without a GPU." >&2
fi

PIPER_DIR="/workspace/piper1-gpl"
CHECKPOINTS_DIR="/workspace/checkpoints"
PRETRAINED_CKPT="${CHECKPOINTS_DIR}/pretrained.ckpt"
PRETRAINED_CFG="${CHECKPOINTS_DIR}/config.json"

# System dependencies (libespeak-ng-dev needed for espeakbridge C extension)
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq espeak-ng libespeak-ng-dev ffmpeg cmake > /dev/null
echo "Done."
echo

# piper1-gpl
if [[ -d "${PIPER_DIR}" ]]; then
  echo "piper1-gpl already cloned, skipping."
else
  echo "Cloning piper1-gpl v1.3.0..."
  git clone --branch v1.3.0 --depth 1 https://github.com/OHF-voice/piper1-gpl.git "${PIPER_DIR}"
fi

echo "Installing piper1-gpl..."
cd "${PIPER_DIR}"
pip install scikit-build cmake -q
pip install -e ".[train]" -q
pip install pytorch-lightning==2.1.0 -q
bash build_monotonic_align.sh

# Build espeakbridge C extension
ESPEAK_EXT="src/piper/espeakbridge$(python3-config --extension-suffix)"
if [[ ! -f "${ESPEAK_EXT}" ]]; then
  echo "Building espeakbridge C extension..."
  gcc -shared -fPIC -O2 \
    $(python3-config --includes) \
    -I/usr/include/espeak-ng \
    src/piper/espeakbridge.c \
    -o "${ESPEAK_EXT}" \
    -lespeak-ng
fi

# Patch piper to allow PosixPath in checkpoints (newer torch requires explicit allowlist)
PIPER_MAIN="src/piper/train/__main__.py"
if ! grep -q "add_safe_globals" "${PIPER_MAIN}" 2>/dev/null; then
  sed -i '1i import pathlib, torch; torch.serialization.add_safe_globals([pathlib.PosixPath])' "${PIPER_MAIN}"
fi

cd /workspace
echo "Done."
echo

# Pretrained checkpoint
mkdir -p "${CHECKPOINTS_DIR}"
if [[ -f "${PRETRAINED_CKPT}" && $(stat -c%s "${PRETRAINED_CKPT}" 2>/dev/null || echo 0) -gt 1000 ]]; then
  echo "Pretrained checkpoint already downloaded, skipping."
else
  echo "Downloading pretrained de_DE-thorsten-medium checkpoint..."
  wget -q --show-progress \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/de/de_DE/thorsten/medium/epoch%3D3135-step%3D2702056.ckpt" \
    -O "${PRETRAINED_CKPT}"

  # Verify download succeeded
  if [[ $(stat -c%s "${PRETRAINED_CKPT}" 2>/dev/null || echo 0) -lt 1000 ]]; then
    echo "ERROR: Checkpoint download failed (file is empty/tiny)." >&2
    echo "Download manually and transfer via: runpodctl send/receive" >&2
    rm -f "${PRETRAINED_CKPT}"
    exit 1
  fi

  # Strip incompatible hyperparameters from the pretrained checkpoint
  echo "Patching checkpoint for compatibility..."
  python3 -c "
import pathlib, torch
torch.serialization.add_safe_globals([pathlib.PosixPath])
ckpt = torch.load('${PRETRAINED_CKPT}', weights_only=False, map_location='cpu')
ckpt.pop('hyper_parameters', None)
torch.save(ckpt, '${PRETRAINED_CKPT}')
print('Checkpoint patched successfully')
"
fi

if [[ -f "${PRETRAINED_CFG}" ]]; then
  echo "Pretrained config already downloaded, skipping."
else
  wget -q \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/de/de_DE/thorsten/medium/config.json" \
    -O "${PRETRAINED_CFG}"
fi
echo

mkdir -p /workspace/cache /workspace/lightning_logs

echo "=== Setup complete ==="
echo "Upload your corpus and run training:"
echo "  # Upload voice_trainer_output/metadata.csv and wavs_processed/ (or wavs/)"
echo "  ./train.sh"
