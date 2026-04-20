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

# System dependencies
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq espeak-ng ffmpeg > /dev/null
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
pip install -e ".[train]" -q
pip install pytorch-lightning==2.1.0 -q
bash build_monotonic_align.sh
cd /workspace
echo "Done."
echo

# Voice trainer dependencies (XTTS corpus generation, post-processing)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/requirements-voice-trainer.txt" ]]; then
  echo "Installing voice trainer dependencies..."
  pip install -r "${SCRIPT_DIR}/requirements-voice-trainer.txt" -q
  echo "Done."
  echo
fi

# Pretrained checkpoint
mkdir -p "${CHECKPOINTS_DIR}"
if [[ -f "${PRETRAINED_CKPT}" ]]; then
  echo "Pretrained checkpoint already downloaded, skipping."
else
  echo "Downloading pretrained de_DE-thorsten-medium checkpoint..."
  wget -q --show-progress \
    "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/de/de_DE/thorsten/medium/epoch%3D3135-step%3D2702056.ckpt" \
    -O "${PRETRAINED_CKPT}"
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
echo "Next: upload your speaker WAV and run the full pipeline:"
echo "  python3 -m voice_trainer generate --speaker-wav your_voice.wav"
echo "  python3 -m voice_trainer postprocess"
echo "  ./train.sh"
