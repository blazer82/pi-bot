#!/usr/bin/env bash
set -euo pipefail

# Use the training venv if on RunPod (separate from generation venv)
VENV_TRAIN="/workspace/venv-train"
if [[ -f "${VENV_TRAIN}/bin/python3" ]]; then
  PYTHON="${VENV_TRAIN}/bin/python3"
else
  PYTHON="python3"
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "Error: no GPU detected. Start the RunPod pod with a GPU attached." >&2
  exit 1
fi

if ! "${PYTHON}" -m piper.train --help >/dev/null 2>&1; then
  echo "Error: piper.train not installed. Run setup_runpod.sh first." >&2
  exit 1
fi

DATASET_DIR="voice_trainer_output"
if [[ ! -f "${DATASET_DIR}/metadata.csv" ]]; then
  echo "Error: ${DATASET_DIR}/metadata.csv not found." >&2
  echo "Run corpus generation first, or upload your corpus to /workspace/pi-bot/${DATASET_DIR}/." >&2
  exit 1
fi

AUDIO_DIR="${DATASET_DIR}/wavs_processed"
[[ -d "${AUDIO_DIR}" ]] || AUDIO_DIR="${DATASET_DIR}/wavs"
if [[ ! -d "${AUDIO_DIR}" ]] || ! compgen -G "${AUDIO_DIR}/*.wav" >/dev/null; then
  echo "Error: no WAV files in ${DATASET_DIR}/wavs_processed/ or ${DATASET_DIR}/wavs/." >&2
  exit 1
fi

PRETRAINED="/workspace/checkpoints/pretrained.ckpt"
if [[ ! -f "${PRETRAINED}" ]]; then
  echo "Error: pretrained checkpoint not found at ${PRETRAINED}." >&2
  echo "Run setup_runpod.sh first, or download manually." >&2
  exit 1
fi

WAV_COUNT=$(find "${AUDIO_DIR}" -maxdepth 1 -name '*.wav' | wc -l)
META_COUNT=$(grep -c . "${DATASET_DIR}/metadata.csv")
echo "Dataset: ${WAV_COUNT} WAVs, ${META_COUNT} metadata entries"
echo "Audio dir: ${AUDIO_DIR}"
echo

exec "${PYTHON}" -m voice_trainer train --pretrained-checkpoint "${PRETRAINED}" "$@"
