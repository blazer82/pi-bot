#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "Error: no GPU detected. Start the RunPod pod with a GPU attached, or run this script inside the training container." >&2
  exit 1
fi

DATASET_DIR="voice_trainer_output"
if [[ ! -f "${DATASET_DIR}/metadata.csv" ]]; then
  echo "Error: ${DATASET_DIR}/metadata.csv not found." >&2
  echo "Upload your post-processed corpus to /workspace/pi-bot/${DATASET_DIR}/ before running training (runpodctl send / scp / RunPod volume)." >&2
  exit 1
fi

AUDIO_DIR="${DATASET_DIR}/wavs_processed"
[[ -d "${AUDIO_DIR}" ]] || AUDIO_DIR="${DATASET_DIR}/wavs"
if [[ ! -d "${AUDIO_DIR}" ]] || ! compgen -G "${AUDIO_DIR}/*.wav" >/dev/null; then
  echo "Error: no WAV files in ${DATASET_DIR}/wavs_processed/ or ${DATASET_DIR}/wavs/." >&2
  exit 1
fi

WAV_COUNT=$(find "${AUDIO_DIR}" -maxdepth 1 -name '*.wav' | wc -l)
META_COUNT=$(grep -c . "${DATASET_DIR}/metadata.csv")
echo "Dataset: ${WAV_COUNT} WAVs, ${META_COUNT} metadata entries"
echo "Audio dir: ${AUDIO_DIR}"
echo

exec python -m voice_trainer train "$@"
