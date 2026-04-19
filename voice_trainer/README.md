# Voice Trainer

Train a custom Piper TTS voice for Pi-Bot. The pipeline uses Coqui XTTS to generate a large synthetic corpus, applies audio effects to shape the voice character, and trains a Piper model that drops into Pi-Bot as an ONNX file.

**Run this on a desktop/GPU machine, not the Pi.** Requires **Python 3.10 or 3.11** (Coqui TTS needs >=3.9 but <3.12).

## Setup

Create a separate venv with Python 3.11:

```bash
python3.11 -m venv venv-voice-trainer
source venv-voice-trainer/bin/activate
pip install -r requirements-voice-trainer.txt
```

For training (Step 4), you also need the Piper training environment:

```bash
git clone https://github.com/rhasspy/piper
cd piper/src/python
pip install -e ".[train]"
```

And `espeak-ng` for phonemization:

```bash
# macOS
brew install espeak-ng

# Ubuntu/Debian
sudo apt install espeak-ng
```

## Pipeline

### Step 1: Set up XTTS

Test that XTTS loads and synthesizes correctly. First run downloads the model (~2GB).

```bash
# Test with default text
python -m voice_trainer xtts-setup

# Clone a voice from a reference recording
python -m voice_trainer xtts-setup --speaker-wav my_voice.wav

# List available built-in speakers
python -m voice_trainer xtts-setup --list-speakers
```

### Step 2: Generate corpus

Batch-render sentences to WAV in LJSpeech format. A starter corpus of ~200 German sentences is included — enough for a ~15-minute test run. For a full voice, use 1000+ sentences (~1-2 hours of audio).

```bash
# Quick test with 10 sentences
python -m voice_trainer generate --max-sentences 10

# Full generation with voice cloning
python -m voice_trainer generate --speaker-wav my_voice.wav

# Use a custom sentence file
python -m voice_trainer generate --sentences my_sentences.txt
```

The output lands in `voice_trainer_output/` with:

- `wavs/` — individual WAV files per sentence
- `metadata.csv` — LJSpeech-format mapping (`file_id|sentence`)
- `concat_full.wav` — all clips concatenated with 1.5s silence gaps (for DAW processing)
- `concat_markers.json` — timestamps and sentence text for each clip in the concatenated WAV

Generation supports **resuming** — if interrupted, rerun the same command and it skips existing files.

**Trailing artifact trimming** is enabled by default. XTTS sometimes appends garbled/reversed audio at the end of clips — the generator uses spectral flatness analysis and text-length-based duration capping to detect and trim these artifacts before writing.

Larger sentence sources:

- [CSS10 German dataset](https://github.com/Kyubyong/css10) — thousands of read sentences
- Generate domain-specific sentences with Claude (conversational, weather reports, jokes, etc.)

### Step 3: Post-process

Two options for shaping the voice character:

**Option A: Manual DAW processing (recommended)**

Open `concat_full.wav` in your DAW (Logic, Audacity, etc.), apply effects, EQ, and cleanup across the entire corpus at once. Export as a single WAV, then split back into individual training clips:

```bash
python -m voice_trainer split processed_from_daw.wav

# Custom markers file or output directory
python -m voice_trainer split processed.wav --markers path/to/concat_markers.json --output-dir my_output/
```

The split command detects silence gaps to find sentence boundaries, and validates against the original markers. If silence detection fails (e.g. effects filled the gaps), it falls back to marker-based timestamp splitting. Output goes to `voice_trainer_output/wavs_processed/`.

**Option B: Automated effects**

Apply a preset effects chain (pitch shift, bitcrush, lowpass filter):

```bash
# Default effects: pitch -2st, bitcrush 14bit, lowpass 7kHz
python -m voice_trainer postprocess

# Custom effects
python -m voice_trainer postprocess --pitch -3 --bitcrush 12 --lowpass 6000
```

Processed files go to `voice_trainer_output/wavs_processed/`.

Listen to a few samples after processing. If the voice doesn't sound right, adjust and rerun — this step is fast.

### Step 4: Train Piper model

Needs a GPU. Use a cloud instance (RunPod, Vast.ai, ~$2-5 for a full run) if you don't have one locally.

```bash
# Start training
python -m voice_trainer train

# Resume from checkpoint
python -m voice_trainer train --resume-from lightning_logs/version_0/checkpoints/last.ckpt

# Custom hyperparameters
python -m voice_trainer train --batch-size 16 --max-epochs 5000
```

When you're happy with a checkpoint, export and install:

```bash
# Export to ONNX
python -m voice_trainer train --export path/to/last.ckpt

# Export and install directly into Pi-Bot's models/piper/
python -m voice_trainer train --export path/to/last.ckpt --install
```

After installing, update `CONFIG["piper_model"]` in `pi_bot/config.py` to the new model name.

#### Docker / RunPod

The repo ships a `Dockerfile` and `train.sh` at the root so you can run Step 4 on a rented GPU (e.g. RunPod) without hand-installing CUDA, piper_train, and espeak-ng.

Build and push the image from your machine:

```bash
docker buildx build --platform linux/amd64 -t <dockerhub-user>/pi-bot-trainer --push .
```

On RunPod, create a pod with **Custom Container** (not a template) and paste the Docker Hub image name. Pick a 30xx/40xx series or A4000/A5000 GPU — a 4090 is usually cheapest and takes ~30–60 min for a 15-minute corpus. Avoid K80/P100/V100 (old CUDA).

RunPod mounts a persistent volume at `/workspace/`, so clone the repo there and upload your corpus:

```bash
cd /workspace
git clone <your-repo-url> pi-bot
cd pi-bot
# upload voice_trainer_output/metadata.csv and voice_trainer_output/wavs_processed/ (or wavs/)
```

**Important:** Piper's preprocessor expects audio in a `wavs/` directory. If you only have `wavs_processed/`, symlink it:

```bash
cd voice_trainer_output
ln -s wavs_processed wavs
```

Then run training:

```bash
./train.sh                                  # defaults
./train.sh --batch-size 16 --max-epochs 5000  # extra flags forwarded to voice_trainer
```

`train.sh` checks that a GPU is visible and that the corpus is present. The training pipeline automatically runs Piper's preprocessing (phonemization, `config.json` generation) if it hasn't been done yet.

Download `lightning_logs/version_<n>/checkpoints/last.ckpt` off the pod when you're happy with it, then run the export/install locally:

```bash
python -m voice_trainer train --export last.ckpt --install
```

## Tips

- **Start small.** Generate 15 minutes first, train a quick model, listen. Scale to 2 hours once the voice direction is right.
- **Sample rate.** Everything stays at 22050Hz throughout — that's what Piper expects.
- **XTTS is slow.** Expect ~2-5x realtime on GPU. A 2-hour corpus can take 4-10 hours to generate.
- **Post-processing is iterative.** Try different effect combinations. Subtle changes make a big difference.

## Note on piper_train

The original [rhasspy/piper](https://github.com/rhasspy/piper) repo has been archived and moved to [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl). `piper_train` pins PyTorch <2 and `pytorch-lightning~=1.7`, which is why this pipeline needs a custom Docker image rather than a standard RunPod template. The Dockerfile handles all necessary workarounds (dependency conflicts, cuFFT patch for RTX 40xx GPUs).

If training becomes unmaintainable, alternatives to consider: [veralvx/piper-train](https://github.com/veralvx/piper-train) (fork with PyTorch 2.5 support, same ONNX output), or [Kokoro-82M](https://github.com/hexgrad/kokoro) (lightweight, high-quality TTS — would need ONNX export testing for Pi 5).
