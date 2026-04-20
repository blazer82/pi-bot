# Voice Trainer

Train a custom Piper TTS voice for Pi-Bot. The pipeline uses Coqui XTTS to generate a large synthetic corpus, applies audio effects to shape the voice character, and fine-tunes a Piper model from a pretrained German checkpoint that drops into Pi-Bot as an ONNX file.

**Run this on a desktop/GPU machine, not the Pi.** Requires **Python 3.10 or 3.11** (Coqui TTS needs >=3.9 but <3.12).

## Setup

Create a separate venv with Python 3.11:

```bash
python3.11 -m venv venv-voice-trainer
source venv-voice-trainer/bin/activate
pip install -r requirements-voice-trainer.txt
```

Training (Step 4) runs inside Docker — no local Piper install needed.

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

#### Downloading more sentences

The built-in 200 sentences are great for quick tests. For production quality, download a larger corpus from [Tatoeba](https://tatoeba.org/) (CC BY 2.0 FR):

```bash
# Download ~1000 CC0-licensed German sentences from Tatoeba
python -m voice_trainer download-sentences

# Download more sentences
python -m voice_trainer download-sentences --count 2000

# Custom output path
python -m voice_trainer download-sentences --output my_sentences.txt
```

The downloaded sentences are filtered for TTS suitability (length, punctuation, no URLs). Use `--seed` for reproducible selection.

#### Generating the corpus

```bash
# Quick test with 10 sentences (built-in corpus)
python -m voice_trainer generate --max-sentences 10

# Full generation with voice cloning
python -m voice_trainer generate --speaker-wav my_voice.wav

# Use downloaded or custom sentences
python -m voice_trainer generate --sentences voice_trainer/data/sentences_tatoeba_de.txt
```

The output lands in `voice_trainer_output/` with:

- `wavs/` — individual WAV files per sentence
- `metadata.csv` — LJSpeech-format mapping (`file_id|sentence`)
- `concat_full.wav` — all clips concatenated with 1.5s silence gaps (for DAW processing)
- `concat_markers.json` — timestamps and sentence text for each clip in the concatenated WAV

Generation supports **resuming** — if interrupted, rerun the same command and it skips existing files.

**Trailing artifact trimming** is enabled by default. XTTS sometimes appends garbled/reversed audio at the end of clips — the generator uses spectral flatness analysis and text-length-based duration capping to detect and trim these artifacts before writing.

For even more variety, you can also generate domain-specific sentences with Claude (conversational, weather reports, jokes, etc.) or extract transcripts from the [CSS10 German dataset](https://github.com/Kyubyong/css10).

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

Needs a GPU. Use a cloud instance (RunPod, Vast.ai) if you don't have one locally.

Training fine-tunes the pretrained `de_DE-thorsten-medium` checkpoint from rhasspy/piper-checkpoints. The checkpoint is baked into the Docker image — no manual download needed.

```bash
# Start fine-tuning (pretrained checkpoint provided by train.sh)
./train.sh

# Custom hyperparameters
./train.sh --batch-size 16 --max-epochs 2000

# Resume interrupted training
./train.sh --resume-from /workspace/lightning_logs/version_0/checkpoints/last.ckpt
```

When you're happy with a checkpoint, export to ONNX on the training machine:

```bash
python3 -m voice_trainer train --export path/to/checkpoint.ckpt
```

Then download the `.onnx` file and the checkpoint's `config.json` (rename to `<model>.onnx.json`), place both in `models/piper/`, and update `CONFIG["piper_model"]` in `pi_bot/config.py`.

#### RunPod

On RunPod, create a pod with a **PyTorch** template. Pick a 30xx/40xx series or A4000/A5000 GPU — a 4090 is usually cheapest. Fine-tuning ~1000 epochs should take well under an hour for a 15-minute corpus.

RunPod mounts a persistent volume at `/workspace/`. Clone the repo and run the setup script (once per pod):

```bash
cd /workspace
git clone <your-repo-url> pi-bot
cd pi-bot

# One-time setup — creates two venvs, downloads pretrained checkpoint
bash setup_runpod.sh
```

The setup script creates two isolated venvs (Coqui XTTS and piper1-gpl can't coexist):

| Venv | Path | Used by |
|------|------|---------|
| Generation | `/workspace/venv-generate` | `xtts-setup`, `generate`, `postprocess`, `split` |
| Training | `/workspace/venv-train` | `train` |

`python3 -m voice_trainer` automatically activates the correct venv for each command — no manual `source activate` needed.

The setup script is idempotent — if a pod restarts, rerunning it skips already-completed steps. On persistent volumes, both venvs and the checkpoint survive restarts.

Run the full pipeline on RunPod:

```bash
# Download sentences + generate corpus
python3 -m voice_trainer download-sentences
python3 -m voice_trainer generate --speaker-wav your_voice.wav

# Post-process and train
python3 -m voice_trainer postprocess
./train.sh
./train.sh --batch-size 16 --max-epochs 2000  # custom hyperparameters
```

`train.sh` checks that a GPU is visible, piper.train is installed, the corpus is present, and the pretrained checkpoint exists.

## Tips

- **Start small.** Generate 15 minutes first, train a quick model, listen. Scale to 2 hours once the voice direction is right.
- **Sample rate.** Everything stays at 22050Hz throughout — that's what Piper expects.
- **XTTS is slow.** Expect ~2-5x realtime on GPU. A 2-hour corpus can take 4-10 hours to generate.
- **Post-processing is iterative.** Try different effect combinations. Subtle changes make a big difference.
- **Fine-tuning converges fast.** Starting from the thorsten-medium checkpoint, expect usable results in ~500-1500 epochs vs 10,000+ from scratch.
