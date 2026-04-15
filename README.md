# Pi-Bot

A voice-activated conversational robot running on Raspberry Pi 5. Uses openWakeWord for wake word detection, whisper.cpp for speech-to-text, ollama (Gemma 4) for conversation, and espeak-ng for robotic text-to-speech. Supports German and English.

## Hardware

- Raspberry Pi 5, 16 GB RAM
- USB microphone
- USB audio interface + speakers
- Raspberry Pi OS Bookworm (64-bit)

## Quick Start

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
python3 pi_bot.py
```

Say **"Hey Jarvis"** to activate, then speak your question. Pi-Bot responds via the speaker.

To keep the Ollama model loaded in memory for faster responses, run:

```bash
ollama run gemma4:e2b-it-q4_K_M --keepalive "-1m"
```

Then type `/bye` to exit the interactive session — the model stays resident.

## How It Works

```
[Mic] → Wake word detection → Record speech → Whisper STT
  → Ollama (Gemma 4) → Tool calls if needed → espeak-ng TTS → [Speaker]
```

1. Continuously listens for the wake word ("Hey Jarvis")
2. On detection, plays an acknowledgment ("Ja?") and records until silence
3. Transcribes speech with whisper.cpp (small model)
4. Sends transcript to Gemma 4 via ollama, with tool definitions
5. If the model calls a tool (e.g., joke lookup), executes it and feeds results back
6. Speaks the final response via espeak-ng

## Configuration

Edit the `CONFIG` dict at the top of `pi_bot.py`:

| Key                  | Default                  | Description                                  |
| -------------------- | ------------------------ | -------------------------------------------- |
| `language`           | `"de"`                   | `"de"` for German, `"en"` for English        |
| `ollama_model`       | `"gemma4:e2b-it-q4_K_M"` | Ollama model tag                             |
| `whisper_model`      | `"small"`                | Whisper model size (tiny/base/small/medium)  |
| `wake_word`          | `"hey_jarvis"`           | openWakeWord model name                      |
| `wake_threshold`     | `0.5`                    | Wake word confidence threshold (0.0-1.0)     |
| `silence_threshold`  | `500`                    | RMS energy below this = silence              |
| `silence_duration`   | `1.5`                    | Seconds of silence before stopping recording |
| `max_record_seconds` | `15`                     | Maximum recording length                     |
| `thinking`           | `False`                  | Enable `<think>` reasoning in ollama         |
| `espeak_speed`       | `130`                    | Speech rate (words per minute)               |
| `espeak_pitch`       | `40`                     | Pitch (0-99, lower = deeper)                 |
| `mic_device`         | `None`                   | Microphone device index (None = default)     |
| `speaker_device`     | `None`                   | Speaker device index (None = default)        |

## USB Audio Setup

Pi-Bot uses two separate USB audio devices: one for the microphone and one for the speaker. After running `setup.sh`, follow these steps to configure them.

### 1. Identify your devices

The setup script prints detected audio devices at the end. You can also run:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Example output:

```
  0  USB Mic Device: Audio (hw:1,0), ALSA (1 in, 0 out)
  1  USB Speaker Device: Audio (hw:2,0), ALSA (0 in, 2 out)
  2  bcm2835 Headphones: - (hw:0,0), ALSA (0 in, 2 out)
```

The mic is the USB device with input channels, the speaker is the one with output channels. Note their **index numbers**.

### 2. Configure Pi-Bot

Edit `pi_bot/config.py` and set the device indices in the `CONFIG` dict:

```python
"mic_device": 0,       # replace with your mic's index
"speaker_device": 1,   # replace with your speaker's index
```

### 3. Route espeak-ng to the USB speaker

espeak-ng uses ALSA directly, not the Python sounddevice library, so it needs its own configuration. First, find your USB speaker's ALSA card number:

```bash
aplay -l
```

Then create or edit `~/.asoundrc`:

```
pcm.!default {
    type hw
    card 2
}
```

Replace `card 2` with your USB speaker's card number from `aplay -l`. Note that ALSA card numbers and sounddevice index numbers are often different.

### 4. Test audio

```bash
# Test speaker (espeak-ng)
espeak-ng -v de "Hallo, ich bin Pi Bot."

# Test mic (record 3 seconds and play back)
arecord -d 3 -D hw:1,0 test.wav && aplay test.wav
```

Replace `hw:1,0` with your mic's ALSA hardware address from `arecord -l`.

### Troubleshooting

- If you hear no sound from espeak-ng, double-check the card number in `~/.asoundrc` matches `aplay -l`.
- If the bot doesn't detect speech, verify `CONFIG["mic_device"]` matches the correct sounddevice index and that `arecord` picks up audio from your mic.
- If devices change index after a reboot, unplug and replug them in the same order, or use `udev` rules to assign stable names.

## Adding Jokes

Edit `jokes.json`. Each joke has this structure:

```json
{
  "id": 121,
  "setup": "The setup line",
  "punchline": "The punchline"
}
```

## Development on macOS

You can run Pi-Bot in text-input chat mode on a Mac, bypassing all microphone and wake word hardware. This lets you test the actual Ollama conversation pipeline and TTS output.

### Prerequisites

- [Ollama](https://ollama.com/) installed and running (`ollama serve`)
- An Ollama model pulled (e.g. `ollama pull gemma4:e2b-it-q4_K_M`)
- espeak-ng: `brew install espeak-ng`

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy requests pytest
```

The hardware-specific packages (`sounddevice`, `openwakeword`, `pywhispercpp`) are not needed for chat mode and are skipped.

### Run chat mode

```bash
source venv/bin/activate
python3 pi_bot.py --chat
```

Type messages to chat with the bot. Commands:

- `reset` — clear conversation history (like a new wake-word activation)
- `exit` / `quit` / Ctrl+C — stop

## Testing

```bash
python3 -m pip install pytest
python3 -m pytest test_pi_bot.py -v
```

All hardware and API dependencies are mocked, so tests run without a microphone, speaker, ollama, or espeak-ng.

## Adding Tools

1. Add a tool definition to the `TOOLS` list in `pi_bot.py` (ollama JSON schema format)
2. Write the tool function
3. Add a dispatch case in `execute_tool()`
4. Mention the tool in the system prompt so the model knows to use it

## Stack

| Component                                                              | Purpose                         |
| ---------------------------------------------------------------------- | ------------------------------- |
| [openWakeWord](https://github.com/dscripka/openWakeWord)               | Wake word detection             |
| [whisper.cpp](https://github.com/ggerganov/whisper.cpp) / pywhispercpp | Speech-to-text                  |
| [ollama](https://ollama.com/) + Gemma 4                                | LLM inference with tool calling |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng)                    | Text-to-speech (robotic voice)  |
