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

Say **"Hey Pee Bot"** to activate, then speak your question. Pi-Bot responds via the speaker.

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

1. Continuously listens for the wake word ("Hey Pee Bot")
2. On detection, plays an acknowledgment ("Ja?") and records until silence
3. Transcribes speech with whisper.cpp (base model)
4. Sends transcript to Gemma 4 via ollama, with tool definitions
5. If the model calls a tool (e.g., joke lookup), executes it and feeds results back
6. Speaks the final response via espeak-ng

## Configuration

Edit the `CONFIG` dict at the top of `pi_bot.py`:

| Key                  | Default                     | Description                                  |
| -------------------- | --------------------------- | -------------------------------------------- |
| `language`           | `"de"`                      | `"de"` for German, `"en"` for English        |
| `ollama_model`       | `"gemma4:e2b-it-q4_K_M"`    | Ollama model tag                             |
| `whisper_model`      | `"base"`                    | Whisper model size (tiny/base/small/medium)  |
| `wake_model`         | `"models/hey_pee_bot.onnx"` | Path to custom openWakeWord `.onnx` model    |
| `wake_threshold`     | `0.5`                       | Wake word confidence threshold (0.0-1.0)     |
| `silence_threshold`  | `500`                       | RMS energy below this = silence              |
| `silence_duration`   | `1.5`                       | Seconds of silence before stopping recording |
| `max_record_seconds` | `15`                        | Maximum recording length                     |
| `thinking`           | `False`                     | Enable `<think>` reasoning in ollama         |
| `espeak_voice`       | `"mb-de4"`                  | MBROLA voice or espeak-ng language code      |
| `espeak_speed`       | `130`                       | Speech rate (words per minute)               |
| `espeak_pitch`       | `40`                        | Pitch (0-99, lower = deeper)                 |
| `mic_device`         | `None`                      | Microphone device index (None = default)     |
| `speaker_device`     | `None`                      | Speaker device index (None = default)        |

## USB Audio Setup

Pi-Bot uses two separate USB audio devices: one for the microphone and one for the speaker. The setup script installs PipeWire for audio routing, which is more reliable than raw ALSA for USB device management.

**Important:** After running `setup.sh` for the first time, reboot so PipeWire services start properly:

```bash
sudo reboot
```

### 1. Identify your devices

After rebooting, check PipeWire devices:

```bash
wpctl status
```

Look for your USB mic under **Sources** and your USB speaker under **Sinks**. Note their ID numbers (the number in the leftmost column).

You can also check sounddevice indices (used by the Python code):

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

### 2. Set default audio devices

Tell PipeWire to use your USB devices as defaults:

```bash
wpctl set-default <sink-id>     # your USB speaker's sink ID
wpctl set-default <source-id>   # your USB mic's source ID
```

This routes all audio (including espeak-ng) through PipeWire automatically — no `~/.asoundrc` needed.

### 3. Configure Pi-Bot

Edit `pi_bot/config.py` and set the sounddevice indices in the `CONFIG` dict:

```python
"mic_device": 0,       # replace with your mic's sounddevice index
"speaker_device": 1,   # replace with your speaker's sounddevice index
```

### 4. Test audio

```bash
# Test speaker (espeak-ng, routed through PipeWire)
espeak-ng -v de "Hallo, ich bin Pi Bot."

# Test mic (record 3 seconds and play back)
pw-record --rate 16000 --channels 1 test.wav
pw-play test.wav
```

### Troubleshooting

- If you hear no sound from espeak-ng, verify PipeWire is running (`systemctl --user status pipewire`) and the correct sink is set as default (`wpctl status`).
- If the bot doesn't detect speech, verify `CONFIG["mic_device"]` matches the correct sounddevice index and that `pw-record` picks up audio from your mic.
- If PipeWire isn't running over SSH, make sure `loginctl enable-linger` was set during setup (the setup script does this automatically).
- **ALSA fallback:** If PipeWire causes issues, you can route espeak-ng directly via ALSA. Find your speaker's card number with `aplay -l`, then create `~/.asoundrc`:
  ```
  pcm.!default {
      type hw
      card 2
  }
  ```
  Replace `card 2` with your speaker's ALSA card number.

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
