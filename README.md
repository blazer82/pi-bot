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
| `espeak_speed`       | `130`                    | Speech rate (words per minute)               |
| `espeak_pitch`       | `40`                     | Pitch (0-99, lower = deeper)                 |
| `mic_device`         | `None`                   | Microphone device index (None = default)     |
| `speaker_device`     | `None`                   | Speaker device index (None = default)        |

## Audio Device Troubleshooting

List available devices:

```bash
python3 -c "import sounddevice; print(sounddevice.query_devices())"
```

Set the device index in `CONFIG["mic_device"]` and `CONFIG["speaker_device"]` if the defaults are wrong.

For espeak-ng output routing, set the ALSA default device in `~/.asoundrc`:

```
pcm.!default {
    type hw
    card 1
}
```

Replace `card 1` with your USB audio interface's card number (find it with `aplay -l`).

## Adding Jokes

Edit `jokes.json`. Each joke has this structure:

```json
{
  "id": 121,
  "setup": "The setup line",
  "punchline": "The punchline"
}
```

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
