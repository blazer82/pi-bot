# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pi-Bot is a voice-activated conversational robot for Raspberry Pi 5. It uses openWakeWord for wake word detection, whisper.cpp for STT, Ollama (Gemma 4) for LLM inference with tool calling, and espeak-ng for TTS. Supports German and English.

## Commands

```bash
# Setup (first time)
chmod +x setup.sh && ./setup.sh

# Activate virtualenv
source venv/bin/activate

# Run the bot
python3 pi_bot.py

# Run all tests
python3 -m pytest test_pi_bot.py -v

# Run a single test class or method
python3 -m pytest test_pi_bot.py::TestTranscribe -v
python3 -m pytest test_pi_bot.py::TestTranscribe::test_basic_transcription -v

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Architecture

This is a single-file Python application (`pi_bot.py`) with no framework overhead.

**Pipeline:** Mic -> Wake word detection -> Record until silence -> Whisper STT -> Ollama chat (streaming) -> Tool execution loop (up to 3 rounds) -> espeak-ng TTS -> Speaker

**Key flow in `main()`:**
1. `listen_for_wake_word()` blocks until "Hey Jarvis" is detected
2. `record_until_silence()` captures the user's speech
3. `transcribe()` converts audio to text via whisper.cpp
4. `chat_with_ollama()` orchestrates the LLM conversation:
   - `_ollama_chat_stream()` calls the Ollama `/api/chat` REST endpoint with streaming
   - `stream_and_speak()` processes the streamed response, strips `<think>` tags, and speaks sentence-by-sentence as they arrive
   - Tool calls are executed via `execute_tool()` and results fed back for up to 3 rounds
5. `wait_for_followup()` listens for a follow-up utterance (two-phase: detect speech onset, then record until silence)

**Configuration:** All tunables live in the `CONFIG` dict at the top of `pi_bot.py` (language, model, thresholds, audio devices, TTS parameters). System prompts are defined separately for German (`SYSTEM_PROMPT_DE`) and English (`SYSTEM_PROMPT_EN`).

**Conversation context:** Rolling history capped at `context_window * 2` messages (default 12 = 6 turns), reset on each new wake-word activation.

## Testing

All hardware and external dependencies (sounddevice, openwakeword, pywhispercpp, espeak-ng, Ollama API) are mocked at the module level in `test_pi_bot.py` before `pi_bot` is imported. Tests run without any physical hardware or services.

**Adding tools:** Add tool definition to `TOOLS` list (Ollama JSON schema format), write the function, add dispatch in `execute_tool()`, and mention the tool in the system prompt.
