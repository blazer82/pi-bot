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
python3 -m pi_bot          # alternative

# Run all tests
python3 -m pytest tests/ -v

# Run a single test file or class
python3 -m pytest tests/test_tools.py -v
python3 -m pytest tests/test_chat.py::TestChatWithOllama -v

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Architecture

The application is structured as the `pi_bot/` package with focused modules:

```
pi_bot/
  config.py   — CONFIG dict, system prompts (DE/EN), TOOLS definitions
  tts.py      — speak() via espeak-ng
  stt.py      — transcribe() via whisper.cpp
  audio.py    — listen_for_wake_word(), record_until_silence(), wait_for_followup()
  tools.py    — tool implementations (weather, system status, jokes) + execute_tool() dispatcher
  chat.py     — Ollama streaming, sentence-by-sentence speech, chat_with_ollama() orchestration
  main.py     — main loop, chat_mode(), CLI entry point
```

The root `pi_bot.py` is a thin shim that calls `pi_bot.main.cli()`.

**Pipeline:** Mic -> Wake word detection -> Record until silence -> Whisper STT -> Ollama chat (streaming) -> Tool execution loop (up to 3 rounds) -> espeak-ng TTS -> Speaker

**Key flow in `main()`:**
1. `listen_for_wake_word()` blocks until "Hey Pee Bot" is detected
2. `record_until_silence()` captures the user's speech
3. `transcribe()` converts audio to text via whisper.cpp
4. `chat_with_ollama()` orchestrates the LLM conversation:
   - `_ollama_chat_stream()` calls the Ollama `/api/chat` REST endpoint with streaming
   - `stream_and_speak()` processes the streamed response, strips `<think>` tags, and speaks sentence-by-sentence as they arrive
   - Tool calls are executed via `execute_tool()` and results fed back for up to 3 rounds
5. `wait_for_followup()` listens for a follow-up utterance (two-phase: detect speech onset, then record until silence)

**Configuration:** All tunables live in `CONFIG` in `pi_bot/config.py` (language, model, thresholds, audio devices, TTS parameters). System prompts are defined separately for German (`SYSTEM_PROMPT_DE`) and English (`SYSTEM_PROMPT_EN`).

**Conversation context:** Rolling history capped at `context_turns * 2` messages (default 50), reset on each new wake-word activation.

## Testing

Tests live in `tests/` and mirror the package structure. Hardware dependencies (sounddevice, openwakeword, pywhispercpp) are mocked in `tests/conftest.py` before any `pi_bot` code is imported.

**Adding tools:** Add the tool schema to `TOOLS` in `pi_bot/config.py`, implement the function and add dispatch in `pi_bot/tools.py`, mention the tool in the system prompt, and add tests in `tests/test_tools.py`.
