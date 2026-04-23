"""Main loop and CLI entry point."""

import json
import os

try:
    from openwakeword.model import Model as WakeModel
    from pywhispercpp.model import Model as WhisperModel
except ImportError:
    WakeModel = None
    WhisperModel = None

from pi_bot.config import CONFIG
from pi_bot.tts import speak, _check_piper
from pi_bot.stt import transcribe, warmup
from pi_bot.audio import (
    open_mic,
    close_mic,
    listen_for_wake_word,
    record_until_silence,
    wait_for_followup,
)
from pi_bot.chat import chat_with_ollama, warmup_ollama
from pi_bot.cues import play as play_cue, start_loop, stop_loop

# jokes.json lives in the repository root, one level above this file
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Pi-Bot starting up...")

    print("Loading wake word model...")
    wake_model = WakeModel(wakeword_model_paths=[CONFIG["wake_model"]])

    print("Loading whisper model...")
    whisper_model = WhisperModel(CONFIG["whisper_model"])
    warmup(whisper_model)

    print("Opening mic stream...")
    open_mic()

    print("Warming up Ollama...")
    warmup_ollama()

    with open(os.path.join(_REPO_DIR, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    no_hear = "Ich habe nichts verstanden."
    err_msg = "Es gab einen Fehler."

    _check_piper()

    ready_msg = "Pi Bot ist bereit."
    print(ready_msg)
    speak(ready_msg)

    try:
        while True:
            try:
                print("Listening for wake word...")
                listen_for_wake_word(wake_model)
                print("Wake word detected!")

                # Fresh conversation context per wake-word activation
                conversation_history = []

                speak("Ja?")

                # --- First utterance ---
                print("Recording...")
                audio = record_until_silence()
                duration = len(audio) / CONFIG["sample_rate"]
                print(f"Recorded {duration:.1f}s of audio")

                play_cue("ack")
                start_loop("thinking")

                print("Transcribing...")
                text = transcribe(whisper_model, audio)
                print(f"User: {text}")

                if not text.strip():
                    stop_loop()
                    speak(no_hear)
                    continue

                print("Thinking...")
                response, end = chat_with_ollama(text, conversation_history, jokes_db)
                print(f"Pi-Bot: {response}")

                if end:
                    print("Conversation ended by bot.")
                    continue

                # --- Follow-up loop ---
                while True:
                    print("Listening for follow-up...")
                    audio = wait_for_followup()
                    if audio is None:
                        print("No follow-up detected, returning to wake word.")
                        break

                    duration = len(audio) / CONFIG["sample_rate"]
                    print(f"Recorded {duration:.1f}s of audio")

                    play_cue("ack")
                    start_loop("thinking")

                    print("Transcribing...")
                    text = transcribe(whisper_model, audio)
                    print(f"User: {text}")

                    if not text.strip():
                        stop_loop()
                        continue  # empty transcription — keep listening for follow-up

                    print("Thinking...")
                    response, end = chat_with_ollama(
                        text, conversation_history, jokes_db)
                    print(f"Pi-Bot: {response}")

                    if end:
                        print("Conversation ended by bot.")
                        break

            except KeyboardInterrupt:
                print("\nShutting down.")
                break
            except Exception as e:
                print(f"Error: {e}")
                stop_loop()
                play_cue("error")
                speak(err_msg)
    finally:
        close_mic()


def chat_mode():
    """Text-input chat mode for development without microphone hardware."""
    with open(os.path.join(_REPO_DIR, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    print("Pi-Bot chat mode")
    print('Type messages to chat. "reset" clears history, "exit" quits.')
    print()

    print("Warming up Ollama...")
    warmup_ollama()

    _check_piper()

    ready_msg = "Pi Bot ist bereit."
    print(ready_msg)
    speak(ready_msg)

    conversation_history = []

    try:
        while True:
            try:
                text = input("> ")
            except EOFError:
                break

            if not text.strip():
                continue

            cmd = text.strip().lower()
            if cmd in ("exit", "quit"):
                break
            if cmd == "reset":
                conversation_history = []
                print("Conversation history cleared.")
                continue

            response, end = chat_with_ollama(text, conversation_history, jokes_db)
            print(f"Pi-Bot: {response}\n")

            if end:
                break
    except KeyboardInterrupt:
        pass

    print("\nBye.")


def cli():
    """CLI entry point with argparse."""
    import argparse

    parser = argparse.ArgumentParser(description="Pi-Bot")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Text-input chat mode (no mic/wake word needed)",
    )
    args = parser.parse_args()

    if args.chat:
        chat_mode()
    else:
        main()


if __name__ == "__main__":
    cli()
