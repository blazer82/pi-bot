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
from pi_bot.tts import speak
from pi_bot.stt import transcribe, warmup
from pi_bot.audio import (
    calibrate_noise_floor,
    listen_for_wake_word,
    record_until_silence,
    wait_for_followup,
)
from pi_bot.chat import chat_with_ollama

# jokes.json lives in the repository root, one level above this file
_REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Pi-Bot starting up...")

    print("Loading wake word model...")
    wake_model = WakeModel(wakeword_model_paths=[CONFIG["wake_model"]])

    print("Loading whisper model...")
    whisper_model = WhisperModel(CONFIG["whisper_model"])
    warmup(whisper_model)

    print("Calibrating noise floor...")
    calibrate_noise_floor()

    with open(os.path.join(_REPO_DIR, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    no_hear = (
        "Ich habe nichts verstanden."
        if CONFIG["language"] == "de"
        else "I didn't catch that."
    )
    err_msg = (
        "Es gab einen Fehler."
        if CONFIG["language"] == "de"
        else "There was an error."
    )

    ready_msg = "Pi Bot ist bereit." if CONFIG["language"] == "de" else "Pi Bot is ready."
    print(ready_msg)
    speak(ready_msg)

    while True:
        try:
            print("Listening for wake word...")
            listen_for_wake_word(wake_model)
            print("Wake word detected!")

            # Fresh conversation context per wake-word activation
            conversation_history = []

            ack = "Ja?" if CONFIG["language"] == "de" else "Yes?"
            speak(ack)

            # --- First utterance ---
            print("Recording...")
            audio = record_until_silence()
            duration = len(audio) / CONFIG["sample_rate"]
            print(f"Recorded {duration:.1f}s of audio")

            print("Transcribing...")
            text = transcribe(whisper_model, audio)
            print(f"User: {text}")

            if not text.strip():
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

                print("Transcribing...")
                text = transcribe(whisper_model, audio)
                print(f"User: {text}")

                if not text.strip():
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
            speak(err_msg)


def chat_mode():
    """Text-input chat mode for development without microphone hardware."""
    with open(os.path.join(_REPO_DIR, "jokes.json"), "r", encoding="utf-8") as f:
        jokes_db = json.load(f)

    print("Pi-Bot chat mode")
    print('Type messages to chat. "reset" clears history, "exit" quits.')
    print()

    ready_msg = "Pi Bot ist bereit." if CONFIG["language"] == "de" else "Pi Bot is ready."
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
