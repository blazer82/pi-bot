"""CLI entry point with subcommands for the voice training pipeline."""

import argparse
import sys

from voice_trainer.config import TRAINER_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice_trainer",
        description="Voice training pipeline for Pi-Bot custom Piper voices",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # xtts-setup
    p1 = sub.add_parser("xtts-setup", help="Set up XTTS and test voice synthesis")
    p1.add_argument("--speaker-wav", help="Reference WAV for voice cloning")
    p1.add_argument(
        "--test-text",
        default="Hallo, ich bin Pi-Bot und freue mich, dich kennenzulernen.",
    )
    p1.add_argument(
        "--list-speakers", action="store_true", help="List available built-in speakers"
    )

    # generate
    p2 = sub.add_parser("generate", help="Generate corpus WAVs via XTTS")
    p2.add_argument("--sentences", help="Path to sentences file (one per line)")
    p2.add_argument("--output-dir", help="Output directory")
    p2.add_argument("--speaker-wav", help="Reference WAV for voice cloning")
    p2.add_argument("--max-sentences", type=int, help="Limit number of sentences")

    # postprocess
    p3 = sub.add_parser("postprocess", help="Apply audio effects to generated WAVs")
    p3.add_argument("--input-dir", help="Input wavs directory")
    p3.add_argument("--output-dir", help="Output directory for processed WAVs")
    p3.add_argument("--pitch", type=float, help="Pitch shift in semitones")
    p3.add_argument("--bitcrush", type=int, help="Bitcrush bit depth")
    p3.add_argument("--lowpass", type=float, help="Lowpass cutoff frequency in Hz")

    # train
    p4 = sub.add_parser("train", help="Train Piper model from processed dataset")
    p4.add_argument("--dataset-dir", help="LJSpeech-format dataset directory")
    p4.add_argument("--batch-size", type=int, help="Training batch size")
    p4.add_argument("--max-epochs", type=int, help="Maximum training epochs")
    p4.add_argument("--resume-from", help="Checkpoint path to resume from")
    p4.add_argument("--export", help="Export checkpoint to ONNX (pass checkpoint path)")
    p4.add_argument(
        "--install",
        action="store_true",
        help="Copy exported model to models/piper/",
    )

    return parser


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "xtts-setup":
        from voice_trainer.xtts_setup import run

        run(args, TRAINER_CONFIG)

    elif args.command == "generate":
        from voice_trainer.generate import run

        run(args, TRAINER_CONFIG)

    elif args.command == "postprocess":
        from voice_trainer.postprocess import run

        run(args, TRAINER_CONFIG)

    elif args.command == "train":
        from voice_trainer.train import run

        run(args, TRAINER_CONFIG)


def main() -> None:
    args = build_parser().parse_args()
    dispatch(args)
