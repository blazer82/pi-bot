"""Step 4: Piper model fine-tuning, export, and installation."""

import os
import shutil
import subprocess
import sys


def check_piper_install() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "piper.train", "--help"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _print_setup_instructions() -> None:
    print("piper.train is not installed. To set it up:\n")
    print("  git clone --branch v1.3.0 https://github.com/OHF-voice/piper1-gpl")
    print("  cd piper1-gpl")
    print('  pip install -e ".[train]"')
    print("  bash build_monotonic_align.sh")
    print("\nYou also need espeak-ng for phonemization:")
    print("  # macOS:  brew install espeak-ng")
    print("  # Ubuntu: sudo apt install espeak-ng")


def prepare_dataset(dataset_dir: str) -> str:
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    if not os.path.isfile(metadata_path):
        print(f"Error: metadata.csv not found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    processed_dir = os.path.join(dataset_dir, "wavs_processed")
    wavs_dir = os.path.join(dataset_dir, "wavs")
    audio_dir = processed_dir if os.path.isdir(processed_dir) else wavs_dir

    if not os.path.isdir(audio_dir):
        print(f"Error: no WAV directory found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    wav_count = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
    with open(metadata_path, encoding="utf-8") as f:
        line_count = sum(1 for line in f if line.strip())

    print(f"Dataset: {wav_count} WAVs, {line_count} metadata entries")
    print(f"Audio dir: {audio_dir}")

    return audio_dir


def run_training(dataset_dir: str, config: dict,
                 pretrained_ckpt: str | None = None,
                 resume_from: str | None = None) -> None:
    metadata_path = os.path.join(dataset_dir, "metadata.csv")

    processed_dir = os.path.join(dataset_dir, "wavs_processed")
    wavs_dir = os.path.join(dataset_dir, "wavs")
    audio_dir = processed_dir if os.path.isdir(processed_dir) else wavs_dir

    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.voice_name", config["piper_voice_name"],
        "--data.csv_path", metadata_path,
        "--data.audio_dir", audio_dir,
        "--data.espeak_voice", config["piper_espeak_voice"],
        "--model.sample_rate", str(config["sample_rate"]),
        "--data.cache_dir", config["piper_cache_dir"],
        "--data.config_path", config["piper_config_path"],
        "--data.batch_size", str(config["piper_batch_size"]),
        "--trainer.max_epochs", str(config["piper_max_epochs"]),
    ]

    ckpt = resume_from or pretrained_ckpt or config.get("piper_pretrained_checkpoint")
    if ckpt:
        cmd.extend(["--ckpt_path", ckpt])

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def export_onnx(checkpoint_path: str, output_path: str) -> None:
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "piper.train.export_onnx",
        "--checkpoint", checkpoint_path,
        "--output-file", output_path,
    ]

    print(f"Exporting: {checkpoint_path} -> {output_path}")
    subprocess.run(cmd, check=True)
    print(f"Exported: {output_path}")


def install_model(onnx_path: str, config: dict) -> None:
    target_dir = config["piper_model_output_dir"]
    os.makedirs(target_dir, exist_ok=True)

    basename = os.path.basename(onnx_path)
    model_name = os.path.splitext(basename)[0]

    target_onnx = os.path.join(target_dir, basename)
    shutil.copy2(onnx_path, target_onnx)
    print(f"Installed: {target_onnx}")

    json_path = onnx_path + ".json"
    if os.path.isfile(json_path):
        target_json = os.path.join(target_dir, basename + ".json")
        shutil.copy2(json_path, target_json)
        print(f"Installed: {target_json}")

    print(f'\nTo use in Pi-Bot, set CONFIG["piper_model"] = "{model_name}"')


def run(args, config: dict) -> None:
    config = dict(config)

    if args.batch_size:
        config["piper_batch_size"] = args.batch_size
    if args.max_epochs:
        config["piper_max_epochs"] = args.max_epochs

    if args.export:
        output = args.export.replace(".ckpt", "") + ".onnx"
        export_onnx(args.export, output)
        if args.install:
            install_model(output, config)
        return

    if not check_piper_install():
        _print_setup_instructions()
        sys.exit(1)

    dataset_dir = args.dataset_dir or config["output_dir"]
    prepare_dataset(dataset_dir)

    print()
    run_training(
        dataset_dir, config,
        pretrained_ckpt=args.pretrained_checkpoint,
        resume_from=args.resume_from,
    )
