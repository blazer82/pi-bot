"""Step 4: Piper model training, export, and installation."""

import os
import shutil
import subprocess
import sys


def check_piper_install() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "piper_train", "--help"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def _print_setup_instructions() -> None:
    print("piper_train is not installed. To set it up:\n")
    print("  git clone https://github.com/rhasspy/piper")
    print("  cd piper/src/python")
    print('  pip install -e ".[train]"')
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


def preprocess_dataset(dataset_dir: str) -> None:
    config_path = os.path.join(dataset_dir, "config.json")
    if os.path.isfile(config_path):
        print("Preprocessing already done (config.json exists), skipping.")
        return

    print("Running piper_train preprocessing...")
    cmd = [
        sys.executable, "-m", "piper_train.preprocess",
        "--language", "de",
        "--input-dir", dataset_dir,
        "--output-dir", dataset_dir,
        "--sample-rate", "22050",
        "--dataset-format", "ljspeech",
        "--single-speaker",
    ]
    subprocess.run(cmd, check=True)
    print("Preprocessing complete.\n")


def run_training(dataset_dir: str, config: dict, resume_from: str | None = None) -> None:
    cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", dataset_dir,
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(config["piper_batch_size"]),
        "--validation-split", str(config["piper_validation_split"]),
        "--num-test-examples", str(config["piper_num_test_examples"]),
        "--max_epochs", str(config["piper_max_epochs"]),
    ]

    if resume_from:
        cmd.extend(["--resume_from_checkpoint", resume_from])

    print(f"Running: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def export_onnx(checkpoint_path: str, output_path: str) -> None:
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "piper_train",
        "--onnx-output", output_path,
        "--checkpoint", checkpoint_path,
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
    preprocess_dataset(dataset_dir)

    print()
    run_training(dataset_dir, config, resume_from=args.resume_from)
