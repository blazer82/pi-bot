"""Allow running as python -m voice_trainer."""

import os
import sys

_GENERATE_VENV = "/workspace/venv-generate"
_TRAIN_VENV = "/workspace/venv-train"

_GENERATE_COMMANDS = {"xtts-setup", "generate", "postprocess", "split"}
_TRAIN_COMMANDS = {"train"}


def _venv_python(venv_dir: str) -> str | None:
    p = os.path.join(venv_dir, "bin", "python3")
    return p if os.path.isfile(p) else None


def _running_in_venv(venv_dir: str) -> bool:
    return os.path.realpath(sys.executable).startswith(os.path.realpath(venv_dir))


def _reexec_in_venv(venv_dir: str) -> None:
    python = _venv_python(venv_dir)
    if python and not _running_in_venv(venv_dir):
        os.execv(python, [python, "-m", "voice_trainer"] + sys.argv[1:])


if len(sys.argv) > 1:
    cmd = sys.argv[1]
    if cmd in _GENERATE_COMMANDS:
        _reexec_in_venv(_GENERATE_VENV)
    elif cmd in _TRAIN_COMMANDS:
        _reexec_in_venv(_TRAIN_VENV)

from voice_trainer.cli import main

main()
