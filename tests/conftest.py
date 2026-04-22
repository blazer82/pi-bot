"""Shared test configuration — pre-import mocking of hardware dependencies.

pytest loads conftest.py before any test module, so the mocks are in place
before any pi_bot submodule tries to import sounddevice, openwakeword, or
pywhispercpp.
"""

import sys
from unittest import mock

# ---------------------------------------------------------------------------
# Mock hardware-dependent imports before any pi_bot code is loaded.
# ---------------------------------------------------------------------------
_sd_mock = mock.MagicMock()
_oww_mock = mock.MagicMock()
_pwc_mock = mock.MagicMock()

sys.modules.setdefault("sounddevice", _sd_mock)
sys.modules.setdefault("openwakeword", _oww_mock)
sys.modules.setdefault("openwakeword.model", _oww_mock)
sys.modules.setdefault("pywhispercpp", _pwc_mock)
sys.modules.setdefault("pywhispercpp.model", _pwc_mock)
sys.modules.setdefault("webrtcvad", mock.MagicMock())
sys.modules.setdefault("psutil", mock.MagicMock())

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
SAMPLE_JOKES = [
    {"id": 1, "setup": "Why did the programmer quit?",
     "punchline": "Because he didn't get arrays."},
    {"id": 2, "setup": "Why do Java devs wear glasses?",
     "punchline": "Because they can't C#."},
]
