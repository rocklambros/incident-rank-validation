from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from engine.cli.secrets import load_secret


class TestLoadSecret:
    def test_env_var_takes_precedence(self) -> None:
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "env-key-123"}):
            result = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
        assert result == "env-key-123"

    def test_falls_back_to_pass(self) -> None:
        env = os.environ.copy()
        env.pop("RUNPOD_API_KEY", None)
        with patch.dict(os.environ, env, clear=True), patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "pass-key-456\n"
            result = load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")
        assert result == "pass-key-456"

    def test_raises_when_both_fail(self) -> None:
        env = os.environ.copy()
        env.pop("RUNPOD_API_KEY", None)
        with patch.dict(os.environ, env, clear=True), patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ""
            with pytest.raises(RuntimeError, match="runpod/api-key"):
                load_secret("runpod/api-key", env_var="RUNPOD_API_KEY")

    def test_strips_trailing_newline(self) -> None:
        env = os.environ.copy()
        env.pop("TEST_KEY", None)
        with patch.dict(os.environ, env, clear=True), patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "secret-value\n\n"
            result = load_secret("test/key", env_var="TEST_KEY")
        assert result == "secret-value"
