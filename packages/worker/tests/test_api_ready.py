from __future__ import annotations

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import _model_repository_root
from pipeline.runtime_status import TritonReadiness, build_ready_payload


class ReadyPayloadTest(unittest.TestCase):
    def test_build_ready_payload_marks_ready_state(self) -> None:
        readiness = TritonReadiness.from_status(
            server_url="triton:8001",
            server_live=True,
            server_ready=True,
            model_ready=True,
            model_name="silero_vad",
        )

        payload = build_ready_payload("baseline", readiness)

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["profile"], "baseline")
        self.assertTrue(payload["triton"]["ready"])
        self.assertIsNone(payload["triton"]["model_present"])
        self.assertIsNone(payload["triton"]["model_state"])

    def test_build_ready_payload_preserves_diagnostics(self) -> None:
        readiness = TritonReadiness.from_error(
            server_url="triton:8001",
            model_name="silero_vad",
            issue="Failed to create Triton client for triton:8001: connection refused",
        )

        payload = build_ready_payload("baseline", readiness)

        self.assertEqual(payload["status"], "unavailable")
        self.assertEqual(payload["triton"]["status"], "unavailable")
        self.assertIn("connection refused", payload["triton"]["summary"])


class WorkerEnvDefaultsTest(unittest.TestCase):
    def test_model_repository_root_defaults_to_repo_model_repository(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            model_repository_root = _model_repository_root()

        self.assertEqual(
            model_repository_root,
            str(Path(__file__).resolve().parents[3] / "model_repository"),
        )


if __name__ == "__main__":
    unittest.main()
