from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.runtime_status import TritonReadiness
from pipeline.triton import TritonUnavailableError, TritonVadClient, inspect_model_repository


class TritonReadinessTest(unittest.TestCase):
    def test_readiness_reports_ready_status(self) -> None:
        readiness = TritonReadiness.from_status(
            server_url="triton:8001",
            server_live=True,
            server_ready=True,
            model_ready=True,
            model_name="silero_vad",
        )

        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.status, "ready")
        self.assertIn("are ready", readiness.summary)
        self.assertIsNone(readiness.model_present)

    def test_score_windows_fails_fast_when_model_is_not_ready(self) -> None:
        class FakeClient:
            def is_server_live(self) -> bool:
                return True

            def is_server_ready(self) -> bool:
                return True

            def is_model_ready(self, model_name: str) -> bool:
                _ = model_name
                return False

            def infer(self, *args, **kwargs):  # pragma: no cover - should never be called
                raise AssertionError("infer should not run when Triton is not ready")

        client = object.__new__(TritonVadClient)
        client._grpcclient = object()
        client._url = "triton:8001"
        client._model_name = "silero_vad"
        client._client = FakeClient()

        with self.assertRaisesRegex(TritonUnavailableError, "not ready in Triton"):
            client.score_windows(object())

    def test_readiness_reports_missing_model_in_repository_index(self) -> None:
        class FakeClient:
            def is_server_live(self) -> bool:
                return True

            def is_server_ready(self) -> bool:
                return True

            def is_model_ready(self, model_name: str) -> bool:
                _ = model_name
                return False

            def get_model_repository_index(self):
                class _Resp:
                    models = [{"name": "other_model", "state": "READY"}]
                return _Resp()

        client = object.__new__(TritonVadClient)
        client._grpcclient = object()
        client._url = "triton:8001"
        client._model_name = "silero_vad"
        client._client = FakeClient()

        readiness = client.readiness()

        self.assertFalse(readiness.ready)
        self.assertFalse(readiness.model_present)
        self.assertIn("was not found in the Triton model repository index", readiness.summary)

    def test_readiness_connection_refused_includes_startup_guidance(self) -> None:
        class FakeClient:
            def is_server_live(self) -> bool:
                raise RuntimeError(
                    "failed to connect to all addresses; last error: UNKNOWN: "
                    "ipv4:127.0.0.1:8001: Failed to connect to remote host: connect: Connection refused (111)"
                )

            def is_server_ready(self) -> bool:  # pragma: no cover - never reached after failure
                return False

            def is_model_ready(self, model_name: str) -> bool:  # pragma: no cover - never reached after failure
                _ = model_name
                return False

        client = object.__new__(TritonVadClient)
        client._grpcclient = object()
        client._url = "127.0.0.1:8001"
        client._model_name = "silero_vad"
        client._client = FakeClient()

        with self.assertRaises(TritonUnavailableError) as context:
            client.readiness()

        message = str(context.exception)
        self.assertIn("docker compose up --build", message)
        self.assertIn("TRITON_GRPC_URL", message)


class ModelRepositoryStatusTest(unittest.TestCase):
    def test_repository_status_reports_missing_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status = inspect_model_repository(temp_dir)

        self.assertEqual(status.status, "unavailable")
        self.assertIn("MANIFEST.json", status.summary)

    def test_repository_status_reports_ready_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "silero_vad" / "1"
            model_path.mkdir(parents=True)
            (model_path / "model.onnx").write_bytes(b"test")
            (root / "MANIFEST.json").write_text(
                (
                    '{"profile":"baseline","selected_model_ids":["silero_vad"],'
                    '"models":[{"model_id":"silero_vad","installed":true}]}'
                ),
                encoding="utf-8",
            )

            status = inspect_model_repository(temp_dir)

        self.assertEqual(status.status, "ready")
        self.assertTrue(status.ready)
        self.assertEqual(status.profile, "baseline")
        self.assertEqual(status.selected_model_ids, ("silero_vad",))
        self.assertIn("contains a manifest and artifact", status.summary)


if __name__ == "__main__":
    unittest.main()
