"""Optional end-to-end tests against a real Flight server.

Set ``QUARREL_FLIGHT_URI`` (for example ``grpc://127.0.0.1:50051``) and start
``cargo run -p quarrel-flight --example flight_server`` to run these. Without the
environment variable they skip, keeping the default local suite hermetic.
"""

from __future__ import annotations

import math
import os
import sys
import time
import unittest
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[1]
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from quarrel_flight import QuarrelFlightClient  # noqa: E402

FLIGHT_URI = os.environ.get("QUARREL_FLIGHT_URI")


@unittest.skipUnless(FLIGHT_URI, "set QUARREL_FLIGHT_URI to run Flight integration tests")
class FlightIntegrationTests(unittest.TestCase):
    """Exercise the exact generate/embed schemas over a real gRPC connection."""

    def setUp(self) -> None:
        self.client = QuarrelFlightClient(FLIGHT_URI)
        self.addCleanup(self.client.close)
        # Allow the example server a moment to bind when the suite is launched
        # immediately after `cargo run`. This only applies to opt-in integration runs.
        deadline = time.monotonic() + 10.0
        while True:
            try:
                self.client.health()
                return
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.1)

    def test_health_action_reports_schemas(self) -> None:
        health = self.client.health()
        self.assertTrue(health["ready"])
        self.assertEqual(health["backend"], "reference")
        self.assertIn("generate_response", health["schemas"])

    def test_generate_round_trip(self) -> None:
        rows = self.client.generate("sdk-generate-1", [128000, 271], max_tokens=4)
        self.assertTrue(rows)
        expected = {
            "request_id",
            "sequence",
            "token_id",
            "logprob",
            "top_logprobs",
            "finish_reason",
            "finished",
        }
        for row in rows:
            self.assertEqual(set(row), expected)
            self.assertEqual(row["request_id"], "sdk-generate-1")
        self.assertTrue(rows[-1]["finished"])

    def test_embed_round_trip(self) -> None:
        rows = self.client.embed("sdk-embed-1", [7, 11])
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["request_id"], "sdk-embed-1")
            vector = row["embedding"]
            self.assertEqual(len(vector), 4)
            self.assertTrue(all(math.isfinite(value) for value in vector))
            norm = math.sqrt(sum(value * value for value in vector))
            self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
