"""API tests using Flask's test client: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app import app  # noqa: E402


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_and_meta(self):
        self.assertEqual(self.client.get("/api/health").json, {"status": "ok"})
        meta = self.client.get("/api/meta").json
        self.assertEqual(len(meta["stations"]), 8)
        self.assertEqual((meta["min_year"], meta["max_year"]), (2019, 2024))

    def test_analyze_returns_expected_structure(self):
        response = self.client.get("/api/analyze?k=2,3&distance=euclidean")
        data = response.json
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data["profiles"], 576)
        self.assertEqual(data["selected_k"], 2)
        self.assertEqual(len(data["points"]), 576)
        self.assertIn("adjusted_rand_mean", data["stability"])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_invalid_parameters_rejected(self):
        for query in ("start=2018", "end=2030", "start=2024&end=2020", "k=9", "k=1", "distance=cosine", "start=abc"):
            self.assertEqual(self.client.get(f"/api/analyze?{query}").status_code, 400, query)

    def test_unknown_route(self):
        self.assertEqual(self.client.get("/nope").status_code, 404)


if __name__ == "__main__":
    unittest.main()
