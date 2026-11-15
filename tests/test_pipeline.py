"""Unit tests (standard library only): python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import FEATURES, fit_configurations, kmedians  # noqa: E402
from src.data_pipeline import load_profiles, parse_dly  # noqa: E402


def dly_line(station, year, month, element, values):
    """Build one fixed-width GHCN-Daily line; values are (value, qflag) per day."""
    line = f"{station}{year:04d}{month:02d}{element}"
    for day in range(31):
        value, qflag = values[day] if day < len(values) else (-9999, " ")
        line += f"{value:5d} {qflag} "
    return line


class ParserTests(unittest.TestCase):
    def test_parse_converts_units_and_drops_flagged_or_missing(self):
        values = [(100, " "), (200, "X"), (-9999, " "), (250, " ")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S.dly"
            path.write_text(dly_line("USW00000001", 2020, 1, "TMAX", values) + "\n", encoding="ascii")
            frame = parse_dly(path, 2019, 2021)
        self.assertEqual(sorted(frame.value), [10.0, 25.0])  # tenths -> degC; flagged/missing removed

    def test_year_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "S.dly"
            path.write_text(dly_line("USW00000001", 2010, 1, "TMAX", [(100, " ")]) + "\n", encoding="ascii")
            self.assertTrue(parse_dly(path, 2019, 2024).empty)


class ClusteringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = pd.read_csv(ROOT / "data/processed/station_month_profiles.csv")

    def test_snapshot_shape(self):
        self.assertEqual(len(self.profiles), 576)
        self.assertFalse(self.profiles[FEATURES].isna().any().any())

    def test_load_profiles_uses_snapshot_and_filters_years(self):
        frame, source = load_profiles(ROOT, 2021, 2022)
        self.assertEqual(source, "processed snapshot")
        self.assertTrue(frame.year.between(2021, 2022).all())

    def test_euclidean_run_is_deterministic(self):
        a = fit_configurations(self.profiles, (2, 3), "euclidean", seed=1)[0].cluster
        b = fit_configurations(self.profiles, (2, 3), "euclidean", seed=1)[0].cluster
        self.assertTrue((a == b).all())

    def test_metrics_and_selection(self):
        _, metrics, stability, _ = fit_configurations(self.profiles, (2, 3, 4), "manhattan")
        self.assertEqual(int(metrics.selected.sum()), 1)
        self.assertTrue(metrics.silhouette.between(-1, 1).all())
        self.assertTrue(0 <= stability["adjusted_rand_mean"] <= 1)

    def test_kmedians_separates_two_blobs(self):
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(0, .1, (30, 2)), rng.normal(5, .1, (30, 2))])
        labels, _ = kmedians(x, 2, seed=0)
        self.assertEqual(len(set(labels[:30])), 1)
        self.assertNotEqual(labels[0], labels[-1])


if __name__ == "__main__":
    unittest.main()
