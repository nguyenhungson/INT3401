import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark import play_match


class TestBenchmark(unittest.TestCase):
    def test_paired_random(self):
        result = play_match("Random", "Random", 2, 2, 10, 1.4, 7)
        self.assertEqual(result["wins"] + result["draws"] + result["losses"], 2)
    def test_odd_games_rejected(self):
        with self.assertRaises(ValueError):
            play_match("Random", "Random", 3, 2, 10, 1.4, 7)

    def test_zero_games_rejected(self):
        with self.assertRaises(ValueError):
            play_match("Random", "Random", 0, 2, 10, 1.4, 7)

    def test_paired_roles_and_seed_are_reproducible(self):
        rows, repeated = [], []
        for output in (rows, repeated):
            play_match("MCTS", "Random", 2, 2, 4, 1.4, 7, game_rows=output)
        self.assertEqual([row["first"] for row in rows], ["focal", "opponent"])
        self.assertEqual([row["focal_seed"] for row in rows], [7, 7])
        self.assertEqual([row["opponent_seed"] for row in rows], [8, 8])
        for first, second in zip(rows, repeated):
            for metric in ("winner", "outcome", "moves"):
                self.assertEqual(first[metric], second[metric])

    def test_smoke_export_and_plots(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            output = Path(directory)
            subprocess.run(
                [sys.executable, "-m", "experiments.benchmark", "--preset", "smoke",
                 "--output", directory],
                check=True,
                capture_output=True,
                timeout=90,
            )
            metadata_path = output / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["parameters"]["preset"], "smoke")
            with (output / "results.csv").open(encoding="utf-8", newline="") as source:
                summary = list(csv.DictReader(source))
            with (output / "games.csv").open(encoding="utf-8", newline="") as source:
                games = list(csv.DictReader(source))
            self.assertEqual(len(summary), 13)
            self.assertEqual(len(games), 26)
            for filename in (
                "uct_c_win_draw_loss.png",
                "iterations_winrate_time.png",
                "rollout_comparison.png",
            ):
                signature = (output / filename).read_bytes()[:8]
                self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
