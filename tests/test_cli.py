"""Chạy CLI thật và kiểm tra nhật ký của các ván hoàn tất."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.agents import RandomAgent
from src.game import EMPTY, X
from src.main import make_board
from src.mcts import MCTSAgent


class TestCLI(unittest.TestCase):
    def play_cli(self, args, inputs=""):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            log = Path(directory) / "moves.jsonl"
            command = [sys.executable, "-X", "utf8", "-m", "src.main", *args]
            result = subprocess.run(
                [*command, "--log", str(log)],
                input=inputs,
                capture_output=True,
                encoding="utf-8",
                timeout=90,
                check=True,
            )
            moves = [
                json.loads(line)
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
        self.assertIn("Kết thúc ván", result.stdout)
        turns = [move["turn"] for move in moves]
        self.assertEqual(turns, list(range(1, len(moves) + 1)))
        self.assertEqual(moves[0]["player"], "X")
        return moves, result.stdout

    def test_caro_ai_vs_ai(self):
        moves, _ = self.play_cli([
            "--mode", "ai-ai", "--agent1", "minimax", "--agent2", "mcts",
            "--depth", "2", "--alpha-beta", "--iterations", "4", "--quiet",
        ])
        self.assertEqual(moves[0]["algorithm"], "AlphaBeta")
        self.assertEqual(moves[0]["depth"], 2)
        self.assertEqual(moves[1]["iterations"], 4)

    def test_connect_four_ai_vs_ai(self):
        moves, _ = self.play_cli([
            "--game", "connect4", "--mode", "ai-ai", "--agent1", "mcts",
            "--agent2", "random", "--iterations", "4", "--quiet",
        ])
        self.assertEqual(moves[0]["move"][0], 5)
        self.assertLessEqual(len(moves), 42)

    def test_human_vs_ai_for_both_games(self):
        for game in ("caro", "connect4"):
            with self.subTest(game=game):
                board = make_board(game)
                human = RandomAgent(seed=12)
                opponent = MCTSAgent(iterations=2, seed=43, rollout="heuristic")
                inputs = ["invalid"]
                turn = X
                while board.status == EMPTY:
                    agent = human if turn == X else opponent
                    row, col = agent.choose_move(board, turn).move
                    if turn == X:
                        entry = f"{row + 1} {col + 1}"
                        if game == "connect4":
                            entry = str(col + 1)
                        inputs.append(entry)
                    board.play(row, col, turn)
                    turn = -turn

                moves, printed = self.play_cli([
                    "--game", game, "--mode", "human-ai", "--agent", "mcts",
                    "--iterations", "2", "--rollout", "heuristic", "--quiet",
                ], "\n".join(inputs) + "\n")
                self.assertIn("Nước đi không hợp lệ", printed)
                self.assertEqual(moves[0]["algorithm"], "Human")
                self.assertEqual(len(moves), board.move_count)

    def test_invalid_parameter_has_no_traceback(self):
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "src.main", "--c", "nan"],
            capture_output=True,
            encoding="utf-8",
            timeout=15,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--c phải", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
