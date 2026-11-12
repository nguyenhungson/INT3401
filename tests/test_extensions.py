"""Kiểm thử rollout có heuristic và Connect Four."""

import unittest
from unittest.mock import Mock

from src.connect4 import ConnectFour
from src.game import DRAW, EMPTY, O, X, Board
from src.mcts import MCTSAgent


class TestHeuristicRollout(unittest.TestCase):
    def setUp(self):
        self.agent = MCTSAgent(iterations=8, seed=7, rollout="heuristic")

    def test_win_before_block(self):
        board = Board()
        for col in range(4):
            board.play(0, col, X)
            board.play(2, col, O)
        snapshot = board.copy()
        self.assertEqual(self.agent._rollout_move(board, X), (0, 4))
        self.assertEqual(board.cells, snapshot.cells)
        self.assertEqual(board.last_move, snapshot.last_move)

    def test_blocks_opponent(self):
        board = Board()
        for col in range(4):
            board.play(2, col, O)
        self.assertEqual(self.agent._rollout_move(board, X), (2, 4))

    def test_extends_line(self):
        board = Board()
        for col in (3, 4, 5):
            board.play(4, col, X)
        self.assertIn(self.agent._rollout_move(board, X), [(4, 2), (4, 6)])

    def test_empty_board(self):
        board = Board()
        self.assertEqual(self.agent._rollout_move(board, X), (4, 4))

    def test_falls_back_to_legal_moves(self):
        board = Mock(spec=Board)
        board.candidate_moves.return_value = []
        board.legal_moves.return_value = [(8, 8)]
        self.assertEqual(self.agent._rollout_move(board, X), (8, 8))


class TestConnectFour(unittest.TestCase):
    def play_columns(self, columns):
        board = ConnectFour()
        player = X
        for col in columns:
            self.assertEqual(board.status, EMPTY)
            move = board.drop_position(col - 1)
            self.assertIsNotNone(move)
            board.play(*move, player)
            player = -player
        return board

    def test_drop_and_full_column(self):
        board = self.play_columns([1] * 6)
        self.assertEqual([row[0] for row in board.cells], [O, X, O, X, O, X])
        self.assertIsNone(board.drop_position(0))
        self.assertIsNone(board.drop_position(-1))
        self.assertIsNone(board.drop_position(7))
        self.assertFalse(board.is_legal(0, 1))
        with self.assertRaises(ValueError):
            board.play(0, 0, X)

    def test_wins_in_four_directions(self):
        examples = [
            ([1, 7, 2, 7, 3, 6, 4], X),
            ([1, 7, 1, 7, 1, 7, 1], X),
            ([2, 3, 1, 2, 6, 3, 1, 2, 1, 1, 3, 4], O),
            ([1, 3, 4, 3, 4, 4, 4, 2, 3, 5, 2], X),
        ]
        for columns, winner in examples:
            with self.subTest(columns=columns):
                board = self.play_columns(columns)
                self.assertEqual(board.status, winner)
                self.assertEqual(board.legal_moves(), [])

    def test_draw(self):
        columns = [
            4, 4, 1, 3, 6, 5, 6, 7, 5, 3, 3, 4, 4, 6, 6, 3, 3, 5, 4, 4, 2,
            2, 3, 2, 6, 6, 1, 5, 1, 5, 5, 7, 7, 7, 7, 1, 2, 1, 7, 1, 2, 2,
        ]
        self.assertEqual(self.play_columns(columns).status, DRAW)

    def test_mcts_uses_gravity_without_altering_original_board(self):
        board = self.play_columns([4, 3, 4, 3])
        snapshot = board.copy()
        for rollout in ("random", "heuristic"):
            with self.subTest(rollout=rollout):
                agent = MCTSAgent(iterations=20, rollout=rollout)
                decision = agent.choose_move(board, X)
                self.assertTrue(board.is_legal(*decision.move))
                self.assertEqual(board.cells, snapshot.cells)
                self.assertEqual(board.legal_moves(), snapshot.legal_moves())
