"""Kiểm chứng giá trị thế cuối và giới hạn thời gian của chế độ strong."""

import time
import unittest

from src.connect4 import ConnectFour
from src.game import DRAW, EMPTY, O, X, Board
from src.mcts import MCTSAgent
from src.minimax import MinimaxAgent
from src.search_tools import SearchTimeout, defend_forks, forced_win, solve_endgame


def exhaustive_value(board, player):
    """Đối chứng bằng cách xét mọi nước, không dùng bộ lọc chiến thuật."""
    if board.status == DRAW:
        return 0
    if board.status != EMPTY:
        return 1 if board.status == player else -1
    values = []
    for move in board.legal_moves():
        child = board.copy()
        child.play(*move, player)
        values.append(-exhaustive_value(child, -player))
    return max(values)


class TestStrong(unittest.TestCase):
    def test_exact_values_match_full_enumeration(self):
        full = [
            [X if (row + 2 * col) % 4 in (0, 1) else O for col in range(9)]
            for row in range(9)
        ]
        for empty in (
            [(0, 0), (1, 2), (3, 4), (7, 8)],
            [(1, 5), (4, 0), (5, 5), (6, 2), (8, 8)],
        ):
            cells = [row[:] for row in full]
            for row, col in empty:
                cells[row][col] = EMPTY
            for player in (X, O):
                with self.subTest(empty=empty, player=player):
                    board = Board(cells)
                    snapshot = board.pretty()
                    move, value, _ = solve_endgame(
                        board, player, time.perf_counter() + 5
                    )
                    self.assertEqual(value, exhaustive_value(board, player))
                    child = board.copy()
                    child.play(*move, player)
                    self.assertEqual(-exhaustive_value(child, -player), value)
                    self.assertEqual(board.pretty(), snapshot)

    def test_connect4_exact_value(self):
        columns = [
            4, 4, 1, 3, 6, 5, 6, 7, 5, 3, 3, 4, 4, 6, 6, 3, 3, 5,
            4, 4, 2, 2, 3, 2, 6, 6, 1, 5, 1, 5, 5, 7, 7, 7, 7, 1,
        ]
        board = ConnectFour()
        for index, col in enumerate(columns):
            board.play(*board.drop_position(col - 1), X if index % 2 == 0 else O)
        move, value, _ = solve_endgame(board, X, time.perf_counter() + 5)
        self.assertEqual(value, exhaustive_value(board, X))
        self.assertTrue(board.is_legal(*move))

    def test_expired_solver_keeps_board(self):
        board = Board()
        before = board.pretty()
        with self.assertRaises(SearchTimeout):
            solve_endgame(board, X, time.perf_counter() - 1)
        self.assertEqual(board.pretty(), before)

    def test_fork_is_proven_against_every_reply(self):
        board = Board()
        for col in (3, 4, 5):
            board.play(4, col, X)
        move, plies = forced_win(board, X, time.perf_counter() + 5)
        self.assertEqual(plies, 3)
        board.play(*move, X)
        for reply in board.legal_moves():
            child = board.copy()
            child.play(*reply, O)
            self.assertTrue(child.winning_moves(X))

    def test_defends_open_three_before_double_threat(self):
        board = Board()
        for col in (3, 4, 5):
            board.play(4, col, X)
        safe = defend_forks(
            board, O, board.legal_moves(), time.perf_counter() + 5
        )
        self.assertTrue(safe)
        self.assertLess(len(safe), len(board.legal_moves()))
        for move in safe:
            child = board.copy()
            child.play(*move, O)
            self.assertIsNone(forced_win(child, X, time.perf_counter() + 5))

    def test_deadline_returns_legal_move_without_mutation(self):
        board = Board()
        for row, col, player in ((4, 4, X), (4, 5, O), (5, 4, X), (3, 4, O)):
            board.play(row, col, player)
        before = (board.pretty(), board.last_move, len(board._history))
        for agent in (
            MinimaxAgent(8, strong=True, time_limit=0.05),
            MCTSAgent(100000, strong=True, time_limit=0.05, rollout="heuristic"),
        ):
            with self.subTest(agent=agent.name):
                decision = agent.choose_move(board, X)
                self.assertTrue(board.is_legal(*decision.move))
                self.assertFalse(decision.solved)
                self.assertLess(decision.elapsed_ms, 500)
                self.assertEqual(
                    (board.pretty(), board.last_move, len(board._history)), before
                )
