import unittest

from src.game import DRAW, EMPTY, O, X, Board


class TestBoard(unittest.TestCase):
    def test_opening(self):
        b = Board()
        self.assertEqual(b.candidate_moves(), [(4, 4)])
        self.assertEqual(len(b.legal_moves()), 81)

    def test_illegal_move_and_undo(self):
        b = Board()
        b.play(0, 0, X)
        with self.assertRaises(ValueError):
            b.play(0, 0, O)
        with self.assertRaises(ValueError):
            b.play(-1, 0, O)
        b.undo()
        self.assertEqual(len(b.legal_moves()), 81)

    def test_win_four_directions(self):
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            b = Board()
            start_c = 4 if dc == -1 else 1
            for k in range(5):
                b.play(1 + k * dr, start_c + k * dc, X)
            self.assertEqual(b.status, X)
            with self.assertRaises(ValueError):
                b.play(8, 8, O)

    def test_five_consecutive_is_win(self):
        b = Board()
        for col in range(5):
            b.play(4, col, O)
        self.assertEqual(b.status, O)

    def test_six_consecutive_is_win(self):
        board = Board()
        for col in (0, 1, 2, 4, 5):
            board.play(4, col, X)
        self.assertEqual(board.status, EMPTY)
        board.play(4, 3, X)
        self.assertEqual(board.status, X)

    def test_end_block_not_applied(self):
        b = Board()
        b.play(4, 1, O)
        b.play(4, 7, O)
        for col in range(2, 7):
            b.play(4, col, X)
        self.assertEqual(b.status, X)

    def test_copy_independent(self):
        b = Board()
        b2 = b.copy()
        b2.play(0, 0, X)
        self.assertEqual(b.cells[0][0], 0)

    def test_draw_state(self):
        # Mẫu này lấp kín bàn mà vẫn ngắt chuỗi theo cả bốn hướng.
        cells = [
            [X if (row + 2 * col) % 4 in (0, 1) else O for col in range(9)]
            for row in range(9)
        ]
        b = Board(cells)
        self.assertEqual(b.status, DRAW)
        self.assertEqual(b.legal_moves(), [])

    def test_load_terminal_board(self):
        cells = [[EMPTY] * 9 for _ in range(9)]
        cells[8][:5] = [O] * 5
        self.assertEqual(Board(cells).status, O)
