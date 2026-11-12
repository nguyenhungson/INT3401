"""Luật Connect Four trên bàn 6 hàng và 7 cột."""

from __future__ import annotations

from .game import DRAW, EMPTY, O, X, has_line

ROWS, COLUMNS, WIN_LENGTH = 6, 7, 4
CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)


class ConnectFour:
    def __init__(self) -> None:
        self.cells = [[EMPTY] * COLUMNS for _ in range(ROWS)]
        self.move_count = 0
        self.status = EMPTY
        self._next_rows = [ROWS - 1] * COLUMNS

    def copy(self) -> ConnectFour:
        board = ConnectFour()
        board.cells = [row[:] for row in self.cells]
        board.move_count = self.move_count
        board.status = self.status
        board._next_rows = self._next_rows[:]
        return board

    def drop_position(self, col: int) -> tuple[int, int] | None:
        """Trả tọa độ quân sẽ rơi xuống, nếu còn chỗ trong cột."""
        if self.status == EMPTY and 0 <= col < COLUMNS:
            row = self._next_rows[col]
            if row >= 0:
                return row, col
        return None

    def legal_moves(self) -> list[tuple[int, int]]:
        return [
            move for col in range(COLUMNS) if (move := self.drop_position(col))
        ]

    def candidate_moves(
        self, radius: int = 1, limit: int | None = None
    ) -> list[tuple[int, int]]:
        """Tìm kiếm các cột gần giữa trước. Mọi cột trống đều hợp lệ."""
        moves = [move for col in CENTER_FIRST if (move := self.drop_position(col))]
        return moves[:limit]

    def is_legal(self, row: int, col: int) -> bool:
        return self.drop_position(col) == (row, col)

    def play(self, row: int, col: int, player: int) -> None:
        if player not in (X, O):
            raise ValueError("Người chơi phải là X hoặc O.")
        if not self.is_legal(row, col):
            raise ValueError(f"Không thể thả quân vào ô ({row + 1}, {col + 1}).")

        self.cells[row][col] = player
        self._next_rows[col] -= 1
        self.move_count += 1
        if has_line(self.cells, row, col, player, WIN_LENGTH):
            self.status = player
        elif self.move_count == ROWS * COLUMNS:
            self.status = DRAW

    def winning_moves(
        self, player: int, candidates: list[tuple[int, int]] | None = None
    ) -> list[tuple[int, int]]:
        if candidates is None:
            candidates = self.legal_moves()
        return [
            (row, col)
            for row, col in candidates
            if has_line(self.cells, row, col, player, WIN_LENGTH)
        ]

    def line_score(self, row: int, col: int, player: int) -> int:
        """Khi không có nước gấp, ưu tiên cột giữa vì có nhiều hướng nối."""
        return COLUMNS // 2 - abs(COLUMNS // 2 - col)

    def pretty(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        lines = [" ".join(symbols[cell] for cell in row) for row in self.cells]
        return "\n".join([*lines, "1 2 3 4 5 6 7"])
