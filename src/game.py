"""Caro 9×9 theo luật của đề bài.

Mỗi ô lưu 0 (trống), 1 (X) hoặc -1 (O). Trạng thái ván lưu 0 khi đang chơi,
1/-1 khi X/O thắng và 2 khi hòa.
"""

from __future__ import annotations

SIZE = 9
WIN_LENGTH = 5
EMPTY, X, O, DRAW = 0, 1, -1, 2
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def has_line(
    cells: list[list[int]], row: int, col: int, player: int, target: int
) -> bool:
    """Đếm chuỗi qua ô đích theo hai phía của mỗi hướng."""
    height, width = len(cells), len(cells[0])
    for dr, dc in DIRECTIONS:
        length = 1
        for step in (1, -1):
            row_step, col_step = step * dr, step * dc
            r, c = row + row_step, col + col_step
            while 0 <= r < height and 0 <= c < width and cells[r][c] == player:
                length += 1
                r += row_step
                c += col_step
        if length >= target:
            return True
    return False


class Board:
    def __init__(self, cells: list[list[int]] | None = None) -> None:
        self.cells = [[EMPTY] * SIZE for _ in range(SIZE)]
        if cells is not None:
            self.cells = [row[:] for row in cells]
        if len(self.cells) != SIZE or any(len(row) != SIZE for row in self.cells):
            raise ValueError("Bàn Caro phải có kích thước 9×9.")
        if any(v not in (X, O, EMPTY) for row in self.cells for v in row):
            raise ValueError("Ô cờ chỉ nhận các giá trị -1, 0 và 1.")
        self.move_count = sum(v != EMPTY for row in self.cells for v in row)
        self.last_move: tuple[int, int] | None = None
        self.status = self._scan_status() if cells is not None else EMPTY
        self._history: list[tuple[int, int, int, tuple[int, int] | None]] = []

    def copy(self) -> Board:
        other = Board.__new__(Board)
        other.cells = [row[:] for row in self.cells]
        other.move_count = self.move_count
        other.last_move = self.last_move
        other.status = self.status
        other._history = []
        return other

    @staticmethod
    def inside(row: int, col: int) -> bool:
        return 0 <= row < SIZE and 0 <= col < SIZE

    def legal_moves(self) -> list[tuple[int, int]]:
        if self.status != EMPTY:
            return []
        return [
            (row, col)
            for row, values in enumerate(self.cells)
            for col, value in enumerate(values)
            if value == EMPTY
        ]

    def is_legal(self, row: int, col: int) -> bool:
        return (
            self.status == EMPTY
            and self.inside(row, col)
            and self.cells[row][col] == EMPTY
        )

    def _five_through(self, row: int, col: int, player: int) -> bool:
        return has_line(self.cells, row, col, player, WIN_LENGTH)

    def _scan_status(self) -> int:
        for r in range(SIZE):
            for c in range(SIZE):
                p = self.cells[r][c]
                if p != EMPTY and self._five_through(r, c, p):
                    return p
        return DRAW if self.move_count == SIZE * SIZE else EMPTY

    def play(self, row: int, col: int, player: int) -> None:
        if player not in (X, O):
            raise ValueError("Người chơi phải là X (1) hoặc O (-1).")
        if not self.is_legal(row, col):
            raise ValueError(f"Nước đi không hợp lệ: ({row + 1}, {col + 1}).")
        self._history.append((row, col, self.status, self.last_move))
        self.cells[row][col] = player
        self.move_count += 1
        self.last_move = (row, col)
        if self._five_through(row, col, player):
            self.status = player
        elif self.move_count == SIZE * SIZE:
            self.status = DRAW
        else:
            self.status = EMPTY

    def undo(self) -> None:
        if not self._history:
            raise ValueError("Bàn cờ chưa có nước nào để hoàn tác.")
        r, c, previous_status, previous_last_move = self._history.pop()
        self.cells[r][c] = EMPTY
        self.move_count -= 1
        self.status = previous_status
        self.last_move = previous_last_move

    def candidate_moves(
        self, radius: int = 1, limit: int | None = None
    ) -> list[tuple[int, int]]:
        """Xếp ô gần quân đã đánh theo số hàng xóm, rồi theo khoảng cách tới tâm.

        Đây là danh sách để tìm kiếm. Luật chơi vẫn cho đánh vào mọi ô trống.
        """
        if self.status != EMPTY:
            return []
        if self.move_count == 0:
            return [(4, 4)]
        scores: dict[tuple[int, int], int] = {}
        for r in range(SIZE):
            for c in range(SIZE):
                if self.cells[r][c] == EMPTY:
                    continue
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr, cc = r + dr, c + dc
                        if self.inside(rr, cc) and self.cells[rr][cc] == EMPTY:
                            weight = 3 if max(abs(dr), abs(dc)) == 1 else 1
                            scores[(rr, cc)] = scores.get((rr, cc), 0) + weight
        ordered = sorted(
            scores,
            key=lambda move: (-scores[move], abs(move[0] - 4) + abs(move[1] - 4), move),
        )
        if not ordered:
            ordered = self.legal_moves()
        return ordered[:limit] if limit is not None else ordered

    def winning_moves(
        self, player: int, candidates: list[tuple[int, int]] | None = None
    ) -> list[tuple[int, int]]:
        """Tìm các nước thắng trong danh sách ô trống hợp lệ."""
        if candidates is None:
            candidates = self.candidate_moves()
        # Ô đích được tính như đã có quân, nên không cần chơi rồi hoàn tác.
        return [
            (row, col)
            for row, col in candidates
            if self._five_through(row, col, player)
        ]

    def line_score(self, row: int, col: int, player: int) -> int:
        """Ưu tiên nối chuỗi quân mình, đặc biệt khi chuỗi còn đầu mở."""
        score = 0
        for dr, dc in DIRECTIONS:
            length = 1
            open_ends = 0
            for step in (1, -1):
                r, c = row + step * dr, col + step * dc
                while self.inside(r, c) and self.cells[r][c] == player:
                    length += 1
                    r, c = r + step * dr, c + step * dc
                if self.inside(r, c) and self.cells[r][c] == EMPTY:
                    open_ends += 1
            if length > 1:
                score += length * length * open_ends
        return score

    def pretty(self) -> str:
        symbols = {EMPTY: ".", X: "X", O: "O"}
        header = "   " + " ".join(str(col + 1) for col in range(SIZE))
        lines = [
            f"{index + 1:2} " + " ".join(symbols[cell] for cell in row)
            for index, row in enumerate(self.cells)
        ]
        return "\n".join([header, *lines])
