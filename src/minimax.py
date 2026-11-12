"""MiniMax giới hạn độ sâu, có thể bật hoặc tắt cắt tỉa Alpha-Beta."""

from __future__ import annotations

import math
import time

from .agents import Decision, GameBoard
from .game import DIRECTIONS, DRAW, EMPTY, SIZE, Board

# Mỗi cửa sổ 5 ô chỉ tính điểm nếu có quân của tối đa một bên.
# Chuỗi còn đầu mở thường nằm trong nhiều cửa sổ có thể nối dài hơn.
WEIGHTS = (0, 2, 18, 180, 3000, 1000000)
WIN_SCORE = 10**9


def evaluate(board: Board, root_player: int) -> int:
    if board.status == root_player:
        return WIN_SCORE
    if board.status == -root_player:
        return -WIN_SCORE
    if board.status == DRAW:
        return 0
    score = 0
    cells = board.cells
    for r in range(SIZE):
        for c in range(SIZE):
            for dr, dc in DIRECTIONS:
                er, ec = r + 4 * dr, c + 4 * dc
                if not board.inside(er, ec):
                    continue
                own = foe = 0
                for k in range(5):
                    p = cells[r + k * dr][c + k * dc]
                    own += p == root_player
                    foe += p == -root_player
                if own and foe:
                    continue
                if own:
                    score += WEIGHTS[own]
                elif foe:
                    score -= WEIGHTS[foe]
    return score


class MinimaxAgent:
    def __init__(
        self,
        depth: int = 2,
        alpha_beta: bool = True,
        max_candidates: int = 12,
        *,
        prefer_quick_wins: bool = False,
    ) -> None:
        if depth < 1 or max_candidates < 1:
            raise ValueError("Độ sâu và giới hạn ứng viên phải là số nguyên dương.")
        self.depth = depth
        self.alpha_beta = alpha_beta
        self.max_candidates = max_candidates
        self.prefer_quick_wins = prefer_quick_wins
        self.name = "AlphaBeta" if alpha_beta else "Minimax"
        self._nodes = 0

    def _search(
        self,
        board: Board,
        depth: int,
        turn: int,
        root: int,
        alpha: float,
        beta: float,
    ) -> int:
        self._nodes += 1
        if depth == 0 or board.status != EMPTY:
            score = evaluate(board, root)
            if self.prefer_quick_wins and abs(score) == WIN_SCORE:
                # Thắng sớm được điểm cao hơn. Nếu buộc thua, cố kéo dài ván.
                moves_played = self.depth - depth
                if score > 0:
                    return score - moves_played
                return score + moves_played
            return score
        maximizing = turn == root
        best = -math.inf if maximizing else math.inf
        for move in self._ordered_moves(board, turn):
            board.play(*move, turn)
            val = self._search(board, depth - 1, -turn, root, alpha, beta)
            board.undo()
            best = max(best, val) if maximizing else min(best, val)
            if self.alpha_beta:
                if maximizing:
                    alpha = max(alpha, best)
                else:
                    beta = min(beta, best)
                if alpha >= beta:
                    # Tổ tiên đã có nhánh tốt hơn nên không cần xét phần còn lại.
                    break
        return int(best)

    def _ordered_moves(self, board: Board, turn: int) -> list[tuple[int, int]]:
        candidates = board.candidate_moves(radius=1)
        # Luôn giữ cả nước thắng ngay lẫn nước chặn, kể cả khi vượt giới hạn.
        wins = board.winning_moves(turn, candidates)
        blocks = board.winning_moves(-turn, candidates)
        urgent = list(dict.fromkeys(wins + blocks))
        remaining = [m for m in candidates if m not in urgent]
        limit = max(self.max_candidates, len(urgent))
        return (urgent + remaining)[:limit]

    def choose_move(self, board: GameBoard, player: int) -> Decision:
        if not isinstance(board, Board):
            raise ValueError("MiniMax chỉ hỗ trợ Caro.")
        if board.status != EMPTY:
            raise ValueError("Ván cờ đã kết thúc.")
        started = time.perf_counter()
        self._nodes = 1  # Tính cả nút gốc của cây tìm kiếm.
        best_value = -math.inf
        best_move: tuple[int, int] | None = None
        alpha, beta = -math.inf, math.inf
        moves = self._ordered_moves(board, player)
        for move in moves:
            board.play(*move, player)
            score = self._search(board, self.depth - 1, -player, player, alpha, beta)
            board.undo()
            if score > best_value:
                best_value, best_move = score, move
            if self.alpha_beta:
                alpha = max(alpha, best_value)
        assert best_move is not None
        return Decision(
            best_move,
            (time.perf_counter() - started) * 1000,
            self._nodes,
            depth=self.depth,
            algorithm=self.name,
            search_value=int(best_value),
        )
