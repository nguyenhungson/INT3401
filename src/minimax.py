"""MiniMax giới hạn độ sâu, có thể bật hoặc tắt cắt tỉa Alpha-Beta."""

from __future__ import annotations

import math
import time

from .agents import Decision, GameBoard
from .game import DIRECTIONS, DRAW, EMPTY, SIZE, Board
from .search_tools import (
    SearchTimeout, defend_forks, forced_win, solve_endgame, tactical_moves,
)

# Mỗi cửa sổ 5 ô chỉ tính điểm nếu có quân của tối đa một bên.
# Chuỗi còn đầu mở thường nằm trong nhiều cửa sổ có thể nối dài hơn.
WEIGHTS = (0, 2, 18, 180, 3000, 1000000)
WIN_SCORE = 10**9
WINDOWS = tuple(
    tuple((row + step * dr, col + step * dc) for step in range(5))
    for row in range(SIZE)
    for col in range(SIZE)
    for dr, dc in DIRECTIONS
    if 0 <= row + 4 * dr < SIZE and 0 <= col + 4 * dc < SIZE
)


def evaluate(board: Board, root_player: int) -> int:
    if board.status == root_player:
        return WIN_SCORE
    if board.status == -root_player:
        return -WIN_SCORE
    if board.status == DRAW:
        return 0
    score = 0
    cells = board.cells
    for window in WINDOWS:
        own = foe = 0
        for row, col in window:
            piece = cells[row][col]
            own += piece == root_player
            foe += piece == -root_player
        if own and not foe:
            score += WEIGHTS[own]
        elif foe and not own:
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
        strong: bool = False,
        time_limit: float = 3.0,
    ) -> None:
        if depth < 1 or max_candidates < 1:
            raise ValueError("Độ sâu và giới hạn ứng viên phải là số nguyên dương.")
        self.depth = depth
        self.alpha_beta = alpha_beta
        self.max_candidates = max_candidates
        self.prefer_quick_wins = prefer_quick_wins
        if not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("Thời gian mỗi nước phải hữu hạn và lớn hơn 0.")
        self.strong = strong
        self.time_limit = time_limit
        self._deadline = math.inf
        self._search_depth = depth
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
        if self.strong and self._nodes % 64 == 0:
            if time.perf_counter() >= self._deadline:
                raise SearchTimeout
        if depth == 0 or board.status != EMPTY:
            score = evaluate(board, root)
            if self.prefer_quick_wins and abs(score) == WIN_SCORE:
                # Thắng sớm được điểm cao hơn. Nếu buộc thua, cố kéo dài ván.
                moves_played = self._search_depth - depth
                if score > 0:
                    return score - moves_played
                return score + moves_played
            return score
        maximizing = turn == root
        best = -math.inf if maximizing else math.inf
        for move in self._ordered_moves(board, turn):
            board.play(*move, turn)
            try:
                val = self._search(board, depth - 1, -turn, root, alpha, beta)
            finally:
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
        if self.strong:
            return tactical_moves(board, turn, self.max_candidates)
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
        if self.strong:
            return self._choose_strong(board, player, started)
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

    def _choose_strong(self, board: Board, player: int, started: float) -> Decision:
        self._deadline = started + self.time_limit
        legal = board.legal_moves()
        forced = forced_win(board, player, self._deadline)
        if forced:
            move, plies = forced
            return Decision(
                move, (time.perf_counter() - started) * 1000,
                depth=plies, algorithm=self.name, search_value=1, solved=True,
            )
        if len(legal) <= 8:
            try:
                move, value, nodes = solve_endgame(board, player, self._deadline)
                return Decision(
                    move, (time.perf_counter() - started) * 1000, nodes=nodes,
                    depth=len(legal), algorithm=self.name, search_value=value,
                    solved=True,
                )
            except SearchTimeout:
                return Decision(
                    legal[0], (time.perf_counter() - started) * 1000,
                    algorithm=self.name,
                )

        self._nodes = 1
        moves = tactical_moves(board, player, len(legal))
        moves = defend_forks(board, player, moves, self._deadline)
        moves = moves[:self.max_candidates]
        best_move, best_value, completed_depth = moves[0], None, 0
        for depth in range(1, self.depth + 1):
            self._search_depth = depth
            current_move, current_value = moves[0], -math.inf
            alpha = -math.inf
            try:
                for move in moves:
                    if time.perf_counter() >= self._deadline:
                        raise SearchTimeout
                    board.play(*move, player)
                    try:
                        value = self._search(
                            board, depth - 1, -player, player, alpha, math.inf
                        )
                    finally:
                        board.undo()
                    if value > current_value:
                        current_move, current_value = move, value
                    if self.alpha_beta:
                        alpha = max(alpha, current_value)
            except SearchTimeout:
                break
            best_move, best_value = current_move, int(current_value)
            completed_depth = depth
            moves.remove(best_move)
            moves.insert(0, best_move)

        return Decision(
            best_move, (time.perf_counter() - started) * 1000,
            nodes=self._nodes, depth=completed_depth, algorithm=self.name,
            search_value=best_value,
        )
