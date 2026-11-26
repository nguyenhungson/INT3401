"""Nước đi chiến thuật và phép giải chính xác thế cuối cho hai tác nhân."""

from __future__ import annotations

import time

from .agents import GameBoard, Move
from .game import DRAW, EMPTY


class SearchTimeout(Exception):
    """Ngân sách tìm kiếm đã hết."""


def tactical_moves(board: GameBoard, player: int, limit: int) -> list[Move]:
    """Giữ nước thắng hoặc chặn bắt buộc, rồi xếp nước thường theo chuỗi quân."""
    candidates = board.candidate_moves()
    wins = board.winning_moves(player, candidates)
    if wins:
        return wins
    blocks = board.winning_moves(-player, candidates)
    if blocks:
        return blocks
    return sorted(
        candidates,
        key=lambda move: (
            board.line_score(*move, player) + board.line_score(*move, -player)
        ),
        reverse=True,
    )[:limit]


def forced_win(
    board: GameBoard, player: int, deadline: float
) -> tuple[Move, int] | None:
    """Chứng minh nước thắng ngay hoặc tạo hai nước thắng không thể chặn hết."""
    legal = board.legal_moves()
    wins = board.winning_moves(player, legal)
    if wins:
        return wins[0], 1
    for move in board.candidate_moves():
        if time.perf_counter() >= deadline:
            return None
        child = board.copy()
        child.play(*move, player)
        if child.status != EMPTY:
            continue
        replies = child.legal_moves()
        threats = child.winning_moves(player, replies)
        if len(threats) > 1 and not child.winning_moves(-player, replies):
            return move, 3
    return None


def defend_forks(
    board: GameBoard, player: int, moves: list[Move], deadline: float
) -> list[Move]:
    """Lọc nước để đối phương có thắng ngay hoặc tạo bốn hở hai đầu."""
    safe = []
    for move in moves:
        if time.perf_counter() >= deadline:
            return moves
        child = board.copy()
        child.play(*move, player)
        if child.status != EMPTY or forced_win(child, -player, deadline) is None:
            safe.append(move)
    return safe or moves


def solve_endgame(
    board: GameBoard, player: int, deadline: float
) -> tuple[Move, int, int]:
    """Duyệt các nước hợp lệ tới kết thúc. Giá trị thuộc bên đang đi.

    Thắng = 1, hòa = 0, thua = -1. Chỉ trả kết quả sau khi đã chứng minh.
    Khi có ngoại lệ hết giờ, mọi nước tạm đều được hoàn tác.
    """
    cache: dict[tuple, int] = {}
    nodes = 0

    def search(position: GameBoard, turn: int) -> tuple[Move | None, int]:
        nonlocal nodes
        if time.perf_counter() >= deadline:
            raise SearchTimeout
        nodes += 1
        if position.status == DRAW:
            return None, 0
        if position.status != EMPTY:
            return None, 1 if position.status == turn else -1
        key = (tuple(tuple(row) for row in position.cells), turn)
        if key in cache:
            return None, cache[key]

        legal = position.legal_moves()
        wins = position.winning_moves(turn, legal)
        if wins:
            return wins[0], 1
        blocks = position.winning_moves(-turn, legal)
        if len(blocks) > 1:
            return blocks[0], -1
        moves = blocks or legal
        best_move, best_value = moves[0], -2
        for move in moves:
            # Bản sao dùng được cho cả Caro và Connect Four.
            child = position.copy()
            child.play(*move, turn)
            _, child_value = search(child, -turn)
            value = -child_value
            if value > best_value:
                best_move, best_value = move, value
            if best_value == 1:
                break
        cache[key] = best_value
        return best_move, best_value

    move, value = search(board, player)
    assert move is not None
    return move, value, nodes
