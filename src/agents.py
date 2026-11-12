"""Giao diện tác nhân và thống kê của một quyết định."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Protocol

Move = tuple[int, int]


class GameBoard(Protocol):
    """Các thao tác chung để Random và MCTS dùng cho cả hai trò chơi."""

    status: int
    move_count: int

    def copy(self) -> GameBoard: ...
    def legal_moves(self) -> list[Move]: ...
    def candidate_moves(
        self, radius: int = 1, limit: int | None = None
    ) -> list[Move]: ...
    def is_legal(self, row: int, col: int) -> bool: ...
    def play(self, row: int, col: int, player: int) -> None: ...
    def winning_moves(self, player: int, candidates: list[Move]) -> list[Move]: ...
    def line_score(self, row: int, col: int, player: int) -> int: ...
    def pretty(self) -> str: ...


@dataclass
class Decision:
    move: Move
    elapsed_ms: float
    nodes: int = 0
    depth: int | None = None
    iterations: int | None = None
    algorithm: str = ""
    search_value: int | None = None

    def as_dict(self) -> dict:
        detail = asdict(self)
        detail["move"] = list(self.move)
        return detail


class Agent(Protocol):
    name: str

    def choose_move(self, board: GameBoard, player: int) -> Decision: ...


class RandomAgent:
    name = "Random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def choose_move(self, board: GameBoard, player: int) -> Decision:
        started = time.perf_counter()
        options = board.legal_moves()
        if not options:
            raise ValueError("Bàn cờ không còn nước đi hợp lệ.")
        move = self.rng.choice(options)
        elapsed_ms = (time.perf_counter() - started) * 1000
        return Decision(move, elapsed_ms, algorithm=self.name)
