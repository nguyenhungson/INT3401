"""Tìm kiếm cây Monte Carlo với UCT.

Điểm của một nút thuộc người vừa đi. Khi chọn nút con, tất cả điểm con
vì thế thuộc cùng người đang tới lượt. Một ván hòa cho mỗi bên 0,5 điểm.
"""

from __future__ import annotations

import math
import random
import time

from .agents import Decision, GameBoard, Move
from .game import DRAW, EMPTY


class Node:
    __slots__ = (
        "parent", "move", "player_just_moved", "visits",
        "reward", "children", "untried",
    )

    def __init__(
        self,
        board: GameBoard,
        player_just_moved: int,
        parent: Node | None = None,
        move: Move | None = None,
        max_candidates: int | None = 20,
    ) -> None:
        self.parent = parent
        self.move = move
        self.player_just_moved = player_just_moved
        self.visits = 0
        self.reward = 0.0
        self.children: list[Node] = []
        self.untried = board.candidate_moves(radius=1, limit=max_candidates)

    def best_uct_child(self, c: float) -> Node:
        parent_log = math.log(max(1, self.visits))

        def uct(child: Node) -> float:
            if child.visits == 0:
                return math.inf
            average_reward = child.reward / child.visits
            exploration = c * math.sqrt(parent_log / child.visits)
            return average_reward + exploration

        return max(self.children, key=uct)

    def backpropagate(self, outcome: int) -> None:
        node: Node | None = self
        while node is not None:
            node.visits += 1
            # Mỗi nút tính điểm cho người vừa đi để tạo ra nút đó.
            reward = float(outcome == node.player_just_moved)
            if outcome == DRAW:
                reward = 0.5
            node.reward += reward
            node = node.parent


class MCTSAgent:
    name = "MCTS"

    def __init__(
        self,
        iterations: int = 300,
        c: float = 1.4,
        seed: int = 42,
        max_candidates: int = 20,
        rollout: str = "random",
        *,
        tactical_root: bool = False,
    ) -> None:
        if iterations < 1 or not math.isfinite(c) or c <= 0 or max_candidates < 1:
            raise ValueError("Cần số mô phỏng, hệ số UCT và giới hạn ứng viên dương.")
        if rollout not in ("random", "heuristic"):
            raise ValueError("Rollout phải là random hoặc heuristic.")
        self.iterations = iterations
        self.c = c
        self.max_candidates = max_candidates
        self.rollout = rollout
        self.tactical_root = tactical_root
        self.rng = random.Random(seed)
        self.last_root: Node | None = None

    def _rollout_move(self, board: GameBoard, player: int) -> Move:
        if self.rollout == "random":
            return self.rng.choice(board.legal_moves())

        candidates = board.candidate_moves()
        if not candidates:
            return self.rng.choice(board.legal_moves())
        wins = board.winning_moves(player, candidates)
        if wins:
            return self.rng.choice(wins)
        blocks = board.winning_moves(-player, candidates)
        if blocks:
            return self.rng.choice(blocks)

        # Chỉ rút gọn các nước thường. Nước thắng và chặn đã xét toàn bộ.
        nearby = candidates[:8]
        scores = [board.line_score(*move, player) for move in nearby]
        best_score = max(scores)
        best_moves = [
            move for move, score in zip(nearby, scores) if score == best_score
        ]
        return self.rng.choice(best_moves)

    def choose_move(self, board: GameBoard, player: int) -> Decision:
        if board.status != EMPTY:
            raise ValueError("Ván cờ đã kết thúc.")
        started = time.perf_counter()
        if self.tactical_root:
            # Xét nước bắt buộc trước khi áp giới hạn ứng viên.
            root = Node(board, -player, max_candidates=None)
            wins = board.winning_moves(player, root.untried)
            blocks = board.winning_moves(-player, root.untried)
            if wins:
                root.untried = wins
            elif len(blocks) == 1:
                root.untried = blocks
            else:
                root.untried = root.untried[:self.max_candidates]
        else:
            root = Node(board, -player, max_candidates=self.max_candidates)
        self.last_root = root
        node_count = 1
        for _ in range(self.iterations):
            sim = board.copy()
            node = root
            turn = player
            # 1. Lựa chọn theo UCT khi đã mở rộng hết các nước của nút.
            while not node.untried and node.children and sim.status == EMPTY:
                node = node.best_uct_child(self.c)
                assert node.move is not None
                sim.play(*node.move, turn)
                turn = -turn
            # 2. Mở rộng một nước chưa thử.
            if node.untried and sim.status == EMPTY:
                idx = self.rng.randrange(len(node.untried))
                move = node.untried.pop(idx)
                sim.play(*move, turn)
                child = Node(
                    sim, turn, parent=node, move=move,
                    max_candidates=self.max_candidates,
                )
                node.children.append(child)
                node_count += 1
                node = child
                turn = -turn
            # 3. Mô phỏng tới khi có kết quả ván.
            while sim.status == EMPTY:
                move = self._rollout_move(sim, turn)
                sim.play(*move, turn)
                turn = -turn
            # 4. Lan truyền kết quả từ lá về gốc.
            node.backpropagate(sim.status)
        # Nếu bằng lượt thăm, max giữ nút được mở rộng trước theo seed.
        selected = max(root.children, key=lambda child: child.visits)
        assert selected.move is not None
        return Decision(
            selected.move,
            (time.perf_counter() - started) * 1000,
            nodes=node_count,
            iterations=self.iterations,
            algorithm=self.name,
        )
