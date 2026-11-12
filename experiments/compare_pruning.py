"""Đo MiniMax thuần và Alpha-Beta trên cùng các trạng thái Caro."""

import argparse
import csv
from pathlib import Path

from src.game import O, X, Board
from src.minimax import MinimaxAgent

STATES = [
    [(4, 4, X), (4, 5, O), (5, 4, X), (3, 4, O)],
    [(4, 3, X), (4, 4, O), (5, 5, X), (5, 4, O), (3, 5, X)],
    [(2, 2, X), (4, 4, O), (3, 3, X), (3, 4, O), (4, 3, X), (5, 5, O)],
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="experiments/results/pruning_comparison.csv"
    )
    args = parser.parse_args()

    rows = []
    for index, moves in enumerate(STATES, 1):
        board = Board()
        turn = X
        for row, col, turn in moves:
            board.play(row, col, turn)
        decisions = [
            MinimaxAgent(depth=3, alpha_beta=prune).choose_move(board, -turn)
            for prune in (False, True)
        ]
        raw, pruned = decisions
        assert raw.search_value == pruned.search_value and raw.move == pruned.move
        assert pruned.nodes <= raw.nodes
        for decision in decisions:
            rows.append({
                "state": index,
                "algorithm": decision.algorithm,
                "player": "X" if turn == O else "O",
                "depth": decision.depth,
                "move": str(tuple(value + 1 for value in decision.move)),
                "search_value": decision.search_value,
                "nodes": decision.nodes,
                "time_ms": round(decision.elapsed_ms, 3),
            })

    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(destination)


if __name__ == "__main__":
    main()
