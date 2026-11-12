"""Đối sánh tác nhân theo cặp ván đổi lượt đi trước."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path

from src.agents import Agent
from src.game import DRAW, EMPTY, O, X
from src.main import make_agent, make_board

GROUPS = ("baseline", "uct_c", "iterations", "rollout", "connect4")
UCT_VALUES = (0.5, 1.0, 1.4, 2.0)
FULL_ITERATIONS = (100, 300, 500)


def make(
    name: str, depth: int, iterations: int, c: float, seed: int, rollout: str = "random"
) -> Agent:
    return make_agent(name.lower(), depth, iterations, c, seed, rollout)


def play_match(
    focal: str,
    opponent: str,
    games: int,
    depth: int,
    iterations: int,
    c: float,
    seed: int,
    max_moves: int = 81,
    *,
    game: str = "caro",
    focal_rollout: str = "random",
    opponent_rollout: str = "random",
    game_rows: list[dict] | None = None,
) -> dict:
    if games < 2 or games % 2:
        raise ValueError("Số ván phải chẵn và ít nhất bằng 2.")

    outcomes = {"win": 0, "draw": 0, "loss": 0}
    all_times = []
    lengths = []
    for game_index in range(games):
        # Giữ nguyên seed riêng của mỗi tác nhân khi đổi bên trong cùng cặp.
        pair_index = game_index // 2
        focal_player = X if game_index % 2 == 0 else O
        focal_seed = seed + pair_index * 2
        opponent_seed = focal_seed + 1
        agents = {
            focal_player: make(focal, depth, iterations, c, focal_seed, focal_rollout),
            -focal_player: make(
                opponent, depth, iterations, c, opponent_seed, opponent_rollout
            ),
        }
        board = make_board(game)
        turn = X
        times = []
        while board.status == EMPTY:
            if board.move_count >= max_moves:
                raise RuntimeError("Ván chưa kết thúc khi chạm giới hạn an toàn.")
            agent = agents[turn]
            decision = agent.choose_move(board, turn)
            board.play(*decision.move, turn)
            if turn == focal_player:
                times.append(decision.elapsed_ms)
            turn = -turn

        if board.status == DRAW:
            outcome = "draw"
        elif board.status == focal_player:
            outcome = "win"
        else:
            outcome = "loss"
        outcomes[outcome] += 1
        all_times.extend(times)
        lengths.append(board.move_count)
        if game_rows is not None:
            game_rows.append({
                "game_number": game_index + 1,
                "pair": pair_index + 1,
                "first": "focal" if focal_player == X else "opponent",
                "focal_player": "X" if focal_player == X else "O",
                "focal_seed": focal_seed,
                "opponent_seed": opponent_seed,
                "winner": {X: "X", O: "O", DRAW: "draw"}[board.status],
                "outcome": outcome,
                "moves": board.move_count,
                "focal_moves": len(times),
                "focal_avg_move_ms": round(statistics.mean(times), 6),
            })

    return {
        "game": game,
        "focal": focal,
        "opponent": opponent,
        "focal_rollout": focal_rollout if focal == "MCTS" else "",
        "opponent_rollout": opponent_rollout if opponent == "MCTS" else "",
        "games": games,
        "wins": outcomes["win"],
        "draws": outcomes["draw"],
        "losses": outcomes["loss"],
        "win_rate": outcomes["win"] / games,
        "draw_rate": outcomes["draw"] / games,
        "loss_rate": outcomes["loss"] / games,
        "avg_move_ms": round(statistics.mean(all_times), 6),
        "avg_game_length": round(statistics.mean(lengths), 2),
        "depth": depth,
        "iterations": iterations,
        "c": c,
        "seed": seed,
    }


def configurations(args: argparse.Namespace) -> list[tuple[str, str, str, int, float]]:
    iteration_values = FULL_ITERATIONS if args.preset == "full" else (2, 4, 8)
    tasks = [
        ("baseline", "Random", "Random", args.fixed_iterations, args.c),
        ("baseline", "AlphaBeta", "Random", args.fixed_iterations, args.c),
        ("baseline", "MCTS", "Random", args.fixed_iterations, args.c),
        ("head_to_head", "MCTS", "AlphaBeta", args.fixed_iterations, args.c),
    ]
    tasks.extend(
        ("uct_c", "MCTS", args.opponent, args.fixed_iterations, c) for c in UCT_VALUES
    )
    tasks.extend(
        ("iterations", "MCTS", args.opponent, iterations, args.c)
        for iterations in iteration_values
    )
    tasks.extend([
        ("rollout", "MCTS", "MCTS", args.rollout_iterations, args.c),
        ("connect4", "MCTS", "Random", args.fixed_iterations, args.c),
    ])
    if args.group == "all":
        return tasks
    groups = {args.group}
    if args.group == "baseline":
        groups.add("head_to_head")
    return [task for task in tasks if task[0] in groups]


def append_csv(path: Path, rows: list[dict]) -> None:
    first_write = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        if first_write:
            writer.writeheader()
        writer.writerows(rows)


def run_suite(args: argparse.Namespace) -> list[dict]:
    if args.games < 2 or args.games % 2:
        raise ValueError("Số ván phải chẵn và ít nhất bằng 2.")
    if args.preset == "full" and args.games < 20:
        raise ValueError("Full benchmark cần ít nhất 20 ván cho mỗi cấu hình.")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / filename).exists() for filename in ("results.csv", "games.csv")):
        raise ValueError("Thư mục đã có CSV. Hãy chọn --output mới để giữ kết quả cũ.")

    tasks = configurations(args)
    metadata = {
        "date_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "parameters": vars(args),
        "configurations": tasks,
        "timing": "Wall clock per agent decision using perf_counter, no time cap.",
        "pairing": "Each pair swaps X/O and retains each agent's seed.",
        "game_rule": "X moves first. Only completed games contribute to results.",
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    results = []
    for index, (kind, focal, opponent, iterations, c) in enumerate(tasks, 1):
        print(
            f"[{index}/{len(tasks)}] {kind}: {focal} vs {opponent}, "
            f"iterations={iterations}, c={c}, games={args.games}",
            flush=True,
        )
        started = time.perf_counter()
        game_rows = []
        game = "connect4" if kind == "connect4" else "caro"
        rollout = "heuristic" if kind == "rollout" else "random"
        summary = play_match(
            focal, opponent, args.games, args.depth, iterations, c, args.seed,
            game=game, focal_rollout=rollout, game_rows=game_rows,
        )
        record = {
            "configuration": index,
            "experiment": kind,
            **summary,
            "wall_seconds": round(time.perf_counter() - started, 3),
        }
        # Lưu từng cấu hình đã hoàn tất để có thể xem kết quả khi dừng giữa chừng.
        append_csv(out / "games.csv", [
            {"configuration": index, "experiment": kind, **row} for row in game_rows
        ])
        append_csv(out / "results.csv", [record])
        results.append(record)
        print(
            f"  W/D/L={record['wins']}/{record['draws']}/{record['losses']} | "
            f"avg={record['avg_move_ms']:.3f} ms | {record['wall_seconds']} s",
            flush=True,
        )

    from experiments.plot_results import build_plots

    build_plots(out / "results.csv", out)
    print("Kết quả tại", out.resolve())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Thực nghiệm Caro và Connect Four.")
    parser.add_argument("--preset", choices=("smoke", "quick", "full"), default="full")
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--games", type=int, help="Số ván chẵn cho mỗi cấu hình.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--c", type=float, default=1.4)
    parser.add_argument("--fixed-iterations", type=int)
    parser.add_argument("--rollout-iterations", type=int)
    parser.add_argument("--opponent", choices=("Random", "AlphaBeta"), default="Random")
    parser.add_argument("--output", help="Thư mục xuất CSV, JSON và PNG.")
    args = parser.parse_args()

    full = args.preset == "full"
    if args.games is None:
        args.games = 20 if full else 2
    if args.fixed_iterations is None:
        args.fixed_iterations = 300 if full else 8
    if args.rollout_iterations is None:
        args.rollout_iterations = 100 if full else 8
    if args.output is None:
        args.output = f"experiments/results/{args.preset}_{args.group}"

    if min(args.depth, args.fixed_iterations, args.rollout_iterations) < 1:
        parser.error("Độ sâu và số mô phỏng phải lớn hơn 0.")
    if not math.isfinite(args.c) or args.c <= 0:
        parser.error("--c phải là số hữu hạn lớn hơn 0.")
    try:
        run_suite(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
