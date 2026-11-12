"""Chạy Caro hoặc Connect Four từ dòng lệnh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .agents import Agent, GameBoard, Move, RandomAgent
from .connect4 import ConnectFour
from .game import DRAW, EMPTY, O, X, Board
from .mcts import MCTSAgent
from .minimax import MinimaxAgent

AGENTS = ("random", "minimax", "alphabeta", "mcts")
SYMBOLS = {X: "X", O: "O"}


def make_board(game: str) -> GameBoard:
    if game == "connect4":
        return ConnectFour()
    if game == "caro":
        return Board()
    raise ValueError(f"Trò chơi không hợp lệ: {game}")


def make_agent(
    name: str,
    depth: int,
    iterations: int,
    c: float,
    seed: int,
    rollout: str = "random",
    alpha_beta: bool | None = None,
    *,
    tactical: bool = False,
) -> Agent:
    if name == "random":
        return RandomAgent(seed)
    if name in ("minimax", "alphabeta"):
        pruning = name == "alphabeta" if alpha_beta is None else alpha_beta
        return MinimaxAgent(depth, alpha_beta=pruning, prefer_quick_wins=tactical)
    if name == "mcts":
        return MCTSAgent(iterations, c, seed, rollout=rollout, tactical_root=tactical)
    raise ValueError(f"Thuật toán không hợp lệ: {name}")


def human_move(board: GameBoard) -> Move:
    while True:
        try:
            if isinstance(board, ConnectFour):
                col = int(input("Nhập cột (1–7): ")) - 1
                move = board.drop_position(col)
                if move is not None:
                    return move
            else:
                entry = input("Nhập hàng và cột (1–9), ví dụ 5 5: ")
                row, col = (int(value) - 1 for value in entry.replace(",", " ").split())
                if board.is_legal(row, col):
                    return row, col
        except ValueError:
            pass
        print("Nước đi không hợp lệ. Hãy thử lại.")


def run(args: argparse.Namespace) -> str:
    board = make_board(args.game)
    agents = {}
    for player, name, seed in ((X, args.x, args.seed), (O, args.o, args.seed + 1)):
        agents[player] = None
        if name != "human":
            agents[player] = make_agent(
                name, args.depth, args.iterations, args.c, seed,
                args.rollout, args.alpha_beta, tactical=args.tactical,
            )

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logfile:
        player = X
        print(board.pretty())
        while board.status == EMPTY:
            agent = agents[player]
            symbol = SYMBOLS[player]
            if agent is None:
                print(f"Lượt của {symbol}")
                move = human_move(board)
                detail = {"algorithm": "Human", "move": list(move)}
            else:
                decision = agent.choose_move(board, player)
                move = decision.move
                detail = decision.as_dict()
                row, col = move
                position = f"({row + 1}, {col + 1})"
                if args.game == "connect4":
                    position = f"cột {col + 1}"
                print(
                    f"{symbol} {agent.name}: {position} | "
                    f"{decision.elapsed_ms:.2f} ms | nút={decision.nodes} | "
                    f"độ sâu={decision.depth or '—'} | "
                    f"mô phỏng={decision.iterations or '—'}"
                )

            board.play(*move, player)
            detail.update(turn=board.move_count, player=symbol)
            logfile.write(json.dumps(detail, ensure_ascii=False) + "\n")
            if not args.quiet or agents[-player] is None or board.status != EMPTY:
                print("\n" + board.pretty() + "\n")
            player = -player

    result = "Hòa" if board.status == DRAW else f"{SYMBOLS[board.status]} thắng"
    print(f"Kết thúc ván. {result}. Nhật ký ở {log_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Chơi Caro 9×9 hoặc Connect Four.")
    parser.add_argument("--game", choices=("caro", "connect4"), default="caro")
    parser.add_argument("--mode", choices=("human-ai", "ai-ai"))
    parser.add_argument("--agent", choices=AGENTS, help="Máy trong chế độ human-ai.")
    parser.add_argument("--agent1", choices=AGENTS, help="Máy X trong chế độ ai-ai.")
    parser.add_argument("--agent2", choices=AGENTS, help="Máy O trong chế độ ai-ai.")
    parser.add_argument(
        "--x", choices=("human", *AGENTS), help="Người chơi X đi trước."
    )
    parser.add_argument("--o", choices=("human", *AGENTS), help="Người chơi O đi sau.")
    parser.add_argument("--depth", type=int, default=2, help="Độ sâu MiniMax.")
    parser.add_argument("--iterations", type=int, default=300, help="Số mô phỏng MCTS.")
    parser.add_argument("--c", type=float, default=1.4, help="Hệ số khám phá UCT.")
    parser.add_argument("--alpha-beta", action=argparse.BooleanOptionalAction)
    parser.add_argument("--rollout", choices=("random", "heuristic"), default="random")
    parser.add_argument(
        "--tactical",
        action="store_true",
        help="Ưu tiên thắng sớm trong MiniMax và xử lý nước bắt buộc tại gốc MCTS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log", default="experiments/results/game_log.jsonl")
    parser.add_argument(
        "--quiet", action="store_true", help="Ẩn bàn trung gian khi máy đi."
    )
    args = parser.parse_args()

    if args.mode and (args.x or args.o):
        parser.error("Chọn --mode hoặc --x/--o.")
    if args.mode == "ai-ai":
        if args.agent:
            parser.error("Dùng --agent1 và --agent2 với --mode ai-ai.")
        args.x, args.o = args.agent1 or "mcts", args.agent2 or "random"
    elif args.mode == "human-ai":
        if args.agent1 or args.agent2:
            parser.error("Dùng --agent với --mode human-ai.")
        args.x, args.o = "human", args.agent or "mcts"
    else:
        if args.agent or args.agent1 or args.agent2:
            parser.error("Các cờ --agent, --agent1 và --agent2 cần --mode.")
        args.x, args.o = args.x or "human", args.o or "mcts"

    if args.depth < 1 or args.iterations < 1:
        parser.error("--depth và --iterations phải là số nguyên dương.")
    if not math.isfinite(args.c) or args.c <= 0:
        parser.error("--c phải là số hữu hạn lớn hơn 0.")
    if args.game == "connect4" and {"minimax", "alphabeta"} & {args.x, args.o}:
        parser.error("Connect Four hỗ trợ Random và MCTS. MiniMax dùng cho Caro.")

    try:
        run(args)
    except (EOFError, KeyboardInterrupt):
        parser.exit(1, "Đã dừng ván cờ.\n")
    except OSError as error:
        parser.exit(1, f"Không thể ghi nhật ký. {error}\n")


if __name__ == "__main__":
    main()
