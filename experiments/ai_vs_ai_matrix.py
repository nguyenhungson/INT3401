"""Chạy và lưu từng ván AI đấu AI. Xem cách dùng với --help."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import combinations_with_replacement, product
from pathlib import Path
from uuid import uuid4

from src.agents import Agent
from src.game import DRAW, EMPTY, O, X
from src.main import SYMBOLS, make_agent, make_board

GROUPS = ("baseline", "rollout", "iterations", "depth", "tactical", "replay")
RESULTS = {X: "X", O: "O", DRAW: "draw"}
OUTPUT_ROOT = Path("experiments/results/ai_vs_ai")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AgentConfig:
    name: str
    depth: int | None = None
    iterations: int | None = None
    c: float | None = None
    rollout: str | None = None
    tactical: bool = False

    def create(self, seed: int) -> Agent:
        return make_agent(
            self.name, self.depth or 2, self.iterations or 300, self.c or 1.4,
            seed, self.rollout or "random", tactical=self.tactical,
        )


@dataclass(frozen=True)
class Matchup:
    matchup_id: str
    group: str
    game: str
    agent_a: AgentConfig
    agent_b: AgentConfig


@dataclass(frozen=True)
class GameSpec:
    matchup: Matchup
    pair: int
    a_side: str
    seed_a: int
    seed_b: int

    @property
    def game_id(self) -> str:
        return f"{self.matchup.matchup_id}__{self.pair:02d}_{self.a_side}"

    def header(self) -> dict:
        players = {
            self.a_side: {
                "slot": "A", "seed": self.seed_a, **asdict(self.matchup.agent_a),
            },
            "O" if self.a_side == "X" else "X": {
                "slot": "B", "seed": self.seed_b, **asdict(self.matchup.agent_b),
            },
        }
        return {
            "game_id": self.game_id,
            "matchup_id": self.matchup.matchup_id,
            "group": self.matchup.group,
            "game": self.matchup.game,
            "pair": self.pair,
            "coordinate_base": 0,
            "players": players,
        }


def build_matchups(groups: list[str]) -> list[Matchup]:
    random = AgentConfig("random")
    minimax = AgentConfig("minimax", depth=2)
    alphabeta = AgentConfig("alphabeta", depth=2)
    mcts = AgentConfig("mcts", iterations=300, c=1.4, rollout="random")
    cases: list[Matchup] = []

    def add(group: str, label: str, a: AgentConfig, b: AgentConfig, game="caro"):
        if group in groups:
            cases.append(Matchup(f"{group}__{game}__{label}", group, game, a, b))

    for game, agents in (
        ("caro", (random, minimax, alphabeta, mcts)),
        ("connect4", (random, mcts)),
    ):
        for a, b in combinations_with_replacement(agents, 2):
            add("baseline", f"{a.name}_vs_{b.name}", a, b, game)

    add("rollout", "random_vs_heuristic", mcts, replace(mcts, rollout="heuristic"))
    for count in (100, 300, 500, 1000):
        add("iterations", f"mcts_{count}", alphabeta, replace(mcts, iterations=count))
    for depth in (2, 3, 5):
        add("depth", f"alphabeta_{depth}", replace(alphabeta, depth=depth), mcts)
    for ab_flag, mcts_flag in product((False, True), repeat=2):
        label = f"ab_{int(ab_flag)}_mcts_{int(mcts_flag)}"
        add("tactical", label, replace(alphabeta, tactical=ab_flag),
            replace(mcts, tactical=mcts_flag))
        if ab_flag == mcts_flag:
            add("replay", label, replace(alphabeta, depth=5, tactical=ab_flag),
                replace(mcts, iterations=1000, tactical=mcts_flag))
    return cases


def build_schedule(cases: list[Matchup], seeds: list[list[int]]) -> dict[str, GameSpec]:
    schedule = {}
    for case in cases:
        if case.group == "replay":
            spec = GameSpec(case, 1, "X", 42, 43)
            schedule[spec.game_id] = spec
            continue
        for pair, (seed_a, seed_b) in enumerate(seeds, 1):
            for side in ("X", "O"):
                spec = GameSpec(case, pair, side, seed_a, seed_b)
                schedule[spec.game_id] = spec
    return schedule


def play_game(spec: GameSpec) -> dict:
    board = make_board(spec.matchup.game)
    agent_a = spec.matchup.agent_a.create(spec.seed_a)
    agent_b = spec.matchup.agent_b.create(spec.seed_b)
    a_player = X if spec.a_side == "X" else O
    agents = {a_player: agent_a, -a_player: agent_b}
    moves = []
    player = X
    while board.status == EMPTY:
        decision = agents[player].choose_move(board, player)
        row, col = decision.move
        board.play(row, col, player)
        detail = decision.as_dict()
        detail.pop("move")
        moves.append({
            "turn": board.move_count, "player": SYMBOLS[player], "row": row, "col": col,
            **detail,
        })
        player = -player
    return {
        **spec.header(),
        "winner": RESULTS[board.status],
        "move_count": board.move_count,
        "moves": moves,
        "final_board": board.pretty(),
    }


def replay_game(game: dict, spec: GameSpec) -> str:
    for key, expected in spec.header().items():
        if game[key] != expected:
            raise ValueError(f"{spec.game_id}: sai thông tin {key}.")
    board = make_board(game["game"])
    player = X
    for turn, move in enumerate(game["moves"], 1):
        if move["turn"] != turn or move["player"] != SYMBOLS[player]:
            raise ValueError(f"{spec.game_id}: sai thứ tự ở lượt {turn}.")
        row, col = move["row"], move["col"]
        if type(row) is not int or type(col) is not int or not board.is_legal(row, col):
            raise ValueError(f"{spec.game_id}: nước không hợp lệ ở lượt {turn}.")
        if not math.isfinite(move["elapsed_ms"]) or move["elapsed_ms"] < 0:
            raise ValueError(f"{spec.game_id}: thời gian không hợp lệ ở lượt {turn}.")
        board.play(row, col, player)
        player = -player
    if RESULTS.get(board.status) != game["winner"] or board.status == EMPTY:
        raise ValueError(f"{spec.game_id}: kết quả không khớp bàn cuối.")
    if game["move_count"] != board.move_count or game["final_board"] != board.pretty():
        raise ValueError(f"{spec.game_id}: bàn cuối hoặc số lượt không khớp.")
    return board.pretty()


def read_jsonl(path: Path, *, repair_tail: bool = False) -> list[dict]:
    """Khi tiếp tục, chỉ sửa dòng cuối bị cắt giữa chừng do gián đoạn."""
    records = []
    with path.open("r+b" if repair_tail else "rb") as stream:
        while line := stream.readline():
            complete = line.endswith(b"\n")
            try:
                records.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if not (repair_tail and not complete):
                    line_number = len(records) + 1
                    message = f"{path}: JSON lỗi ở dòng {line_number}."
                    raise ValueError(message) from error
                stream.seek(-len(line), os.SEEK_CUR)
                stream.truncate()
            else:
                if repair_tail and not complete:
                    stream.write(b"\n")
    return records


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def lock_run(directory: Path):
    """Cho phép một tiến trình ghi. Hệ điều hành nhả khóa khi tiến trình thoát."""
    with (directory / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Thư mục đang được sử dụng: {directory}") from error
        yield


def load_games(path: Path, schedule: dict[str, GameSpec], *, resume=False) -> dict:
    games = {}
    for game in read_jsonl(path, repair_tail=resume):
        game_id = game["game_id"]
        if game_id not in schedule or game_id in games:
            raise ValueError(f"Ván trùng hoặc không thuộc lịch chạy: {game_id}")
        replay_game(game, schedule[game_id])
        games[game_id] = game
    return games


def summarize(cases: list[Matchup], schedule: dict, games: dict) -> list[dict]:
    rows = []
    for case in cases:
        planned = sum(spec.matchup == case for spec in schedule.values())
        completed = [
            game for game in games.values() if game["matchup_id"] == case.matchup_id
        ]
        move_counts = Counter(
            tuple((move["row"], move["col"]) for move in game["moves"])
            for game in completed
        )
        deterministic = all(
            agent.name in ("minimax", "alphabeta")
            for agent in (case.agent_a, case.agent_b)
        )
        for slot, agent in (("A", case.agent_a), ("B", case.agent_b)):
            outcomes = Counter()
            side_games = Counter()
            side_wins = Counter()
            times = []
            for game in completed:
                side = "X" if game["players"]["X"]["slot"] == slot else "O"
                side_games[side] += 1
                times.extend(move["elapsed_ms"] for move in game["moves"]
                             if move["player"] == side)
                if game["winner"] == side:
                    outcomes["wins"] += 1
                    side_wins[side] += 1
                elif game["winner"] == "draw":
                    outcomes["draws"] += 1
                else:
                    outcomes["losses"] += 1

            rows.append({
                "matchup_id": case.matchup_id, "group": case.group, "game": case.game,
                "slot": slot, "agent": agent.name, "depth": agent.depth,
                "iterations": agent.iterations, "c": agent.c, "rollout": agent.rollout,
                "tactical": agent.tactical,
                "planned_games": planned, "completed_games": len(completed),
                "wins": outcomes["wins"], "draws": outcomes["draws"],
                "losses": outcomes["losses"],
                "games_as_x": side_games["X"], "wins_as_x": side_wins["X"],
                "win_rate_as_x": ratio(side_wins["X"], side_games["X"]),
                "games_as_o": side_games["O"], "wins_as_o": side_wins["O"],
                "win_rate_as_o": ratio(side_wins["O"], side_games["O"]),
                "avg_game_moves": ratio(sum(game["move_count"] for game in completed),
                                        len(completed)),
                "avg_move_ms": ratio(sum(times), len(times)),
                "unique_move_sequences": len(move_counts),
                "repeated_move_sequences": len(completed) - len(move_counts),
                "both_agents_deterministic": deterministic,
            })
    return rows


def ratio(numerator: float, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_summary(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_plan(directory: Path) -> tuple[dict, list[Matchup], dict[str, GameSpec]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Phiên bản manifest không tương thích.")
    cases = [
        Matchup(
            item["matchup_id"], item["group"], item["game"],
            AgentConfig(**item["agent_a"]), AgentConfig(**item["agent_b"]),
        )
        for item in manifest["matchups"]
    ]
    schedule = build_schedule(cases, manifest["seed_schedule"])
    if len(schedule) != manifest["planned_games"]:
        raise ValueError("Số ván trong manifest không khớp lịch chạy.")
    return manifest, cases, schedule


def run(directory: Path, manifest: dict, cases: list[Matchup], schedule: dict) -> int:
    games_path = directory / "games.jsonl"
    errors_path = directory / "errors.jsonl"
    with lock_run(directory):
        games_path.touch(exist_ok=True)
        errors_path.touch(exist_ok=True)
        games = load_games(games_path, schedule, resume=True)
        errors = read_jsonl(errors_path, repair_tail=True)
        print(
            f"{directory} | Dự kiến {len(schedule)} ván, đã có {len(games)}",
            flush=True,
        )
        manifest["status"] = "running"
        write_manifest(directory / "manifest.json", manifest)
        try:
            for game_id, spec in schedule.items():
                if game_id in games:
                    continue
                try:
                    game = play_game(spec)
                    replay_game(game, spec)
                except Exception as error:
                    entry = {"game_id": game_id, "time_utc": utc_now(),
                             "type": type(error).__name__, "error": str(error)}
                    append_jsonl(errors_path, entry)
                    errors.append(entry)
                    print(f"Lỗi {game_id}: {error}", file=sys.stderr, flush=True)
                    continue
                append_jsonl(games_path, game)
                games[game_id] = game
                print(f"[{len(games)}/{len(schedule)}] {game_id}: "
                      f"{game['winner']} ({game['move_count']} lượt)", flush=True)
        finally:
            write_summary(directory / "summary.csv", summarize(cases, schedule, games))
            pending = [game_id for game_id in schedule if game_id not in games]
            manifest.update(
                status="complete" if not pending else "partial",
                completed_games=len(games), pending_games=len(pending),
                missing_game_ids=pending, error_attempts=len(errors),
                updated_at_utc=utc_now(),
            )
            write_manifest(directory / "manifest.json", manifest)
    return 0 if not pending else 1


def verify(directory: Path, game_id: str | None) -> None:
    with lock_run(directory):
        manifest, cases, schedule = load_plan(directory)
        games = load_games(directory / "games.jsonl", schedule)
        pending = [game_id for game_id in schedule if game_id not in games]
        expected_progress = {
            "status": "partial" if pending else "complete",
            "completed_games": len(games),
            "pending_games": len(pending),
            "missing_game_ids": pending,
            "error_attempts": len(read_jsonl(directory / "errors.jsonl")),
        }
        if any(manifest.get(key) != value for key, value in expected_progress.items()):
            raise ValueError("Tiến độ trong manifest không khớp nhật ký.")
        with (directory / "summary.csv").open(newline="", encoding="utf-8") as stream:
            actual = list(csv.DictReader(stream))
        expected = [
            {key: "" if value is None else str(value) for key, value in row.items()}
            for row in summarize(cases, schedule, games)
        ]
        if actual != expected:
            raise ValueError("summary.csv không khớp dữ liệu từng ván.")
        if not games:
            raise ValueError("Chưa có ván để phát lại.")
        selected = game_id or next(iter(games))
        if selected not in games:
            raise ValueError(f"Không tìm thấy ván {selected}.")
        game = games[selected]
        print(f"Đã kiểm tra {len(games)}/{len(schedule)} ván và summary.csv.")
        print(f"Phát lại {selected} | {game['winner']} | {game['move_count']} lượt")
        print(game["final_board"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument("--resume", type=Path, help="Tiếp tục từ thư mục kết quả.")
    existing.add_argument(
        "--verify", type=Path, help="Kiểm tra kết quả và phát lại log."
    )
    parser.add_argument("--game-id", help="Ván hiển thị khi dùng --verify.")
    parser.add_argument("--preset", choices=("smoke", "full"))
    parser.add_argument(
        "--group", nargs="+", choices=GROUPS, help="Mặc định: baseline."
    )
    parser.add_argument("--seed", type=int, help="Seed của cặp đầu. Mặc định: 2026.")
    parser.add_argument("--run-id", help=f"Tên thư mục mới trong {OUTPUT_ROOT}.")
    args = parser.parse_args()
    if args.game_id and not args.verify:
        parser.error("--game-id cần đi cùng --verify.")
    if (args.resume or args.verify) and any(
        value is not None for value in (args.preset, args.group, args.seed, args.run_id)
    ):
        parser.error("Khi mở thư mục hiện có, cấu hình được lấy từ manifest.")

    try:
        if args.verify:
            verify(args.verify, args.game_id)
            return 0
        if args.resume:
            manifest, cases, schedule = load_plan(args.resume)
            return run(args.resume, manifest, cases, schedule)

        groups = args.group or ["baseline"]
        preset = args.preset or "full"
        seed = args.seed if args.seed is not None else 2026
        pairs = 1 if preset == "smoke" else 10
        seeds = [[seed + 2 * index, seed + 2 * index + 1] for index in range(pairs)]
        cases = build_matchups(groups)
        schedule = build_schedule(cases, seeds)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = args.run_id or f"{stamp}_{uuid4().hex[:6]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
            raise ValueError("run-id chỉ gồm chữ, số, gạch ngang và gạch dưới.")
        directory = OUTPUT_ROOT / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": SCHEMA_VERSION, "run_id": run_id, "preset": preset,
            "groups": groups, "games": sorted({case.game for case in cases}),
            "seed_schedule": seeds, "replay_seeds": {"X": 42, "O": 43},
            "coordinate_base": 0, "x_moves_first": True,
            "matchups": [asdict(case) for case in cases],
            "planned_games": len(schedule), "completed_games": 0, "status": "planned",
            "argv": [
                sys.executable, "-m", "experiments.ai_vs_ai_matrix", *sys.argv[1:],
            ],
            "created_at_utc": utc_now(), "python": platform.python_version(),
            "platform": platform.platform(),
        }
        write_manifest(directory / "manifest.json", manifest)
        return run(directory, manifest, cases, schedule)
    except KeyboardInterrupt:
        print("Đã dừng. Dùng --resume để tiếp tục.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError) as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
