"""Vẽ đồ thị từ CSV đã đo bằng chương trình benchmark."""

import csv
from pathlib import Path


def build_plots(csv_path: str | Path, output_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(csv_path).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exploration_rows = sorted(
        (row for row in rows if row["experiment"] == "uct_c"),
        key=lambda row: float(row["c"]),
    )
    iteration_rows = sorted(
        (row for row in rows if row["experiment"] == "iterations"),
        key=lambda row: int(row["iterations"]),
    )
    if exploration_rows:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        coefficients = [float(row["c"]) for row in exploration_rows]
        metrics = [("win_rate", "Win"), ("draw_rate", "Draw"), ("loss_rate", "Loss")]
        for metric, name in metrics:
            percentages = [100 * float(row[metric]) for row in exploration_rows]
            ax.plot(coefficients, percentages, marker="o", label=name)
        ax.set(
            xlabel="UCT exploration coefficient c",
            ylabel="Share of games (%)",
            title="Sensitivity to UCT exploration (fixed iterations)",
        )
        ax.set_ylim(0, 100)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "uct_c_win_draw_loss.png", dpi=180)
        plt.close(fig)
    if iteration_rows:
        fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
        iterations = [int(row["iterations"]) for row in iteration_rows]
        win_rates = [100 * float(row["win_rate"]) for row in iteration_rows]
        times_ms = [float(row["avg_move_ms"]) for row in iteration_rows]
        ax1.plot(iterations, win_rates, marker="o", label="Win rate")
        ax1.set(
            xlabel="MCTS iterations",
            ylabel="Win rate (%)",
            title="MCTS iterations: quality vs speed",
        )
        ax1.set_ylim(0, 100)
        ax2 = ax1.twinx()
        ax2.plot(
            iterations, times_ms, marker="s", linestyle="--", label="Inference time"
        )
        ax2.set_ylabel("Average move decision time (ms)")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")
        fig.tight_layout()
        fig.savefig(out / "iterations_winrate_time.png", dpi=180)
        plt.close(fig)
    rollout = next((row for row in rows if row["experiment"] == "rollout"), None)
    if rollout:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        rates = [float(rollout[key]) for key in ("win_rate", "draw_rate", "loss_rate")]
        bars = ax.bar(
            ["Heuristic wins", "Draws", "Random wins"],
            [100 * rate for rate in rates],
            color=["#206b74", "#8793a3", "#bc703b"],
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set(
            ylabel="Share of games (%)",
            title=f"Caro: heuristic vs random rollout ({rollout['games']} games)",
            ylim=(0, 110),
        )
        fig.tight_layout()
        fig.savefig(out / "rollout_comparison.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--output", default="experiments/results")
    args = parser.parse_args()
    build_plots(args.csv, args.output)
