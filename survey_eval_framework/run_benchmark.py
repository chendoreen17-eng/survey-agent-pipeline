import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="End-to-end survey agent benchmark pipeline")
    p.add_argument("--tasks-json", required=True)
    p.add_argument("--models-json", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--judge-model", default="gemini-2.5-pro")
    p.add_argument("--judge-api-key", default="env:LLMMELON_API_KEY")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen = out / "generation.jsonl"
    ev = out / "eval.jsonl"
    lb = out / "leaderboard.json"
    battles = out / "battles.jsonl"
    elo = out / "elo.json"

    py = sys.executable
    root = Path(__file__).resolve().parent

    run([py, str(root / "run_generation.py"), "--tasks-json", args.tasks_json, "--models-json", args.models_json, "--output-jsonl", str(gen)])
    run(
        [
            py,
            str(root / "run_eval.py"),
            "--tasks-json",
            args.tasks_json,
            "--generation-jsonl",
            str(gen),
            "--judge-output-jsonl",
            str(ev),
            "--judge-model",
            args.judge_model,
            "--judge-api-key",
            args.judge_api_key,
        ]
    )
    run([py, str(root / "aggregate_scores.py"), "--eval-jsonl", str(ev), "--output-json", str(lb)])
    run(
        [
            py,
            str(root / "run_pairwise_elo.py"),
            "--tasks-json",
            args.tasks_json,
            "--generation-jsonl",
            str(gen),
            "--battle-jsonl",
            str(battles),
            "--elo-json",
            str(elo),
            "--judge-model",
            args.judge_model,
            "--judge-api-key",
            args.judge_api_key,
        ]
    )

    print("[DONE] benchmark complete")
    print(f"- generation: {gen}")
    print(f"- eval: {ev}")
    print(f"- leaderboard: {lb}")
    print(f"- battles: {battles}")
    print(f"- elo: {elo}")


if __name__ == "__main__":
    main()
