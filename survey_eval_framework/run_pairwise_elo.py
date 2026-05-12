import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common import append_jsonl, chat_completion, extract_json_from_text, get_env_or_value, load_json, load_jsonl


K_FACTOR = 24


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra: float, rb: float, sa: float) -> Tuple[float, float]:
    ea = expected(ra, rb)
    eb = expected(rb, ra)
    sb = 1.0 - sa
    return ra + K_FACTOR * (sa - ea), rb + K_FACTOR * (sb - eb)


def build_prompt(task: Dict[str, Any], a_model: str, a_text: str, b_model: str, b_text: str) -> str:
    gold = task.get("gold", {})
    return (
        "You are a strict benchmark judge for survey quality.\n"
        f"Topic: {task.get('topic','')} | L2: {task.get('l2_domain','')}\n"
        f"Gold key points: {gold.get('key_points', [])}\n"
        f"Gold gaps: {gold.get('gaps', [])}\n"
        f"Gold future works: {gold.get('future_works', [])}\n\n"
        f"Response A ({a_model}):\n{a_text}\n\n"
        f"Response B ({b_model}):\n{b_text}\n\n"
        "Return STRICT JSON only:\n"
        "{\n"
        "  \"winner\": \"A|B|Tie\",\n"
        "  \"confidence\": 0.0,\n"
        "  \"reason\": \"short reason\"\n"
        "}\n"
        "Judge based on accuracy, coverage, grounding, and gap-future alignment."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run pairwise battles and compute Elo ranking.")
    p.add_argument("--tasks-json", required=True)
    p.add_argument("--generation-jsonl", required=True)
    p.add_argument("--battle-jsonl", required=True)
    p.add_argument("--elo-json", required=True)
    p.add_argument("--judge-base-url", default="https://llmmelon.cloud")
    p.add_argument("--judge-model", default="gemini-2.5-pro")
    p.add_argument("--judge-api-key", default="env:LLMMELON_API_KEY")
    p.add_argument("--timeout-sec", type=int, default=240)
    args = p.parse_args()

    tasks = load_json(Path(args.tasks_json))
    task_map = {t.get("task_id", ""): t for t in tasks if t.get("task_id")}
    gens = load_jsonl(Path(args.generation_jsonl))
    api_key = get_env_or_value(args.judge_api_key)
    if not api_key:
        raise ValueError("judge api key missing")

    by_task = defaultdict(list)
    for g in gens:
        if g.get("ok") and g.get("response"):
            by_task[g.get("task_id", "")].append(g)

    elo = defaultdict(lambda: 1000.0)

    for task_id, rows in by_task.items():
        task = task_map.get(task_id)
        if not task or len(rows) < 2:
            continue
        for a, b in itertools.combinations(rows, 2):
            a_id, b_id = a.get("model_id", ""), b.get("model_id", "")
            prompt = build_prompt(task, a_id, a.get("response", ""), b_id, b.get("response", ""))
            resp = chat_completion(
                base_url=args.judge_base_url,
                api_key=api_key,
                model=args.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout_sec=args.timeout_sec,
                max_retries=3,
            )
            parsed = extract_json_from_text(resp.get("content", "")) if resp.get("ok") else None
            winner = "Tie"
            confidence = 0.0
            reason = resp.get("error", "")
            if isinstance(parsed, dict):
                winner = str(parsed.get("winner", "Tie"))
                confidence = float(parsed.get("confidence", 0) or 0)
                reason = str(parsed.get("reason", ""))

            if winner == "A":
                sa = 1.0
            elif winner == "B":
                sa = 0.0
            else:
                sa = 0.5

            elo[a_id], elo[b_id] = update_elo(elo[a_id], elo[b_id], sa)

            append_jsonl(
                Path(args.battle_jsonl),
                {
                    "task_id": task_id,
                    "model_a": a_id,
                    "model_b": b_id,
                    "winner": winner,
                    "confidence": confidence,
                    "reason": reason,
                    "judge_ok": resp.get("ok", False),
                },
            )
            print(f"[BATTLE] {task_id} {a_id} vs {b_id} -> {winner}")

    ranked = sorted([{"model_id": k, "elo": round(v, 2)} for k, v in elo.items()], key=lambda x: x["elo"], reverse=True)
    from common import dump_json

    dump_json(Path(args.elo_json), {"ranking": ranked})
    print(f"[DONE] elo -> {args.elo_json}")


if __name__ == "__main__":
    main()
