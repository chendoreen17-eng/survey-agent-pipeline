import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import dump_json, load_json, load_jsonl


DEFAULT_WEIGHTS = {
    # General writing quality
    "coverage": 0.08,
    "factual_consistency": 0.10,
    "structure_clarity": 0.04,
    "citation_grounding": 0.08,
    "gap_future_alignment": 0.07,
    "novel_insight": 0.03,
    # Research-trajectory core metrics
    "l1_l2_routing_accuracy": 0.12,
    "evolution_chain_coherence": 0.12,
    "gap_identification_quality": 0.10,
    "gap_future_linkage": 0.12,
    "citation_semantic_grounding": 0.10,
    "evidence_traceability": 0.04,
}


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def content_score(scores: Dict[str, Any], weights: Dict[str, float]) -> float:
    total = 0.0
    for k, w in weights.items():
        v = scores.get(k, 0)
        try:
            v = float(v)
        except Exception:
            v = 0.0
        total += (v / 5.0) * w
    return total


def quiz_score(quiz_rows: List[Dict[str, Any]]) -> float:
    if not quiz_rows:
        return 0.0
    vals = []
    for q in quiz_rows:
        try:
            vals.append(float(q.get("score", 0)) / 2.0)
        except Exception:
            vals.append(0.0)
    return mean(vals)


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate evaluation outputs into leaderboard.")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--weights-json", default="")
    p.add_argument("--alpha-content", type=float, default=0.7)
    p.add_argument("--beta-quiz", type=float, default=0.3)
    args = p.parse_args()

    weights = DEFAULT_WEIGHTS.copy()
    if args.weights_json:
        w = load_json(Path(args.weights_json))
        if isinstance(w, dict):
            weights.update({k: float(v) for k, v in w.items() if k in weights})

    rows = load_jsonl(Path(args.eval_jsonl))
    by_model = defaultdict(list)
    for r in rows:
        by_model[r.get("model_id", "unknown")].append(r)

    leaderboard = []
    for model_id, ms in by_model.items():
        c_scores = []
        q_scores = []
        traj_scores = []
        ok_count = 0
        for r in ms:
            if r.get("status") != "ok":
                continue
            ok_count += 1
            c_scores.append(content_score(r.get("scores", {}), weights))
            t = r.get("trajectory_scores", {})
            if not isinstance(t, dict):
                t = {}
            # Only trajectory components, normalized to [0,1] by dividing by 5 and by sum of trajectory weights.
            traj_keys = [
                "l1_l2_routing_accuracy",
                "evolution_chain_coherence",
                "gap_identification_quality",
                "gap_future_linkage",
                "citation_semantic_grounding",
                "evidence_traceability",
            ]
            tw = sum(weights.get(k, 0.0) for k in traj_keys)
            if tw > 0:
                tv = 0.0
                for k in traj_keys:
                    try:
                        vv = float(t.get(k, 0))
                    except Exception:
                        vv = 0.0
                    tv += (vv / 5.0) * weights.get(k, 0.0)
                traj_scores.append(tv / tw)
            else:
                traj_scores.append(0.0)
            q_scores.append(quiz_score(r.get("quiz", [])))

        c_mean = mean(c_scores)
        t_mean = mean(traj_scores)
        q_mean = mean(q_scores)
        final = args.alpha_content * c_mean + args.beta_quiz * q_mean
        leaderboard.append(
            {
                "model_id": model_id,
                "num_tasks_total": len(ms),
                "num_tasks_ok": ok_count,
                "content_score": round(c_mean, 4),
                "trajectory_score": round(t_mean, 4),
                "quiz_score": round(q_mean, 4),
                "final_score": round(final, 4),
            }
        )

    leaderboard.sort(key=lambda x: x["final_score"], reverse=True)
    out = {
        "weights": weights,
        "alpha_content": args.alpha_content,
        "beta_quiz": args.beta_quiz,
        "leaderboard": leaderboard,
    }
    dump_json(Path(args.output_json), out)
    print(f"[DONE] {args.output_json}")


if __name__ == "__main__":
    main()
