import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import append_jsonl, chat_completion, extract_json_from_text, get_env_or_value, load_json, load_jsonl


JUDGE_SCHEMA = {
    "content_scores": {
        "coverage": "0-5",
        "factual_consistency": "0-5",
        "structure_clarity": "0-5",
        "citation_grounding": "0-5",
        "gap_future_alignment": "0-5",
        "novel_insight": "0-5",
    },
    "trajectory_scores": {
        "l1_l2_routing_accuracy": "0-5",
        "evolution_chain_coherence": "0-5",
        "gap_identification_quality": "0-5",
        "gap_future_linkage": "0-5",
        "citation_semantic_grounding": "0-5",
        "evidence_traceability": "0-5",
    },
    "quiz_eval": [
        {
            "question": "string",
            "gold_answer": "string",
            "predicted_from_report": "string",
            "score": "0|1|2",
            "rationale": "string",
        }
    ],
    "summary": {
        "strengths": ["string"],
        "weaknesses": ["string"],
    },
}


def build_judge_prompt(task: Dict[str, Any], candidate: str) -> str:
    gold = task.get("gold", {})
    quiz = gold.get("quiz", [])
    routing_target = gold.get("routing_target", {})
    evolution_chain = gold.get("evolution_chain_expected", [])
    semantic_profile = gold.get("semantic_intent_profile", {})
    expected_trace_papers = gold.get("expected_trace_papers", [])
    return (
        "You are an evaluator for survey quality. Score the candidate report with strict standards.\n"
        "Use only provided task evidence and gold references.\n\n"
        f"Task topic: {task.get('topic','')}\n"
        f"L1: {task.get('l1_domain','')}\n"
        f"L2: {task.get('l2_domain','')}\n\n"
        f"Gold routing target: {routing_target}\n"
        f"Gold evolution chain signals: {evolution_chain}\n"
        f"Gold key points: {gold.get('key_points', [])}\n"
        f"Gold gaps: {gold.get('gaps', [])}\n"
        f"Gold future works: {gold.get('future_works', [])}\n"
        f"Gold citation semantic profile: {semantic_profile}\n"
        f"Expected trace papers: {expected_trace_papers}\n"
        f"Quiz set: {quiz}\n\n"
        "Candidate report:\n"
        f"{candidate}\n\n"
        "Return STRICT JSON only with this schema:\n"
        f"{JUDGE_SCHEMA}\n"
        "Scoring guidance:\n"
        "- coverage: whether key points are covered\n"
        "- factual_consistency: internal correctness and no obvious fabricated claims\n"
        "- structure_clarity: logical organization\n"
        "- citation_grounding: whether claims are tied to references\n"
        "- gap_future_alignment: whether future works align with identified gaps\n"
        "- novel_insight: synthesis quality beyond restating\n"
        "Trajectory-scoring guidance:\n"
        "- l1_l2_routing_accuracy: does the report stay in the correct L1/L2 research line\n"
        "- evolution_chain_coherence: is the past->present technical evolution chain coherent and evidence-backed\n"
        "- gap_identification_quality: are the bottlenecks technically specific and aligned with domain reality\n"
        "- gap_future_linkage: are future directions explicitly and correctly linked to the identified gaps\n"
        "- citation_semantic_grounding: are claims supported by citation semantic intent/context evidence\n"
        "- evidence_traceability: can key claims be traced to concrete paper IDs/references\n"
        "Quiz score: 0 wrong/unsupported, 1 partially correct, 2 correct."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate generated survey outputs with LLM judge + quiz scoring.")
    p.add_argument("--tasks-json", required=True)
    p.add_argument("--generation-jsonl", required=True)
    p.add_argument("--judge-output-jsonl", required=True)
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

    for g in gens:
        task_id = g.get("task_id", "")
        task = task_map.get(task_id)
        if not task:
            continue
        candidate = g.get("response", "")
        if not candidate:
            row = {
                "task_id": task_id,
                "model_id": g.get("model_id", ""),
                "status": "empty_response",
                "error": g.get("error", ""),
                "scores": {},
                "quiz": [],
                "summary": {},
                "raw_judge": "",
            }
            append_jsonl(Path(args.judge_output_jsonl), row)
            print(f"[EVAL] task={task_id} model={g.get('model_id','')} empty")
            continue

        prompt = build_judge_prompt(task, candidate)
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
        if isinstance(parsed, dict):
            content_scores = parsed.get("content_scores", {})
            trajectory_scores = parsed.get("trajectory_scores", {})
            merged_scores = {}
            if isinstance(content_scores, dict):
                merged_scores.update(content_scores)
            if isinstance(trajectory_scores, dict):
                merged_scores.update(trajectory_scores)
            quiz_eval = parsed.get("quiz_eval", [])
            summary = parsed.get("summary", {})
            status = "ok"
            err = ""
        else:
            content_scores = {}
            trajectory_scores = {}
            merged_scores = {}
            quiz_eval = []
            summary = {}
            status = "parse_error" if resp.get("ok") else "judge_error"
            err = resp.get("error", "")

        row = {
            "task_id": task_id,
            "model_id": g.get("model_id", ""),
            "status": status,
            "error": err,
            "scores": merged_scores,
            "content_scores": content_scores,
            "trajectory_scores": trajectory_scores,
            "quiz": quiz_eval,
            "summary": summary,
            "raw_judge": "" if status == "ok" else resp.get("content", ""),
        }
        append_jsonl(Path(args.judge_output_jsonl), row)
        print(f"[EVAL] task={task_id} model={g.get('model_id','')} status={status}")


if __name__ == "__main__":
    main()
