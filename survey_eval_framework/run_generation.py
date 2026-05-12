import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import append_jsonl, chat_completion, get_env_or_value, load_json, load_jsonl


def build_prompt(task: Dict[str, Any]) -> str:
    bundle = task.get("source_bundle", {})
    rep = bundle.get("representative_papers", [])
    citation_sem = bundle.get("citation_semantics", [])
    domain_snapshot = bundle.get("domain_snapshot", {})

    rep_lines = []
    for p in rep[:20]:
        rep_lines.append(
            f"- {p.get('paper_id','')} | {p.get('year','')} | {p.get('title','')} | abs={str(p.get('abstract',''))[:220]}"
        )
    cit_lines = []
    for c in citation_sem[:40]:
        cit_lines.append(
            f"- src={c.get('source_paper_id','')} -> tgt={c.get('matched_target_paper_id','')} | intent={c.get('intent','')} | desc={str(c.get('description',''))[:220]}"
        )

    return (
        f"Topic: {task.get('topic','')}\n"
        f"L1: {task.get('l1_domain','')}\n"
        f"L2: {task.get('l2_domain','')}\n\n"
        "You are writing a rigorous survey subsection for this topic.\n"
        "Requirements:\n"
        "1) concise technical evolution timeline\n"
        "2) unresolved gaps\n"
        "3) future work directions aligned with gaps\n"
        "4) explicitly grounded in representative papers and citation semantics\n"
        "5) include traceable references to paper_id when making claims\n\n"
        f"Representative papers:\n{chr(10).join(rep_lines) or '- N/A'}\n\n"
        f"Citation semantics:\n{chr(10).join(cit_lines) or '- N/A'}\n\n"
        f"Domain snapshot:\n{domain_snapshot}\n\n"
        "Output a structured markdown section with headings: Evolution, Gaps, Future Works."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run generation for multiple models on survey tasks.")
    p.add_argument("--tasks-json", required=True)
    p.add_argument("--models-json", required=True, help="List of model configs.")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--timeout-sec", type=int, default=240)
    args = p.parse_args()

    tasks = load_json(Path(args.tasks_json))
    models = load_json(Path(args.models_json))
    if not isinstance(tasks, list):
        raise ValueError("tasks-json must be a list")
    if not isinstance(models, list):
        raise ValueError("models-json must be a list")

    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    out_path = Path(args.output_jsonl)

    for m in models:
        model_id = m.get("model_id", "")
        mode = m.get("mode", "llm_api")

        if mode == "precomputed":
            precomputed_path = m.get("precomputed_jsonl", "")
            if not precomputed_path:
                print(f"[SKIP] model={model_id} missing precomputed_jsonl")
                continue
            pre_rows = load_jsonl(Path(precomputed_path))
            by_task = {str(r.get("task_id", "")): r for r in pre_rows if r.get("task_id")}
            for t in tasks:
                task_id = t.get("task_id", "")
                rr = by_task.get(str(task_id), {})
                row = {
                    "task_id": task_id,
                    "model_id": model_id,
                    "model_name": m.get("model_name", model_id),
                    "ok": bool(rr.get("response", "")),
                    "error": "" if rr.get("response", "") else "missing_precomputed_response",
                    "response": rr.get("response", ""),
                    "topic": t.get("topic", ""),
                    "l1_domain": t.get("l1_domain", ""),
                    "l2_domain": t.get("l2_domain", ""),
                }
                append_jsonl(out_path, row)
                print(f"[GEN] model={model_id} task={task_id} ok={row['ok']} (precomputed)")
            continue

        base_url = m.get("base_url", "https://llmmelon.cloud")
        model_name = m.get("model_name", "")
        api_key = get_env_or_value(m.get("api_key", ""))
        system_prompt = m.get("system_prompt", "You are a careful research assistant.")
        temp = float(m.get("temperature", 0.2))

        if not api_key:
            print(f"[SKIP] model={model_id} missing api key")
            continue

        for t in tasks:
            task_id = t.get("task_id", "")
            prompt = build_prompt(t)
            resp = chat_completion(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                timeout_sec=args.timeout_sec,
                max_retries=3,
            )
            row = {
                "task_id": task_id,
                "model_id": model_id,
                "model_name": model_name,
                "ok": resp["ok"],
                "error": resp["error"],
                "response": resp["content"],
                "topic": t.get("topic", ""),
                "l1_domain": t.get("l1_domain", ""),
                "l2_domain": t.get("l2_domain", ""),
            }
            append_jsonl(out_path, row)
            print(f"[GEN] model={model_id} task={task_id} ok={resp['ok']}")


if __name__ == "__main__":
    main()
