import argparse
import json
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    p = argparse.ArgumentParser(description="Build precomputed survey_agent outputs aligned to task_id.")
    p.add_argument("--tasks-json", required=True)
    p.add_argument("--domain-json", required=True, help="domain_landscape_l1_l2_with_reps_gap_future.json")
    p.add_argument("--output-jsonl", required=True)
    args = p.parse_args()

    tasks = load_json(Path(args.tasks_json))
    domain = load_json(Path(args.domain_json))

    by_l2id = {}
    for l1 in domain.get("domain_hierarchy", []):
        for l2 in l1.get("l2_domains", []):
            by_l2id[str(l2.get("l2_id", ""))] = l2

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            tid = str(t.get("task_id", ""))
            l2id = tid.replace("task_l2_", "")
            l2 = by_l2id.get(l2id, {})
            rep = l2.get("representative_papers", [])
            rep_ids = [str(x.get("paper_id", "")) for x in rep[:8] if isinstance(x, dict)]

            text = []
            text.append(f"## Evolution\n{l2.get('development_history', '')}")
            text.append("## Gaps")
            for g in l2.get("gap_unresolved_bottlenecks", [])[:5]:
                if isinstance(g, dict):
                    text.append(f"- {g.get('name','')} :: {g.get('technical_explanation','')}")
                else:
                    text.append(f"- {g}")
            text.append("## Future Works")
            for fw in l2.get("future_work_directions", [])[:5]:
                if isinstance(fw, dict):
                    text.append(f"- {fw.get('direction','')} (why_now: {fw.get('why_now','')}; first_step: {fw.get('first_step','')})")
                else:
                    text.append(f"- {fw}")
            if rep_ids:
                text.append("## Representative Paper IDs")
                text.extend([f"- {x}" for x in rep_ids])

            row = {"task_id": tid, "response": "\n".join(text)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[DONE] {args.output_jsonl}")


if __name__ == "__main__":
    main()
