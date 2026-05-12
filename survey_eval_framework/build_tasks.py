import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def as_list_text(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for it in x:
            if isinstance(it, dict):
                if "name" in it:
                    out.append(str(it.get("name", "")))
                elif "direction" in it:
                    out.append(str(it.get("direction", "")))
                elif "summary" in it:
                    out.append(str(it.get("summary", "")))
                else:
                    out.append(json.dumps(it, ensure_ascii=False))
            else:
                out.append(str(it))
        return [s.strip() for s in out if str(s).strip()]
    if isinstance(x, dict):
        return [json.dumps(x, ensure_ascii=False)]
    return [str(x).strip()] if str(x).strip() else []


def make_quiz(key_points: List[str], gaps: List[str], futures: List[str], dev: List[str]) -> List[Dict[str, str]]:
    quiz: List[Dict[str, str]] = []
    if key_points:
        quiz.append({"q": "What is one central technical focus of this L2 domain?", "a": key_points[0]})
    if gaps:
        quiz.append({"q": "Name one unresolved bottleneck in this L2 domain.", "a": gaps[0]})
    if futures:
        quiz.append({"q": "Name one high-priority future work direction for this L2 domain.", "a": futures[0]})
    if dev:
        quiz.append({"q": "Give one evolution signal (past -> present) for this L2 domain.", "a": dev[0]})
    return quiz


def main() -> None:
    p = argparse.ArgumentParser(description="Build survey benchmark tasks from existing domain+paper+citation artifacts.")
    p.add_argument("--domain-json", required=True)
    p.add_argument("--reps-json", required=True)
    p.add_argument("--citation-details-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--max-reps-per-l2", type=int, default=12)
    p.add_argument("--max-citation-per-l2", type=int, default=80)
    args = p.parse_args()

    domain = load_json(Path(args.domain_json))
    reps = load_json(Path(args.reps_json))
    citation_obj = load_json(Path(args.citation_details_json))
    citation_rows = citation_obj if isinstance(citation_obj, list) else citation_obj.get("citation_details", [])

    if not isinstance(reps, list):
        raise ValueError("reps-json must be list")
    if not isinstance(citation_rows, list):
        raise ValueError("citation-details-json must be list or {citation_details:[]}")

    reps_by_l2id: Dict[str, List[Dict[str, Any]]] = {}
    for r in reps:
        if str(r.get("status", "")).lower() != "ok":
            continue
        l2id = str(r.get("final_domain_id", "")).strip()
        reps_by_l2id.setdefault(l2id, []).append(r)

    citations_by_src: Dict[str, List[Dict[str, Any]]] = {}
    for c in citation_rows:
        if str(c.get("status", "")).lower() != "ok":
            continue
        src = str(c.get("source_paper_id", "")).strip()
        if not src:
            continue
        citations_by_src.setdefault(src, []).append(c)

    tasks: List[Dict[str, Any]] = []
    for l1 in domain.get("domain_hierarchy", []):
        l1_name = str(l1.get("l1_name", ""))
        for l2 in l1.get("l2_domains", []):
            l2_id = str(l2.get("l2_id", "")).strip()
            l2_name = str(l2.get("l2_name", "")).strip()

            rep_rows = reps_by_l2id.get(l2_id, [])[: args.max_reps_per_l2]
            rep_ids = {str(r.get("paper_id", "")) for r in rep_rows if str(r.get("paper_id", ""))}

            sem_rows: List[Dict[str, Any]] = []
            for pid in rep_ids:
                sem_rows.extend(citations_by_src.get(pid, []))
            sem_rows = sem_rows[: args.max_citation_per_l2]
            intent_profile = Counter(str(x.get("intent", "")).strip() for x in sem_rows if str(x.get("intent", "")).strip())

            key_points = as_list_text(l2.get("summary", ""))
            gaps = as_list_text(l2.get("gap_unresolved_bottlenecks", []))
            futures = as_list_text(l2.get("future_work_directions", []))
            dev = as_list_text(l2.get("development_history", ""))

            task = {
                "task_id": f"task_l2_{l2_id}",
                "topic": l2_name,
                "l1_domain": l1_name,
                "l2_domain": l2_name,
                "source_bundle": {
                    "representative_papers": rep_rows,
                    "citation_semantics": sem_rows,
                    "domain_snapshot": {
                        "l1_summary": l1.get("summary", ""),
                        "l2_summary": l2.get("summary", ""),
                        "development_history": l2.get("development_history", ""),
                    },
                },
                "gold": {
                    "key_points": key_points,
                    "gaps": gaps,
                    "future_works": futures,
                    "development_signals": dev,
                    "routing_target": {"l1_domain": l1_name, "l2_domain": l2_name},
                    "evolution_chain_expected": dev,
                    "semantic_intent_profile": dict(intent_profile),
                    "expected_trace_papers": sorted([x for x in rep_ids if x])[:10],
                    "quiz": make_quiz(key_points, gaps, futures, dev),
                },
            }
            tasks.append(task)

    dump_json(Path(args.output_json), tasks)
    print(f"[DONE] tasks={len(tasks)} -> {args.output_json}")


if __name__ == "__main__":
    main()
