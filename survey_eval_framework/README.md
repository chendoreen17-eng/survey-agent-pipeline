# Survey Agent Evaluation Framework

This framework benchmarks your survey agent against baseline models with two complementary protocols inspired by SurveyBench and MLR-bench design ideas:

1. **Task-based report generation** on the same topic/evidence bundle.
2. **LLM-judge scoring + quiz validation** for quality and correctness.
3. **Pairwise battle + Elo ranking** for relative strength comparisons.

## Folder Structure

- `build_tasks.py`: Build benchmark tasks from your existing artifacts.
- `run_generation.py`: Generate candidate survey reports for each model.
- `run_eval.py`: Score outputs with rubric + quiz checks.
- `aggregate_scores.py`: Compute weighted leaderboard.
- `run_pairwise_elo.py`: Pairwise battles and Elo ranking.
- `run_benchmark.py`: End-to-end runner.
- `make_precomputed_outputs.py`: Convert your existing survey-agent outputs to precomputed format.
- `configs/models.example.json`: Model registry template.
- `configs/weights.json`: Rubric weight config.

## 1) Build Tasks from Your Existing Data

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/build_tasks.py" --domain-json "C:/Users/doreen chen/Desktop/source_discovery/domain_landscape_l1_l2_with_reps_gap_future.json" --reps-json "C:/Users/doreen chen/Desktop/source_discovery/gap&future works/representatives/l2_repres_abstract.json" --citation-details-json "C:/Users/doreen chen/Desktop/source_discovery/citation_graph/citation_details/citation_details.json" --output-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json"
```

## 2) (Optional) Prepare Precomputed Outputs for Your Survey Agent

If your survey agent is a pipeline (not a direct chat model endpoint), convert existing domain results to `task_id -> response`:

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/make_precomputed_outputs.py" --tasks-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json" --domain-json "C:/Users/doreen chen/Desktop/source_discovery/domain_landscape_l1_l2_with_reps_gap_future.json" --output-jsonl "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/outputs/survey_agent_outputs.jsonl"
```

Then set `survey_agent` entry in `configs/models.example.json` to use this file.

## 3) Configure Model Registry

Copy and edit:

- `configs/models.example.json`

Each model can be:

- `mode = "llm_api"`: online model via OpenAI-compatible API.
- `mode = "precomputed"`: load prepared outputs from JSONL.

## 4) Run End-to-End Benchmark

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/run_benchmark.py" --tasks-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json" --models-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/configs/models.example.json" --out-dir "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/outputs/benchmark_run_01" --judge-model "gemini-2.5-pro" --judge-api-key "env:LLMMELON_API_KEY"
```

## 5) Output Files

Inside `out-dir`:

- `generation.jsonl`: model raw outputs per task.
- `eval.jsonl`: judge results per task (rubric + quiz).
- `leaderboard.json`: weighted final ranking.
- `battles.jsonl`: pairwise match decisions.
- `elo.json`: Elo ranking.

## Scoring Summary

- Content dimensions: coverage, factual consistency, structure clarity, citation grounding, gap-future alignment, novel insight.
- Quiz scoring: per-question 0/1/2 normalized to [0,1].
- Final score: `alpha_content * content + beta_quiz * quiz`.

Default aggregation:

- `alpha_content = 0.7`
- `beta_quiz = 0.3`

## Notes

- Keep all compared systems on the **same tasks and same evidence** for fairness.
- Use the same judge model and settings across all systems.
- Run multiple trials and report mean/std for stronger claims.
