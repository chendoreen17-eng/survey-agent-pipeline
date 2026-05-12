# Survey Agent 评测框架（中文版）

这个框架用于评估并对比你的 **survey agent** 与其他模型在“综述生成任务”上的表现。框架参考了 SurveyBench / MLR-bench 思路，并结合你当前数据资产做了可落地实现。

## 一、框架目标

统一在**同一任务、同一证据、同一评审标准**下进行比较，输出：

1. 任务级评分（通用质量 + 研究脉络）
2. 模型总榜（加权总分）
3. 两两对战与 Elo 排名（相对优势）

---

## 二、目录与脚本说明

- `build_tasks.py`：从现有数据构建 benchmark 任务集（L2 级任务）。
- `make_precomputed_outputs.py`：把已有 survey agent 结果转成 `task_id -> response`。
- `run_generation.py`：让多个模型在同一任务集上生成回答。
- `run_eval.py`：用 LLM Judge 按 rubric + quiz 对每条回答评分。
- `aggregate_scores.py`：汇总模型排行榜。
- `run_pairwise_elo.py`：pairwise battle + Elo。
- `run_benchmark.py`：一键串联完整流程。

---

## 三、评测指标（重点）

### A. 通用质量指标（0~5）

1. `coverage`：关键要点覆盖度
2. `factual_consistency`：事实一致性
3. `structure_clarity`：结构清晰度
4. `citation_grounding`：引用依据充分性
5. `gap_future_alignment`：gap 与 future work 对齐程度
6. `novel_insight`：综合洞察能力

### B. 研究脉络核心指标（0~5）

1. `l1_l2_routing_accuracy`：是否准确落在目标 L1/L2 研究脉络
2. `evolution_chain_coherence`：技术演化链（past->present）是否连贯且有证据
3. `gap_identification_quality`：识别出的瓶颈是否具体、真实、关键
4. `gap_future_linkage`：future works 是否明确对应已识别 gap
5. `citation_semantic_grounding`：是否有效利用引用语义（intent/context/description）支撑论点
6. `evidence_traceability`：关键结论是否可追溯到具体 paper_id/引用证据

### C. Quiz 指标

- 每题 0/1/2 分（错/部分对/正确），归一化到 [0,1]。

---

## 四、加权方式

- 维度权重见 `configs/weights.json`
- 总分：

`final_score = alpha_content * content_score + beta_quiz * quiz_score`

默认：

- `alpha_content = 0.7`
- `beta_quiz = 0.3`

注意：`content_score` 已包含“研究脉络核心指标”的加权分量。

---

## 五、运行流程

### 1) 构建任务集

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/build_tasks.py" --domain-json "C:/Users/doreen chen/Desktop/source_discovery/domain_landscape_l1_l2_with_reps_gap_future.json" --reps-json "C:/Users/doreen chen/Desktop/source_discovery/gap&future works/representatives/l2_repres_abstract.json" --citation-details-json "C:/Users/doreen chen/Desktop/source_discovery/citation_graph/citation_details/citation_details.json" --output-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json"
```

### 2) （可选）准备 survey agent 预置输出

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/make_precomputed_outputs.py" --tasks-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json" --domain-json "C:/Users/doreen chen/Desktop/source_discovery/domain_landscape_l1_l2_with_reps_gap_future.json" --output-jsonl "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/outputs/survey_agent_outputs.jsonl"
```

### 3) 配置模型列表

编辑 `configs/models.example.json`。

### 4) 一键跑完整评测

```powershell
python "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/run_benchmark.py" --tasks-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/tasks/tasks_l2.json" --models-json "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/configs/models.example.json" --out-dir "C:/Users/doreen chen/Desktop/source_discovery/survey_eval_framework/outputs/benchmark_run_01" --judge-model "gemini-2.5-pro" --judge-api-key "env:LLMMELON_API_KEY"
```

---

## 六、公平性建议

1. 用同一 backbone 做 A/B：
   - A: 不接信息库
   - B: 接入引用图 + domain + gap/future
2. Judge 模型固定，参数固定。
3. 同任务集多次复现实验，报告均值/方差。
4. 对 `l1_l2_routing_accuracy`、`evolution_chain_coherence`、`gap_future_linkage` 做重点误差分析。
