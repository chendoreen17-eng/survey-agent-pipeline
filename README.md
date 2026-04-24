# Source Discovery Pipeline (Cleaned GitHub Repo)

This repository is a cleaned export of the research workflow from paper indexing to gap/future-works generation.
Only final merged outputs and repaired complete versions are kept.

## Workflow
1. Paper Index
2. Citation Graph (extract/parse/match/build/analyze)
3. Embeddings
4. Clustering (L2) and Hierarchy (L1)
5. Domain Naming (L2/L1)
6. Gap & Future Works

## Structure
- scripts/: runnable pipeline scripts by stage
- data/: final datasets and reports by stage
- docs/: curation notes (what was kept/removed)

## Key Outputs
- data/01_paper_index/papers.csv
- data/02_citation_graph/citation_edges_all_dedup_repaired.csv
- data/02_citation_graph/analysis/graph_basic_stats.json
- data/03_embeddings/paper_embeddings_all.npy
- data/04_clustering/final_structured_domains_all_hierarchy.csv
- data/05_domain_naming/l1_domain_names.csv
- data/05_domain_naming/l2_domain_names.csv
- data/06_gap_future/research_insights_report.md

## Environment
Install dependencies from requirements.txt.
