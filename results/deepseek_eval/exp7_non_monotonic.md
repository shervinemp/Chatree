### Publishable External Benchmark: `non_monotonic` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Query / Ingest / Total USD | Ingest s | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | 100.0% [100.0, 100.0] | 0.100 | 0.055 | 3295 | $0.0770 / $0.0221 / **$0.0991** | 310.1s | 68.6% | 4834.0 / 6836.5 | ref |
| `flat_hybrid_rag` | 36.1% [22.2, 52.8] | 0.181 | 0.126 | 777 | $0.0079 / $0.0000 / **$0.0079** | 1.0s | 4.6% | 1398.7 / 2224.4 | 0.0000* |
| `full_context_linear` | 100.0% [100.0, 100.0] | 0.232 | 0.154 | 3710 | $0.0105 / $0.0000 / **$0.0105** | 0.0s | 76.8% | 1403.1 / 2007.5 | 1.0000 |
| `tree_summary_only` | 88.9% [77.8, 97.2] | 0.278 | 0.189 | 1426 | $0.0104 / $0.0099 / **$0.0203** | 130.6s | 22.1% | 1428.9 / 1792.4 | 0.1250 |
| `mem0_official` | 100.0% [100.0, 100.0] | 0.193 | 0.114 | 1235 | $0.0123 / $0.0172 / **$0.0295** | 280.1s | 0.0% | 1565.2 / 2892.3 | 1.0000 |
| `rlm_official` | 77.8% [63.9, 91.7] | 0.195 | 0.130 | 4792 | $0.1110 / $0.0000 / **$0.1110** | 0.0s | 0.0% | 5268.4 / 18096.3 | 0.0078* |

#### Track 7 Non-Monotonic Engineering Per-Category Accuracy
| System | cross_branch_synthesis | dead_end_quarantine | downstream_staleness | spec_reversal | verbatim_under_noise |
|---|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `flat_hybrid_rag` | 0.0% | 100.0% | 33.3% | 33.3% | 25.0% |
| `full_context_linear` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `tree_summary_only` | 83.3% | 100.0% | 100.0% | 100.0% | 75.0% |
| `mem0_official` | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| `rlm_official` | 66.7% | 100.0% | 83.3% | 66.7% | 75.0% |