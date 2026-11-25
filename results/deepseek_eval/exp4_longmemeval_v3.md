### Publishable External Benchmark: `longmemeval` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Query / Ingest / Total USD | Ingest s | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | 80.0% [66.7, 93.3] | 0.148 | 0.094 | 6295 | $0.2348 / $0.8837 / **$1.1185** | 4552.1s | 31.4% | 5114.1 / 7687.1 | ref |
| `full_context_linear` | 70.0% [53.3, 86.7] | 0.199 | 0.136 | 108772 | $0.8822 / $0.0000 / **$0.8822** | 0.0s | 0.0% | 2421.0 / 4755.2 | 0.5078 |
| `mem0_official` | 70.0% [53.3, 86.7] | 0.188 | 0.125 | 1125 | $0.0088 / $1.5460 / **$1.5548** | 6024.6s | 0.0% | 1456.1 / 3601.9 | 0.3750 |
| `rlm_official` | 70.0% [53.3, 86.7] | 0.205 | 0.148 | 21373 | $0.3752 / $0.0000 / **$0.3752** | 0.0s | 0.0% | 11248.4 / 31979.3 | 0.5810 |
| `chatree_two_tier` | 63.3% [46.7, 80.0] | 0.172 | 0.126 | 6927 | $0.1618 / $0.6295 / **$0.7913** | 4436.6s | 49.6% | 6191.1 / 8147.9 | 0.1797 |
| `flat_hybrid_rag` | 40.0% [23.3, 56.7] | 0.181 | 0.135 | 717 | $0.0037 / $0.0000 / **$0.0037** | 31.2s | 39.3% | 3543.4 / 7098.2 | 0.0018* |

#### LongMemEval-S Per-Category Accuracy (6 Systems)
| System | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | 80.0% | 60.0% | 100.0% | 100.0% | 80.0% | 60.0% |
| `full_context_linear` | 100.0% | 40.0% | 100.0% | 20.0% | 100.0% | 60.0% |
| `mem0_official` | 40.0% | 40.0% | 100.0% | 100.0% | 80.0% | 60.0% |
| `rlm_official` | 80.0% | 40.0% | 100.0% | 40.0% | 80.0% | 80.0% |
| `chatree_two_tier` | 80.0% | 40.0% | 60.0% | 60.0% | 80.0% | 60.0% |
| `flat_hybrid_rag` | 80.0% | 40.0% | 20.0% | 0.0% | 80.0% | 20.0% |