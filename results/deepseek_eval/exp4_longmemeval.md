### Publishable External Benchmark: `longmemeval` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Est. USD | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier` | 63.3% [46.7, 80.0] | 0.172 | 0.126 | 6927 | $0.16181 | 49.6% | 6191.1 / 8147.9 | ref |
| `flat_hybrid_rag` | 40.0% [23.3, 56.7] | 0.201 | 0.157 | 770 | $0.00537 | 0.0% | 9173.4 / 276324.3 | 0.0654 |
| `full_context_linear` | 70.0% [53.3, 86.7] | 0.199 | 0.136 | 108772 | $0.88222 | 0.0% | 2421.0 / 4755.2 | 0.7744 |
| `rlm_official` | 70.0% [53.3, 86.7] | 0.205 | 0.148 | 21373 | $0.37519 | 0.0% | 11248.4 / 31979.3 | 0.7539 |

#### LongMemEval-S Per-Category Accuracy
| System | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier` | 80.0% | 40.0% | 60.0% | 60.0% | 80.0% | 60.0% |
| `flat_hybrid_rag` | 80.0% | 40.0% | 20.0% | 0.0% | 80.0% | 20.0% |
| `full_context_linear` | 100.0% | 40.0% | 100.0% | 20.0% | 100.0% | 60.0% |
| `rlm_official` | 80.0% | 40.0% | 100.0% | 40.0% | 80.0% | 80.0% |