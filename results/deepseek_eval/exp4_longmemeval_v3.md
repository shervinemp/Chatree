### Publishable External Benchmark: `longmemeval` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Est. USD | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | 80.0% [66.7, 93.3] | 0.148 | 0.094 | 6295 | $0.23478 | 31.4% | 5083.2 / 7924.4 | ref |
| `chatree_two_tier` | 63.3% [46.7, 80.0] | 0.172 | 0.126 | 6927 | $0.16181 | 49.6% | 5982.9 / 8180.7 | 0.1797 |
| `full_context_linear` | 70.0% [53.3, 86.7] | 0.199 | 0.136 | 108772 | $0.88222 | 0.0% | 2396.9 / 5026.9 | 0.5078 |
| `rlm_official` | 70.0% [53.3, 86.7] | 0.205 | 0.148 | 21373 | $0.37519 | 0.0% | 11230.9 / 32576.7 | 0.5810 |
| `flat_hybrid_rag` | 40.0% [23.3, 56.7] | 0.201 | 0.157 | 770 | $0.00537 | 0.0% | 8880.2 / 277823.6 | 0.0018* |

#### LongMemEval-S Per-Category Accuracy (n=30, 5/category)
| System | Overall (95% CI) | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning | Peak Tok | Cache | McNemar p vs v3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier_v3` | **80.0%** `[66.7, 93.3]` | 80.0% | 60.0% | 100.0% | 100.0% | 80.0% | 60.0% | 6,295 | 31.4% | ref |
| `chatree_two_tier` | **63.3%** `[46.7, 80.0]` | 80.0% | 40.0% | 60.0% | 60.0% | 80.0% | 60.0% | 6,927 | 49.6% | 0.1797 |
| `full_context_linear` | **70.0%** `[53.3, 86.7]` | 100.0% | 40.0% | 100.0% | 20.0% | 100.0% | 60.0% | 108,772 | 0.0% | 0.5078 |
| `rlm_official` | **70.0%** `[53.3, 86.7]` | 80.0% | 40.0% | 100.0% | 40.0% | 80.0% | 80.0% | 21,373 | 0.0% | 0.5810 |
| `flat_hybrid_rag` | **40.0%** `[23.3, 56.7]` | 80.0% | 40.0% | 20.0% | 0.0% | 80.0% | 20.0% | 770 | 0.0% | 0.0018* |