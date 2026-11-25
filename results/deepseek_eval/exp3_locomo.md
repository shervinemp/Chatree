### Publishable External Benchmark: `locomo` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Est. USD | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_two_tier` | 75.9% [64.8, 87.0] | 0.123 | 0.075 | 318 | $0.09095 | 81.7% | 4893.6 / 7976.8 | ref |
| `flat_hybrid_rag` | 51.8% [38.9, 64.8] | 0.149 | 0.101 | 696 | $0.00995 | 3.0% | 2696.9 / 3353.7 | 0.0164* |
| `full_context_linear` | 68.5% [55.6, 79.6] | 0.160 | 0.097 | 22955 | $0.04565 | 95.6% | 1271.2 / 1738.8 | 0.4807 |
| `tree_summary_only` | 55.6% [42.6, 68.5] | 0.136 | 0.096 | 1575 | $0.01547 | 18.6% | 1478.8 / 1792.8 | 0.0074* |
| `mem0_official` | 68.5% [55.6, 79.6] | 0.163 | 0.104 | 816 | $0.01165 | 0.4% | 938.8 / 1725.8 | 0.4545 |
| `rlm_official` | 74.1% [61.1, 85.2] | 0.136 | 0.080 | 16309 | $0.49053 | 0.0% | 9881.4 / 25347.8 | 1.0000 |

#### LoCoMo Per-Category Accuracy
| System | adversarial | multi_hop | open_domain | single_hop | temporal |
|---|---:|---:|---:|---:|---:|
| `chatree_two_tier` | 91.7% | 50.0% | 83.3% | 91.7% | 66.7% |
| `flat_hybrid_rag` | 66.7% | 41.7% | 50.0% | 58.3% | 41.7% |
| `full_context_linear` | 33.3% | 75.0% | 83.3% | 91.7% | 66.7% |
| `tree_summary_only` | 100.0% | 41.7% | 83.3% | 50.0% | 16.7% |
| `mem0_official` | 66.7% | 66.7% | 83.3% | 75.0% | 58.3% |
| `rlm_official` | 33.3% | 83.3% | 100.0% | 91.7% | 75.0% |