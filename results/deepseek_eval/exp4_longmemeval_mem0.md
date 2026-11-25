### Publishable External Benchmark: `longmemeval` (model=`deepseek/deepseek-chat`)

| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Query / Ingest / Total USD | Ingest s | Cache | p50 / p95 ms | McNemar p vs. Ref |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mem0_official` | 70.0% [53.3, 86.7] | 0.188 | 0.125 | 1125 | $0.0088 / $1.5460 / **$1.5548** | 6024.6s | 0.0% | 1456.1 / 3601.9 | ref |

#### LongMemEval-S Per-Category Accuracy (mem0_official)
| System | knowledge-update | multi-session | single-session-assistant | single-session-preference | single-session-user | temporal-reasoning |
|---|---:|---:|---:|---:|---:|---:|
| `mem0_official` | 40.0% | 40.0% | 100.0% | 100.0% | 80.0% | 60.0% |