### Track 6 Infinite-Context Compaction: `infinite-context-compaction-8t` (model=`deepseek/deepseek-chat`)

| System | Gen | Compacts | Anchored Recall | Non-Anchored Recall | Overall | Peak Tok | Final Tok | Cache | Est. USD | Gen-1 Preserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `chatree_non_destructive` | 10 | 9 | 100.0% | 100.0% | 100.0% | 1727 | 1723 | 20.6% | $0.01299 | 4 |
| `destructive_rolling` | 7 | 6 | 100.0% | 0.0% | 50.0% | 1483 | 1985 | 58.6% | $0.00590 | 0 |
| `sliding_window` | 1 | 0 | 0.0% | 0.0% | 0.0% | 1144 | 1108 | 15.9% | $0.00436 | 0 |