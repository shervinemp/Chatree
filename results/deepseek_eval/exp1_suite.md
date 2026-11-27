### Chatree Evaluation: `distractor-scaling-k4` (distractor_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 2629 | $0.00206 | 62% | 886.2 / 891.4 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 504 | $0.00224 | 35% | 717.4 / 773.8 | 0 | 0 (0) |
| `flat_rag` | 0/2 (0%) | 0% | 0% | 0% | 1100 | $0.00241 | 0% | 1377.3 / 1795.0 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 1009 | $0.00223 | 0% | 805.2 / 855.5 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 1152 | $0.00314 | 32% | 901.8 / 942.6 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 1091 | $0.00372 | 48% | 868.8 / 876.2 | 4 | 0 (0) |

### Chatree Evaluation: `proposal-vs-decision-trap` (proposal_vs_decision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/2 (50%) | 50% | 0% | 0% | 2786 | $0.00177 | 61% | 876.5 / 898.2 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 989 | $0.00163 | 39% | 713.8 / 815.0 | 0 | 0 (0) |
| `flat_rag` | 2/2 (100%) | 100% | 0% | 0% | 1148 | $0.00191 | 0% | 850.4 / 858.7 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 1197 | $0.00144 | 0% | 804.8 / 913.7 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 1179 | $0.00259 | 49% | 1341.9 / 1506.3 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 1139 | $0.00273 | 70% | 1275.0 / 1561.5 | 4 | 0 (0) |

### Chatree Evaluation: `multihop-revision-staleness` (multihop_revision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 0/1 (0%) | 100% | 0% | 100% | 3120 | $0.00188 | 48% | 2083.7 / 2083.7 | 0 | 0 (0) |
| `sliding_window` | 1/1 (100%) | 100% | 0% | 0% | 1668 | $0.00199 | 26% | 825.6 / 825.6 | 0 | 0 (0) |
| `flat_rag` | 1/1 (100%) | 100% | 0% | 0% | 1365 | $0.00156 | 0% | 844.4 / 844.4 | 0 | 1 (1) |
| `tree_summary_only` | 1/1 (100%) | 100% | 0% | 0% | 1357 | $0.00206 | 14% | 3867.6 / 3867.6 | 0 | 1 (1) |
| `tree_ipc_no_staleness` | 0/1 (0%) | 100% | 0% | 100% | 2984 | $0.00378 | 63% | 6938.9 / 6938.9 | 2 | 1 (1) |
| `tree_ipc_full` | 1/1 (100%) | 100% | 0% | 0% | 1258 | $0.00206 | 43% | 2097.2 / 2097.2 | 1 | 1 (1) |

### Chatree Evaluation: `unscripted-agentic-policy` (unscripted_agentic)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/1 (100%) | 100% | 0% | 0% | 2035 | $0.00101 | 37% | 2203.6 / 2754.6 | 0 | 0 (0) |
| `tree_ipc_full` | 1/1 (100%) | 100% | 0% | 0% | 4606 | $0.00755 | 79% | 5565.8 / 13877.5 | 1 | 0 (0) |

### Chatree Evaluation: `crossover-b4-t3` (crossover_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 7235 | $0.00641 | 81% | 1380.9 / 1609.7 | 0 | 0 (0) |
| `sliding_window` | 1/2 (50%) | 50% | 0% | 0% | 1808 | $0.00714 | 72% | 859.0 / 902.6 | 0 | 0 (0) |
| `flat_rag` | 2/2 (100%) | 100% | 0% | 0% | 1351 | $0.00651 | 26% | 962.6 / 1123.2 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 1472 | $0.00658 | 31% | 752.2 / 885.2 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 1356 | $0.00780 | 38% | 1021.7 / 1096.1 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 1375 | $0.00869 | 48% | 1256.1 / 1516.7 | 4 | 0 (0) |