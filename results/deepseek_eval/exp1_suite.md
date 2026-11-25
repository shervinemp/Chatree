### Chatree Evaluation: `distractor-scaling-k4` (distractor_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 2533 | $0.00211 | 65% | 1084.7 / 1231.9 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 374 | $0.00174 | 27% | 1072.4 / 1133.6 | 0 | 0 (0) |
| `flat_rag` | 0/2 (0%) | 0% | 0% | 0% | 779 | $0.00233 | 0% | 1306.2 / 1579.6 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 890 | $0.00231 | 0% | 2.4 / 2.8 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 896 | $0.00290 | 36% | 1310.3 / 1340.7 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 874 | $0.00340 | 50% | 1442.6 / 1611.1 | 4 | 0 (0) |

### Chatree Evaluation: `proposal-vs-decision-trap` (proposal_vs_decision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 2313 | $0.00165 | 52% | 1312.3 / 1498.2 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 1269 | $0.00194 | 37% | 917.5 / 918.1 | 0 | 0 (0) |
| `flat_rag` | 1/2 (50%) | 100% | 50% | 0% | 934 | $0.00192 | 0% | 1662.5 / 1729.2 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 874 | $0.00159 | 0% | 2.2 / 2.4 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 869 | $0.00245 | 47% | 2031.7 / 2143.7 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 1174 | $0.00297 | 65% | 2110.3 / 2281.0 | 4 | 0 (0) |

### Chatree Evaluation: `multihop-revision-staleness` (multihop_revision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/1 (100%) | 100% | 0% | 0% | 2988 | $0.00192 | 48% | 2887.3 / 2887.3 | 0 | 0 (0) |
| `sliding_window` | 1/1 (100%) | 100% | 0% | 0% | 1747 | $0.00207 | 26% | 1990.7 / 1990.7 | 0 | 0 (0) |
| `flat_rag` | 0/1 (0%) | 100% | 0% | 100% | 1138 | $0.00157 | 0% | 1012.1 / 1012.1 | 0 | 1 (1) |
| `tree_summary_only` | 0/1 (0%) | 100% | 0% | 100% | 1137 | $0.00209 | 9% | 4356.0 / 4356.0 | 0 | 1 (1) |
| `tree_ipc_no_staleness` | 0/1 (0%) | 100% | 0% | 100% | 1374 | $0.00186 | 0% | 2376.8 / 2376.8 | 0 | 1 (1) |
| `tree_ipc_full` | 1/1 (100%) | 100% | 0% | 0% | 1138 | $0.00200 | 38% | 1753.5 / 1753.5 | 1 | 1 (1) |

### Chatree Evaluation: `unscripted-agentic-policy` (unscripted_agentic)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/1 (100%) | 100% | 0% | 0% | 1613 | $0.00084 | 41% | 1988.0 / 5795.6 | 0 | 0 (0) |
| `tree_ipc_full` | 1/1 (100%) | 100% | 0% | 0% | 4186 | $0.00721 | 78% | 7725.0 / 17263.6 | 1 | 0 (0) |

### Chatree Evaluation: `crossover-b4-t3` (crossover_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 6620 | $0.00634 | 80% | 1364.7 / 1671.5 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 1755 | $0.00676 | 71% | 1194.4 / 1607.5 | 0 | 0 (0) |
| `flat_rag` | 2/2 (100%) | 100% | 0% | 0% | 1172 | $0.00658 | 28% | 1101.2 / 1302.0 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 1178 | $0.00651 | 29% | 3.8 / 4.9 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 1170 | $0.00768 | 38% | 1418.5 / 1476.0 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 1246 | $0.00824 | 50% | 1992.2 / 2155.6 | 4 | 0 (0) |