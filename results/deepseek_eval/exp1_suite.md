### Chatree Evaluation: `distractor-scaling-k4` (distractor_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 0/2 (0%) | 100% | 100% | 0% | 1829 | $0.00178 | 52% | 2167.5 / 2607.8 | 0 | 0 (0) |
| `sliding_window` | 0/2 (0%) | 100% | 100% | 0% | 595 | $0.00226 | 48% | 1223.0 / 1321.7 | 0 | 0 (0) |
| `flat_rag` | 0/2 (0%) | 0% | 100% | 0% | 676 | $0.00182 | 0% | 1947.2 / 2035.2 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 640 | $0.00169 | 0% | 2.9 / 3.1 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 0/2 (0%) | 100% | 100% | 0% | 680 | $0.00208 | 35% | 849.8 / 958.8 | 2 | 0 (0) |
| `tree_ipc_full` | 0/2 (0%) | 100% | 100% | 0% | 599 | $0.00234 | 50% | 1784.3 / 1807.1 | 4 | 0 (0) |

### Chatree Evaluation: `proposal-vs-decision-trap` (proposal_vs_decision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 0/2 (0%) | 100% | 100% | 0% | 1706 | $0.00133 | 45% | 1537.1 / 1744.2 | 0 | 0 (0) |
| `sliding_window` | 2/2 (100%) | 100% | 0% | 0% | 934 | $0.00150 | 39% | 872.2 / 938.6 | 0 | 0 (0) |
| `flat_rag` | 0/2 (0%) | 100% | 100% | 0% | 626 | $0.00151 | 0% | 1056.4 / 1117.3 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 650 | $0.00121 | 0% | 3.4 / 4.0 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 0/2 (0%) | 50% | 50% | 0% | 656 | $0.00225 | 42% | 1575.2 / 1652.8 | 2 | 0 (0) |
| `tree_ipc_full` | 1/2 (50%) | 100% | 50% | 0% | 679 | $0.00224 | 62% | 1480.8 / 1530.4 | 4 | 0 (0) |

### Chatree Evaluation: `multihop-revision-staleness` (multihop_revision)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/1 (100%) | 100% | 0% | 0% | 2801 | $0.00184 | 48% | 2242.0 / 2242.0 | 0 | 0 (0) |
| `sliding_window` | 1/1 (100%) | 100% | 0% | 0% | 1682 | $0.00205 | 26% | 1089.7 / 1089.7 | 0 | 0 (0) |
| `flat_rag` | 0/1 (0%) | 100% | 0% | 100% | 952 | $0.00120 | 0% | 1269.7 / 1269.7 | 0 | 1 (1) |
| `tree_summary_only` | 0/1 (0%) | 100% | 0% | 100% | 963 | $0.00161 | 4% | 4239.9 / 4239.9 | 0 | 1 (1) |
| `tree_ipc_no_staleness` | 0/1 (0%) | 0% | 0% | 100% | 1130 | $0.00160 | 0% | 2699.1 / 2699.1 | 0 | 1 (1) |
| `tree_ipc_full` | 0/1 (0%) | 100% | 0% | 100% | 960 | $0.00172 | 34% | 2233.9 / 2233.9 | 1 | 1 (1) |

### Chatree Evaluation: `unscripted-agentic-policy` (unscripted_agentic)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 1/1 (100%) | 100% | 0% | 0% | 1552 | $0.00092 | 42% | 2447.1 / 3907.8 | 0 | 0 (0) |
| `tree_ipc_full` | 1/1 (100%) | 100% | 0% | 0% | 7607 | $0.01057 | 73% | 12324.9 / 25053.9 | 2 | 0 (0) |

### Chatree Evaluation: `crossover-b4-t3` (crossover_scaling)

| Mode | Strict Pass | Recall | Leakage | Stale Err | Peak Root Tok | Est. USD | Cache | p50 / p95 ms | Shadow | Stale (Trans.) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `linear` | 2/2 (100%) | 100% | 0% | 0% | 6798 | $0.00597 | 79% | 3944.9 / 4421.7 | 0 | 0 (0) |
| `sliding_window` | 1/2 (50%) | 50% | 0% | 0% | 1791 | $0.00715 | 71% | 1340.8 / 1610.8 | 0 | 0 (0) |
| `flat_rag` | 2/2 (100%) | 100% | 0% | 0% | 879 | $0.00577 | 29% | 2064.9 / 2472.7 | 0 | 0 (0) |
| `tree_summary_only` | 0/2 (0%) | 0% | 0% | 0% | 1054 | $0.00582 | 35% | 3.9 / 3.9 | 0 | 0 (0) |
| `tree_ipc_no_staleness` | 2/2 (100%) | 100% | 0% | 0% | 1012 | $0.00725 | 39% | 1803.2 / 2197.1 | 2 | 0 (0) |
| `tree_ipc_full` | 2/2 (100%) | 100% | 0% | 0% | 754 | $0.00766 | 40% | 2288.0 / 2523.2 | 4 | 0 (0) |