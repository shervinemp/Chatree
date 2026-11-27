# Future plans: long and jumping conversations

Written 2026-11-26 against commit `0daebb8`. All 113 tests pass at that commit. No code was changed.

## How to read this

Each finding has one of these labels:

| Label | Meaning |
|---|---|
| **Reproduced** | Run offline with the mock model and an in-memory database |
| **Read in code** | Follows from the source; not run |
| **From results** | Taken from `results/deepseek_eval/` |
| **Risk** | Follows from the design; not seen happening |

Nothing here was run against a real model. The server, MCP and UI paths were not checked. The probe scripts were temporary, so each item has a **Done when** line that describes the test to add. Line numbers refer to `0daebb8`.

## Can Chatree hold an unlimited conversation?

- **Storage:** yes in principle. Compaction and archiving keep the original turns.
- **Request size:** not with the defaults. Several parts of a request grow, or are cut badly (Phase 2).
- **Recall:** measured on some long histories, not for conversations that keep growing. LoCoMo and LongMemEval-S test pre-loaded histories (full-history peaks of about 35k and 109k tokens; `chatree_two_tier_v3` scored 77.1% and 80%, the latter on 30 items). The 10-generation compaction test is a 9-turn conversation that compacts almost every turn, and its questions point at the first turn.

The target to build toward:

1. Every request fits the model's window, however long the conversation is.
2. No turn is deleted, and any turn can be reached in a small, bounded number of lookups.
3. Work added to each turn does not grow in proportion to the history.

Perfect recall is **not** a realistic promise. An old detail comes back only if navigation or search finds it. Measure how often that happens (Phase 5).

## What already works: keep it

- Compaction does not delete the source turns.
- Archived branches are left out of default search and branch summaries, but stay in the database.
- Most of the per-turn context block is capped. It held at about 712–716 estimated tokens from 25 to 400 topics, with no stale notices or search hints. **Reproduced**
- Keyword and hybrid search both found a first-turn fact that the shadow query could not see (item 2.5). **Reproduced**
- In the live compaction test, the model recovered first-turn details after nine compactions. The two comparison systems lost the details that were not marked as anchors. **From results** (`exp6_compaction_bench.md`)

## Where this list came from

It started as a critique of another app's conversation memory. This is how each point applies here.

| Critique | Applies to Chatree? |
|---|---|
| Finished topics cannot be reopened | **No** for storage. Related problem: the model is not told about most topics (1.1, 1.2) |
| The model sees only a few topic notes | **Yes**, and it sees the least recently updated ones (1.1) |
| The model cannot read old words | **Mostly no.** Search and shadow queries exist. Long branches are the exception (2.5) |
| Only recent turns are sent | **Partly.** Whole branches are sent; past the limit, the oldest turns are cut (2.1) |
| Some parts of a request have no limit | **Yes** (2.2–2.4) |
| The model's context size is never set or checked | **Yes**, no model window is looked up (2.2) |
| Send the whole conversation while it fits | **Yes**, there is no such mode (Phase 4) |
| Evidence is small | **Yes**, and some results have scoring or setup errors (Phase 0) |

---

## Phase 0: correct the evidence

Do this first. These results are quoted in the README and the paper, and later design choices lean on them.

### 0.1 The keyword scorer fails correct answers

**From results**, and the mechanism is **Reproduced**.

- I checked every failed check for `chatree_live`, `linear` and `rlm_official` in both Track 5 runs: seven in total. In each one, the LLM judge (0.9 to 1.0) says the old values appear only as retired or excluded.
- Saved answers are cut at 240 characters (`eval/scoring.py:221`), so I relied mostly on the judge's written reasons.
- `is_token_asserted_as_active` (`scoring.py:110`) looks at the current clause and only the previous line. With a constructed answer, the first bullet under `### Excluded, as requested` passed and the second bullet was flagged as active.

What changes:

- "0.0% dead-end leakage (vs 33.3% for `linear` and `rlm_official`)" is not supported.
- In the natural-prompt run, the saved table shows `linear` 3/3 and `chatree_live` 2/3 strict passes. By the judge's reasons it is a tie. The README text gives neither number.
- On correctness, these two runs show no clear difference between `chatree_live`, `linear` and (in the default run) `rlm_official`.

Affected text: the leakage comparison at `README.md:176`. The natural-prompt summaries (`README.md:175`, `paper/chatree_paper.md:181`, `paper/chatree_paper.tex` near line 386) should report strict passes after rescoring. Check the matching tables and figures. Other runners in `eval/` use the same scorer, so rescore their results too.

Fix:

- Track the enclosing section or list introduction, not just the previous line. Cover tables.
- Or have the judge return a yes/no for "presented as current" for each forbidden value.
- Report both scorers and list every disagreement.
- Save complete answers.

**Done when:** regression tests cover multi-item excluded lists and tables, saved runs contain full answers, and Track 5 is rerun and the text updated.

### 0.2 Two comparisons never ask the model the question

**Read in code**

When shadow turns are off, `request()` drops the prompt (`engine/ipc/shadow.py:105-107`) and returns the branch summary. The harness then grades that summary as the answer:

- `eval/ablations.py:324-326`: the `no_shadow_turns_folding` ablation
- `eval/runner.py:591-593`: `tree_summary_only` in the stress suite and the crossover sweep

This fits the 1.6–2.2 ms median latency for `tree_summary_only` in `exp2_crossover.md`. The result measures whether the summary happens to contain the expected tokens, not whether a model can answer from summaries.

What changes: do not rely on "removing shadow turns drops strict pass to 0.0% (p=0.031)", or on the Track 1 and Track 2 `tree_summary_only` failures. Affected text: `README.md:159-160`, `paper/chatree_paper.md` section 5.4 (including the Figure 4 caption), and `paper/chatree_paper.tex` near line 419.

**Not affected:** the LoCoMo and Track 7 `tree_summary_only` numbers come from `TreeSummaryOnlyAdapter`, which does call the model.

**Done when:** both paths give the retrieved summary and the same question to the same model, the tracks are rerun, and the claims match the new numbers.

### 0.3 The staleness ablations change two things

**Read in code**

For post-revision questions, `no_staleness` (`ablations.py:327-329`) makes the root answer with `include_core_tools=False`. So it removes the notice **and** the ability to look anything up. `no_transitive_staleness` (`ablations.py:330-336`) and `tree_ipc_no_staleness` (`runner.py:594` onward) have similar forced branches. The harness decides what the model would do.

**Done when:** each ablation removes only the named feature, the model keeps its tools, and the results are rerun.

### 0.4 Small samples

**From results**

- Each Track 5 run grades three questions.
- In Track 2, several ablations score 100% while the full system scores 85.7%. With seven checks, one check is 14.3 points.
- The README already notes the single model and single seed.

**Done when:** there are many more graded checks, several seeds, a hand-checked sample of judge decisions, and intervals next to every headline figure.

---

## Phase 1: stop losing track of topics

These are small fixes with a large effect on jumping conversations.

### 1.1 Topic lists favour the oldest topics

**Reproduced**

- Branch summaries are sorted by `updated_at` ascending, then cut to 8 (`engine/context.py:321`, `:330`). Chat branches come before subagents.
- Children are cut to 10 and siblings to 8 (`context.py:290`, `:282`). The repository returns the oldest first.
- With 15 topics under the root, the summaries showed topics 1–8 and the children line showed 1–10. Topic 15 stayed absent from the block even after its summary was revised.

A newer topic can still appear through the recent-visit line (up to 5), automatic search pointers (3 by default) or fact hints, but only if it was visited recently or the search matches. None of the AI's tools lists all branches.

Fix:

- Fill the list from, in order: relevant to this message, recently visited, recently updated.
- Say how many topics are left out.
- Add a paged `list_branches` tool.
- Make the order of children and siblings newest or most relevant first.

**Done when:** with 200 topics, the most recently updated one is in the block, the omitted count is correct, and paging reaches every topic.

### 1.2 "N unchanged" points at things the model cannot see

**Read in code**

In the steady state, the summaries become `N unchanged (call dialogue_tree_query_branch(id) ...)` with no ids (`context.py:323-327`). The context block is never saved to history, so the model cannot read the earlier list. The `TREE` and `REFLOG` lines still give some ids.

**Done when:** every steady-state block still gives an id and short title for each topic it expects the model to use.

### 1.3 The AI cannot start a thirteenth topic from one parent

**Reproduced**

`engine/ipc/branching.py:106-107` limits `ACTIVE` children to `max_active_children` (12) for topics the AI starts. Switching away does not change a chat branch's status, so ordinary conversation topics stay `ACTIVE`. The thirteenth was refused. Topics created through the SDK are not limited (15 of 15 worked). The AI can get past the limit only through the gap in 1.4.

Fix: apply the limit to running subagent jobs, which is what costs money. Or add an idle status that does not count.

**Done when:** the AI can start 100 chat topics from one parent, while concurrent jobs stay limited.

### 1.4 `autonomous=true` skips the spawn rules

**Reproduced.** I assume this is unintended.

All spawn rules sit under `if initiated_by_model and not autonomous` (`branching.py:63`), and the model can set `autonomous` itself (`tools.py:158`). With it set:

- the thirteenth topic was accepted despite the limit of 12;
- a spawn was accepted with `spawn_policy="manual"`.

By the same condition, `suggest` would also be skipped (**Read in code**).

Fix: apply the policy and limits first, whatever `autonomous` says. If autonomous sub-turns should be allowed, make that a separate host setting that the model cannot change.

**Done when:** tests show that no tool argument gets past `manual`, `suggest` or the limits.

---

## Phase 2: give every part of a request a fixed budget

This is what unlimited length depends on. Anything that grows must be capped, rolled up into a higher-level summary, or reachable only through lookup.

### 2.1 One long branch is cut badly by default

**Reproduced** (with a 3,000-token window in place of the 120,000 default; same code path)

`auto_compact_enabled` is `False` (`config.py:31`). A branch grows until the guard (`context.py:157`) drops the oldest messages. In the probe:

- the compaction handoff was missing as soon as trimming began, because it is the first thing after the system message;
- from then on, only the system message stayed identical from one turn to the next, so the history could not stay in the prompt cache.

Fix:

- Turn automatic compaction on by default.
- Always keep the handoff and pinned items.
- Trim in large blocks, down to a lower mark, so the prefix stays stable between trims.

**Done when:** over 500 turns, the handoff is always present and the prefix changes only at compaction or block trims.

### 2.2 No real limit for the whole request

**Read in code**

- I found no lookup of the model's context window. `max_branch_context_tokens` is one fixed value, 120,000 (`config.py:30`).
- The guard covers only the stored prefix. It does not count the context block, the new message, tool results or tool schemas.
- Small local models can have far smaller windows. Ollama documents a default of about 4k below 24 GiB of video memory. Check how each provider path sets it.

Fix:

- Set the window for each model, from provider data or config.
- Budget = window minus the space reserved for the reply.
- Give each part its own limit: system and tools, pins, topic map, handoff, recent turns, retrieved text, stale notices, mailbox, new message, tool results.
- If the required parts cannot fit, return a clear error.

**Done when:** a test records every request sent to the mock model, including shadow, summary and compaction calls, and none exceeds the window across long random conversations.

### 2.3 Stale notices have no limit

**Reproduced** (a worst case)

The loop at `context.py:350` prints every stale dependency. In the probe, the root had consulted every topic and every topic then changed materially:

| Topics | Context block, estimated tokens |
|---:|---:|
| 5 | 952 |
| 25 | 3,163 |
| 100 | 10,489 |
| 400 | 39,776 |

Fix: rank by severity, distance and relevance. Show a few, then a count and a way to page through the rest.

**Done when:** the block stays under its budget with 1,000 stale dependencies.

### 2.4 Compaction itself has no size limit

**Read in code**, plus a **Risk**

- The compactor reads the guarded prefix (`engine/staleness.py:662`). If a branch is over the limit, it summarises without the oldest turns, and possibly without the earlier handoff.
- The handoff has no token limit and is told to merge and keep all active decisions (`staleness.py:664`). **Risk:** it keeps growing over many generations, until compaction stops saving much. The 9-turn test cannot show this.

Fix:

- Compact in pieces that fit.
- Give the handoff a budget.
- Store pinned decisions as structured entries.
- Move retired and old items to a searchable archive.

**Done when:** over 100 or more generations, handoff size levels off and the chosen decisions survive.

### 2.5 A shadow query on a long branch cannot see its start

**Reproduced** (3,000-token window)

`assemble_for_shadow_query` uses the same guard. A fact from the first turn was missing from the shadow context, although keyword and hybrid search both found it. The shadow prompt says to answer from the branch's turns and not to call tools, so it may report not found for something that exists.

Fix: when the target is over budget, build the shadow context from the handoff or summary, search hits with their neighbouring turns, and the recent turns. Mark the view as partial, so "not found" is not treated as final.

**Done when:** facts at the start, middle and end of an over-budget branch can all be answered.

### 2.6 The topic map must stay small with any number of topics

**Design work**

Good ordering (1.1) is not enough for thousands of topics. Show the path to the root, recent topics, relevant topics and a few group summaries, then let the model open a group. Chatree already has subtree rollups. A flat jumping conversation also needs grouping among siblings, for example by subject or time period.

**Done when:** map size stays within budget at 10, 1,000 and 10,000 topics, and any topic can be reached in a few steps.

### 2.7 A branch's saved facts do not grow with the branch

**Read in code**, plus a **Risk**

The summary prompt asks for 2–4 sentences and up to 8 facts (`staleness.py:529`, `:522`). New facts replace all of that branch's earlier ones (`storage/repos/memory_facts.py:25`). **Risk:** in a long branch that is not compacted, older facts drop out of fact search. The turns themselves remain searchable.

Fix: extract facts from new turns only, add them, and replace individual facts only when one supersedes another.

**Done when:** a fact from early in a long branch is still returned by fact search after many re-summaries.

---

## Phase 3: keep per-turn work flat

### 3.1 Indexing rewrites the whole branch every turn

**Reproduced** (one run on one machine, in-memory database, default hash embedder)

| Turns in the branch | Chunk rows rewritten for one new turn | Indexing time |
|---:|---:|---:|
| 100 | 201 | 44 ms |
| 400 | 801 | 295 ms |
| 1,600 | 3,201 | 689 ms |

Embeddings are reused. But `index_node_chunks` (`engine/rag.py:154`) loads and re-chunks every message, and `replace_node_embeddings` (`storage/repos/search_vectors.py:128`) deletes and reinserts every row, then clears the tree's vector cache. The search after it took 6–10 ms. This runs after the reply is stored, inside the turn.

Fix: re-chunk only the tail, write only changed rows, and update the cache incrementally.

**Done when:** rows written and indexing time stay about the same from 100 to 10,000 turns.

### 3.2 Other work that grows

**Read in code**; not timed

- Building a prefix loads every message in the branch.
- A turn loads the full node list, in some paths more than once.
- Without `sqlite-vec`, rebuilding the vector cache loads every embedding in the tree.

**Done when:** a timing test covers whole turns at growing history sizes.

---

## Phase 4: do not pay for the tree while everything fits

**From results**, plus **Design work**

On the natural 18-turn run, `chatree_live` used 360,289 input tokens against 76,540 for `linear`, with a median turn of 7.0 s against 3.6 s. After allowing for 0.1, it was not more correct on the graded checks. The README already says trees should matter mainly at long length, heavy branching or overflow.

Add an adaptive mode:

- While the conversation fits its budget, send it as one transcript in time order, with each turn labelled by topic. Only add to the end, so the prompt cache can be reused.
- Leave archived branches out, which keeps dead ends quarantined. Removing a branch from the middle will reset part of the cache once.
- When it stops fitting, switch to tree assembly, with a margin so it does not flip back and forth.
- This needs one message order across the whole tree.
- Keep the tree, summaries and index up to date in the background, so switching is cheap.

Risk: with everything visible, the model may mix up topics or revive replaced decisions. Benchmark the mode as its own system.

**Done when:** below the budget, input tokens and latency are close to `linear` with no worse leakage or stale answers, and crossing the budget loses nothing.

---

## Phase 5: prove the long-conversation claim

Build the offline half early. It would catch many of the problems in Phases 1–3.

**Offline checks (mock model, no cost)**

- Random conversations with 1,000–10,000 turns and 100 or more topics, with a small window such as 8k.
- Every request fits (2.2).
- No turn is lost, and every topic is reachable.
- Time per turn and rows written, at checkpoints.

**Real-model checks**

- Facts planted at the start, middle and end, including middle compaction generations.
- Exact questions, reworded questions, and "the thing we compared it with" questions that share no key words.
- Questions that do not say when or where something was said.
- Revised and abandoned decisions many topics later.
- Report separately: did it find the right branch, was the right text in the request, was the answer right?
- Compare with full history where it fits, a sliding window, and destructive summaries.

The default embedder is `hash/local-128` (`config.py:60`), which hashes text features. Do not assume it will match reworded questions. Compare it with a trained embedding model.

**Done when:** the README states the longest length tested, recall at that length by question type, and what failed.

---

## Suggested order

1. **1.4**, because it gets past a guard rail.
2. **0.1–0.3**, so that published claims and later benchmarks can be trusted.
3. The offline checks from **Phase 5**, expected to fail at first.
4. **1.1–1.3**.
5. **2.1, 2.2, 2.3, 2.5**.
6. **3.1**.
7. **2.4, 2.6, 2.7**.
8. **Phase 4**, then the real-model half of **Phase 5**.

## Not checked

- Whether the K=2 crossover result (`linear`, 100% leakage) has the scorer problem. Those answers were not saved.
- The LoCoMo and LongMemEval grading in detail. I only confirmed that their summary-only adapter calls the model.
- Failed checks for `sliding_window` and `flat_rag` in Track 5.
- Multi-session use, subagent jobs, the mailbox's growth, export and import.
