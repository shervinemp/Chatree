"""
Generate publication-grade vector (PDF, SVG) and 300-DPI PNG figures directly from
`results/deepseek_eval/` CSV, JSON, and Markdown artifacts for the Chatree paper.

Run with:
    uv run --with matplotlib python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

RESULTS_DIR = Path("results/deepseek_eval")
OUT_DIR = Path("paper/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Clean academic typography & colorblind-safe Okabe-Ito-inspired palette
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#334155",
        "axes.linewidth": 0.8,
        "xtick.color": "#1e293b",
        "ytick.color": "#1e293b",
        "text.color": "#0f172a",
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#cbd5e1",
        "legend.fontsize": 8.5,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "chatree_two_tier_v3": "#2563eb",  # Royal blue (primary hero)
    "chatree_v3": "#2563eb",
    "chatree_live": "#2563eb",
    "chatree_non_destructive": "#2563eb",
    "tree_ipc_full": "#2563eb",
    "chatree_two_tier": "#60a5fa",  # Sky blue (v2 ablation)
    "full_context_linear": "#475569",  # Slate (linear upper bound)
    "linear": "#475569",
    "mem0_official": "#059669",  # Emerald
    "rlm_official": "#d97706",  # Amber
    "sliding_window": "#9333ea",  # Purple
    "destructive_rolling": "#ea580c",  # Orange
    "flat_hybrid_rag": "#dc2626",  # Crimson
    "flat_rag": "#dc2626",
    "tree_summary_only": "#94a3b8",  # Muted gray
}

DISPLAY_NAMES = {
    "chatree_two_tier_v3": "Chatree v3\n(+Tier-2.5)",
    "chatree_two_tier": "Chatree v2\n(pre-Facts)",
    "full_context_linear": "Full\nLinear",
    "mem0_official": "Mem0\n(Official)",
    "rlm_official": "MIT RLM\n(Official)",
    "flat_hybrid_rag": "Flat\nRAG",
    "tree_summary_only": "Summary\nFolding",
    "chatree_live": "Chatree Live",
    "linear": "Full Linear",
    "sliding_window": "Sliding Window",
    "flat_rag": "Flat RAG",
    "chatree_non_destructive": "Chatree\n(Non-Destructive)",
    "destructive_rolling": "Destructive\nRolling",
    "tree_ipc_full": "Chatree (Full IPC)",
}


def _save(fig: plt.Figure, stem: str) -> None:
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT_DIR / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR / stem}.{{pdf,png,svg}}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _parse_md_table(path: Path, header_substr: str) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for idx, ln in enumerate(lines):
        if header_substr in ln:
            start = idx
            break
    if start is None:
        return []
    # Find first pipe table after header
    tbl_lines: list[str] = []
    for ln in lines[start:]:
        if ln.strip().startswith("|"):
            tbl_lines.append(ln.strip())
        elif tbl_lines:
            break
    if len(tbl_lines) < 3:
        return []
    headers = [c.strip().strip("`") for c in tbl_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for ln in tbl_lines[2:]:
        cells = [c.strip().strip("`").replace("**", "") for c in ln.strip("|").split("|")]
        if len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


# ---------------------------------------------------------------------------
# Figure 1: System Architecture Diagram
# ---------------------------------------------------------------------------
def make_fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 58)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f8fafc", ec="#334155", title_color="#0f172a", lw=1.25):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2", facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=8.6, fontweight="bold", color=title_color)
        if body:
            ax.text(x + w / 2, y + (h - 3.3) / 2 + 0.15, body, ha="center", va="center", fontsize=7.0, color="#334155", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", color="#475569", lx=None, ly=None):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3),
        )
        if label:
            mx = lx if lx is not None else (x1 + x2) / 2
            my = ly if ly is not None else (y1 + y2) / 2
            ax.text(
                mx, my, label, ha="center", va="center", fontsize=6.9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor=color, lw=0.6),
            )

    ax.text(
        52, 56.2,
        "Chatree Architecture: Hierarchical Actor-Model Dialogue Trees with Bounded O(1) Context",
        ha="center", va="center", fontsize=11.4, fontweight="bold", color="#0f172a",
    )

    # Row 1: User -> ContextAssembler -> Orchestrator / TreeQueryEngine
    box(1.5, 40, 17.5, 13.0, "User / Agent Turn", "SDK · CLI · MCP\nFastAPI REST/SSE\nor Benchmark Eval\n(chat / deep ask)", fc="#eff6ff", ec="#2563eb", title_color="#1d4ed8")
    box(23.0, 40, 34.0, 13.0, "ContextAssembler (KV-Cache Safe)", "1. Immutable Static Prefix (CoW Fork / Slice / Clean)\n2. Ephemeral <dialogue_tree_context> Tail:\n   TREE · REFLOG · BRANCH SUMMARIES\n   PINNED [STALE] · FACTS · STALE · MAILBOX", fc="#eef2ff", ec="#4f46e5", title_color="#3730a3")
    box(61.0, 40, 41.5, 13.0, "TurnOrchestrator & TreeQueryEngine", "• Auto-Compact Guard & Frozen Multi-Round Envelope\n• Unified Deep Ask: Hop-1 Parallel Read-Only Shadows\n  → Tier-2.5 Dated Facts → Hop-2 Drill-Down → Synthesis\n• Provider-Agnostic: OpenAI-Compat / LiteLLM / Callable", fc="#f0fdf4", ec="#059669", title_color="#047857")

    arrow(19.6, 46.5, 22.4, 46.5)
    arrow(57.6, 46.5, 60.4, 46.5)

    # Row 2: IPC Router, StalenessService, 4-Tier Hybrid Tree RAG
    box(1.5, 18.5, 31.5, 14.5, "IPCRouter + Guardrails", "• 9 dialogue_tree_* Core Primitives\n• Read-Only Shadow Turns + Memo Cache\n• Cycle Guard · Hop Cap · Budget Ledger\n• Non-Destructive Compact & ARCHIVED\n• Branch Permissions + GitWorktreeSandbox", fc="#fffbeb", ec="#d97706", title_color="#b45309")
    box(36.0, 18.5, 31.5, 14.5, "Staleness & Compaction Engine", "• Lazy Summaries + Piggybacked FACTS_JSON\n  (0 extra LLM calls, absolute dates)\n• Materiality Gating (v1.1 vs Material v2.0)\n• Transitive Staleness DAG (A → B → C)\n• 5-Section Anchor Compact + Savings Guard", fc="#fef2f2", ec="#dc2626", title_color="#b91c1c")
    box(70.5, 18.5, 32.0, 14.5, "4-Tier Hybrid Tree Index (RAG)", "Tier 1: Contextual Chunks (O(1) Tail Index)\nTier 2: Summaries  |  Tier 2.5: Atomic Facts\nTier 3: Depth-Capped Rollups (d ≤ 8)\nFTS5 BM25 + Cached NumPy / sqlite-vec RRF\n+ Local ONNX Reranker (CUDA/DirectML/CPU)", fc="#f5f3ff", ec="#7c3aed", title_color="#6d28d9")

    arrow(72.0, 39.4, 18.5, 33.6, label="tool dispatch", color="#b45309", lx=37.0, ly=36.5)
    arrow(79.5, 39.4, 53.0, 33.6, label="commit / compact", color="#b91c1c", lx=66.0, ly=36.5)
    arrow(86.5, 39.4, 86.5, 33.6, label="4-tier retrieve", color="#6d28d9", lx=91.5, ly=36.5)

    # Row 3: SQLite WAL + Portable Store
    box(7.0, 1.8, 90.0, 11.2, "Local-First Persistent Store (SQLite WAL + FTS5 + Cached Vector Matrices + Portable .dialogue/ v3)", "nodes (CoW path, ACTIVE | COMPLETED | ARCHIVED) · messages (verbatim append-only) · ipc_events · reflog\nnode_dependencies (transitive DAG) · shadow_cache · branch_facts (Tier-2.5) · tree_invariants (PINNED) · artifacts (slug@vN)", fc="#f1f5f9", ec="#475569", title_color="#0f172a")

    arrow(17.5, 17.9, 28.0, 13.6, color="#475569")
    arrow(51.8, 17.9, 51.8, 13.6, color="#475569")
    arrow(86.5, 17.9, 76.0, 13.6, color="#475569")

    _save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Official LoCoMo (n=245) Overall Accuracy + Per-Category Breakdown
# ---------------------------------------------------------------------------
def make_fig2_locomo() -> None:
    rows = _read_csv(RESULTS_DIR / "exp3_locomo_rejudged_v3.csv")
    cat_rows = _parse_md_table(RESULTS_DIR / "exp3_locomo_rejudged_v3.md", "LoCoMo Per-Category Accuracy")
    cat_by_sys = {r["System"]: r for r in cat_rows}

    order = [
        "chatree_two_tier_v3",
        "full_context_linear",
        "mem0_official",
        "rlm_official",
        "chatree_two_tier",
        "flat_hybrid_rag",
        "tree_summary_only",
    ]
    by_sys = {r["system"]: r for r in rows}
    ordered = [by_sys[s] for s in order if s in by_sys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [1.15, 1.3]})

    # Panel (a): Overall accuracy + 95% bootstrap CI
    x = np.arange(len(ordered))
    accs = [float(r["accuracy"]) * 100 for r in ordered]
    lows = [accs[i] - float(r["acc_ci_low"]) * 100 for i, r in enumerate(ordered)]
    highs = [float(r["acc_ci_high"]) * 100 - accs[i] for i, r in enumerate(ordered)]
    bar_colors = [COLORS[r["system"]] for r in ordered]
    labels = [DISPLAY_NAMES[r["system"]] for r in ordered]

    bars = ax1.bar(x, accs, color=bar_colors, width=0.62, edgecolor="#1e293b", linewidth=0.6, zorder=3)
    ax1.errorbar(x, accs, yerr=[lows, highs], fmt="none", ecolor="#0f172a", elinewidth=1.1, capsize=3.5, zorder=4)
    for i, (b, r) in enumerate(zip(bars, ordered)):
        n_tag = "‡" if int(r["n_questions"]) < 245 else ""
        sig = "*" if r["mcnemar_p_vs_ref"] and float(r["mcnemar_p_vs_ref"]) < 0.05 else ""
        ax1.text(
            b.get_x() + b.get_width() / 2,
            accs[i] + highs[i] + 1.5,
            f"{accs[i]:.1f}%{n_tag}{sig}",
            ha="center", va="bottom", fontsize=7.8, fontweight="bold" if i == 0 else "normal",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7.5)
    ax1.set_ylim(0, 96)
    ax1.set_ylabel("Accuracy (%) ± 95% Bootstrap CI")
    ax1.set_title("(a) Official LoCoMo Overall Accuracy (n = 245)")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(0.02, 0.94, "* p < 0.001 vs. Chatree v3 (McNemar)  |  ‡ n = 117", transform=ax1.transAxes, fontsize=7.2, color="#475569")

    # Panel (b): Per-category grouped bars for key hero + competitive systems
    cats = ["single_hop", "multi_hop", "temporal", "open_domain", "adversarial"]
    cat_labels = ["Single-Hop", "Multi-Hop", "Temporal", "Open-Domain", "Adversarial"]
    compare_sys = [
        ("chatree_two_tier_v3", "Chatree v3 (+Tier-2.5)"),
        ("chatree_two_tier", "Chatree v2 (pre-Facts)"),
        ("full_context_linear", "Full-Context Linear"),
        ("mem0_official", "Mem0 (Official)"),
        ("flat_hybrid_rag", "Flat Hybrid RAG"),
    ]
    cx = np.arange(len(cats))
    width = 0.16
    for idx, (skey, slabel) in enumerate(compare_sys):
        crow = cat_by_sys[skey]
        vals = [float(crow[c].rstrip("%")) for c in cats]
        offset = (idx - (len(compare_sys) - 1) / 2) * width
        ax2.bar(
            cx + offset,
            vals,
            width=width,
            label=slabel,
            color=COLORS[skey],
            edgecolor="#1e293b",
            linewidth=0.45,
            zorder=3,
        )
    ax2.set_xticks(cx)
    ax2.set_xticklabels(cat_labels, fontsize=8.3)
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Category Accuracy (%)")
    ax2.set_title("(b) LoCoMo Per-Category Accuracy (Fixed Adversarial Rubric)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper center", ncol=3, fontsize=7.4, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig2_locomo_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Accuracy vs. Peak Context Pareto Frontier (LoCoMo + LongMemEval-S)
# ---------------------------------------------------------------------------
def make_fig3_pareto() -> None:
    loco_rows = _read_csv(RESULTS_DIR / "exp3_locomo_rejudged_v3.csv")
    lme_rows = _read_csv(RESULTS_DIR / "exp4_longmemeval_v3.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))

    def plot_panel(ax, rows, title, offsets):
        for r in rows:
            s = r["system"]
            pk = float(r["peak_context_tokens"])
            acc = float(r["accuracy"]) * 100
            cache = float(r["cache_hit_ratio"]) * 100
            is_hero = s == "chatree_two_tier_v3"
            ax.scatter(
                pk,
                acc,
                s=220 if is_hero else 120,
                marker="*" if is_hero else "o",
                color=COLORS[s],
                edgecolors="#0f172a",
                linewidths=1.0 if is_hero else 0.7,
                zorder=5 if is_hero else 4,
            )
            dx, dy, ha = offsets.get(s, (0, 8, "center"))
            short = DISPLAY_NAMES[s].replace("\n", " ")
            ax.annotate(
                f"{short}\n{acc:.1f}% · {pk/1000:.1f}k tok ({cache:.0f}% cache)",
                xy=(pk, acc),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=7.3,
                fontweight="bold" if is_hero else "normal",
                color="#1d4ed8" if is_hero else "#1e293b",
                arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.7) if abs(dx) > 12 or abs(dy) > 12 else None,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Peak Context Window (tokens, log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.35, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_panel(
        ax1,
        loco_rows,
        "(a) Official LoCoMo (n = 245): Pareto Frontier",
        {
            "chatree_two_tier_v3": (-16, 20, "right"),
            "full_context_linear": (-6, 21, "right"),
            "mem0_official": (12, 8, "left"),
            "rlm_official": (-10, -20, "right"),
            "chatree_two_tier": (-12, -16, "right"),
            "flat_hybrid_rag": (12, 4, "left"),
            "tree_summary_only": (12, -6, "left"),
        },
    )
    ax1.set_xlim(450, 72000)
    ax1.set_ylim(34, 92)

    plot_panel(
        ax2,
        lme_rows,
        "(b) Official LongMemEval-S (n = 30): Pareto Frontier",
        {
            "chatree_two_tier_v3": (-14, 14, "right"),
            "full_context_linear": (-10, -20, "right"),
            "rlm_official": (-10, 18, "right"),
            "chatree_two_tier": (12, -14, "left"),
            "flat_hybrid_rag": (12, 4, "left"),
        },
    )
    ax2.set_xlim(400, 195000)
    ax2.set_ylim(32, 95)
    ax2.annotate(
        "",
        xy=(6800, 80.5),
        xytext=(105000, 80.5),
        arrowprops=dict(arrowstyle="<->", color="#2563eb", lw=1.25),
    )
    ax2.text(
        26000, 83.2,
        "17.3× smaller peak context (+10 pts)",
        fontsize=7.6,
        fontweight="bold",
        color="#1d4ed8",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#eff6ff", edgecolor="#93c5fd", lw=0.7),
    )

    fig.tight_layout(w_pad=2.4)
    _save(fig, "fig3_pareto_efficiency")


# ---------------------------------------------------------------------------
# Figure 4: Track 2 K-Scaling Crossover Curve & Systematic Ablation Matrix
# ---------------------------------------------------------------------------
def make_fig4_crossover_and_ablations() -> None:
    cross_rows = _parse_md_table(RESULTS_DIR / "exp2_crossover.md", "Scaling & Crossover Curve")
    abl_rows = _read_csv(RESULTS_DIR / "exp2_ablations.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.05, 1.25]})

    # Panel (a): Total input tokens & Peak root tokens vs K (2, 4, 8)
    ks = [2, 4, 8]
    for mode, label, ls, marker in [
        ("linear", "Full Linear (total tok)", "-", "s"),
        ("tree_ipc_full", "Chatree Full IPC (total tok · 100% pass)", "-", "o"),
        ("sliding_window", "Sliding Window (total tok · 0% at K=8)", "--", "^"),
    ]:
        ys = []
        for k in ks:
            row = next(r for r in cross_rows if int(r["K"]) == k and r["Mode"] == mode)
            ys.append(int(row["Total Tok"]) / 1000.0)
        ax1.plot(ks, ys, label=label, color=COLORS[mode], linestyle=ls, marker=marker, linewidth=2.0, markersize=6.5, zorder=4)
        for k, y in zip(ks, ys):
            if mode == "sliding_window":
                dy = -2.8 if k < 8 else 1.8
            elif mode == "linear":
                dy = 1.6 if k < 8 else -2.7
            else:
                dy = 1.6 if k < 8 else -2.8
            ax1.text(k, y + dy, f"{y:.1f}k", ha="center", fontsize=7.4, color=COLORS[mode], fontweight="bold")

    ax1.axvline(x=8, color="#2563eb", linestyle=":", alpha=0.45, linewidth=1.2)
    ax1.annotate(
        "Empirical Crossover (K=8)\nChatree 29.6k < Linear 39.7k\n(Peak: 957 vs. 4,266 tok)",
        xy=(8, 29.589),
        xytext=(3.95, 29.5),
        fontsize=7.3,
        fontweight="bold",
        color="#1d4ed8",
        bbox=dict(boxstyle="round,pad=0.32", facecolor="#eff6ff", edgecolor="#93c5fd"),
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.1),
    )
    ax1.set_xticks(ks)
    ax1.set_xticklabels([f"K = {k}" for k in ks])
    ax1.set_ylim(0, 46)
    ax1.set_xlabel("Number of Distractor Branches (K)")
    ax1.set_ylabel("Total Input Tokens (thousands)")
    ax1.set_title("(a) Track 2 Scaling & Total-Token Crossover Curve")
    ax1.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=7.3)

    # Panel (b): Horizontal bar chart of key ablations
    abl_select = [
        ("chatree_oracle_routing", "Oracle Routing (Upper Bound)"),
        ("chatree_full", "Chatree Full (Non-Oracle)"),
        ("no_staleness", "− Staleness Tracking"),
        ("no_materiality_gating", "− Materiality Gating"),
        ("no_hybrid_rag_fts_only", "− Dense Vectors (FTS5 Only)"),
        ("no_transitive_staleness", "− Transitive Staleness (A→B→C)"),
        ("no_shadow_turns_folding", "− Shadow Turns (Folding Only)*"),
    ]
    by_abl = {r["ablation"]: r for r in abl_rows}
    y = np.arange(len(abl_select))
    vals = [float(by_abl[k]["pass_rate"]) * 100 for k, _ in abl_select]
    stale_errs = [float(by_abl[k]["stale_contradiction_rate"]) * 100 for k, _ in abl_select]
    colors_b = [
        "#059669" if k == "chatree_oracle_routing"
        else "#2563eb" if k == "chatree_full"
        else "#dc2626" if k == "no_shadow_turns_folding"
        else "#f59e0b"
        for k, _ in abl_select
    ]
    bars = ax2.barh(y, vals, color=colors_b, height=0.58, edgecolor="#1e293b", linewidth=0.6, zorder=3)
    for i, (b, val, se) in enumerate(zip(bars, vals, stale_errs)):
        tag = f"{val:.1f}%"
        if se > 0:
            tag += f"  (stale err {se:.1f}%)"
        ax2.text(min(val + 1.8, 102), b.get_y() + b.get_height() / 2, tag, va="center", ha="left", fontsize=7.6, fontweight="bold" if i <= 1 or val == 0 else "normal")

    ax2.set_yticks(y)
    ax2.set_yticklabels([lbl for _, lbl in abl_select], fontsize=8.0)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 134)
    ax2.set_xlabel("Strict Pass Rate (%) across 4 Stress Scenarios")
    ax2.set_title("(b) Track 2 Mechanism Ablations (*McNemar p = 0.0312)")
    ax2.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig4_crossover_and_ablations")


# ---------------------------------------------------------------------------
# Figure 5: Track 6 Infinite-Context Compaction & Track 5 Natural Jump-Around
# ---------------------------------------------------------------------------
def make_fig5_compaction_and_live() -> None:
    comp_data = json.loads((RESULTS_DIR / "exp6_compaction_bench.json").read_text(encoding="utf-8"))["results"]
    ja_rows = _read_csv(RESULTS_DIR / "exp5_natural_jump_around.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Panel (a): Track 6 Anchored vs Non-Anchored Verbatim Recall
    x = np.arange(len(comp_data))
    anchored = [r["anchored_recall"] * 100 for r in comp_data]
    non_anchored = [r["non_anchored_recall"] * 100 for r in comp_data]
    w = 0.34

    b1 = ax1.bar(x - w / 2, anchored, width=w, label="Anchored Decisions Recall", color="#60a5fa", edgecolor="#1e293b", linewidth=0.6, zorder=3)
    b2 = ax1.bar(x + w / 2, non_anchored, width=w, label="Non-Anchored Turn-1 Verbatim Recall", color="#2563eb", edgecolor="#1e293b", linewidth=0.6, zorder=3)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width() / 2, h + 2.0, f"{h:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")

    xlabels = [
        f"{DISPLAY_NAMES[r['system']]}\n(Peak {r['peak_context_tokens']:,} tok · Gen-1: {r['gen1_messages_preserved']}/4)"
        for r in comp_data
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=7.6)
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Recall (%)")
    ax1.set_title("(a) Track 6: 9-Generation Infinite-Context Compaction")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", fontsize=7.6)

    # Panel (b): Track 5 Natural-Prompt Live Jump-Around (Recall, Judge, KV Cache)
    x2 = np.arange(len(ja_rows))
    recalls = [float(r["mean_recall"]) * 100 for r in ja_rows]
    judges = [float(r["avg_judge_score"]) * 100 for r in ja_rows]
    caches = [float(r["cache_hit_ratio"]) * 100 for r in ja_rows]
    w2 = 0.25

    ax2.bar(x2 - w2, recalls, width=w2, label="Fact Recall (%)", color="#2563eb", edgecolor="#1e293b", linewidth=0.5, zorder=3)
    ax2.bar(x2, judges, width=w2, label="LLM Judge (%)", color="#059669", edgecolor="#1e293b", linewidth=0.5, zorder=3)
    ax2.bar(x2 + w2, caches, width=w2, label="KV Cache Hit (%)", color="#d97706", edgecolor="#1e293b", linewidth=0.5, zorder=3)
    for i, r in enumerate(ja_rows):
        ax2.text(i - w2 - 0.02, recalls[i] + 2.0, f"{recalls[i]:.0f}%", ha="center", fontsize=6.6, fontweight="bold")
        ax2.text(i + 0.02, judges[i] + 2.0, f"{judges[i]:.0f}%", ha="center", fontsize=6.6)
        ax2.text(i + w2 + 0.02, caches[i] + 2.0, f"{caches[i]:.0f}%", ha="center", fontsize=6.6)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(
        [f"{DISPLAY_NAMES[r['system']]}\n(Peak {int(r['peak_context_tokens']):,} tok)" for r in ja_rows],
        fontsize=7.6,
    )
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Score / Hit Rate (%)")
    ax2.set_title("(b) Track 5: Live 18-Turn Jump-Around (Natural Prompts)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(w_pad=2.2)
    _save(fig, "fig5_compaction_and_live")


if __name__ == "__main__":
    make_fig1_architecture()
    make_fig2_locomo()
    make_fig3_pareto()
    make_fig4_crossover_and_ablations()
    make_fig5_compaction_and_live()
    print("All 5 publication figures generated cleanly.")
