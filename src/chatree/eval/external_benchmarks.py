from __future__ import annotations

import asyncio
import csv
import io
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence
from urllib.request import urlretrieve

from pydantic import BaseModel, Field

from chatree.eval.adapters import MemorySystemAdapter
from chatree.eval.runner import _percentile
from chatree.eval.scoring import CostModel
from chatree.eval.stats import (
    BootstrapCI,
    McNemarResult,
    PairedBootstrapResult,
    bleu1,
    bootstrap_ci,
    format_latex_table,
    mcnemar_test,
    paired_bootstrap_test,
    token_f1,
)
from chatree.protocols import LLMProvider
from chatree.providers.mock import MockLLMProvider

LOCOMO_DEFAULT_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
LONGMEMEVAL_URLS: dict[str, str] = {
    "s": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
    "m": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json",
}

LOCOMO_CATEGORY_MAP: dict[int, str] = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


class ExternalTurn(BaseModel):
    role: str
    content: str
    speaker: str = ""
    date_time: str = ""


class ExternalSession(BaseModel):
    session_id: str
    title: str = ""
    date_time: str = ""
    summary: str | None = None
    turns: list[ExternalTurn] = Field(default_factory=list)


class ExternalQAItem(BaseModel):
    question_id: str
    question: str
    ground_truth: str
    distractor_answer: str = ""
    category: str = "general"
    question_date: str = ""
    is_abstention: bool = False


class ExternalConversation(BaseModel):
    conversation_id: str
    benchmark: Literal["locomo", "longmemeval", "custom"] = "custom"
    sessions: list[ExternalSession] = Field(default_factory=list)
    qa_items: list[ExternalQAItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset Loaders: LoCoMo (ACL 2024) & LongMemEval (ICLR 2025)
# ---------------------------------------------------------------------------


class LoCoMoDataset:
    """
    Loader for the official LoCoMo benchmark (`snap-research/locomo`, Maharana et al., ACL 2024).
    Parses `locomo10.json` without modifying conversations, timestamps, or QA annotations.
    """

    @staticmethod
    def download(dest_path: str | Path, *, url: str = LOCOMO_DEFAULT_URL) -> Path:
        p = Path(dest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, p)
        return p

    @staticmethod
    def from_records(
        records: list[dict[str, Any]],
        *,
        max_conversations: int | None = None,
        max_questions_per_conv: int | None = None,
        categories: set[str] | None = None,
    ) -> list[ExternalConversation]:
        out: list[ExternalConversation] = []
        subset = records[:max_conversations] if max_conversations else records
        for idx, rec in enumerate(subset):
            cid = str(rec.get("sample_id") or f"locomo_{idx}")
            conv_raw = rec.get("conversation") or {}
            summaries = rec.get("session_summary") or {}

            # Extract ordered session_1, session_2, ...
            session_nums: list[int] = []
            for k in conv_raw.keys():
                m = re.fullmatch(r"session_(\d+)", str(k))
                if m:
                    session_nums.append(int(m.group(1)))
            session_nums.sort()

            sessions: list[ExternalSession] = []
            speaker_a = str(conv_raw.get("speaker_a") or "Speaker A")
            for num in session_nums:
                raw_turns = conv_raw.get(f"session_{num}") or []
                dt = str(conv_raw.get(f"session_{num}_date_time") or "")
                sess_sum = summaries.get(f"session_{num}_summary")
                turns: list[ExternalTurn] = []
                for t in raw_turns:
                    spk = str(t.get("speaker") or "")
                    txt = str(t.get("text") or "").strip()
                    if t.get("blip_caption"):
                        txt = f"{txt} [Image: {t['blip_caption']}]".strip()
                    if not txt:
                        continue
                    role = "user" if spk == speaker_a else "assistant"
                    turns.append(
                        ExternalTurn(role=role, speaker=spk, content=f"{spk}: {txt}" if spk else txt, date_time=dt)
                    )
                sessions.append(
                    ExternalSession(
                        session_id=f"session_{num}",
                        title=f"Session {num}",
                        date_time=dt,
                        summary=str(sess_sum) if sess_sum else None,
                        turns=turns,
                    )
                )

            qa_items: list[ExternalQAItem] = []
            for q_idx, q in enumerate(rec.get("qa") or []):
                cat_raw = q.get("category")
                cat = (
                    LOCOMO_CATEGORY_MAP.get(int(cat_raw), str(cat_raw))
                    if isinstance(cat_raw, int) or (isinstance(cat_raw, str) and cat_raw.isdigit())
                    else str(cat_raw or "general")
                )
                if categories and cat not in categories:
                    continue
                is_abs = cat == "adversarial"
                if is_abs:
                    # LoCoMo Category 5 stores the wrong-entity distractor in `adversarial_answer`;
                    # NEVER pass it as `ground_truth` to the abstention judge!
                    gt = ""
                    distractor = str(
                        q.get("adversarial_answer")
                        if q.get("adversarial_answer") is not None
                        else q.get("answer", "")
                    )
                else:
                    gt = str(q.get("answer") if q.get("answer") is not None else "")
                    distractor = ""
                qa_items.append(
                    ExternalQAItem(
                        question_id=f"{cid}_q{q_idx}",
                        question=str(q.get("question") or ""),
                        ground_truth=gt,
                        distractor_answer=distractor,
                        category=cat,
                        is_abstention=is_abs,
                    )
                )
                if max_questions_per_conv and len(qa_items) >= max_questions_per_conv:
                    break

            out.append(
                ExternalConversation(
                    conversation_id=cid,
                    benchmark="locomo",
                    sessions=sessions,
                    qa_items=qa_items,
                )
            )
        return out

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        max_conversations: int | None = None,
        max_questions_per_conv: int | None = None,
        categories: set[str] | None = None,
    ) -> list[ExternalConversation]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_records(
            raw,
            max_conversations=max_conversations,
            max_questions_per_conv=max_questions_per_conv,
            categories=categories,
        )


class LongMemEvalDataset:
    """
    Loader for the official LongMemEval benchmark (`xiaowu0162/longmemeval-cleaned`, Wu et al., ICLR 2025).
    Parses `longmemeval_s_cleaned.json` / `longmemeval_m_cleaned.json` faithfully.
    """

    @staticmethod
    def download(dest_path: str | Path, *, split: Literal["s", "m"] = "s") -> Path:
        p = Path(dest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(LONGMEMEVAL_URLS[split], p)
        return p

    @staticmethod
    def from_records(
        records: list[dict[str, Any]],
        *,
        max_items: int | None = None,
        categories: set[str] | None = None,
    ) -> list[ExternalConversation]:
        out: list[ExternalConversation] = []
        for idx, rec in enumerate(records):
            qid = str(rec.get("question_id") or f"lme_{idx}")
            qtype = str(rec.get("question_type") or "general")
            if categories and qtype not in categories:
                continue
            is_abs = qid.endswith("_abs") or "abstention" in qtype.lower()

            sess_ids = rec.get("haystack_session_ids") or []
            sess_dates = rec.get("haystack_dates") or []
            raw_sessions = rec.get("haystack_sessions") or []

            sessions: list[ExternalSession] = []
            for s_idx, s_turns in enumerate(raw_sessions):
                sid = str(sess_ids[s_idx]) if s_idx < len(sess_ids) else f"session_{s_idx + 1}"
                dt = str(sess_dates[s_idx]) if s_idx < len(sess_dates) else ""
                turns = [
                    ExternalTurn(
                        role=str(t.get("role") or "user"),
                        content=str(t.get("content") or ""),
                        date_time=dt,
                    )
                    for t in (s_turns or [])
                    if str(t.get("content") or "").strip()
                ]
                sessions.append(
                    ExternalSession(
                        session_id=sid,
                        title=f"Session {s_idx + 1}" + (f" ({dt})" if dt else ""),
                        date_time=dt,
                        turns=turns,
                    )
                )

            qa = ExternalQAItem(
                question_id=qid,
                question=str(rec.get("question") or ""),
                ground_truth=str(rec.get("answer") or ""),
                category=qtype,
                question_date=str(rec.get("question_date") or ""),
                is_abstention=is_abs,
            )
            out.append(
                ExternalConversation(
                    conversation_id=qid,
                    benchmark="longmemeval",
                    sessions=sessions,
                    qa_items=[qa],
                )
            )
            if max_items and len(out) >= max_items:
                break
        return out

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        max_items: int | None = None,
        categories: set[str] | None = None,
    ) -> list[ExternalConversation]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_records(raw, max_items=max_items, categories=categories)


# ---------------------------------------------------------------------------
# Official LongMemEval / LoCoMo Binary Judge & Head-to-Head Evaluation Report
# ---------------------------------------------------------------------------

_ABSTENTION_MARKERS = (
    "i do not know",
    "i don't know",
    "not mentioned",
    "no information",
    "cannot answer",
    "not enough information",
    "was not discussed",
    "never mentioned",
    "don't have that information",
    "do not have enough information",
    "did not ",
    "didn't ",
    "was not ",
    "wasn't ",
    "is not the one",
    "has no ",
    "never ",
)


async def score_external_qa(
    *,
    question: str,
    ground_truth: str,
    prediction: str,
    category: str,
    is_abstention: bool,
    judge_llm: LLMProvider | None,
    judge_model: str | None,
    distractor_answer: str = "",
) -> tuple[bool, float, float, str]:
    """
    Score one QA item returning `(judge_correct, f1_score, bleu1_score, rationale)`.
    Uses the official LongMemEval/LoCoMo binary LLM-judge rubric when `judge_llm` is a real LLM,
    properly accepting both explicit abstention AND truthful false-premise/speaker-swap
    corrections on adversarial items without leaking distractor answers as `Ground Truth:`.
    """
    f1 = token_f1(prediction, ground_truth) if ground_truth else 0.0
    b1 = bleu1(prediction, ground_truth) if ground_truth else 0.0

    if is_abstention:
        pred_low = prediction.lower()
        abstained = any(mk in pred_low for mk in _ABSTENTION_MARKERS)
        if judge_llm is None or not judge_model or judge_model.startswith("mock/"):
            return abstained, (1.0 if abstained else 0.0), (1.0 if abstained else 0.0), "abstention_check"

    if judge_llm is not None and judge_model and not judge_model.startswith("mock/"):
        if is_abstention:
            distractor_line = (
                f"Wrong-Premise Distractor (from another person/event — NOT valid for the asked premise): {distractor_answer}\n"
                if distractor_answer
                else ""
            )
            rubric = (
                "You are evaluating a conversational memory assistant on a standardized benchmark "
                f"(category: {category}).\n"
                "This is an ABSTENTION / ADVERSARIAL FALSE-PREMISE question: the asked-about fact or "
                "person-event pairing did NOT occur in the conversation history (e.g. the event never "
                "happened or happened to a DIFFERENT speaker).\n"
                "Mark correct=true if EITHER:\n"
                "  (a) the assistant explicitly states that the information is unknown / not mentioned, OR\n"
                "  (b) the assistant explicitly rejects the false premise or corrects the swapped speaker/entity "
                "(e.g., 'Caroline did not run the race — Melanie did', even if it also helpfully mentions what Melanie did).\n"
                "Mark correct=false ONLY if the assistant accepts the false premise as true, fabricates a claim "
                "for the asked person/entity, or attributes another person's event to the wrong person.\n"
                'Respond with strict JSON: {"correct": true|false, "rationale": "short reason"}\n'
                f"Question: {question}\n"
                f"{distractor_line}"
                f"Assistant Response: {prediction[:1500]}"
            )
        else:
            rubric = (
                "You are evaluating a conversational memory assistant on a standardized benchmark "
                f"(category: {category}).\n"
                "Mark correct=true if the assistant's response contains the correct answer "
                "and is consistent with the ground truth (allow minor paraphrasing/formatting differences; "
                "for knowledge-update questions, the latest updated fact must be given).\n"
                'Respond with strict JSON: {"correct": true|false, "rationale": "short reason"}\n'
                f"Question: {question}\n"
                f"Ground Truth: {ground_truth}\n"
                f"Assistant Response: {prediction[:1500]}"
            )
        resp = await judge_llm.complete(
            model=judge_model,
            messages=[{"role": "user", "content": rubric}],
            tools=None,
            temperature=0.0,
        )
        m = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return bool(parsed.get("correct", False)), f1, b1, str(parsed.get("rationale") or "")
            except json.JSONDecodeError:
                pass
        return ("true" in resp.content.lower() and "false" not in resp.content.lower()), f1, b1, resp.content[:160]

    # Deterministic offline fallback (strict substring match of normalized ground truth or F1 >= 0.5)
    gt_low = ground_truth.strip().lower()
    pred_low = prediction.lower()
    correct = bool(gt_low and (gt_low in pred_low or f1 >= 0.5))
    return correct, f1, b1, "offline_exact_or_f1"


class ExternalQAResult(BaseModel):
    conversation_id: str
    question_id: str
    category: str
    question: str
    ground_truth: str
    prediction: str
    correct: bool
    f1: float
    bleu1: float
    judge_rationale: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read_tokens: int = 0
    peak_context_tokens: int = 0
    latency_ms: float = 0.0
    ingest_tokens_in: int = 0
    ingest_tokens_out: int = 0
    ingest_latency_ms: float = 0.0


class SystemExternalResult(BaseModel):
    system_name: str
    benchmark: str
    n_questions: int
    accuracy: float
    accuracy_ci: BootstrapCI
    mean_f1: float
    f1_ci: BootstrapCI
    mean_bleu1: float
    category_accuracy: dict[str, float] = Field(default_factory=dict)
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    cache_hit_ratio: float = 0.0
    peak_context_tokens: int = 0
    estimated_cost_usd: float = 0.0
    ingest_tokens_in: int = 0
    ingest_tokens_out: int = 0
    ingest_cost_usd: float = 0.0
    ingest_latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    mcnemar_vs_ref: McNemarResult | None = None
    paired_f1_vs_ref: PairedBootstrapResult | None = None
    item_results: list[ExternalQAResult] = Field(default_factory=list)


class ExternalBenchmarkReport(BaseModel):
    benchmark: str
    model: str
    reference_system: str
    results: list[SystemExternalResult]

    def to_markdown_table(self) -> str:
        lines = [
            f"### Publishable External Benchmark: `{self.benchmark}` (model=`{self.model}`)",
            "",
            "| System | Accuracy (95% CI) | Token F1 | BLEU-1 | Peak Tok | Est. USD | Cache | p50 / p95 ms | McNemar p vs. Ref |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in self.results:
            ci_str = f"{r.accuracy * 100:.1f}% [{r.accuracy_ci.ci_low * 100:.1f}, {r.accuracy_ci.ci_high * 100:.1f}]"
            p_str = (
                "ref"
                if r.system_name == self.reference_system
                else (
                    f"{r.mcnemar_vs_ref.p_value:.4f}" + ("*" if r.mcnemar_vs_ref.significant_05 else "")
                    if r.mcnemar_vs_ref is not None
                    else "—"
                )
            )
            lines.append(
                f"| `{r.system_name}` | {ci_str} | {r.mean_f1:.3f} | {r.mean_bleu1:.3f} "
                f"| {r.peak_context_tokens} | ${r.estimated_cost_usd:.5f} "
                f"| {r.cache_hit_ratio * 100:.1f}% | {r.latency_p50_ms:.1f} / {r.latency_p95_ms:.1f} | {p_str} |"
            )
        return "\n".join(lines)

    def to_latex_table(
        self,
        *,
        caption: str | None = None,
        label: str | None = None,
    ) -> str:
        headers = [
            "System",
            "Accuracy (95% CI)",
            "F1",
            "BLEU-1",
            "Peak Tok",
            "Cost (USD)",
            "p50 ms",
            "p-val",
        ]
        rows: list[list[str]] = []
        for r in self.results:
            ci_str = f"{r.accuracy * 100:.1f}% [{r.accuracy_ci.ci_low * 100:.1f}, {r.accuracy_ci.ci_high * 100:.1f}]"
            p_str = (
                "ref"
                if r.system_name == self.reference_system
                else (
                    f"{r.mcnemar_vs_ref.p_value:.4f}" + ("*" if r.mcnemar_vs_ref.significant_05 else "")
                    if r.mcnemar_vs_ref is not None
                    else "-"
                )
            )
            rows.append(
                [
                    r.system_name,
                    ci_str,
                    f"{r.mean_f1:.3f}",
                    f"{r.mean_bleu1:.3f}",
                    str(r.peak_context_tokens),
                    f"${r.estimated_cost_usd:.5f}",
                    f"{r.latency_p50_ms:.1f}",
                    p_str,
                ]
            )
        return format_latex_table(
            headers,
            rows,
            caption=caption or f"{self.benchmark} comparison on {self.model}",
            label=label or f"tab:{self.benchmark}",
        )

    def save_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "benchmark": r.benchmark,
                "system": r.system_name,
                "n_questions": r.n_questions,
                "accuracy": r.accuracy,
                "acc_ci_low": r.accuracy_ci.ci_low,
                "acc_ci_high": r.accuracy_ci.ci_high,
                "mean_f1": r.mean_f1,
                "mean_bleu1": r.mean_bleu1,
                "peak_context_tokens": r.peak_context_tokens,
                "total_tokens_in": r.total_tokens_in,
                "total_tokens_out": r.total_tokens_out,
                "cache_hit_ratio": r.cache_hit_ratio,
                "estimated_cost_usd": r.estimated_cost_usd,
                "ingest_tokens_in": r.ingest_tokens_in,
                "ingest_tokens_out": r.ingest_tokens_out,
                "ingest_cost_usd": r.ingest_cost_usd,
                "ingest_latency_ms": r.ingest_latency_ms,
                "latency_p50_ms": r.latency_p50_ms,
                "latency_p95_ms": r.latency_p95_ms,
                "mcnemar_p_vs_ref": r.mcnemar_vs_ref.p_value if r.mcnemar_vs_ref else None,
            }
            for r in self.results
        ]
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        p.write_text(buf.getvalue(), encoding="utf-8")
        return p

    def save_manifest(
        self,
        path: str | Path,
        *,
        judge_model: str | None = None,
        embed_model: str = "hash/local-128",
        reranker_name: str = "none",
        extra: dict[str, Any] | None = None,
    ) -> Path:
        return write_run_manifest(
            path,
            benchmark=self.benchmark,
            model=self.model,
            judge_model=judge_model,
            embed_model=embed_model,
            reranker_name=reranker_name,
            total_tokens_in=sum(r.total_tokens_in for r in self.results),
            total_tokens_out=sum(r.total_tokens_out for r in self.results),
            total_estimated_cost_usd=round(sum(r.estimated_cost_usd for r in self.results), 6),
            systems=[r.system_name for r in self.results],
            extra=extra,
        )


class DryRunEstimate(BaseModel):
    benchmark: str
    model: str
    n_conversations: int
    n_sessions: int
    n_questions: int
    systems: list[str]
    estimated_llm_calls: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float

    def to_summary(self) -> str:
        return (
            f"[DRY RUN] benchmark={self.benchmark} | model={self.model} | "
            f"convs={self.n_conversations} | sessions={self.n_sessions} | questions={self.n_questions} | "
            f"systems={', '.join(self.systems)} | est_calls={self.estimated_llm_calls} | "
            f"est_tokens={self.estimated_input_tokens:,} in / {self.estimated_output_tokens:,} out | "
            f"est_cost=${self.estimated_cost_usd:.4f}"
        )


def _git_head_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=3
        )
        return out.decode("utf-8", errors="ignore").strip() or None
    except Exception:
        return None


def write_run_manifest(
    path: str | Path,
    *,
    benchmark: str,
    model: str,
    judge_model: str | None = None,
    embed_model: str = "hash/local-128",
    reranker_name: str = "none",
    total_tokens_in: int = 0,
    total_tokens_out: int = 0,
    total_estimated_cost_usd: float = 0.0,
    systems: Sequence[str] = (),
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist an immutable provenance manifest alongside CSV/JSONL/LaTeX benchmark outputs."""
    from chatree import __version__

    manifest: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "chatree_version": __version__,
        "git_sha": _git_head_sha(),
        "benchmark": benchmark,
        "model": model,
        "judge_model": judge_model,
        "embed_model": embed_model,
        "reranker": reranker_name,
        "systems": list(systems),
        "total_tokens_in": total_tokens_in,
        "total_tokens_out": total_tokens_out,
        "total_estimated_cost_usd": total_estimated_cost_usd,
    }
    if extra:
        manifest["extra"] = extra
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


class ExternalBenchmarkRunner:
    """
    Runs head-to-head publishable evaluations across standardized external datasets
    (LoCoMo, LongMemEval) for any list of `MemorySystemAdapter`s, with:
      - Per-item JSONL checkpointing & automatic resume (`checkpoint_path`)
      - Bounded async concurrency (`max_concurrency`)
      - Pre-flight dry-run cost estimator (`estimate_cost_dry_run`)
      - Official metrics, 95% bootstrap CIs, and paired McNemar / bootstrap significance tests.
    """

    def __init__(
        self,
        *,
        model: str = "mock/chatree-demo",
        judge_model: str | None = None,
        judge_llm: LLMProvider | None = None,
        cost_model: CostModel | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.model = model
        self.judge_model = judge_model
        self.judge_llm = judge_llm or (
            MockLLMProvider(stream_delay_s=0.0) if (judge_model or "").startswith("mock/") else None
        )
        self.cost_model = cost_model or CostModel()
        self.max_concurrency = max(1, max_concurrency)

    def estimate_cost_dry_run(
        self,
        conversations: Sequence[ExternalConversation],
        adapters: Sequence[MemorySystemAdapter],
    ) -> DryRunEstimate:
        """Estimate API calls, tokens, and USD cost before making a single paid LLM request."""
        benchmark_name = conversations[0].benchmark if conversations else "custom"
        n_sessions = sum(len(c.sessions) for c in conversations)
        n_questions = sum(len(c.qa_items) for c in conversations)

        total_chars = sum(len(t.content) for c in conversations for s in c.sessions for t in s.turns)
        avg_conv_tokens = max(200, (total_chars // 4) // max(1, len(conversations)))
        avg_sess_tokens = max(80, (total_chars // 4) // max(1, n_sessions))

        est_calls = 0
        est_in = 0
        est_out = 0
        for adapter in adapters:
            nm = adapter.name.lower()
            if "linear" in nm:
                est_calls += n_questions
                est_in += n_questions * (avg_conv_tokens + 120)
                est_out += n_questions * 80
            elif "chatree" in nm:
                k = int(getattr(adapter, "max_ask_targets", 4))
                # k parallel shadow turns + 1 concise synthesizer call per question
                est_calls += n_questions * (k + 1)
                est_in += n_questions * (k * (avg_sess_tokens + 200) + 600)
                est_out += n_questions * (k * 60 + 80)
            else:
                est_calls += n_questions
                est_in += n_questions * 1600
                est_out += n_questions * 80

        if self.judge_model and not self.judge_model.startswith("mock/"):
            judge_calls = n_questions * len(adapters)
            est_calls += judge_calls
            est_in += judge_calls * 450
            est_out += judge_calls * 50

        est_usd = self.cost_model.estimate_usd(
            self.model, tokens_in=est_in, tokens_out=est_out, cache_read_tokens=0
        )
        return DryRunEstimate(
            benchmark=benchmark_name,
            model=self.model,
            n_conversations=len(conversations),
            n_sessions=n_sessions,
            n_questions=n_questions,
            systems=[a.name for a in adapters],
            estimated_llm_calls=est_calls,
            estimated_input_tokens=est_in,
            estimated_output_tokens=est_out,
            estimated_cost_usd=est_usd,
        )

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> dict[tuple[str, str, str], ExternalQAResult]:
        cache: dict[tuple[str, str, str], ExternalQAResult] = {}
        if not checkpoint_path.exists():
            return cache
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                sys_name = str(rec["system_name"])
                res = ExternalQAResult(**rec["result"])
                cache[(sys_name, res.conversation_id, res.question_id)] = res
            except Exception:
                continue
        return cache

    async def evaluate(
        self,
        conversations: Sequence[ExternalConversation],
        adapters: Sequence[MemorySystemAdapter],
        *,
        reference_system: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> ExternalBenchmarkReport:
        benchmark_name = conversations[0].benchmark if conversations else "custom"
        ref_name = reference_system or (adapters[0].name if adapters else "")
        system_results: list[SystemExternalResult] = []

        cp_path = Path(checkpoint_path) if checkpoint_path else None
        cp_cache = self._load_checkpoint(cp_path) if cp_path else {}
        cp_lock = asyncio.Lock()

        async def _append_cp(sys_name: str, res: ExternalQAResult) -> None:
            if cp_path is None:
                return
            async with cp_lock:
                cp_path.parent.mkdir(parents=True, exist_ok=True)
                with cp_path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"system_name": sys_name, "result": res.model_dump()}) + "\n"
                    )

        for adapter in adapters:
            can_clone = callable(getattr(adapter, "clone", None))
            concurrency = self.max_concurrency if can_clone else 1
            sem = asyncio.Semaphore(concurrency)

            async def _eval_conversation(
                conv: ExternalConversation,
                _adapter: MemorySystemAdapter = adapter,
                _can_clone: bool = can_clone,
            ) -> list[ExternalQAResult]:
                # Fast-path resume: skip entire conversation if every QA item is already checkpointed!
                cached_items: list[ExternalQAResult] = []
                missing_qas: list[ExternalQAItem] = []
                for qa in conv.qa_items:
                    hit = cp_cache.get((_adapter.name, conv.conversation_id, qa.question_id))
                    if hit is not None:
                        cached_items.append(hit)
                    else:
                        missing_qas.append(qa)
                if not missing_qas:
                    return cached_items

                async with sem:
                    worker = _adapter.clone() if _can_clone else _adapter  # type: ignore[attr-defined]
                    await worker.reset(conv.conversation_id)
                    results_by_qid: dict[str, ExternalQAResult] = {
                        r.question_id: r for r in cached_items
                    }
                    try:
                        t_ing0 = time.perf_counter()
                        for sess in conv.sessions:
                            await worker.start_session(
                                sess.session_id, title=sess.title, date_time=sess.date_time
                            )
                            for turn in sess.turns:
                                await worker.ingest_turn(
                                    sess.session_id,
                                    turn.role,
                                    turn.content,
                                    date_time=turn.date_time or sess.date_time,
                                )
                            await worker.finish_session(sess.session_id, summary=sess.summary)
                        ingest_ms = round((time.perf_counter() - t_ing0) * 1000.0, 2)
                        get_ingest_fn = getattr(worker, "get_ingest_usage", None)
                        ing_in, ing_out = get_ingest_fn() if callable(get_ingest_fn) else (0, 0)

                        for q_idx, qa in enumerate(missing_qas):
                            ans = await worker.answer_query(qa.question, date_time=qa.question_date)
                            ok, f1, b1, rat = await score_external_qa(
                                question=qa.question,
                                ground_truth=qa.ground_truth,
                                distractor_answer=qa.distractor_answer,
                                prediction=ans.answer,
                                category=qa.category,
                                is_abstention=qa.is_abstention,
                                judge_llm=self.judge_llm,
                                judge_model=self.judge_model,
                            )
                            qres = ExternalQAResult(
                                conversation_id=conv.conversation_id,
                                question_id=qa.question_id,
                                category=qa.category,
                                question=qa.question,
                                ground_truth=qa.ground_truth,
                                prediction=ans.answer,
                                correct=ok,
                                f1=f1,
                                bleu1=b1,
                                judge_rationale=rat,
                                tokens_in=ans.tokens_in,
                                tokens_out=ans.tokens_out,
                                cache_read_tokens=ans.cache_read_tokens,
                                peak_context_tokens=ans.peak_context_tokens,
                                latency_ms=ans.latency_ms,
                                # Record per-conversation ingest usage on first QA only to avoid double-counting
                                ingest_tokens_in=ing_in if q_idx == 0 else 0,
                                ingest_tokens_out=ing_out if q_idx == 0 else 0,
                                ingest_latency_ms=ingest_ms if q_idx == 0 else 0.0,
                            )
                            results_by_qid[qa.question_id] = qres
                            cp_cache[(_adapter.name, conv.conversation_id, qa.question_id)] = qres
                            await _append_cp(_adapter.name, qres)
                    finally:
                        await worker.close()
                    return [results_by_qid[qa.question_id] for qa in conv.qa_items]

            if concurrency > 1 and len(conversations) > 1:
                conv_batches = await asyncio.gather(*(_eval_conversation(c) for c in conversations))
            else:
                conv_batches = [await _eval_conversation(c) for c in conversations]

            item_results: list[ExternalQAResult] = [r for batch in conv_batches for r in batch]

            acc_vals = [1.0 if r.correct else 0.0 for r in item_results]
            f1_vals = [r.f1 for r in item_results]
            b1_vals = [r.bleu1 for r in item_results]
            lat_vals = [r.latency_ms for r in item_results]

            cat_buckets: dict[str, list[float]] = {}
            for r in item_results:
                cat_buckets.setdefault(r.category, []).append(1.0 if r.correct else 0.0)
            cat_acc = {k: round(sum(v) / len(v), 4) for k, v in cat_buckets.items() if v}

            t_in = sum(r.tokens_in for r in item_results)
            t_out = sum(r.tokens_out for r in item_results)
            t_cache = sum(r.cache_read_tokens for r in item_results)
            peak_tok = max((r.peak_context_tokens for r in item_results), default=0)
            ing_t_in = sum(r.ingest_tokens_in for r in item_results)
            ing_t_out = sum(r.ingest_tokens_out for r in item_results)
            ing_ms_tot = round(sum(r.ingest_latency_ms for r in item_results), 2)
            ing_usd = self.cost_model.estimate_usd(
                self.model, tokens_in=ing_t_in, tokens_out=ing_t_out, cache_read_tokens=0
            )

            system_results.append(
                SystemExternalResult(
                    system_name=adapter.name,
                    benchmark=benchmark_name,
                    n_questions=len(item_results),
                    accuracy=round(sum(acc_vals) / len(acc_vals), 4) if acc_vals else 0.0,
                    accuracy_ci=bootstrap_ci(acc_vals),
                    mean_f1=round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0.0,
                    f1_ci=bootstrap_ci(f1_vals),
                    mean_bleu1=round(sum(b1_vals) / len(b1_vals), 4) if b1_vals else 0.0,
                    category_accuracy=cat_acc,
                    total_tokens_in=t_in,
                    total_tokens_out=t_out,
                    cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
                    peak_context_tokens=peak_tok,
                    estimated_cost_usd=self.cost_model.estimate_usd(
                        self.model, tokens_in=t_in, tokens_out=t_out, cache_read_tokens=t_cache
                    ),
                    ingest_tokens_in=ing_t_in,
                    ingest_tokens_out=ing_t_out,
                    ingest_cost_usd=ing_usd,
                    ingest_latency_ms=ing_ms_tot,
                    latency_p50_ms=_percentile(lat_vals, 0.50),
                    latency_p95_ms=_percentile(lat_vals, 0.95),
                    item_results=item_results,
                )
            )

        # Compute paired statistical significance (McNemar + Paired Bootstrap F1) vs reference system
        ref_res = next((r for r in system_results if r.system_name == ref_name), None)
        if ref_res is not None and ref_res.item_results:
            ref_correct = [q.correct for q in ref_res.item_results]
            ref_f1 = [q.f1 for q in ref_res.item_results]
            for sys_res in system_results:
                if sys_res.system_name == ref_name or len(sys_res.item_results) != len(ref_correct):
                    continue
                sys_res.mcnemar_vs_ref = mcnemar_test(ref_correct, [q.correct for q in sys_res.item_results])
                sys_res.paired_f1_vs_ref = paired_bootstrap_test(ref_f1, [q.f1 for q in sys_res.item_results])

        return ExternalBenchmarkReport(
            benchmark=benchmark_name,
            model=self.model,
            reference_system=ref_name,
            results=system_results,
        )

    async def rejudge_checkpoint(
        self,
        conversations: Sequence[ExternalConversation],
        checkpoint_path: str | Path,
        *,
        output_checkpoint_path: str | Path | None = None,
        systems: Sequence[str] | None = None,
        categories_only: set[str] | None = None,
        reference_system: str | None = None,
    ) -> ExternalBenchmarkReport:
        """
        Re-score saved predictions in `checkpoint_path` (for example with a cross-family frontier
        judge or the fixed adversarial rubric) strictly over the exact `(conversation_id, question_id)`
        items in `conversations`, without re-running any memory system ingestion or queries!
        """
        cp_in = Path(checkpoint_path)
        loaded = self._load_checkpoint(cp_in)
        benchmark_name = conversations[0].benchmark if conversations else "custom"
        qa_index: dict[tuple[str, str], ExternalQAItem] = {
            (c.conversation_id, qa.question_id): qa for c in conversations for qa in c.qa_items
        }
        ordered_keys = [(c.conversation_id, qa.question_id) for c in conversations for qa in c.qa_items]

        available_systems = sorted({sys_name for (sys_name, _, _) in loaded.keys()})
        target_systems = list(systems) if systems else available_systems
        ref_name = reference_system or (target_systems[0] if target_systems else "")
        cp_out = Path(output_checkpoint_path) if output_checkpoint_path else None
        sem = asyncio.Semaphore(max(1, self.max_concurrency))

        system_results: list[SystemExternalResult] = []
        for sys_name in target_systems:
            present = [
                (cid, qid, loaded[(sys_name, cid, qid)])
                for (cid, qid) in ordered_keys
                if (sys_name, cid, qid) in loaded
            ]
            if not present:
                continue

            async def _rescore(
                cid: str, qid: str, old: ExternalQAResult
            ) -> ExternalQAResult:
                qa = qa_index[(cid, qid)]
                if categories_only is not None and qa.category not in categories_only:
                    return old
                async with sem:
                    ok, f1, b1, rat = await score_external_qa(
                        question=qa.question,
                        ground_truth=qa.ground_truth,
                        distractor_answer=qa.distractor_answer,
                        prediction=old.prediction,
                        category=qa.category,
                        is_abstention=qa.is_abstention,
                        judge_llm=self.judge_llm,
                        judge_model=self.judge_model,
                    )
                return old.model_copy(
                    update={
                        "ground_truth": qa.ground_truth,
                        "correct": ok,
                        "f1": f1,
                        "bleu1": b1,
                        "judge_rationale": rat,
                    }
                )

            item_results = list(
                await asyncio.gather(*(_rescore(cid, qid, old) for cid, qid, old in present))
            )
            if cp_out is not None:
                cp_out.parent.mkdir(parents=True, exist_ok=True)
                with cp_out.open("a", encoding="utf-8") as fh:
                    for qres in item_results:
                        fh.write(
                            json.dumps({"system_name": sys_name, "result": qres.model_dump()}) + "\n"
                        )

            acc_vals = [1.0 if r.correct else 0.0 for r in item_results]
            f1_vals = [r.f1 for r in item_results]
            b1_vals = [r.bleu1 for r in item_results]
            lat_vals = [r.latency_ms for r in item_results]
            cat_buckets: dict[str, list[float]] = {}
            for r in item_results:
                cat_buckets.setdefault(r.category, []).append(1.0 if r.correct else 0.0)
            cat_acc = {k: round(sum(v) / len(v), 4) for k, v in cat_buckets.items() if v}

            t_in = sum(r.tokens_in for r in item_results)
            t_out = sum(r.tokens_out for r in item_results)
            t_cache = sum(r.cache_read_tokens for r in item_results)
            peak_tok = max((r.peak_context_tokens for r in item_results), default=0)
            ing_t_in = sum(r.ingest_tokens_in for r in item_results)
            ing_t_out = sum(r.ingest_tokens_out for r in item_results)
            ing_ms_tot = round(sum(r.ingest_latency_ms for r in item_results), 2)
            ing_usd = self.cost_model.estimate_usd(
                self.model, tokens_in=ing_t_in, tokens_out=ing_t_out, cache_read_tokens=0
            )
            system_results.append(
                SystemExternalResult(
                    system_name=sys_name,
                    benchmark=benchmark_name,
                    n_questions=len(item_results),
                    accuracy=round(sum(acc_vals) / len(acc_vals), 4) if acc_vals else 0.0,
                    accuracy_ci=bootstrap_ci(acc_vals),
                    mean_f1=round(sum(f1_vals) / len(f1_vals), 4) if f1_vals else 0.0,
                    f1_ci=bootstrap_ci(f1_vals),
                    mean_bleu1=round(sum(b1_vals) / len(b1_vals), 4) if b1_vals else 0.0,
                    category_accuracy=cat_acc,
                    total_tokens_in=t_in,
                    total_tokens_out=t_out,
                    cache_hit_ratio=round(t_cache / t_in, 4) if t_in > 0 else 0.0,
                    peak_context_tokens=peak_tok,
                    estimated_cost_usd=self.cost_model.estimate_usd(
                        self.model, tokens_in=t_in, tokens_out=t_out, cache_read_tokens=t_cache
                    ),
                    ingest_tokens_in=ing_t_in,
                    ingest_tokens_out=ing_t_out,
                    ingest_cost_usd=ing_usd,
                    ingest_latency_ms=ing_ms_tot,
                    latency_p50_ms=_percentile(lat_vals, 0.50),
                    latency_p95_ms=_percentile(lat_vals, 0.95),
                    item_results=item_results,
                )
            )

        ref_res = next((r for r in system_results if r.system_name == ref_name), None)
        if ref_res is not None and ref_res.item_results:
            ref_by_id = {(q.conversation_id, q.question_id): q for q in ref_res.item_results}
            for sys_res in system_results:
                if sys_res.system_name == ref_name:
                    continue
                paired = [
                    (ref_by_id[(q.conversation_id, q.question_id)], q)
                    for q in sys_res.item_results
                    if (q.conversation_id, q.question_id) in ref_by_id
                ]
                if paired:
                    sys_res.mcnemar_vs_ref = mcnemar_test(
                        [a.correct for a, _ in paired], [b.correct for _, b in paired]
                    )
                    sys_res.paired_f1_vs_ref = paired_bootstrap_test(
                        [a.f1 for a, _ in paired], [b.f1 for _, b in paired]
                    )

        return ExternalBenchmarkReport(
            benchmark=benchmark_name,
            model=self.model,
            reference_system=ref_name,
            results=system_results,
        )
