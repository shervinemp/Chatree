from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from chatree.protocols import LLMProvider


class ModelPricing(BaseModel):
    """Per-1M token USD pricing with explicit KV-cache read discount."""

    input_per_1m: float = 2.50
    cached_input_per_1m: float = 0.25
    output_per_1m: float = 10.00


DEFAULT_PRICING_TABLE: dict[str, ModelPricing] = {
    # Reference indicative prices (USD per 1M tokens) across major hosted & local providers
    "mock/": ModelPricing(input_per_1m=2.50, cached_input_per_1m=0.25, output_per_1m=10.00),
    "gemini/gemini-2.5-flash": ModelPricing(input_per_1m=0.15, cached_input_per_1m=0.0375, output_per_1m=0.60),
    "gemini/gemini-2.5-pro": ModelPricing(input_per_1m=1.25, cached_input_per_1m=0.31, output_per_1m=10.00),
    "anthropic/claude-opus": ModelPricing(input_per_1m=15.00, cached_input_per_1m=1.50, output_per_1m=75.00),
    "anthropic/claude-sonnet": ModelPricing(input_per_1m=3.00, cached_input_per_1m=0.30, output_per_1m=15.00),
    "anthropic/claude-haiku": ModelPricing(input_per_1m=0.80, cached_input_per_1m=0.08, output_per_1m=4.00),
    "openai/gpt-5": ModelPricing(input_per_1m=1.25, cached_input_per_1m=0.125, output_per_1m=10.00),
    "openai/gpt-4.1-mini": ModelPricing(input_per_1m=0.40, cached_input_per_1m=0.10, output_per_1m=1.60),
    "openai/gpt-4.1": ModelPricing(input_per_1m=2.00, cached_input_per_1m=0.50, output_per_1m=8.00),
    "openai/gpt-4o-mini": ModelPricing(input_per_1m=0.15, cached_input_per_1m=0.075, output_per_1m=0.60),
    "openai/gpt-4o": ModelPricing(input_per_1m=2.50, cached_input_per_1m=1.25, output_per_1m=10.00),
    "openai/o3": ModelPricing(input_per_1m=2.00, cached_input_per_1m=0.50, output_per_1m=8.00),
    "deepseek/deepseek-chat": ModelPricing(input_per_1m=0.28, cached_input_per_1m=0.028, output_per_1m=0.42),
    "deepseek/deepseek-reasoner": ModelPricing(input_per_1m=0.55, cached_input_per_1m=0.14, output_per_1m=2.19),
    "groq/": ModelPricing(input_per_1m=0.59, cached_input_per_1m=0.59, output_per_1m=0.79),
    "mistral/": ModelPricing(input_per_1m=2.00, cached_input_per_1m=2.00, output_per_1m=6.00),
    "together_ai/": ModelPricing(input_per_1m=0.88, cached_input_per_1m=0.88, output_per_1m=0.88),
    "xai/": ModelPricing(input_per_1m=3.00, cached_input_per_1m=0.75, output_per_1m=15.00),
    "ollama/": ModelPricing(input_per_1m=0.0, cached_input_per_1m=0.0, output_per_1m=0.0),
    "vllm/": ModelPricing(input_per_1m=0.0, cached_input_per_1m=0.0, output_per_1m=0.0),
    "sglang/": ModelPricing(input_per_1m=0.0, cached_input_per_1m=0.0, output_per_1m=0.0),
    "lm_studio/": ModelPricing(input_per_1m=0.0, cached_input_per_1m=0.0, output_per_1m=0.0),
    "llamacpp/": ModelPricing(input_per_1m=0.0, cached_input_per_1m=0.0, output_per_1m=0.0),
}


class CostModel:
    """Computes effective USD cost accounting for uncached input, cached KV reads, and output tokens."""

    def __init__(self, pricing_table: dict[str, ModelPricing] | None = None) -> None:
        self.pricing_table = pricing_table or dict(DEFAULT_PRICING_TABLE)

    def resolve_pricing(self, model: str) -> ModelPricing:
        low = model.lower()
        for prefix, pricing in self.pricing_table.items():
            if low.startswith(prefix.lower()) or prefix.lower() in low:
                return pricing
        return ModelPricing()

    def estimate_usd(
        self,
        model: str,
        *,
        tokens_in: int,
        tokens_out: int,
        cache_read_tokens: int = 0,
    ) -> float:
        p = self.resolve_pricing(model)
        cached = min(max(0, cache_read_tokens), max(0, tokens_in))
        uncached_in = max(0, tokens_in - cached)
        cost = (
            (uncached_in / 1_000_000.0) * p.input_per_1m
            + (cached / 1_000_000.0) * p.cached_input_per_1m
            + (max(0, tokens_out) / 1_000_000.0) * p.output_per_1m
        )
        return round(cost, 6)


class QueryEvalResult(BaseModel):
    query: str
    target_branch: str
    passed: bool  # strict_pass: recall_pass AND no rejected leakage AND no stale contradiction
    recall_pass: bool = False
    recall_score: float = 0.0
    rejected_leakage: bool = False
    stale_contradiction: bool = False
    matched_keywords: list[str] = Field(default_factory=list)
    expected_keywords: list[str] = Field(default_factory=list)
    leaked_keywords: list[str] = Field(default_factory=list)
    stale_keywords_found: list[str] = Field(default_factory=list)
    judge_score: float | None = None
    judge_rationale: str | None = None
    latency_ms: float = 0.0
    answer_preview: str = ""
    full_answer: str = ""


_EXCLUSION_OR_RETIREMENT_RE = re.compile(
    r"\b("
    r"retired|superseded|replaced|abandoned|scrapped|excluded|deprecated|rejected|"
    r"obsolete|archived|dropped|removed|formerly|previously|not\s+active|no\s+longer|"
    r"not\s+part\s+of|not\s+configured|not\s+in\s+use|"
    r"must\s+not|do\s+not\s+use|not\s+used|not\s+merged|not\s+adopted|never\s+adopted|"
    r"rather\s+than|instead\s+of|superceded|old\s+value|old\s+ttl|prior\s+value"
    r")\b",
    re.IGNORECASE,
)

_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_HEADING_OR_LABEL_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|\*\*|__|(?:excluded|retired|rejected|scrapped|abandoned|deprecated|superseded|obsolete|not\s+adopted|dropped|prior|old)\b)",
    re.IGNORECASE,
)


def _line_is_under_exclusion_heading(lines: list[str], line_idx: int) -> bool:
    """
    Walk upward from `line_idx` across bullet/numbered list items, table rows, or indented
    continuations to detect an enclosing markdown heading or list introduction that marks
    the entire block as excluded/retired/rejected (e.g., `### Excluded / Retired:` followed
    by 5 bullets, or `The following options were retired:\n- X\n- Y`).
    """
    cur_line = lines[line_idx]
    cur_stripped = cur_line.strip()
    in_list_or_table = bool(
        _BULLET_LINE_RE.match(cur_line)
        or (cur_stripped.startswith("|") and "|" in cur_stripped[1:])
        or (cur_line.startswith(("  ", "\t")) and cur_stripped)
    )
    blank_gap = 0
    for prev_idx in range(line_idx - 1, max(-1, line_idx - 25), -1):
        raw = lines[prev_idx]
        s = raw.strip()
        if not s:
            blank_gap += 1
            if blank_gap >= 2:
                break
            continue
        # Markdown table separator row (`| --- | --- |`) — keep walking up to header/heading
        if s.startswith("|") and set(s.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        # Heading line (`# ...` or `**...**` ending a section title)
        if s.startswith("#") or (
            s.startswith("**") and (s.endswith("**") or s.endswith("**:") or ":**" in s)
        ):
            return bool(_EXCLUSION_OR_RETIREMENT_RE.search(s))
        # List introducer right above a list (e.g., "Retired / excluded proposals:" or "We rejected:")
        if in_list_or_table and (s.endswith(":") or _HEADING_OR_LABEL_RE.match(raw)):
            if not _BULLET_LINE_RE.match(raw) and not s.startswith("|"):
                return bool(_EXCLUSION_OR_RETIREMENT_RE.search(s))
        # Continuing through sibling bullets or table rows belonging to the same list/table
        if _BULLET_LINE_RE.match(raw) or (s.startswith("|") and "|" in s[1:]):
            blank_gap = 0
            continue
        # Non-heading prose line: if directly above with no blank line and it's a heading/label
        if blank_gap == 0 and _HEADING_OR_LABEL_RE.match(raw) and _EXCLUSION_OR_RETIREMENT_RE.search(s):
            return True
        # Reached ordinary non-list prose — stop walking unless still inside an explicit section
        if not in_list_or_table:
            break
    return False


def is_token_asserted_as_active(answer_text: str, keyword: str) -> bool:
    """
    Return True iff `keyword` appears in `answer_text` as an active claim (i.e. NOT solely
    inside an explicit retirement/exclusion/negation clause, table row marked retired/excluded,
    or multi-item bullet list under an exclusion heading such as `### Excluded / Retired:`).
    """
    if not keyword:
        return False
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = list(pattern.finditer(answer_text))
    if not matches:
        return False

    lines = answer_text.splitlines()
    # Map character offset -> 0-based line index
    line_starts: list[int] = []
    pos = 0
    for ln in answer_text.splitlines(keepends=True):
        line_starts.append(pos)
        pos += len(ln)

    for m in matches:
        start, end = m.start(), m.end()
        line_start = answer_text.rfind("\n", 0, start) + 1
        line_end = answer_text.find("\n", end)
        if line_end == -1:
            line_end = len(answer_text)
        line_text = answer_text[line_start:line_end]

        # 1. Markdown table row: check the entire row (status column often marks Retired/Excluded)
        stripped_line = line_text.strip()
        if stripped_line.startswith("|") and stripped_line.count("|") >= 2:
            if _EXCLUSION_OR_RETIREMENT_RE.search(stripped_line):
                continue

        # 2. Local sentence/clause on the matched line
        rel_start = start - line_start
        rel_end = end - line_start
        sent_lo = max(
            line_text.rfind(". ", 0, rel_start) + 2 if line_text.rfind(". ", 0, rel_start) != -1 else 0,
            line_text.rfind("; ", 0, rel_start) + 2 if line_text.rfind("; ", 0, rel_start) != -1 else 0,
        )
        sent_hi_candidates = [
            idx
            for idx in (line_text.find(". ", rel_end), line_text.find("; ", rel_end))
            if idx != -1
        ]
        sent_hi = min(sent_hi_candidates) if sent_hi_candidates else len(line_text)
        clause = line_text[sent_lo:sent_hi]

        if _EXCLUSION_OR_RETIREMENT_RE.search(clause):
            continue

        # 3. Enclosing exclusion/retirement heading or list lead-in (works for bullet 1..N and tables)
        line_idx = 0
        for i, ls in enumerate(line_starts):
            if ls <= start:
                line_idx = i
            else:
                break
        if lines and _line_is_under_exclusion_heading(lines, line_idx):
            continue

        # Found at least one active, non-negated assertion of this keyword
        return True
    return False


class ExactFactScorer:
    """
    Deterministic scorer separating three distinct error modes:
      1) Missing recall (`recall_pass=False`)
      2) Rejected-proposal leakage (`rejected_leakage=True`)
      3) Stale-decision contradiction (`stale_contradiction=True`)
    `passed` (strict_pass) requires full recall with zero active leakage and zero active
    stale contradiction. When `negation_aware=True` (default), explicitly retired/excluded
    mentions (e.g. 'JWT_TTL_15M is retired') are not misclassified as active leakage.
    """

    def score(
        self,
        *,
        query: str,
        target_branch: str,
        answer_text: str,
        expected_keywords: list[str],
        rejected_keywords: list[str] | None = None,
        stale_keywords: list[str] | None = None,
        latency_ms: float = 0.0,
        negation_aware: bool = True,
    ) -> QueryEvalResult:
        ans_low = answer_text.lower()
        matched_exp = [kw for kw in expected_keywords if kw.lower() in ans_low]
        if negation_aware:
            leaked = [
                kw for kw in (rejected_keywords or []) if is_token_asserted_as_active(answer_text, kw)
            ]
            stale_found = [
                kw for kw in (stale_keywords or []) if is_token_asserted_as_active(answer_text, kw)
            ]
        else:
            leaked = [kw for kw in (rejected_keywords or []) if kw.lower() in ans_low]
            stale_found = [kw for kw in (stale_keywords or []) if kw.lower() in ans_low]

        recall_score = (
            round(len(matched_exp) / len(expected_keywords), 4) if expected_keywords else 1.0
        )
        recall_pass = len(matched_exp) == len(expected_keywords)
        rejected_leakage = len(leaked) > 0
        stale_contradiction = len(stale_found) > 0
        strict_pass = recall_pass and not rejected_leakage and not stale_contradiction

        return QueryEvalResult(
            query=query,
            target_branch=target_branch,
            passed=strict_pass,
            recall_pass=recall_pass,
            recall_score=recall_score,
            rejected_leakage=rejected_leakage,
            stale_contradiction=stale_contradiction,
            matched_keywords=matched_exp,
            expected_keywords=list(expected_keywords),
            leaked_keywords=leaked,
            stale_keywords_found=stale_found,
            latency_ms=round(latency_ms, 2),
            answer_preview=answer_text[:240],
            full_answer=answer_text,
        )


class LLMJudgeScorer:
    """Optional rubric-based LLM-as-a-Judge evaluator compatible with any LLMProvider."""

    def __init__(self, llm: LLMProvider, judge_model: str) -> None:
        self.llm = llm
        self.judge_model = judge_model

    async def judge(
        self,
        *,
        query: str,
        answer_text: str,
        expected_keywords: list[str],
        forbidden_keywords: list[str],
    ) -> tuple[float, str]:
        prompt = (
            "[Chatree Eval Judge]\n"
            "Score the candidate answer on [0.0, 1.0] for factual accuracy and absence of rejected/stale facts.\n"
            "Return strict JSON: {\"score\": float, \"factual_match\": bool, \"rejected_leakage\": bool, \"rationale\": str}.\n"
            f"QUERY: {query}\n"
            f"EXPECTED_FACTS: {', '.join(expected_keywords)}\n"
            f"FORBIDDEN_FACTS: {', '.join(forbidden_keywords)}\n"
            f"CANDIDATE_ANSWER:\n{answer_text[:1500]}"
        )
        resp = await self.llm.complete(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            tools=None,
            temperature=0.0,
        )
        raw = resp.content.strip()
        # Extract JSON block even if wrapped in markdown fences
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed: dict[str, Any] = {}
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = {}
        score = float(parsed.get("score", 0.0))
        rationale = str(parsed.get("rationale") or raw[:200])
        return max(0.0, min(1.0, round(score, 4))), rationale
