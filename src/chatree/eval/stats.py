from __future__ import annotations

import math
import random
import re
import string
from collections import Counter
from typing import Sequence

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Official QA Text Metrics (SQuAD / LoCoMo Token F1, BLEU-1, Exact Match)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer_text(text: str) -> str:
    """Standard SQuAD / LoCoMo normalization: lowercase, strip punctuation/articles/extra whitespace."""
    s = str(text or "").lower().translate(_PUNCT_TABLE)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer_text(prediction) == normalize_answer_text(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Official SQuAD / LoCoMo bag-of-tokens F1 score."""
    pred_toks = normalize_answer_text(prediction).split()
    gold_toks = normalize_answer_text(ground_truth).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    prec = num_same / len(pred_toks)
    rec = num_same / len(gold_toks)
    return round((2.0 * prec * rec) / (prec + rec), 4)


def bleu1(prediction: str, ground_truth: str) -> float:
    """Unigram BLEU-1 with standard brevity penalty (used in LoCoMo official evaluation)."""
    pred_toks = normalize_answer_text(prediction).split()
    gold_toks = normalize_answer_text(ground_truth).split()
    if not pred_toks or not gold_toks:
        return 0.0
    pred_counts = Counter(pred_toks)
    gold_counts = Counter(gold_toks)
    clipped = sum(min(cnt, gold_counts[tok]) for tok, cnt in pred_counts.items())
    precision = clipped / len(pred_toks)
    if precision <= 0.0:
        return 0.0
    bp = 1.0 if len(pred_toks) >= len(gold_toks) else math.exp(1.0 - len(gold_toks) / len(pred_toks))
    return round(bp * precision, 4)


# ---------------------------------------------------------------------------
# Statistical Significance Tests (Bootstrap 95% CI, Paired Bootstrap, McNemar)
# ---------------------------------------------------------------------------


class BootstrapCI(BaseModel):
    mean: float
    ci_low: float
    ci_high: float
    n: int
    n_resamples: int


class PairedBootstrapResult(BaseModel):
    mean_a: float
    mean_b: float
    delta_mean: float  # mean_a - mean_b
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    significant_05: bool


class McNemarResult(BaseModel):
    n: int
    b01: int  # A wrong, B right
    b10: int  # A right, B wrong
    statistic: float
    p_value: float
    method: str  # "exact_binomial" | "chi2_continuity"
    significant_05: bool


class CohensKappaResult(BaseModel):
    n: int
    observed_agreement: float
    expected_agreement: float
    kappa: float
    disagreements: int
    a_only_positive: int
    b_only_positive: int
    interpretation: str


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile helper shared across evaluation runners."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 2)


def bootstrap_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Non-parametric percentile bootstrap confidence interval for a sample mean."""
    n = len(values)
    if n == 0:
        return BootstrapCI(mean=0.0, ci_low=0.0, ci_high=0.0, n=0, n_resamples=n_resamples)
    vals = [float(v) for v in values]
    mean_val = sum(vals) / n
    if n == 1:
        return BootstrapCI(
            mean=round(mean_val, 4),
            ci_low=round(mean_val, 4),
            ci_high=round(mean_val, 4),
            n=1,
            n_resamples=n_resamples,
        )
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        sample_sum = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(sample_sum / n)
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1)
    return BootstrapCI(
        mean=round(mean_val, 4),
        ci_low=round(means[lo_idx], 4),
        ci_high=round(means[hi_idx], 4),
        n=n,
        n_resamples=n_resamples,
    )


def paired_bootstrap_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> PairedBootstrapResult:
    """Two-sided paired bootstrap significance test comparing system A vs. system B on identical items."""
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Paired lengths must match ({len(scores_a)} vs {len(scores_b)}).")
    n = len(scores_a)
    if n == 0:
        return PairedBootstrapResult(
            mean_a=0.0, mean_b=0.0, delta_mean=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0, n=0, significant_05=False
        )
    a = [float(x) for x in scores_a]
    b = [float(x) for x in scores_b]
    diffs = [x - y for x, y in zip(a, b)]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    obs_delta = sum(diffs) / n

    rng = random.Random(seed)
    boot_deltas: list[float] = []
    # Also compute centered null distribution for two-sided p-value
    centered = [d - obs_delta for d in diffs]
    extreme = 0
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        bd = sum(diffs[i] for i in idxs) / n
        boot_deltas.append(bd)
        null_d = sum(centered[i] for i in idxs) / n
        if abs(null_d) >= abs(obs_delta) - 1e-12:
            extreme += 1

    boot_deltas.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha / 2.0) * n_resamples)) - 1)
    p_val = min(1.0, (extreme + 1.0) / (n_resamples + 1.0)) if abs(obs_delta) > 1e-12 else 1.0
    return PairedBootstrapResult(
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        delta_mean=round(obs_delta, 4),
        ci_low=round(boot_deltas[lo_idx], 4),
        ci_high=round(boot_deltas[hi_idx], 4),
        p_value=round(p_val, 5),
        n=n,
        significant_05=p_val < 0.05,
    )


def mcnemar_test(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """
    McNemar's paired binary significance test (exact two-sided binomial when discordant count < 25;
    Edwards continuity-corrected chi-squared df=1 otherwise).
    """
    if len(a_correct) != len(b_correct):
        raise ValueError(f"Paired lengths must match ({len(a_correct)} vs {len(b_correct)}).")
    n = len(a_correct)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    disc = b10 + b01
    if disc == 0:
        return McNemarResult(
            n=n, b01=0, b10=0, statistic=0.0, p_value=1.0, method="exact_binomial", significant_05=False
        )

    if disc < 25:
        k = min(b10, b01)
        tail = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2.0**disc)
        p_val = min(1.0, 2.0 * tail)
        return McNemarResult(
            n=n,
            b01=b01,
            b10=b10,
            statistic=float(k),
            p_value=round(p_val, 5),
            method="exact_binomial",
            significant_05=p_val < 0.05,
        )

    chi2 = ((abs(b10 - b01) - 1.0) ** 2) / disc
    # Survival function for chi-squared with df=1 is erfc(sqrt(chi2 / 2))
    p_val = math.erfc(math.sqrt(max(0.0, chi2) / 2.0))
    return McNemarResult(
        n=n,
        b01=b01,
        b10=b10,
        statistic=round(chi2, 4),
        p_value=round(p_val, 5),
        method="chi2_continuity",
        significant_05=p_val < 0.05,
    )


def cohens_kappa(rater_a: Sequence[bool], rater_b: Sequence[bool]) -> CohensKappaResult:
    """
    Compute exact binary Cohen's kappa inter-rater agreement between two judges over identical items.
    Uses Landis & Koch (1977) qualitative interpretation bands.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Paired lengths must match ({len(rater_a)} vs {len(rater_b)}).")
    n = len(rater_a)
    if n == 0:
        return CohensKappaResult(
            n=0,
            observed_agreement=0.0,
            expected_agreement=0.0,
            kappa=0.0,
            disagreements=0,
            a_only_positive=0,
            b_only_positive=0,
            interpretation="empty",
        )
    n11 = sum(1 for a, b in zip(rater_a, rater_b) if a and b)
    n00 = sum(1 for a, b in zip(rater_a, rater_b) if (not a) and (not b))
    n10 = sum(1 for a, b in zip(rater_a, rater_b) if a and (not b))
    n01 = sum(1 for a, b in zip(rater_a, rater_b) if (not a) and b)

    po = (n11 + n00) / n
    pa1 = (n11 + n10) / n
    pb1 = (n11 + n01) / n
    pe = pa1 * pb1 + (1.0 - pa1) * (1.0 - pb1)

    if abs(1.0 - pe) <= 1e-12:
        kappa = 1.0 if po >= 1.0 - 1e-12 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    if kappa >= 0.81:
        band = "almost_perfect"
    elif kappa >= 0.61:
        band = "substantial"
    elif kappa >= 0.41:
        band = "moderate"
    elif kappa >= 0.21:
        band = "fair"
    elif kappa >= 0.0:
        band = "slight"
    else:
        band = "poor"

    return CohensKappaResult(
        n=n,
        observed_agreement=round(po, 4),
        expected_agreement=round(pe, 4),
        kappa=round(kappa, 4),
        disagreements=n10 + n01,
        a_only_positive=n10,
        b_only_positive=n01,
        interpretation=band,
    )


# ---------------------------------------------------------------------------
# Camera-Ready LaTeX Table Formatter (`booktabs`)
# ---------------------------------------------------------------------------


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("$", "\\$")
        .replace("#", "\\#")
    )


def format_latex_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    caption: str = "Chatree Evaluation Results",
    label: str = "tab:chatree-eval",
    col_align: str | None = None,
) -> str:
    """Emit a camera-ready LaTeX `booktabs` table for direct inclusion in papers."""
    if not headers:
        return ""
    align = col_align or ("l" + "r" * (len(headers) - 1))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape(h)}}}" for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(cell) for cell in row) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"
