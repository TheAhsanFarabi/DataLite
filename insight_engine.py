"""
DataLite — Insight Engine (Phase 1)
====================================

The deterministic core of DataLite's auto-insight feature.

Design principle: the engine NEVER uses an LLM and NEVER runs model-generated
code. Every finding is computed directly from the full dataframe with pandas /
numpy, carries an interestingness score, and ships with the evidence that backs
it. The (future) LLM layer only ever *narrates* these structured findings — it
cannot fabricate numbers because it never sees the raw data.

A "detector" scans the dataframe and returns a list of `Finding`s. `analyze()`
runs every detector, deduplicates, ranks by `score * weight`, and returns the
most interesting findings first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Per-detector priority weights. Higher = surfaced earlier when scores tie.
# These encode editorial judgement: "X group scores higher than Y" is more
# interesting to a human than "this column has one missing value."
# --------------------------------------------------------------------------- #
WEIGHTS = {
    "segment_difference": 1.10,
    "correlation": 1.00,
    "missingness_pattern": 0.95,
    "duplicates": 0.65,
    "imbalance": 0.70,
    "outliers": 0.60,
    "missingness": 0.50,
    "hygiene": 0.50,
}


@dataclass
class Finding:
    """One structured, evidence-backed observation about the data."""

    kind: str                       # detector type, keys into WEIGHTS
    headline: str                   # one plain-English sentence
    score: float                    # normalized effect size in [0, 1]
    detail: str = ""                # optional secondary sentence
    chart: Optional[dict] = None    # spec the app uses to draw a mini-chart
    evidence: dict = field(default_factory=dict)  # raw numbers for "Verify"

    @property
    def weight(self) -> float:
        return WEIGHTS.get(self.kind, 0.5)

    @property
    def rank_score(self) -> float:
        return float(self.score) * self.weight


# --------------------------------------------------------------------------- #
# Column-type helpers
# --------------------------------------------------------------------------- #
def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = df.select_dtypes(include=[np.number]).columns
    # drop boolean columns that numpy counts as numeric
    return [c for c in cols if not pd.api.types.is_bool_dtype(df[c])]


def _is_categorical_like(s: pd.Series) -> bool:
    """True for string/object/category/bool columns; False for numeric & datetime.

    Robust across pandas 2.x (strings are ``object``) and pandas 3.x (strings
    get a dedicated ``str`` dtype, so ``dtype == object`` no longer works).
    """
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s):
        return False
    if (pd.api.types.is_datetime64_any_dtype(s)
            or pd.api.types.is_timedelta64_dtype(s)):
        return False
    return True


def _categorical_cols(df: pd.DataFrame, max_card: int = 12) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if _is_categorical_like(s):
            nun = s.nunique(dropna=True)
            if 2 <= nun <= max_card:
                out.append(c)
    return out


def _cohen_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #
def detect_correlations(df: pd.DataFrame, threshold: float = 0.30,
                        top_k: int = 4) -> list[Finding]:
    num = _numeric_cols(df)
    if len(num) < 2:
        return []
    corr = df[num].corr(numeric_only=True)
    seen, cands = set(), []
    for i, a in enumerate(num):
        for b in num[i + 1:]:
            r = corr.loc[a, b]
            if pd.isna(r) or abs(r) < threshold:
                continue
            cands.append((abs(r), a, b, float(r)))
    cands.sort(reverse=True)
    findings = []
    for absr, a, b, r in cands[:top_k]:
        direction = "move together" if r > 0 else "move in opposite directions"
        strength = ("a strong" if absr >= 0.7 else
                    "a moderate" if absr >= 0.5 else "a noticeable")
        findings.append(Finding(
            kind="correlation",
            headline=f"{a} and {b} {direction} (r = {r:+.2f}).",
            detail=f"There's {strength} linear relationship between the two.",
            score=min(absr, 1.0),
            chart={"type": "scatter", "x": a, "y": b},
            evidence={"pearson_r": round(r, 3), "x": a, "y": b,
                      "n": int(df[[a, b]].dropna().shape[0])},
        ))
        seen.add(frozenset((a, b)))
    return findings


def detect_segment_differences(df: pd.DataFrame, min_d: float = 0.5,
                               top_k: int = 4) -> list[Finding]:
    num = _numeric_cols(df)
    cats = _categorical_cols(df, max_card=8)
    cands = []
    for c in cats:
        for n in num:
            grp = df[[c, n]].dropna().groupby(c)[n]
            means = grp.mean()
            sizes = grp.size()
            # only groups with at least a few observations
            valid = means[sizes >= 3]
            if len(valid) < 2:
                continue
            hi, lo = valid.idxmax(), valid.idxmin()
            d = _cohen_d(df.loc[df[c] == hi, n], df.loc[df[c] == lo, n])
            if abs(d) < min_d:
                continue
            cands.append((abs(d), c, n, hi, lo, float(means[hi]),
                          float(means[lo]), means))
    cands.sort(key=lambda t: t[0], reverse=True)
    findings = []
    for d, c, n, hi, lo, hi_mean, lo_mean, means in cands[:top_k]:
        findings.append(Finding(
            kind="segment_difference",
            headline=(f"'{hi}' has higher {n} than '{lo}' "
                      f"(avg {hi_mean:.1f} vs {lo_mean:.1f})."),
            detail=f"Grouping by {c} splits {n} into clearly different levels "
                   f"(effect size d = {d:.2f}).",
            score=min(d / 2.0, 1.0),
            chart={"type": "group_bar", "group": c, "value": n,
                   "means": {str(k): float(v) for k, v in means.items()}},
            evidence={"group_col": c, "value_col": n, "cohens_d": round(d, 2),
                      "group_means": {str(k): round(float(v), 2)
                                      for k, v in means.items()}},
        ))
    return findings


def detect_imbalance(df: pd.DataFrame, threshold: float = 0.70,
                     top_k: int = 3) -> list[Finding]:
    cats = _categorical_cols(df, max_card=12)
    cands = []
    for c in cats:
        vc = df[c].value_counts(normalize=True, dropna=True)
        if vc.empty:
            continue
        top_share = float(vc.iloc[0])
        k = len(vc)
        if top_share < threshold or k < 2:
            continue
        # normalize: 1/k (perfectly even) -> 0, 1.0 (single class) -> 1
        norm = (top_share - 1 / k) / (1 - 1 / k)
        cands.append((norm, c, vc.index[0], top_share, vc))
    cands.sort(reverse=True, key=lambda t: t[0])
    findings = []
    for norm, c, top, share, vc in cands[:top_k]:
        findings.append(Finding(
            kind="imbalance",
            headline=f"{c} is dominated by '{top}' ({share*100:.0f}% of rows).",
            detail="Heavy class imbalance — worth knowing before any modelling.",
            score=min(norm, 1.0),
            chart={"type": "value_counts", "col": c,
                   "counts": {str(k): int(v) for k, v in
                              df[c].value_counts(dropna=True).items()}},
            evidence={"column": c, "top_value": str(top),
                      "top_share": round(share, 3)},
        ))
    return findings


def detect_outliers(df: pd.DataFrame, min_frac: float = 0.02,
                    top_k: int = 3) -> list[Finding]:
    num = _numeric_cols(df)
    cands = []
    for n in num:
        s = df[n].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        frac = float(mask.mean())
        if frac < min_frac:
            continue
        cands.append((frac, n, int(mask.sum()), float(lo), float(hi)))
    cands.sort(reverse=True, key=lambda t: t[0])
    findings = []
    for frac, n, count, lo, hi in cands[:top_k]:
        findings.append(Finding(
            kind="outliers",
            headline=f"{n} has {count} outlier value(s) "
                     f"({frac*100:.1f}% of rows) outside the typical range.",
            detail="These sit beyond 1.5×IQR — check for errors or rare events.",
            score=min(frac / 0.10, 1.0),
            chart={"type": "box", "col": n},
            evidence={"column": n, "n_outliers": count,
                      "lower_bound": round(lo, 2), "upper_bound": round(hi, 2)},
        ))
    return findings


def detect_missingness(df: pd.DataFrame, pattern_spread: float = 0.30,
                       plain_frac: float = 0.05) -> list[Finding]:
    findings = []
    cats = _categorical_cols(df, max_card=8)
    miss = df.isna().mean()
    cols_with_missing = [c for c in df.columns if miss[c] > 0]

    # (a) the interesting case: missingness depends on another column
    for m in cols_with_missing:
        ind = df[m].isna()
        best = None
        for c in cats:
            if c == m:
                continue
            rates = ind.groupby(df[c]).mean()
            sizes = df[c].value_counts()
            rates = rates[sizes[rates.index] >= 3]
            if len(rates) < 2:
                continue
            spread = float(rates.max() - rates.min())
            if spread >= pattern_spread and (best is None or spread > best[0]):
                best = (spread, c, rates.idxmax(), float(rates.max()),
                        rates.idxmin(), float(rates.min()))
        if best:
            spread, c, hi_g, hi_r, lo_g, lo_r = best
            findings.append(Finding(
                kind="missingness_pattern",
                headline=(f"{m} is missing far more often when {c} = '{hi_g}' "
                          f"({hi_r*100:.0f}% vs {lo_r*100:.0f}% for '{lo_g}')."),
                detail="Missingness isn't random here — it tracks another "
                       "column, which can bias analysis if ignored.",
                score=min(spread, 1.0),
                chart={"type": "missing_by_group", "missing_col": m,
                       "group_col": c,
                       "rates": {str(k): float(v) for k, v in
                                 ind.groupby(df[c]).mean().items()}},
                evidence={"missing_col": m, "group_col": c,
                          "max_rate_group": str(hi_g), "max_rate": round(hi_r, 3),
                          "min_rate_group": str(lo_g), "min_rate": round(lo_r, 3)},
            ))

    # (b) plain missingness (lower priority)
    pattern_cols = {f.evidence.get("missing_col") for f in findings}
    for c in cols_with_missing:
        if c in pattern_cols or miss[c] < plain_frac:
            continue
        findings.append(Finding(
            kind="missingness",
            headline=f"{c} is missing in {miss[c]*100:.0f}% of rows.",
            score=min(float(miss[c]), 1.0),
            chart={"type": "metric", "label": f"{c} missing",
                   "value": f"{miss[c]*100:.0f}%"},
            evidence={"column": c, "missing_fraction": round(float(miss[c]), 3)},
        ))
    return findings


def detect_hygiene(df: pd.DataFrame) -> list[Finding]:
    findings = []
    n_rows = len(df)
    if n_rows == 0:
        return findings

    # duplicate rows
    dups = int(df.duplicated().sum())
    if dups > 0:
        frac = dups / n_rows
        findings.append(Finding(
            kind="duplicates",
            headline=f"{dups} row(s) ({frac*100:.0f}%) are exact duplicates.",
            detail="Duplicate rows can quietly skew counts and averages.",
            score=min(frac / 0.10, 1.0),
            chart={"type": "metric", "label": "Duplicate rows", "value": str(dups)},
            evidence={"duplicate_rows": dups, "fraction": round(frac, 3)},
        ))

    # constant & id-like columns
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            findings.append(Finding(
                kind="hygiene",
                headline=f"{c} has the same value in every row — it adds no information.",
                score=0.45,
                chart={"type": "metric", "label": c, "value": "constant"},
                evidence={"column": c, "unique_values": int(nun)},
            ))
        elif (nun == n_rows and n_rows >= 25
              and not pd.api.types.is_float_dtype(df[c])):
            findings.append(Finding(
                kind="hygiene",
                headline=f"{c} looks like an identifier (every value is unique).",
                detail="Likely an ID column — usually excluded from analysis.",
                score=0.40,
                chart={"type": "metric", "label": c, "value": "unique id"},
                evidence={"column": c, "unique_values": int(nun)},
            ))
    return findings


DETECTORS = [
    detect_correlations,
    detect_segment_differences,
    detect_imbalance,
    detect_outliers,
    detect_missingness,
    detect_hygiene,
]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def dataset_summary(df: pd.DataFrame) -> dict:
    """Cheap, factual one-liner stats for the hero section (no LLM guessing)."""
    num = _numeric_cols(df)
    cats = _categorical_cols(df, max_card=10**9)
    miss = float(df.isna().mean().mean()) if df.size else 0.0
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric": len(num),
        "categorical": len([c for c in df.columns if c not in num]),
        "missing_pct": round(miss * 100, 1),
        "duplicates": int(df.duplicated().sum()),
        "sentence": (
            f"{df.shape[0]:,} rows × {df.shape[1]} columns — "
            f"{len(num)} numeric, {df.shape[1] - len(num)} non-numeric, "
            f"{miss*100:.0f}% missing overall."
        ),
    }


def analyze(df: pd.DataFrame, limit: int = 8) -> list[Finding]:
    """Run all detectors, dedupe, and return the most interesting findings."""
    if df is None or df.empty:
        return []

    findings: list[Finding] = []
    for det in DETECTORS:
        try:
            findings.extend(det(df))
        except Exception:
            # one misbehaving detector must never take down the whole report
            continue

    # dedupe identical headlines, keeping the highest-scoring instance
    best: dict[str, Finding] = {}
    for f in findings:
        if f.headline not in best or f.rank_score > best[f.headline].rank_score:
            best[f.headline] = f

    ranked = sorted(best.values(), key=lambda f: f.rank_score, reverse=True)
    return ranked[:limit]
