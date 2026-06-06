"""
DataLite — presentation layer ("data newspaper" theme)
======================================================

Pure rendering: turns Finding objects into self-contained editorial HTML cards
with embedded (base64) charts. Kept separate from app.py so it can be reused by
the static preview generator and unit-tested without a Streamlit runtime.

Aesthetic: newsprint cream, ink type, a single vermillion accent, Fraunces
(serif display) + Spline Sans (body) + DM Mono (data). Each insight is a
"story": a section kicker, a serif headline, a styled chart, a signal-strength
bar, and a collapsible "verify the numbers" panel.
"""

from __future__ import annotations

import base64
import html
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- palette ---------------------------------------------------------------- #
PAPER = "#FAF7F0"
CARD = "#FFFDF8"
INK = "#211C16"
MUTE = "#6E6557"
LINE = "#E4DCCD"
ACCENT = "#C23B22"  # vermillion masthead accent

# muted, editorial per-section accents (no bright SaaS hues)
KIND_META = {
    "segment_difference": ("Group difference", "#2C5478"),
    "correlation":        ("Correlation",      "#176B66"),
    "missingness_pattern":("Missing pattern",  "#A8772A"),
    "missingness":        ("Missing data",     "#A8772A"),
    "imbalance":          ("Imbalance",        "#6E3A6B"),
    "outliers":           ("Outliers",         "#C23B22"),
    "duplicates":         ("Duplicates",       "#9B4B5A"),
    "hygiene":            ("Data hygiene",     "#5A5751"),
}

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700"
         "&family=Spline+Sans:wght@400;500;600;700"
         "&family=DM+Mono:wght@400;500&display=swap")

CSS = f"""
<style>
@import url('{FONTS}');

:root {{
  --paper:{PAPER}; --card:{CARD}; --ink:{INK}; --mute:{MUTE};
  --line:{LINE}; --accent:{ACCENT};
}}

/* strip default streamlit chrome for a clean canvas */
#MainMenu, header[data-testid="stHeader"], footer {{ display:none !important; }}
[data-testid="stDecoration"] {{ display:none !important; }}

html, body, .stApp, [class*="st-"], button, input, textarea, select {{
  font-family:'Spline Sans', system-ui, sans-serif;
}}
.stApp {{
  background:
    radial-gradient(120% 80% at 50% -10%, #FFFDF7 0%, var(--paper) 55%) fixed;
  color:var(--ink);
}}
[data-testid="stMainBlockContainer"], .block-container {{
  max-width:1120px; padding-top:2.2rem; padding-bottom:4rem;
}}

/* ---------- masthead ---------- */
.masthead {{ text-align:center; margin:0 0 6px; }}
.masthead .edition {{
  font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.32em; font-size:10.5px; color:var(--accent); font-weight:500;
}}
.masthead .wordmark {{
  font-family:'Fraunces', serif; font-weight:600; font-optical-sizing:auto;
  font-size:clamp(40px, 6vw, 68px); line-height:.95; letter-spacing:-.02em;
  margin:6px 0 4px; color:var(--ink);
}}
.masthead .wordmark .spark {{ color:var(--accent); font-weight:500; }}
.masthead .dateline {{
  font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.16em; font-size:11px; color:var(--mute);
}}
.rule {{ height:3px; background:var(--ink); margin:14px 0 4px; }}
.rule.thin {{ height:1px; background:var(--line); margin:4px 0 22px; }}

/* ---------- summary chips ---------- */
.chips {{
  display:flex; flex-wrap:wrap; justify-content:center; gap:0;
  margin:0 0 26px; border-top:1px solid var(--line);
  border-bottom:1px solid var(--line); padding:10px 0;
}}
.chips .chip {{ padding:0 22px; text-align:center; border-right:1px solid var(--line); }}
.chips .chip:last-child {{ border-right:none; }}
.chips .v {{ font-family:'Fraunces', serif; font-weight:600; font-size:22px;
  color:var(--ink); line-height:1; }}
.chips .k {{ font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.14em; font-size:9.5px; color:var(--mute); margin-top:4px; }}

/* ---------- section heading ---------- */
.section-h {{ font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.22em; font-size:11px; color:var(--mute);
  display:flex; align-items:center; gap:12px; margin:6px 0 14px; }}
.section-h::after {{ content:""; flex:1; height:1px; background:var(--line); }}

/* ---------- cards ---------- */
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px,1fr));
  gap:16px; margin-top:16px; }}
.card {{
  background:var(--card); border:1px solid var(--line);
  border-top:3px solid var(--accent); border-radius:3px; padding:18px 18px 16px;
  box-shadow:0 1px 2px rgba(33,28,22,.04); position:relative;
  animation:rise .55s cubic-bezier(.2,.7,.2,1) both;
}}
.card:hover {{ box-shadow:0 10px 28px -12px rgba(33,28,22,.28);
  transform:translateY(-3px); transition:all .25s ease; }}
.card:nth-child(1){{animation-delay:.02s}} .card:nth-child(2){{animation-delay:.08s}}
.card:nth-child(3){{animation-delay:.14s}} .card:nth-child(4){{animation-delay:.20s}}
.card:nth-child(5){{animation-delay:.26s}} .card:nth-child(6){{animation-delay:.32s}}
.card:nth-child(7){{animation-delay:.38s}}

.kicker {{ font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.18em; font-size:10px; color:var(--accent); font-weight:500;
  margin-bottom:7px; }}
.headline {{ font-family:'Fraunces', serif; font-weight:500;
  font-size:18px; line-height:1.24; letter-spacing:-.005em; color:var(--ink);
  margin:0 0 6px; }}
.detail {{ font-size:12.5px; line-height:1.5; color:var(--mute); margin:0 0 4px; }}
.chart {{ width:100%; display:block; margin:10px 0 4px; }}
.bignum {{ font-family:'Fraunces', serif; font-weight:600; font-size:46px;
  line-height:1.1; margin:8px 0; }}

/* signal strength bar */
.signal {{ display:flex; align-items:center; gap:9px; margin-top:12px; }}
.signal .lab {{ font-family:'DM Mono', monospace; font-size:9px;
  letter-spacing:.16em; color:var(--mute); }}
.signal .track {{ flex:1; height:3px; background:var(--line); border-radius:2px;
  overflow:hidden; }}
.signal .fill {{ display:block; height:100%; background:var(--accent);
  border-radius:2px; animation:grow .9s ease both; }}
.signal .num {{ font-family:'DM Mono', monospace; font-size:10.5px;
  color:var(--mute); min-width:24px; text-align:right; }}

/* verify */
details.verify {{ margin-top:11px; border-top:1px dashed var(--line);
  padding-top:9px; }}
details.verify summary {{ list-style:none; cursor:pointer;
  font-family:'DM Mono', monospace; text-transform:uppercase; letter-spacing:.12em;
  font-size:10px; color:var(--mute); display:flex; align-items:center; gap:6px; }}
details.verify summary::-webkit-details-marker {{ display:none; }}
details.verify summary::before {{ content:"+"; color:var(--accent);
  font-weight:600; }}
details.verify[open] summary::before {{ content:"\\2212"; }}
.evid {{ margin-top:8px; display:grid; gap:4px; }}
.evid div {{ display:flex; justify-content:space-between; gap:14px;
  font-family:'DM Mono', monospace; font-size:11px; }}
.evid span {{ color:var(--mute); }} .evid b {{ color:var(--ink); font-weight:500; }}

/* lead story */
.lead {{ background:var(--card); border:1px solid var(--line);
  border-top:3px solid var(--accent); border-radius:3px; padding:24px 26px;
  display:flex; gap:28px; align-items:center; margin-top:6px;
  box-shadow:0 1px 2px rgba(33,28,22,.04);
  animation:rise .5s cubic-bezier(.2,.7,.2,1) both; }}
.lead .textcol {{ flex:1.05; min-width:0; }}
.lead .chartcol {{ flex:1; min-width:0; }}
.lead .headline {{ font-size:27px; font-weight:500; line-height:1.16; }}
.lead .detail {{ font-size:13.5px; }}
@media (max-width:760px) {{ .lead {{ flex-direction:column; align-items:stretch; }} }}

.quiet {{ text-align:center; padding:48px 20px; color:var(--mute);
  font-family:'Fraunces', serif; font-style:italic; font-size:18px; }}

@keyframes rise {{ from {{ opacity:0; transform:translateY(12px); }}
  to {{ opacity:1; transform:none; }} }}
@keyframes grow {{ from {{ width:0 !important; }} }}

/* ---------- streamlit widget restyle (sidebar / expanders / inputs) ------- */
[data-testid="stSidebar"] {{ background:#F1ECE0; border-right:1px solid var(--line); }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {{
  font-family:'Fraunces', serif; font-weight:600; }}
[data-testid="stExpander"] details {{ border:1px solid var(--line) !important;
  border-radius:3px !important; background:var(--card); }}
[data-testid="stExpander"] summary {{ font-family:'DM Mono', monospace;
  text-transform:uppercase; letter-spacing:.1em; font-size:12px; }}
.stButton button {{ background:var(--ink); color:var(--paper); border:none;
  border-radius:3px; font-family:'DM Mono', monospace; text-transform:uppercase;
  letter-spacing:.1em; font-size:12px; }}
.stButton button:hover {{ background:var(--accent); color:#fff; }}
h1,h2,h3,h4 {{ color:var(--ink); }}
</style>
"""


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTE, labelsize=8, length=0)
    ax.xaxis.label.set_color(MUTE); ax.yaxis.label.set_color(MUTE)
    ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
    ax.grid(axis="y", color=LINE, linewidth=.8)
    ax.set_axisbelow(True)
    ax.figure.patch.set_alpha(0); ax.patch.set_alpha(0)


def chart_b64(finding, frame, large=False):
    spec = finding.chart or {}
    kind = spec.get("type")
    if kind in (None, "metric"):
        return None
    figsize = (6.4, 2.7) if large else (4.4, 2.35)
    fig, ax = plt.subplots(figsize=figsize)
    accent = KIND_META.get(finding.kind, ("", INK))[1]
    try:
        if kind == "scatter":
            sub = frame[[spec["x"], spec["y"]]].dropna()
            ax.scatter(sub[spec["x"]], sub[spec["y"]], s=16, alpha=.55,
                       color=accent, edgecolor="none")
            ax.set_xlabel(spec["x"]); ax.set_ylabel(spec["y"])
        elif kind == "group_bar":
            mv = spec["means"]
            ax.bar(list(mv.keys()), list(mv.values()), color=accent, width=.62)
            ax.set_ylabel(spec["value"])
        elif kind == "value_counts":
            cc = dict(sorted(spec["counts"].items(),
                             key=lambda kv: kv[1], reverse=True)[:8])
            ax.bar(list(cc.keys()), list(cc.values()), color=accent, width=.62)
            ax.set_ylabel("count")
        elif kind == "box":
            bp = ax.boxplot(frame[spec["col"]].dropna(), patch_artist=True,
                            widths=.45)
            for b in bp["boxes"]:
                b.set(facecolor=accent, alpha=.22, edgecolor=accent)
            for part in bp["whiskers"] + bp["caps"] + bp["medians"]:
                part.set(color=accent)
            for fl in bp["fliers"]:
                fl.set(markeredgecolor=accent, markersize=3, alpha=.5)
            ax.set_ylabel(spec["col"]); ax.set_xticks([])
        elif kind == "missing_by_group":
            rr = {k: v * 100 for k, v in spec["rates"].items()}
            ax.bar(list(rr.keys()), list(rr.values()), color=accent, width=.62)
            ax.set_ylabel("% missing")
        else:
            plt.close(fig); return None

        _style_axes(ax)
        if kind in ("group_bar", "value_counts", "missing_by_group"):
            plt.setp(ax.get_xticklabels(), rotation=16, ha="right")
        fig.tight_layout()
        return _fig_to_b64(fig)
    except Exception:
        plt.close(fig)
        return None


# --------------------------------------------------------------------------- #
# html builders
# --------------------------------------------------------------------------- #
def _e(x) -> str:
    return html.escape(str(x))


def _signal(score: float) -> str:
    pct = max(2, min(100, int(round(float(score) * 100))))
    return (f"<div class='signal'><span class='lab'>SIGNAL</span>"
            f"<span class='track'><span class='fill' style='width:{pct}%'>"
            f"</span></span><span class='num'>{pct}</span></div>")


def _verify(finding) -> str:
    rows = "".join(f"<div><span>{_e(k)}</span><b>{_e(v)}</b></div>"
                   for k, v in finding.evidence.items())
    return (f"<details class='verify'><summary>Verify the numbers</summary>"
            f"<div class='evid'>{rows}</div></details>")


def _chart_block(finding, frame, large=False) -> str:
    spec = finding.chart or {}
    if spec.get("type") == "metric":
        accent = KIND_META.get(finding.kind, ("", INK))[1]
        return f"<div class='bignum' style='color:{accent}'>{_e(spec['value'])}</div>"
    b64 = chart_b64(finding, frame, large=large)
    return f"<img class='chart' src='data:image/png;base64,{b64}'/>" if b64 else ""


def card_html(finding, frame, lead=False) -> str:
    label, accent = KIND_META.get(finding.kind, (finding.kind, INK))
    detail = f"<p class='detail'>{_e(finding.detail)}</p>" if finding.detail else ""
    head = (f"<div class='kicker'>{_e(label)}</div>"
            f"<h3 class='headline'>{_e(finding.headline)}</h3>{detail}")
    if lead:
        return (f"<article class='lead' style='--accent:{accent}'>"
                f"<div class='textcol'>{head}{_signal(finding.score)}"
                f"{_verify(finding)}</div>"
                f"<div class='chartcol'>{_chart_block(finding, frame, large=True)}"
                f"</div></article>")
    return (f"<article class='card' style='--accent:{accent}'>{head}"
            f"{_chart_block(finding, frame)}{_signal(finding.score)}"
            f"{_verify(finding)}</article>")


def masthead_html(summary: dict, n_findings: int, date_str: str) -> str:
    return (
        "<div class='masthead'>"
        "<div class='edition'>Auto-Insight Edition</div>"
        "<div class='wordmark'><span class='spark'>✦</span> DataLite</div>"
        f"<div class='dateline'>{_e(date_str)} &nbsp;·&nbsp; "
        f"{summary['rows']:,} rows × {summary['cols']} cols &nbsp;·&nbsp; "
        f"{n_findings} findings on the front page</div>"
        "</div><div class='rule'></div><div class='rule thin'></div>"
    )


def chips_html(summary: dict) -> str:
    chips = [
        (f"{summary['numeric']}", "Numeric"),
        (f"{summary['cols'] - summary['numeric']}", "Categorical"),
        (f"{summary['missing_pct']:.0f}%", "Missing"),
        (f"{summary['duplicates']}", "Duplicate rows"),
    ]
    inner = "".join(f"<div class='chip'><div class='v'>{_e(v)}</div>"
                    f"<div class='k'>{_e(k)}</div></div>" for v, k in chips)
    return f"<div class='chips'>{inner}</div>"


def insights_html(findings, frame) -> str:
    if not findings:
        return ("<div class='quiet'>A quiet edition — no strong patterns made "
                "the front page today. Explore the data yourself below.</div>")
    lead = card_html(findings[0], frame, lead=True)
    if len(findings) > 1:
        rest = "".join(card_html(f, frame) for f in findings[1:])
        body = f"<div class='grid'>{rest}</div>"
    else:
        body = ""
    head = "<div class='section-h'>The findings</div>"
    return f"{lead}{head}{body}"
