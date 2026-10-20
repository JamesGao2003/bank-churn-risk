"""Static HTML dashboard (plotly, one self-contained page) built from the
modeling, survival and fairness results."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C
from .survival import km_table

# Palette (categorical slots in fixed order, sequential blue ramp for ordered tiers)
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SERIES = [S1, S2, S3, S4, S5]
TIER_COLORS = {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#0d366b"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

LAYOUT = dict(
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    font=dict(family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=INK),
    margin=dict(l=56, r=24, t=96, b=48), hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK2)),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=INK2)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=INK2)),
)


def _fig(title: str, **kw) -> go.Figure:
    fig = go.Figure()
    layout = {**LAYOUT, **kw}
    fig.update_layout(title=dict(text=title, x=0, y=0.97, yref="container", font=dict(size=16)), **layout)
    return fig


def fig_roc(res: dict) -> go.Figure:
    fig = _fig("ROC curves on the hold-out set (n = 2,000)", xaxis_title="False positive rate", yaxis_title="True positive rate")
    for i, (name, pts) in enumerate(res["roc"].items()):
        auc = res["holdout"].loc[name, "roc_auc"]
        fig.add_trace(go.Scatter(x=pts["fpr"], y=pts["tpr"], mode="lines", name=f"{name.replace('_', ' ')} (AUC {auc:.3f})",
                                 line=dict(color=SERIES[i], width=2), hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(color=GRID, width=1), hoverinfo="skip"))
    return fig


def fig_calibration(res: dict) -> go.Figure:
    fig = _fig(f"Calibration of the {res['best_name']} model (hold-out, decile bins)", xaxis_title="Mean predicted probability", yaxis_title="Observed churn rate")
    for i, (label, tbl) in enumerate([("before calibration", res["calibration_before"]), ("after isotonic calibration", res["calibration_after"])]):
        fig.add_trace(go.Scatter(x=tbl["mean_predicted"], y=tbl["observed_rate"], mode="lines+markers", name=label,
                                 line=dict(color=SERIES[i], width=2), marker=dict(size=8),
                                 hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect", line=dict(color=GRID, width=1), hoverinfo="skip"))
    return fig


def fig_tiers(res: dict) -> go.Figure:
    t = res["tiers"]
    fig = _fig("Risk tiers on the hold-out set: customers and observed churn rate", xaxis_title="Risk tier (0-100 score)", yaxis_title="Customers", bargap=0.35)
    fig.add_trace(go.Bar(x=t["tier"].astype(str), y=t["customers"], name="customers",
                         marker=dict(color=[TIER_COLORS[x] for x in t["tier"].astype(str)], cornerradius=4),
                         text=[f"{r:.0%} churn" for r in t["churn_rate"]], textposition="outside", textfont=dict(color=INK2),
                         customdata=np.stack([t["churn_rate"], t["share_of_churners"], t["lift_vs_base"]], axis=1),
                         hovertemplate="%{x}: %{y} customers<br>churn rate %{customdata[0]:.1%}<br>share of all churners %{customdata[1]:.1%}<br>lift %{customdata[2]:.1f}x<extra></extra>"))
    fig.update_layout(showlegend=False)
    return fig


def fig_cutoff(res: dict) -> go.Figure:
    s = res["cutoff_sweep"]
    best = s.loc[s["net_benefit"].idxmax()]
    fig = _fig("Net benefit of targeting every customer at or above a score cutoff", xaxis_title="Risk score cutoff", yaxis_title="Net benefit (USD, hold-out set)")
    fig.add_trace(go.Scatter(x=s["cutoff"], y=s["net_benefit"], mode="lines+markers", name="net benefit", line=dict(color=S1, width=2), marker=dict(size=8),
                             customdata=np.stack([s["targeted"], s["recall"]], axis=1),
                             hovertemplate="cutoff %{x}<br>net benefit $%{y:,.0f}<br>targeted %{customdata[0]}<br>recall %{customdata[1]:.0%}<extra></extra>"))
    fig.add_annotation(x=best["cutoff"], y=best["net_benefit"], text=f"best cutoff {int(best['cutoff'])}: ${best['net_benefit']:,.0f}", showarrow=True, arrowhead=0, ay=-40, font=dict(color=INK2))
    fig.update_layout(showlegend=False)
    return fig


def fig_importance(res: dict) -> go.Figure:
    imp = res["importance"].head(10).iloc[::-1]
    fig = _fig(f"Top 10 feature importances ({res['best_name']})", xaxis_title="Importance (gain share)", yaxis_title="", bargap=0.3)
    fig.add_trace(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker=dict(color=S1, cornerradius=4),
                         hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(showlegend=False, margin=dict(l=150))
    return fig


def fig_km(surv: dict, col: str) -> go.Figure:
    tbl = km_table(surv["km"][col])
    p = surv["logrank_p"][col]
    fig = _fig(f"Kaplan-Meier retention by {col} (log-rank p = {p:.1e})", xaxis_title="Tenure (years)", yaxis_title="Share still a customer")
    for i, (g, d) in enumerate(tbl.groupby("group", sort=False)):
        fig.add_trace(go.Scatter(x=d["tenure"], y=d["survival"], mode="lines", name=g.split("=")[-1], line=dict(color=SERIES[i % len(SERIES)], width=2, shape="hv"),
                                 hovertemplate="tenure %{x}<br>retention %{y:.1%}<extra>" + g + "</extra>"))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%")
    return fig


def fig_hazard(surv: dict) -> go.Figure:
    h = surv["hazard"]
    fig = _fig("Discrete hazard by tenure year (churners at t / customers reaching t)", xaxis_title="Tenure (years)", yaxis_title="Hazard", bargap=0.3)
    colors = [S2 if t == surv["peak_hazard_tenure"] else S1 for t in h["tenure"]]
    fig.add_trace(go.Bar(x=h["tenure"], y=h["hazard"], marker=dict(color=colors, cornerradius=4), customdata=np.stack([h["at_risk"], h["observed"]], axis=1),
                         hovertemplate="tenure %{x}<br>hazard %{y:.1%}<br>at risk %{customdata[0]}<br>churned %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%"); fig.update_layout(showlegend=False)
    return fig


def fig_cox(surv: dict) -> go.Figure:
    t = surv["cox_table"].iloc[::-1]
    fig = _fig(f"Cox proportional-hazards: hazard ratios with 95% CI (concordance {surv['concordance']:.3f})", xaxis_title="Hazard ratio (log scale)", yaxis_title="")
    fig.add_trace(go.Scatter(x=t["hazard_ratio"], y=t["covariate"], mode="markers", marker=dict(color=[S2 if s else INK2 for s in t["significant_5pct"]], size=9),
                             error_x=dict(type="data", symmetric=False, array=t["hr_upper95"] - t["hazard_ratio"], arrayminus=t["hazard_ratio"] - t["hr_lower95"], color=INK2, thickness=1.5),
                             customdata=t["p_value"], hovertemplate="%{y}<br>HR %{x:.3f}<br>p = %{customdata:.3g}<extra></extra>"))
    fig.add_vline(x=1, line=dict(color=GRID, width=1))
    fig.update_xaxes(type="log", tickvals=[0.3, 0.5, 0.7, 1, 1.5, 2, 3], ticktext=["0.3", "0.5", "0.7", "1", "1.5", "2", "3"])
    fig.update_layout(showlegend=False, margin=dict(l=150))
    return fig


def fig_fairness(fair: dict, key: str, title: str) -> go.Figure:
    t = fair[key]
    fig = _fig(title, xaxis_title="", yaxis_title="Rate", barmode="group", bargap=0.3, bargroupgap=0.08)
    for i, (col, label) in enumerate([("recall", "recall (true positive rate)"), ("false_positive_rate", "false positive rate"), ("actual_churn_rate", "actual churn rate")]):
        fig.add_trace(go.Bar(x=t["group"], y=t[col], name=label, marker=dict(color=SERIES[i], cornerradius=4), hovertemplate="%{x}<br>" + label + " %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    return fig


def _table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(fmt[c])
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.3f}")
    return d.to_html(index=False, classes="tbl", border=0)


def build(res: dict, surv: dict, fair: dict, out_path=C.REPORTS / "dashboard.html") -> str:
    pct = lambda v: f"{v:.1%}"
    tiers = res["tiers"].set_index("tier")
    high = tiers.loc["High"]
    econ = res["economics"]
    best_cut = res["cutoff_sweep"].loc[res["cutoff_sweep"]["net_benefit"].idxmax()]
    cal = res["calibrated_eval"]; unc = res["uncalibrated_eval"]
    kpis = [
        ("Hold-out ROC AUC", f"{cal['roc_auc']:.3f}", f"{res['best_name']}, calibrated"),
        ("PR AUC", f"{cal['pr_auc']:.3f}", "base rate 20.4%"),
        ("Brier score", f"{cal['brier']:.3f}", f"was {unc['brier']:.3f} before calibration"),
        ("High-risk tier", f"{high['share_of_customers']:.0%} of customers", f"hold {high['share_of_churners']:.0%} of churners, {high['lift_vs_base']:.1f}x lift"),
        ("Best cutoff net benefit", f"${best_cut['net_benefit']:,.0f}", f"score >= {int(best_cut['cutoff'])}, 2,000 hold-out customers"),
        ("Peak churn hazard", f"tenure {surv['peak_hazard_tenure']} yrs", f"{surv['peak_hazard']:.1%} of customers reaching it"),
    ]
    figs = [
        ("Model performance", [fig_roc(res), fig_calibration(res)]),
        ("Risk score and retention economics", [fig_tiers(res), fig_cutoff(res), fig_importance(res)]),
        ("Survival analysis", [fig_km(surv, "Geography"), fig_km(surv, "NumOfProducts"), fig_km(surv, "IsActiveMember"), fig_hazard(surv), fig_cox(surv)]),
        ("Fairness check", [fig_fairness(fair, "by_gender", f"Fairness by Gender at score >= {fair['cutoff']}"), fig_fairness(fair, "by_age", f"Fairness by age band at score >= {fair['cutoff']}")]),
    ]
    tables = {
        "Model performance": _table(res["holdout"].reset_index().drop(columns=["threshold"]).rename(columns={"index": "model"})),
        "Risk score and retention economics": _table(econ[["tier", "customers", "churners", "churn_rate", "share_of_churners", "lift_vs_base", "offer_cost", "expected_saved_value", "net_benefit"]],
                                                    {"churn_rate": pct, "share_of_churners": pct, "offer_cost": lambda v: f"${v:,.0f}", "expected_saved_value": lambda v: f"${v:,.0f}", "net_benefit": lambda v: f"${v:,.0f}"}),
        "Survival analysis": _table(surv["cox_table"], {"p_value": lambda v: f"{v:.2e}"}),
        "Fairness check": _table(pd.concat([fair["by_gender"].assign(dimension="Gender"), fair["by_age"].assign(dimension="Age band")]).reset_index(drop=True), {"actual_churn_rate": pct, "selection_rate": pct, "recall": pct, "false_positive_rate": pct, "precision": pct}),
    }
    parts = []
    first = True
    for section, fl in figs:
        parts.append(f"<section><h2>{section}</h2><div class='grid'>")
        for f in fl:
            parts.append("<div class='card'>" + f.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displayModeBar": False, "responsive": True}) + "</div>")
            first = False
        parts.append("</div><details><summary>Table view</summary>" + tables[section] + "</details></section>")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Churn Risk</title>
<style>
:root{{--surface:{SURFACE};--ink:{INK};--ink2:{INK2};--grid:{GRID};--card:#ffffff}}
body{{margin:0;background:var(--surface);color:var(--ink);font-family:Inter,-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;line-height:1.45}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:var(--ink2);margin:0 0 20px;font-size:14px}}
h2{{font-size:18px;margin:32px 0 12px;border-bottom:1px solid var(--grid);padding-bottom:6px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:14px 16px}}
.kpi .l{{font-size:12px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em}} .kpi .v{{font-size:24px;font-weight:600;margin:4px 0}} .kpi .s{{font-size:12px;color:var(--ink2)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}}
.card{{background:var(--card);border:1px solid var(--grid);border-radius:8px;padding:8px;min-width:0}}
details{{margin-top:12px}} summary{{cursor:pointer;color:var(--ink2);font-size:13px}}
table.tbl{{border-collapse:collapse;font-size:13px;margin-top:8px;width:100%;overflow-x:auto;display:block}} .tbl th,.tbl td{{padding:6px 10px;border-bottom:1px solid var(--grid);text-align:right}} .tbl th:first-child,.tbl td:first-child{{text-align:left}}
.note{{font-size:13px;color:var(--ink2);margin-top:16px}}
</style></head><body><main>
<h1>Bank Customer Churn: risk scoring, survival analysis and fairness check</h1>
<p class="sub">Public "Churn Modelling" dataset, 10,000 retail-bank customers (France, Germany, Spain). Hold-out test set of 2,000 customers, stratified. Generated {date.today().isoformat()} from the project notebook; every number comes from the saved run in <code>reports/</code>.</p>
<div class="kpis">{''.join(f"<div class='kpi'><div class='l'>{l}</div><div class='v'>{v}</div><div class='s'>{s}</div></div>" for l, v, s in kpis)}</div>
{''.join(parts)}
<p class="note">Retention economics assume a ${C.RETENTION_OFFER_COST:.0f} offer per targeted customer, ${C.RETAINED_CUSTOMER_VALUE:.0f} of 12-month value kept per retained churner, and a {C.OFFER_SUCCESS_RATE:.0%} offer success rate. Survival analysis reads Tenure as time-to-event with Exited as the event; customers who stayed are right-censored at their current tenure.</p>
</main></body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def all_figures(res: dict, surv: dict, fair: dict) -> dict[str, go.Figure]:
    """Every dashboard figure keyed by a file-safe name."""
    figs = {
        "roc": fig_roc(res), "calibration": fig_calibration(res), "risk_tiers": fig_tiers(res),
        "cutoff_sweep": fig_cutoff(res), "feature_importance": fig_importance(res),
        "km_geography": fig_km(surv, "Geography"), "km_numofproducts": fig_km(surv, "NumOfProducts"),
        "km_isactivemember": fig_km(surv, "IsActiveMember"), "hazard_by_tenure": fig_hazard(surv),
        "cox_hazard_ratios": fig_cox(surv),
        "fairness_gender": fig_fairness(fair, "by_gender", f"Fairness by Gender at score >= {fair['cutoff']}"),
        "fairness_age": fig_fairness(fair, "by_age", f"Fairness by age band at score >= {fair['cutoff']}"),
    }
    return figs


def save_figures(res: dict, surv: dict, fair: dict, out_dir=C.FIGURES) -> list[str]:
    """Static PNG copies of every figure (needs kaleido) for the README."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in all_figures(res, surv, fair).items():
        p = out_dir / f"{name}.png"
        fig.write_image(str(p), width=900, height=520, scale=2)
        paths.append(str(p))
    return paths
