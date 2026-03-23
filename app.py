"""
Corporate Distress Early Warning System — Interactive Dashboard
E628 Data Science for Business | Person 3 (Ibrahim)

Run locally:  python app.py  -->  http://127.0.0.1:8050
Deploy:       push to GitHub --> connect to Render.com (see README.md)
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import pandas as pd
import numpy as np

COLORS = {
    "distressed": "#E84855",
    "stable":     "#2E86AB",
    "neutral":    "#888888",
    "highlight":  "#F4A261",
    "background": "#F8F9FA",
    "panel":      "#FFFFFF",
    "border":     "#DEE2E6",
    "text":       "#212529",
    "text_muted": "#6C757D",
}

TIER_COLORS = {
    "Critical": "#E84855",
    "High":     "#F4A261",
    "Moderate": "#FFD166",
    "Low":      "#2E86AB",
}

distress_df = pd.DataFrame({
    "ticker":        ["AMC","BYND","WKHS","CMLS","PTON","SEAT","SPCE","BBBY","PLUG",
                      "HD","AAPL","JNJ","MCD","MSFT","V","PG","KO","WMT"],
    "distress_label":[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
    "prob_lr":       [0.8374,0.9997,0.9822,0.994,0.8112,0.8998,0.8865,0.7196,0.9011,
                      0.2989,0.055,0.0176,0.1982,0.0414,0.0774,0.1149,0.1102,0.0545],
    "prob_rf":       [1.0,1.0,1.0,1.0,1.0,1.0,0.98,0.9,0.86,
                      0.12,0.04,0.04,0.04,0.02,0.02,0.02,0.0,0.0],
    "prob_gb":       [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,
                      0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    "prob_ensemble": [0.9458,0.9999,0.9941,0.998,0.9371,0.9666,0.9555,0.8732,0.9204,
                      0.1397,0.0317,0.0192,0.0794,0.0205,0.0325,0.045,0.0368,0.0182],
    "risk_tier":     ["Critical","Critical","Critical","Critical","Critical","Critical",
                      "Critical","Critical","Critical","Low","Low","Low","Low","Low",
                      "Low","Low","Low","Low"],
})

features_df = pd.DataFrame({
    "ticker":       ["AAPL","AMC","BBBY","BYND","CMLS","HD","JNJ","KO","MCD","MSFT",
                     "PG","PLUG","PTON","SEAT","SPCE","V","WKHS","WMT"],
    "return_12m":   [0.173,-0.661,-0.138,-0.801,-0.989,-0.042,0.498,0.115,0.063,0.016,
                     -0.112,0.365,-0.385,-0.881,-0.256,-0.100,-0.871,0.408],
    "return_6m":    [0.052,-0.648,-0.503,-0.742,-0.963,-0.204,0.360,0.163,0.053,-0.227,
                     -0.059,0.381,-0.490,-0.627,-0.213,-0.117,-0.766,0.182],
    "volatility_6m":[0.014,0.034,0.044,0.201,0.150,0.014,0.010,0.011,0.010,0.016,
                     0.012,0.073,0.040,0.053,0.045,0.014,0.062,0.015],
    "volume_change_1m":[0.013,-0.515,-0.193,0.481,-0.587,0.189,-0.178,-0.392,-0.412,0.110,
                        -0.519,0.105,-0.148,-0.327,-0.267,0.013,0.704,-0.461],
    "total_liabilities_to_total_assets":[0.767,1.236,0.488,2.307,1.053,0.886,0.591,0.673,
                                          1.036,0.412,0.581,0.613,1.151,0.697,0.735,0.599,
                                          0.725,0.627],
    "interest_coverage":[None,0.108,None,-24.05,-0.153,None,24.21,7.87,None,66.55,
                         25.64,-41.57,-0.238,-3.74,-18.82,35.67,-45.62,9.42],
    "net_income_to_total_assets":[0.111,-0.016,-0.049,-0.185,-0.019,None,0.026,0.022,
                                   None,0.058,0.034,-0.326,-0.018,-0.008,-0.075,
                                   0.060,-0.067,0.015],
    "current_ratio":[0.974,0.412,1.250,4.535,1.742,1.051,1.028,1.459,1.000,1.386,
                     0.724,2.310,1.983,0.672,2.868,1.111,1.207,0.790],
    "distress_label":[0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,0],
})

shap_df = pd.DataFrame({
    "ticker":       ["AAPL","AMC","BBBY","BYND","CMLS","HD","JNJ","KO","MCD","MSFT",
                     "PG","PLUG","PTON","SEAT","SPCE","V","WKHS","WMT"],
    "return_12m":   [-0.03,0.20,0.13,0.13,0.17,-0.02,-0.02,0.0,-0.03,-0.02,
                     0.0,0.0,0.16,0.18,0.15,0.0,0.15,-0.02],
    "return_6m":    [-0.08,0.12,0.11,0.05,0.11,-0.03,-0.06,-0.08,-0.08,0.0,
                     -0.08,0.03,0.08,0.14,0.0,-0.08,0.08,-0.08],
    "volatility_6m":[-0.11,0.25,0.28,0.27,0.26,-0.11,-0.08,-0.08,-0.11,-0.08,
                     -0.08,0.30,0.27,0.27,0.29,-0.08,0.28,-0.09],
    "volume_change_1m":[0.01,-0.01,0.0,-0.01,0.0,0.0,0.0,0.0,-0.01,0.0,
                        0.0,0.04,0.0,0.0,0.0,0.01,0.0,0.0],
    "total_liabilities_to_total_assets":[0.0,0.01,0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0,
                                          0.0,0.01,0.0,0.0,0.0,0.0,0.0,0.0],
    "interest_coverage":[0.0,0.0,0.0,0.0,0.03,0.0,-0.03,-0.03,-0.03,-0.04,
                         -0.03,0.12,0.02,0.07,0.06,-0.04,0.03,-0.02],
    "net_income_to_total_assets":[-0.01,0.06,0.07,0.02,0.05,0.0,-0.01,-0.01,-0.01,-0.01,
                                   -0.01,0.09,0.06,0.0,0.06,-0.01,0.02,-0.01],
    "current_ratio":[0.0,0.0,0.02,0.02,0.02,0.0,0.0,0.0,0.0,0.0,
                     0.0,0.06,0.03,0.0,0.05,0.0,0.02,0.0],
})

cv_df = pd.DataFrame({
    "Model":          ["Logistic Regression","Random Forest","Gradient Boosting"],
    "AUC_mean":       [1.0, 1.0, 1.0],
    "F1_mean":        [1.0, 1.0, 1.0],
    "Precision_mean": [1.0, 1.0, 1.0],
    "Recall_mean":    [1.0, 1.0, 1.0],
})

FEATURES = [
    "return_12m","return_6m","volatility_6m","volume_change_1m",
    "total_liabilities_to_total_assets","interest_coverage",
    "net_income_to_total_assets","current_ratio",
]

FEATURE_LABELS = {
    "return_12m":                       "12m Return",
    "return_6m":                        "6m Return",
    "volatility_6m":                    "6m Volatility",
    "volume_change_1m":                 "1m Volume Change",
    "total_liabilities_to_total_assets":"Liabilities / Assets",
    "interest_coverage":                "Interest Coverage",
    "net_income_to_total_assets":       "Net Income / Assets",
    "current_ratio":                    "Current Ratio",
}

FEATURE_IMPORTANCE = {
    "volatility_6m":0.334,"return_6m":0.190,"return_12m":0.162,
    "interest_coverage":0.133,"net_income_to_total_assets":0.104,
    "current_ratio":0.034,"total_liabilities_to_total_assets":0.029,
    "volume_change_1m":0.014,
}

# ---------------------------------------------------------------------------
# EDA data — used by Panel 0 charts
# ---------------------------------------------------------------------------
eda_df = pd.DataFrame({
    "ticker":       ["AAPL","AMC","BBBY","BYND","CMLS","HD","JNJ","KO","MCD","MSFT",
                     "PG","PLUG","PTON","SEAT","SPCE","V","WKHS","WMT"],
    "return_12m":   [0.173,-0.661,-0.138,-0.801,-0.989,-0.042,0.498,0.115,0.063,0.016,
                     -0.112,0.365,-0.385,-0.881,-0.256,-0.100,-0.871,0.408],
    "return_6m":    [0.052,-0.648,-0.503,-0.742,-0.963,-0.204,0.360,0.163,0.053,-0.227,
                     -0.059,0.381,-0.490,-0.627,-0.213,-0.117,-0.766,0.182],
    "volatility_6m":[0.014,0.034,0.044,0.201,0.150,0.014,0.010,0.011,0.010,0.016,
                     0.012,0.073,0.040,0.053,0.045,0.014,0.062,0.015],
    "volume_change_1m":[0.013,-0.515,-0.193,0.481,-0.587,0.189,-0.178,-0.392,-0.412,0.110,
                        -0.519,0.105,-0.148,-0.327,-0.267,0.013,0.704,-0.461],
    "total_liabilities_to_total_assets":[0.767,1.236,0.488,2.307,1.053,0.886,0.591,0.673,
                                          1.036,0.412,0.581,0.613,1.151,0.697,0.735,0.599,
                                          0.725,0.627],
    "interest_coverage":[None,0.108,None,-24.05,-0.153,None,24.21,7.87,None,66.55,
                         25.64,-41.57,-0.238,-3.74,-18.82,35.67,-45.62,9.42],
    "net_income_to_total_assets":[0.111,-0.016,-0.049,-0.185,-0.019,None,0.026,0.022,
                                   None,0.058,0.034,-0.326,-0.018,-0.008,-0.075,
                                   0.060,-0.067,0.015],
    "current_ratio":[0.974,0.412,1.250,4.535,1.742,1.051,1.028,1.459,1.000,1.386,
                     0.724,2.310,1.983,0.672,2.868,1.111,1.207,0.790],
    "distress_label":[0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,0],
})
eda_df["label_str"] = eda_df["distress_label"].map({1:"Distressed",0:"Stable"})
D = eda_df[eda_df["distress_label"]==1]
S = eda_df[eda_df["distress_label"]==0]

PLOT_BG = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    font=dict(color="#212529", family="DM Sans, sans-serif", size=12),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=0),
)
GRID = dict(gridcolor="#DEE2E6", zerolinecolor="#DEE2E6")

# ---------------------------------------------------------------------------
# EDA figure builders
# ---------------------------------------------------------------------------
def build_eda1():
    """Box plot: 12m return by label"""
    fig = go.Figure()
    for grp, color, df_g in [("Distressed", COLORS["distressed"], D),
                               ("Stable",     COLORS["stable"],     S)]:
        fig.add_trace(go.Box(
            y=df_g["return_12m"], name=grp,
            marker_color=color, boxmean=True,
            line_width=1.5,
        ))
    fig.update_layout(**PLOT_BG,
                      title=dict(text="12-month return: distressed vs stable", font_size=13, x=0),
                      yaxis=dict(tickformat=".0%", title="12m Return", **GRID),
                      xaxis=dict(**GRID),
                      margin=dict(l=10,r=10,t=44,b=10), showlegend=False)
    return fig

def build_eda2():
    """Bar chart: mean feature values by label"""
    feats  = ["return_12m","return_6m","volatility_6m","interest_coverage","current_ratio"]
    labels = ["12m Return","6m Return","6m Volatility","Interest Coverage","Current Ratio"]
    d_means = [D[f].mean() for f in feats]
    s_means = [S[f].mean() for f in feats]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Distressed", x=labels, y=d_means,
                         marker_color=COLORS["distressed"]))
    fig.add_trace(go.Bar(name="Stable",     x=labels, y=s_means,
                         marker_color=COLORS["stable"]))
    fig.update_layout(**PLOT_BG, barmode="group",
                      title=dict(text="Mean feature values: distressed vs stable", font_size=13, x=0),
                      xaxis=dict(**GRID), yaxis=dict(**GRID),
                      margin=dict(l=10,r=10,t=44,b=10),
                      legend_orientation="h", legend_y=1.12, legend_x=0)
    return fig

def build_eda3():
    """Scatter: 12m return vs 6m volatility, coloured by label"""
    fig = go.Figure()
    for grp, color, df_g in [("Distressed", COLORS["distressed"], D),
                               ("Stable",     COLORS["stable"],     S)]:
        fig.add_trace(go.Scatter(
            x=df_g["volatility_6m"], y=df_g["return_12m"],
            mode="markers+text", name=grp,
            text=df_g["ticker"], textposition="top center",
            textfont=dict(size=9, color=COLORS["text_muted"]),
            marker=dict(color=color, size=10, line=dict(width=1, color=COLORS["border"])),
        ))
    fig.update_layout(**PLOT_BG,
                      title=dict(text="12m return vs 6m volatility", font_size=13, x=0),
                      xaxis=dict(title="6m Volatility", **GRID),
                      yaxis=dict(title="12m Return", tickformat=".0%", **GRID),
                      margin=dict(l=10,r=10,t=44,b=30),
                      legend_orientation="h", legend_y=1.12, legend_x=0)
    return fig

def build_eda4():
    """Histogram: distribution of distress probability scores"""
    probs_d = distress_df[distress_df["distress_label"]==1]["prob_ensemble"].tolist()
    probs_s = distress_df[distress_df["distress_label"]==0]["prob_ensemble"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=probs_d, name="Distressed", nbinsx=10,
                               marker_color=COLORS["distressed"], opacity=0.8))
    fig.add_trace(go.Histogram(x=probs_s, name="Stable", nbinsx=10,
                               marker_color=COLORS["stable"], opacity=0.8))
    fig.update_layout(**PLOT_BG, barmode="overlay",
                      title=dict(text="Distribution of ensemble distress probabilities", font_size=13, x=0),
                      xaxis=dict(title="Ensemble Probability", tickformat=".0%", **GRID),
                      yaxis=dict(title="Count", **GRID),
                      margin=dict(l=10,r=10,t=44,b=30),
                      legend_orientation="h", legend_y=1.12, legend_x=0)
    return fig

def build_eda5():
    """Bar: leverage (liabilities/assets) by company, coloured by label"""
    df_sorted = eda_df.sort_values("total_liabilities_to_total_assets", ascending=False)
    colors = [COLORS["distressed"] if l==1 else COLORS["stable"] for l in df_sorted["distress_label"]]
    fig = go.Figure(go.Bar(
        x=df_sorted["ticker"], y=df_sorted["total_liabilities_to_total_assets"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in df_sorted["total_liabilities_to_total_assets"]],
        textposition="outside",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=COLORS["highlight"],
                  annotation_text="Liabilities > Assets", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.update_layout(**PLOT_BG,
                      title=dict(text="Leverage (liabilities / assets) by company", font_size=13, x=0),
                      xaxis=dict(**GRID), yaxis=dict(title="Liabilities / Assets", **GRID),
                      margin=dict(l=10,r=10,t=44,b=10), showlegend=False)
    return fig

def build_eda6():
    """Scatter: interest coverage vs net income/assets, coloured by label"""
    df_ic = eda_df.dropna(subset=["interest_coverage","net_income_to_total_assets"])
    fig = go.Figure()
    for grp, color in [("Distressed", COLORS["distressed"]), ("Stable", COLORS["stable"])]:
        sub = df_ic[df_ic["label_str"]==grp]
        fig.add_trace(go.Scatter(
            x=sub["interest_coverage"], y=sub["net_income_to_total_assets"],
            mode="markers+text", name=grp,
            text=sub["ticker"], textposition="top center",
            textfont=dict(size=9, color=COLORS["text_muted"]),
            marker=dict(color=color, size=10, line=dict(width=1, color=COLORS["border"])),
        ))
    fig.add_vline(x=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1)
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"], line_width=1)
    fig.update_layout(**PLOT_BG,
                      title=dict(text="Interest coverage vs net income / assets", font_size=13, x=0),
                      xaxis=dict(title="Interest Coverage Ratio", **GRID),
                      yaxis=dict(title="Net Income / Assets", **GRID),
                      margin=dict(l=10,r=10,t=44,b=30),
                      legend_orientation="h", legend_y=1.12, legend_x=0)
    return fig

def tier_color(t):
    return TIER_COLORS.get(t, COLORS["neutral"])

def label_s():
    return {"fontSize":"11px","color":COLORS["text_muted"],"fontWeight":"600",
            "letterSpacing":"0.05em","textTransform":"uppercase",
            "marginBottom":"8px","display":"block"}

def card_s():
    return {"backgroundColor":COLORS["panel"],"border":f"1px solid {COLORS['border']}",
            "borderRadius":"8px","padding":"16px 18px"}

def card_lbl():
    return {"fontSize":"12px","fontWeight":"500","color":COLORS["text_muted"],
            "marginBottom":"12px","marginTop":0,
            "textTransform":"uppercase","letterSpacing":"0.04em"}

def kpi(value, label, color):
    return html.Div(style={"backgroundColor":COLORS["panel"],"padding":"14px 20px",
                            "textAlign":"center"}, children=[
        html.Div(value, style={"fontSize":"22px","fontWeight":"600","color":color}),
        html.Div(label, style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"2px"}),
    ])

def section_hdr(num, title, subtitle):
    return html.Div(style={
        "marginBottom":"14px","paddingBottom":"10px",
        "borderBottom":f"1px solid {COLORS['border']}",
        "display":"flex","alignItems":"baseline","gap":"12px","flexWrap":"wrap",
    }, children=[
        html.Span(f"Panel {num}", style={"fontSize":"10px","fontWeight":"600",
                                         "letterSpacing":"0.08em","color":COLORS["highlight"],
                                         "textTransform":"uppercase"}),
        html.Span(title, style={"fontSize":"16px","fontWeight":"600","color":COLORS["text"]}),
        html.Span(subtitle, style={"fontSize":"12px","color":COLORS["text_muted"]}),
    ])

# ---------------------------------------------------------------------------
# Pre-build initial figures — these render immediately on page load
# ---------------------------------------------------------------------------
def build_screener_fig(model_col="prob_ensemble"):
    df = distress_df.sort_values(model_col, ascending=False)
    scores = df[model_col].tolist()
    colors = [tier_color(t) for t in df["risk_tier"]]
    fig = go.Figure(go.Bar(
        x=df["ticker"].tolist(), y=scores,
        marker_color=colors,
        text=[f"{v:.0%}" for v in scores],
        textposition="outside", cliponaxis=False,
    ))
    fig.add_hline(y=0.8, line_dash="dot", line_color=COLORS["distressed"],
                  annotation_text="Critical (80%)", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.add_hline(y=0.6, line_dash="dot", line_color=COLORS["highlight"],
                  annotation_text="High (60%)", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.update_layout(**PLOT_BG,
                      xaxis=dict(**GRID),
                      yaxis=dict(tickformat=".0%", range=[0,1.18], **GRID),
                      margin=dict(l=10,r=10,t=36,b=10),
                      showlegend=False,
                      title_text="Distress probability by company")
    return fig

def build_cv_fig():
    metrics    = ["AUC","F1","Precision","Recall"]
    cols       = ["AUC_mean","F1_mean","Precision_mean","Recall_mean"]
    bar_colors = ["#378ADD","#E84855","#F4A261","#888888"]
    width = 0.18
    x_pos = np.arange(len(cv_df))
    fig = go.Figure()
    for i, (m, c, color) in enumerate(zip(metrics, cols, bar_colors)):
        fig.add_trace(go.Bar(
            name=m, x=[v + (i-1.5)*width for v in x_pos], y=cv_df[c],
            width=width, marker_color=color,
            text=[f"{v:.2f}" for v in cv_df[c]], textposition="outside",
        ))






    return fig

def build_feat_imp_fig():
    feats  = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1])
    labels = [FEATURE_LABELS.get(f,f) for f,_ in feats]
    values = [v for _,v in feats]
    colors = [COLORS["distressed"] if v >= 0.15 else
              COLORS["stable"]     if v >= 0.05 else
              COLORS["neutral"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
    ))
    fig.update_layout(**PLOT_BG,
                      xaxis=dict(range=[0,0.42], **GRID),
                      yaxis=dict(**GRID),
                      margin=dict(l=10,r=60,t=10,b=10))
    return fig

def build_shap_fig(ticker="AMC"):
    row  = shap_df[shap_df["ticker"] == ticker].iloc[0]
    vals = [(FEATURE_LABELS.get(f,f), float(row[f])) for f in FEATURES]
    vals.sort(key=lambda x: abs(x[1]))
    names, values = zip(*vals)
    bar_colors = [COLORS["distressed"] if v > 0 else COLORS["stable"] for v in values]
    tier = distress_df[distress_df["ticker"] == ticker]["risk_tier"].values[0]
    fig = go.Figure(go.Bar(
        x=list(values), y=list(names), orientation="h",
        marker_color=bar_colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
    ))
    fig.add_vline(x=0, line_color=COLORS["text_muted"], line_width=1)
    fig.update_layout(**PLOT_BG,
                      title=dict(text=f"{ticker}  [{tier}]  — SHAP feature contributions",
                                 font_size=13, x=0),
                      xaxis=dict(title="SHAP contribution (positive = pushes toward distress)", **GRID),
                      yaxis=dict(**GRID),
                      margin=dict(l=10,r=70,t=44,b=30))
    return fig

def build_deepdive_figs(ticker="AMC"):
    row_d = distress_df[distress_df["ticker"] == ticker].iloc[0]
    row_f = features_df[features_df["ticker"] == ticker].iloc[0]
    tc    = tier_color(row_d["risk_tier"])

    r_vals = []
    for f in FEATURES:
        v = row_f[f]
        all_v = features_df[f].dropna()
        if pd.isna(v) or all_v.std() == 0:
            r_vals.append(0.5)
        else:
            r_vals.append(float(np.clip((v - all_v.min()) / (all_v.max() - all_v.min()), 0, 1)))

    lbls = [FEATURE_LABELS.get(f,f) for f in FEATURES]
    r_c  = r_vals + [r_vals[0]]
    l_c  = lbls   + [lbls[0]]
    rr,gg,bb = int(tc[1:3],16), int(tc[3:5],16), int(tc[5:7],16)

    radar_fig = go.Figure(go.Scatterpolar(
        r=r_c, theta=l_c, fill="toself",
        fillcolor=f"rgba({rr},{gg},{bb},0.18)",
        line_color=tc, name=ticker,
    ))
    radar_fig.update_layout(**PLOT_BG,
                             polar=dict(
                                 bgcolor="rgba(0,0,0,0)",
                                 radialaxis=dict(visible=True, range=[0,1],
                                                 gridcolor=COLORS["border"],
                                                 color=COLORS["text_muted"]),
                                 angularaxis=dict(gridcolor=COLORS["border"],
                                                  color=COLORS["text_muted"]),
                             ),
                             margin=dict(l=40,r=40,t=20,b=20), showlegend=False)

    models = ["Logistic Reg.","Random Forest","Gradient Boost","Ensemble"]
    probs  = [row_d["prob_lr"],row_d["prob_rf"],row_d["prob_gb"],row_d["prob_ensemble"]]
    bc     = [COLORS["distressed"] if p >= 0.5 else COLORS["stable"] for p in probs]
    prob_fig = go.Figure(go.Bar(
        x=models, y=probs, marker_color=bc,
        text=[f"{p:.1%}" for p in probs], textposition="outside",
    ))
    prob_fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["text_muted"],
                       annotation_text="Decision boundary", annotation_font_size=10)
    prob_fig.update_layout(**PLOT_BG,
                            xaxis=dict(**GRID),
                            yaxis=dict(range=[0,1.18], tickformat=".0%", **GRID),
                            margin=dict(l=10,r=10,t=20,b=10), showlegend=False)
    return radar_fig, prob_fig

def build_tuning_fig():
    params = [
        "None / 50", "None / 100", "None / 200",
        "5 / 50",    "5 / 100",    "5 / 200",
        "10 / 50",   "10 / 100",   "10 / 200",
    ]
    auc_vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    std_vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    fig = go.Figure(go.Bar(
        x=params, y=auc_vals,
        error_y=dict(type="data", array=std_vals, visible=True,
                     color=COLORS["text_muted"], thickness=1.5),
        marker_color=COLORS["stable"],
        text=[f"{v:.2f}" for v in auc_vals],
        textposition="outside",
    ))
    fig.update_layout(**PLOT_BG,
                      xaxis=dict(title="Max depth / N estimators", **GRID,
                                 tickangle=-30, tickfont_size=11),
                      yaxis=dict(range=[0, 1.18], title="Cross-validated AUC", **GRID),
                      margin=dict(l=10, r=10, t=10, b=80),
                      showlegend=False)
    return fig

# Build all initial figures and table data once at startup
_init_display = distress_df.sort_values("prob_ensemble", ascending=False).copy()
_init_display["distress_label"] = _init_display["distress_label"].map({1:"Distressed",0:"Stable"})
_init_display["score_fmt"] = _init_display["prob_ensemble"].apply(lambda x: f"{x:.3%}")
_init_display["ensemble_fmt"] = _init_display["prob_ensemble"].apply(lambda x: f"{x:.3%}")
INIT_TABLE_DATA = _init_display[["ticker","risk_tier","distress_label","score_fmt","ensemble_fmt"]].to_dict("records")
INIT_TABLE_COLS = [
    {"name":"Ticker","id":"ticker"},
    {"name":"Risk tier","id":"risk_tier"},
    {"name":"True label","id":"distress_label"},
    {"name":"Selected score","id":"score_fmt"},
    {"name":"Ensemble score","id":"ensemble_fmt"},
]
INIT_TUNING_FIG     = build_tuning_fig()
INIT_SCREENER_FIG   = build_screener_fig()
INIT_CV_FIG         = build_cv_fig()
INIT_FEAT_IMP_FIG   = build_feat_imp_fig()
INIT_SHAP_FIG       = build_shap_fig()
INIT_RADAR, INIT_PROBS = build_deepdive_figs()
INIT_EDA1 = build_eda1()
INIT_EDA2 = build_eda2()
INIT_EDA3 = build_eda3()
INIT_EDA4 = build_eda4()
INIT_EDA5 = build_eda5()
INIT_EDA6 = build_eda6()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, title="Corporate Distress EWS")
server = app.server

app.layout = html.Div(style={
    "backgroundColor":COLORS["background"],"minHeight":"100vh",
    "fontFamily":"'DM Sans','Segoe UI',sans-serif","color":COLORS["text"],
}, children=[

    html.Link(rel="stylesheet",
              href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap"),

    html.Div(style={
        "backgroundColor":COLORS["panel"],"borderBottom":f"1px solid {COLORS['border']}",
        "padding":"16px 32px","display":"flex","alignItems":"center","justifyContent":"space-between",
    }, children=[
        html.Div([
            html.Span("▲ ", style={"color":COLORS["distressed"],"fontSize":"18px"}),
            html.Span("Corporate Distress Early Warning System",
                      style={"fontSize":"17px","fontWeight":"600","letterSpacing":"-0.3px"}),
        ], style={"display":"flex","alignItems":"center","gap":"6px"}),
        html.Div("E628 Data Science for Business  ·  Random Forest  ·  18 companies  ·  12-month horizon",
                 style={"fontSize":"11px","color":COLORS["text_muted"]}),
    ]),

    html.Div(style={
        "display":"grid","gridTemplateColumns":"repeat(5,1fr)",
        "gap":"1px","backgroundColor":COLORS["border"],
        "borderBottom":f"1px solid {COLORS['border']}",
    }, children=[
        kpi("9",  "Critical companies",   COLORS["distressed"]),
        kpi("0",  "High / Moderate risk (full separation)", COLORS["highlight"]),
        kpi("9",  "Low risk",             COLORS["stable"]),
        kpi("1.00*","Best model AUC (n=18)", "#a8d8a8"),
        kpi("Random Forest","Top model",  COLORS["text_muted"]),
    ]),

    html.Div(style={"padding":"24px 28px","maxWidth":"1400px","margin":"0 auto"}, children=[

        html.Div(style={
            "backgroundColor":COLORS["panel"],
            "borderLeft":f"3px solid {COLORS['highlight']}",
            "borderRadius":"6px","padding":"12px 18px","marginBottom":"24px",
        }, children=[
            html.P([
                html.Strong("What this dashboard does: ", style={"color":COLORS["text"]}),
                "Predicts the probability of corporate distress over a 12-month horizon "
                "using 8 market and financial features. Screen companies by risk tier, "
                "explore model performance, understand what drives predictions, and "
                "inspect individual firm profiles.",
            ], style={"margin":0,"fontSize":"13px","color":COLORS["text_muted"],"lineHeight":"1.6"}),
        ]),

        section_hdr("5","Exploratory data analysis",
                    "Six charts exploring the raw features — distressed vs stable companies"),

        # Row 1
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px","marginBottom":"20px"}, children=[
            html.Div(style=card_s(), children=[
                html.P("EDA 1 — 12-month return by distress label", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA1, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("Distressed companies show dramatically lower 12m returns on average, with significant downside outliers — consistent with a pre-distress price collapse signal.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
            html.Div(style=card_s(), children=[
                html.P("EDA 2 — Mean feature values: distressed vs stable", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA2, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("Distressed firms have negative average returns and near-zero interest coverage, while stable firms show positive returns and strong debt-servicing capacity.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
        ]),

        # Row 2
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px","marginBottom":"20px"}, children=[
            html.Div(style=card_s(), children=[
                html.P("EDA 3 — 12m return vs 6m volatility (by label)", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA3, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("High volatility combined with negative returns cleanly separates distressed firms from stable ones — this quadrant pattern underpins volatility_6m's top feature importance score.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
            html.Div(style=card_s(), children=[
                html.P("EDA 4 — Ensemble distress probability distribution", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA4, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("The model produces a bimodal distribution — stable firms cluster near 0% and distressed firms near 100% — with almost no overlap, reflecting near-perfect class separation.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
        ]),

        # Row 3
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"20px","marginBottom":"32px"}, children=[
            html.Div(style=card_s(), children=[
                html.P("EDA 5 — Leverage (liabilities / assets) by company", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA5, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("Several distressed firms (BYND, CMLS, PTON) carry liabilities exceeding total assets — a classic balance sheet distress signal. Stable firms generally remain below 1.0x.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
            html.Div(style=card_s(), children=[
                html.P("EDA 6 — Interest coverage vs net income / assets", style=card_lbl()),
                dcc.Graph(figure=INIT_EDA6, config={"displayModeBar":False}, style={"height":"280px"}),
                html.P("Distressed firms cluster in the negative quadrant — unable to cover interest and unprofitable. Stable firms are spread across positive territory, confirming their financial health.",
                       style={"fontSize":"11px","color":COLORS["text_muted"],"marginTop":"8px","lineHeight":"1.5"}),
            ]),
        ]),

        section_hdr("1","Company risk screener",
                    "Filter by tier and probability threshold — results update live"),
        html.P("Table sorted by selected model score, highest to lowest ↓",
               style={"fontSize":"12px","color":COLORS["text_muted"],
                      "marginTop":"-8px","marginBottom":"14px","fontStyle":"italic"}),

        html.Div(style={"display":"grid","gridTemplateColumns":"210px 1fr",
                        "gap":"20px","marginBottom":"32px"}, children=[
            html.Div(style={"display":"flex","flexDirection":"column","gap":"20px"}, children=[
                html.Div([
                    html.Label("Risk tier", style=label_s()),
                    dcc.Checklist(
                        id="tier-filter",
                        options=[{"label":html.Span(t, style={"color":tier_color(t),"paddingLeft":"6px"}),
                                  "value":t}
                                 for t in ["Critical","High","Moderate","Low"]],
                        value=["Critical","High","Moderate","Low"],
                        inputStyle={"accentColor":COLORS["highlight"]},
                        style={"fontSize":"13px","lineHeight":"2.4"},
                    ),
                ]),
                html.Div([
                    html.Label("Min distress probability", style=label_s()),
                    dcc.Slider(id="prob-threshold", min=0, max=1, step=0.05, value=0,
                               marks={0:"0",0.5:"50%",1:"100%"},
                               tooltip={"placement":"bottom","always_visible":False}),
                ]),
                html.Div([
                    html.Label("Model score", style=label_s()),
                    dcc.RadioItems(
                        id="model-select-screener",
                        options=[
                            {"label":" Ensemble","value":"prob_ensemble"},
                            {"label":" Random Forest","value":"prob_rf"},
                            {"label":" Logistic Reg.","value":"prob_lr"},
                            {"label":" Gradient Boost","value":"prob_gb"},
                        ],
                        value="prob_ensemble",
                        inputStyle={"accentColor":COLORS["highlight"],"marginRight":"6px"},
                        style={"fontSize":"13px","lineHeight":"2.4"},
                    ),
                ]),
            ]),
            html.Div([
                html.Div(id="screener-empty-msg"),
                dash_table.DataTable(
                    id="screener-table",
                    style_table={"overflowX":"auto","borderRadius":"6px",
                                 "marginBottom":"16px","border":f"1px solid {COLORS['border']}"},
                    style_header={"backgroundColor":COLORS["border"],"color":COLORS["text"],
                                  "fontWeight":"600","fontSize":"12px","border":"none",
                                  "padding":"10px 14px","textTransform":"uppercase",
                                  "letterSpacing":"0.04em"},
                    style_cell={"backgroundColor":"#FFFFFF","color":"#212529",
                                "border":f"1px solid {COLORS['border']}",
                                "fontSize":"13px","padding":"9px 14px","textAlign":"left"},
                    style_filter={"backgroundColor":"#F8F9FA","color":"#212529",
                                  "border":f"1px solid {COLORS['border']}"},
                    style_filter_conditional=[
                        {"if":{"column_id":"ticker"},"color":"#212529"},
                    ],
                    style_data_conditional=[
                        {"if":{"filter_query":'{risk_tier} = "Critical"',"column_id":"risk_tier"},
                         "color":COLORS["distressed"],"fontWeight":"600"},
                        {"if":{"filter_query":'{risk_tier} = "Low"',"column_id":"risk_tier"},
                         "color":COLORS["stable"],"fontWeight":"600"},
                        {"if":{"filter_query":'{distress_label} = "Distressed"'},
                         "borderLeft":f"3px solid {COLORS['distressed']}"},
                        {"if":{"filter_query":'{distress_label} = "Stable"'},
                         "borderLeft":f"3px solid {COLORS['stable']}"},
                    ],
                    page_size=18, sort_action="native",
                ),
                dcc.Graph(id="screener-bar", figure=INIT_SCREENER_FIG,
                          config={"displayModeBar":False}, style={"height":"280px"}),
            ]),
        ]),

        section_hdr("2","Model performance",
                    "3-fold stratified cross-validation across all three classifiers"),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr",
                        "gap":"20px","marginBottom":"20px"}, children=[
            html.Div(style=card_s(), children=[
                html.P("Cross-validation scores (AUC, F1, Precision, Recall)", style=card_lbl()),
                dcc.Graph(id="cv-chart", figure=INIT_CV_FIG,
                          config={"displayModeBar":False}, style={"height":"270px"}),
            ]),
            html.Div(style={"display":"flex","flexDirection":"column","gap":"16px"}, children=[
                html.Div(style=card_s(), children=[
                    html.P("ROC curves — all models (AUC = 1.00)", style=card_lbl()),
                    html.Img(src="/assets/roc_curves.png",
                             style={"width":"100%","borderRadius":"4px","maxHeight":"180px","objectFit":"contain"}),
                ]),
                html.Div(style=card_s(), children=[
                    html.P("Confusion matrices (cross-validation, 3-fold stratified)", style=card_lbl()),
                    html.Img(src="/assets/confusion_matrices.png",
                             style={"width":"100%","borderRadius":"4px","maxHeight":"150px","objectFit":"contain"}),
                ]),
            ]),
        ]),

        html.Div(style={**card_s(),"marginBottom":"32px"}, children=[
            html.P("Random Forest hyperparameter tuning", style=card_lbl()),
            html.Img(src="/assets/tuning_results.png",
                     style={"width":"100%","borderRadius":"4px","maxHeight":"220px","objectFit":"contain"}),
            html.P("All 9 combinations (depth x estimators) reach AUC ~1.00, "
                   "confirming the signal strength is robust across hyperparameter choices.",
                   style={"fontSize":"12px","color":COLORS["text_muted"],
                          "marginTop":"10px","marginBottom":0,"lineHeight":"1.6"}),
        ]),

        section_hdr("3","Feature importance & SHAP",
                    "What drives the model's distress predictions?"),

        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr",
                        "gap":"20px","marginBottom":"20px"}, children=[
            html.Div(style=card_s(), children=[
                html.P("Random Forest feature importances (Gini impurity)", style=card_lbl()),
                dcc.Graph(id="feat-imp-chart", figure=INIT_FEAT_IMP_FIG,
                          config={"displayModeBar":False}, style={"height":"310px"}),
            ]),
            html.Div(style=card_s(), children=[
                html.P("SHAP beeswarm — feature values coloured by distress label", style=card_lbl()),
                html.Img(src="/assets/shap_beeswarm.png",
                         style={"width":"100%","borderRadius":"4px","maxHeight":"310px","objectFit":"contain"}),
            ]),
        ]),

        html.Div(style={**card_s(),"marginBottom":"32px"}, children=[
            html.P("SHAP feature contributions — per company (interactive)", style=card_lbl()),
            html.Div(style={"display":"flex","alignItems":"center","gap":"14px","marginBottom":"14px"}, children=[
                html.Label("Select company:", style={"fontSize":"13px","color":COLORS["text_muted"]}),
                dcc.Dropdown(
                    id="shap-company-select",
                    options=[{"label":t,"value":t} for t in sorted(shap_df["ticker"].tolist())],
                    value="AMC", clearable=False,
                    style={"width":"160px","fontSize":"13px"},
                ),
            ]),
            dcc.Graph(id="shap-waterfall-chart", figure=INIT_SHAP_FIG,
                      config={"displayModeBar":False}, style={"height":"320px"}),
        ]),

        section_hdr("4","Company deep dive",
                    "Inspect features and probability breakdown for any firm"),

        html.Div(style={"display":"grid","gridTemplateColumns":"190px 1fr",
                        "gap":"20px","marginBottom":"32px"}, children=[
            html.Div(style={"display":"flex","flexDirection":"column","gap":"14px"}, children=[
                html.Div([
                    html.Label("Select company", style=label_s()),
                    dcc.Dropdown(
                        id="deepdive-company",
                        options=[{"label":t,"value":t} for t in sorted(distress_df["ticker"].tolist())],
                        value="AMC", clearable=False,
                        style={"fontSize":"13px","color":"#212529","fontWeight":"500"},
                    ),
                ]),
                html.Div(id="deepdive-scorecard"),
            ]),
            html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px"}, children=[
                html.Div(style=card_s(), children=[
                    html.P("Feature profile (normalised 0-1 across all 18 companies)", style=card_lbl()),
                    dcc.Graph(id="deepdive-radar", figure=INIT_RADAR,
                              config={"displayModeBar":False}, style={"height":"290px"}),
                ]),
                html.Div(style=card_s(), children=[
                    html.P("Distress probability by model", style=card_lbl()),
                    dcc.Graph(id="deepdive-probs", figure=INIT_PROBS,
                              config={"displayModeBar":False}, style={"height":"290px"}),
                ]),
            ]),
        ]),
    ]),

    html.Div(style={
        "borderTop":f"1px solid {COLORS['border']}","padding":"12px 32px",
        "fontSize":"11px","color":COLORS["text_muted"],
        "display":"flex","justifyContent":"space-between",
    }, children=[
        html.Span("E628 Data Science for Business — Corporate Distress Early Warning System"),
        html.Span("Model: Random Forest | Validation: 3-fold StratifiedKFold | Horizon: 12 months"),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks  (update on interaction only — initial view handled by INIT_ figs)
# ---------------------------------------------------------------------------

@app.callback(
    Output("screener-table","data"),
    Output("screener-table","columns"),
    Output("screener-bar","figure"),
    Output("screener-bar","style"),
    Output("screener-empty-msg","children"),
    Input("tier-filter","value"),
    Input("prob-threshold","value"),
    Input("model-select-screener","value"),
    prevent_initial_call=False,
)
def update_screener(tiers, threshold, model_col):
    filtered = distress_df[
        distress_df["risk_tier"].isin(tiers) &
        (distress_df[model_col] >= threshold)
    ].copy().sort_values(model_col, ascending=False)

    if filtered.empty:
        msg = html.Div(
            "No companies match this filter. No firms in this dataset fall in the High or Moderate risk tiers.",
            style={"color":COLORS["text_muted"],"padding":"14px 0",
                   "fontStyle":"italic","fontSize":"13px"},
        )
        return [], [], go.Figure(), {"display":"none"}, msg

    scores_num = filtered[model_col].tolist()
    bar_colors = [tier_color(t) for t in filtered["risk_tier"]]
    tickers    = filtered["ticker"].tolist()

    display = filtered.copy()
    display["distress_label"] = display["distress_label"].map({1:"Distressed",0:"Stable"})
    display["score_fmt"] = display[model_col].apply(lambda x: f"{float(x):.3%}")
    display["ensemble_fmt"]   = display["prob_ensemble"].apply(lambda x: f"{float(x):.3%}")
    display = display[["ticker","risk_tier","distress_label","score_fmt","ensemble_fmt"]]

    cols = [
        {"name":"Ticker",         "id":"ticker"},
        {"name":"Risk tier",      "id":"risk_tier"},
        {"name":"True label",     "id":"distress_label"},
        {"name":"Selected score", "id":"score_fmt"},
        {"name":"Ensemble score", "id":"ensemble_fmt"},
    ]

    fig = go.Figure(go.Bar(
        x=tickers, y=scores_num, marker_color=bar_colors,
        text=[f"{v:.0%}" for v in scores_num],
        textposition="outside", cliponaxis=False,
    ))
    fig.add_hline(y=0.8, line_dash="dot", line_color=COLORS["distressed"],
                  annotation_text="Critical (80%)", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.add_hline(y=0.6, line_dash="dot", line_color=COLORS["highlight"],
                  annotation_text="High (60%)", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.update_layout(**PLOT_BG,
                      xaxis=dict(**GRID),
                      yaxis=dict(tickformat=".0%", range=[0,1.18], **GRID),
                      margin=dict(l=10,r=10,t=36,b=10),
                      showlegend=False,
                      title_text="Distress probability by company")
    return display.to_dict("records"), cols, fig, {"height":"280px"}, None


@app.callback(Output("shap-waterfall-chart","figure"), Input("shap-company-select","value"))
def update_shap_waterfall(ticker):
    return build_shap_fig(ticker)


@app.callback(
    Output("deepdive-scorecard","children"),
    Output("deepdive-radar","figure"),
    Output("deepdive-probs","figure"),
    Input("deepdive-company","value"),
)
def update_deepdive(ticker):
    row_d    = distress_df[distress_df["ticker"] == ticker].iloc[0]
    tier     = row_d["risk_tier"]
    tc       = tier_color(tier)
    prob     = row_d["prob_ensemble"]
    true_lbl = "Distressed" if row_d["distress_label"] == 1 else "Stable"
    lbl_c    = COLORS["distressed"] if row_d["distress_label"] == 1 else COLORS["stable"]

    scorecard = html.Div(style={
        "backgroundColor":COLORS["panel"],"border":f"2px solid {tc}",
        "borderRadius":"8px","padding":"16px",
    }, children=[
        html.Div(ticker, style={"fontSize":"24px","fontWeight":"700","marginBottom":"6px"}),
        html.Div(f"Ensemble: {prob:.1%}",
                 style={"fontSize":"13px","color":COLORS["text_muted"],"marginBottom":"4px"}),
        html.Div(tier, style={"fontSize":"14px","fontWeight":"600","color":tc,"marginBottom":"10px"}),
        html.Div([
            html.Span("True label: ", style={"fontSize":"12px","color":COLORS["text_muted"]}),
            html.Span(true_lbl, style={"fontSize":"12px","fontWeight":"600","color":lbl_c}),
        ]),
    ])

    radar_fig, prob_fig = build_deepdive_figs(ticker)
    return scorecard, radar_fig, prob_fig


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
