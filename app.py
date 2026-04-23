from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sp500_insights.data_pipeline import (  # noqa: E402
    COMPANY_LABELS,
    TICKER_COLORS,
    build_outputs,
    export_project_artifacts,
    load_project_outputs,
)

st.set_page_config(
    page_title="S&P 500 Representative Stocks Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Fira+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --navy: #1E40AF;
  --navy-deep: #1E3A8A;
  --blue: #3B82F6;
  --teal: #0F766E;
  --amber: #F59E0B;
  --slate: #64748B;
  --background: #F8FAFC;
  --panel: rgba(255,255,255,0.88);
  --border: rgba(30,64,175,0.12);
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
    radial-gradient(circle at top right, rgba(245,158,11,0.14), transparent 25%),
    linear-gradient(180deg, #F8FAFC 0%, #EDF3FB 100%);
  color: #0F172A;
  font-family: 'Fira Sans', sans-serif;
}

div.block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
}

h1, h2, h3 {
  color: var(--navy-deep);
  font-family: 'Fira Code', monospace;
  letter-spacing: -0.02em;
}

.hero-shell {
  display: grid;
  grid-template-columns: 1.65fr 0.95fr;
  gap: 1rem;
  margin-bottom: 1rem;
}

.hero-card,
.signal-card,
.section-panel {
  background: var(--panel);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border);
  border-radius: 22px;
  box-shadow: 0 20px 45px rgba(15, 23, 42, 0.06);
}

.hero-card {
  padding: 1.6rem 1.7rem;
  position: relative;
  overflow: hidden;
}

.hero-card::after {
  content: "";
  position: absolute;
  inset: auto -60px -80px auto;
  width: 180px;
  height: 180px;
  background: radial-gradient(circle, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0));
}

.eyebrow {
  font-family: 'Fira Code', monospace;
  color: var(--blue);
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
}

.hero-title {
  font-size: 2.15rem;
  line-height: 1.08;
  margin: 0.35rem 0 0.75rem 0;
}

.hero-copy {
  color: #334155;
  font-size: 1rem;
  max-width: 92%;
}

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
}

.chip {
  font-family: 'Fira Code', monospace;
  font-size: 0.8rem;
  padding: 0.45rem 0.7rem;
  border-radius: 999px;
  background: rgba(30, 64, 175, 0.08);
  color: var(--navy-deep);
}

.signal-card {
  padding: 1.3rem 1.3rem;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.signal-label {
  font-family: 'Fira Code', monospace;
  color: var(--slate);
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.signal-value {
  font-family: 'Fira Code', monospace;
  color: var(--navy-deep);
  font-size: 2.1rem;
  margin-top: 0.2rem;
}

.signal-note {
  color: #475569;
  font-size: 0.94rem;
}

.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.9rem;
  margin: 1rem 0 1.4rem 0;
}

.kpi-card {
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(30,64,175,0.1);
  border-radius: 18px;
  padding: 1rem 1rem 0.95rem 1rem;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
}

.kpi-card .label {
  font-family: 'Fira Code', monospace;
  color: #64748B;
  font-size: 0.78rem;
  text-transform: uppercase;
}

.kpi-card .value {
  font-family: 'Fira Code', monospace;
  color: #1E3A8A;
  font-size: 1.65rem;
  margin-top: 0.25rem;
}

.kpi-card .caption {
  color: #475569;
  font-size: 0.9rem;
  margin-top: 0.35rem;
}

.section-panel {
  padding: 1rem 1rem 0.5rem 1rem;
  margin-bottom: 1rem;
}

[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.9);
  border-right: 1px solid rgba(30, 64, 175, 0.08);
}

@media (max-width: 1100px) {
  .hero-shell, .kpi-grid {
    grid-template-columns: 1fr;
  }
  .hero-copy {
    max-width: 100%;
  }
}
</style>
"""


@st.cache_data(show_spinner=False)
def ensure_outputs() -> dict[str, pd.DataFrame]:
    processed_files = [
        ROOT / "data" / "processed" / "daily_stock_clean.csv",
        ROOT / "data" / "processed" / "monthly_returns.csv",
        ROOT / "data" / "processed" / "fundamentals_clean.csv",
        ROOT / "data" / "processed" / "company_year_panel.csv",
    ]
    if not all(path.exists() for path in processed_files):
        export_project_artifacts()
    return load_project_outputs()


def pct_format(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}"


def num_format(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.1f}"


def build_kpi_card(label: str, value: str, caption: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      <div class="caption">{caption}</div>
    </div>
    """


def render_kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(cards), gap="medium")
    for column, (label, value, caption) in zip(columns, cards):
        with column:
            st.markdown(build_kpi_card(label, value, caption), unsafe_allow_html=True)


def plotly_layout(fig: go.Figure, title: str, *, show_legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        margin=dict(l=28, r=24, t=92, b=88 if show_legend else 36),
        font=dict(family="Fira Sans", color="#0F172A"),
        title_font=dict(family="Fira Code", size=15, color="#1E3A8A"),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            x=0,
            xanchor="left",
            title_text="",
            font=dict(size=11),
        ),
        hoverlabel=dict(bgcolor="#FFFFFF", font_family="Fira Sans"),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.24)", zeroline=False)
    return fig


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
outputs = ensure_outputs()

daily_stock = outputs["daily_stock"]
monthly_returns = outputs["monthly_returns"]
fundamentals = outputs["fundamentals"]
company_year = outputs["company_year"]

all_companies = [ticker for ticker in COMPANY_LABELS if ticker in daily_stock["ticker"].unique()]

st.sidebar.markdown("## Dashboard Controls")
selected_companies = st.sidebar.multiselect(
    "Companies",
    options=all_companies,
    default=all_companies,
)
selected_year = st.sidebar.selectbox("Year Filter", options=["All", 2023, 2024], index=0)
view_mode = st.sidebar.radio("View Mode", options=["Price Trend", "Return Performance"])

filtered_daily = daily_stock[daily_stock["ticker"].isin(selected_companies)].copy()
filtered_monthly = monthly_returns[monthly_returns["ticker"].isin(selected_companies)].copy()
filtered_fundamentals = fundamentals[fundamentals["ticker"].isin(selected_companies)].copy()
filtered_panel = company_year[company_year["ticker"].isin(selected_companies)].copy()

if not selected_companies:
    st.warning("Select at least one company from the sidebar to populate the dashboard.")
    st.stop()

if selected_year != "All":
    filtered_daily = filtered_daily[filtered_daily["year"] == int(selected_year)]
    filtered_monthly = filtered_monthly[filtered_monthly["year"] == int(selected_year)]
    filtered_fundamentals = filtered_fundamentals[filtered_fundamentals["year"] == int(selected_year)]
    filtered_panel = filtered_panel[filtered_panel["year"] == int(selected_year)]

sample_start = filtered_daily["date"].min().strftime("%Y-%m-%d")
sample_end = filtered_daily["date"].max().strftime("%Y-%m-%d")
best_row = filtered_panel.sort_values("cumulative_return", ascending=False).iloc[0]

hero_html = f"""
<div class="hero-shell">
  <div class="hero-card">
    <div class="eyebrow">S&P 500 subset analytics product</div>
    <div class="hero-title">Representative Stocks Performance and Financial Fundamentals</div>
    <div class="hero-copy">
      This dashboard compares Apple, Microsoft, JPMorgan Chase, and Walmart across market performance,
      downside risk, and core business fundamentals. It is designed for an academic audience that needs
      a clear, defensible, and Python-driven analytical workflow rather than an over-engineered finance tool.
    </div>
    <div class="chip-row">
      <div class="chip">{sample_start} to {sample_end}</div>
      <div class="chip">{len(selected_companies)} companies selected</div>
      <div class="chip">{'All years' if selected_year == 'All' else f'Year {selected_year}'}</div>
    </div>
  </div>
  <div class="signal-card">
    <div>
      <div class="signal-label">Top return signal</div>
      <div class="signal-value">{best_row['ticker']}</div>
      <div class="signal-note">
        Best cumulative return in the current filter window: {pct_format(best_row['cumulative_return'])}.
      </div>
    </div>
    <div class="signal-note">
      Focus: descriptive analytics, exploratory stock-fundamentals comparison, and transparent limitations.
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

render_kpi_cards(
    [
        (
            "Sample Firms",
            str(len(selected_companies)),
            "AAPL, MSFT, JPM, and WMT subset",
        ),
        (
            "Mean Cumulative Return",
            pct_format(filtered_panel["cumulative_return"].mean()),
            "Average across visible company-year records",
        ),
        (
            "Mean Annualized Volatility",
            pct_format(filtered_panel["annualized_volatility"].mean()),
            "Derived from daily return variability",
        ),
        (
            "Mean ROA",
            pct_format(filtered_panel["roa"].mean()),
            "Accounting profitability relative to total assets",
        ),
    ]
)

overview_tab, performance_tab, risk_tab, fundamentals_tab, insight_tab = st.tabs(
    ["Overview", "Stock Performance", "Risk Profile", "Financial Fundamentals", "Insights and Limitations"]
)

with overview_tab:
    col1, col2 = st.columns([1.15, 0.85], gap="large")
    with col1:
        metric_col = "normalized_price" if view_mode == "Price Trend" else "cumulative_return_index"
        title = (
            "Normalized Price Index"
            if view_mode == "Price Trend"
            else "Compounded Daily Return Index"
        )
        fig = px.line(
            filtered_daily,
            x="date",
            y=metric_col,
            color="ticker",
            color_discrete_map=TICKER_COLORS,
            line_group="ticker",
        )
        plotly_layout(fig, title)
        st.plotly_chart(fig, width="stretch")
    with col2:
        ranked = filtered_panel.sort_values(
            ["year", "cumulative_return"], ascending=[True, False]
        )[
            [
                "year",
                "ticker",
                "cumulative_return",
                "annualized_volatility",
                "max_drawdown",
                "roa",
            ]
        ].copy()
        ranked["cumulative_return"] = ranked["cumulative_return"].map(pct_format)
        ranked["annualized_volatility"] = ranked["annualized_volatility"].map(pct_format)
        ranked["max_drawdown"] = ranked["max_drawdown"].map(pct_format)
        ranked["roa"] = ranked["roa"].map(pct_format)
        st.markdown("### Company-Year Scorecard")
        st.dataframe(ranked, width="stretch", hide_index=True)

    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    narrative_cols = st.columns(3)
    insights = [
        (
            "Return leaders",
            "MSFT led the sample in 2023, while WMT posted the strongest 2024 cumulative return.",
        ),
        (
            "Risk contrast",
            "WMT shows the lowest annualized volatility in the sample, while technology names fluctuate more.",
        ),
        (
            "Interpretation rule",
            "This product compares stock outcomes and fundamentals side by side, but it does not claim causal inference.",
        ),
    ]
    for column, (title_text, body) in zip(narrative_cols, insights):
        column.markdown(f"#### {title_text}")
        column.write(body)
    st.markdown("</div>", unsafe_allow_html=True)

with performance_tab:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.line(
            filtered_monthly,
            x="month",
            y="monthly_return",
            color="ticker",
            color_discrete_map=TICKER_COLORS,
            markers=True,
        )
        fig.update_xaxes(type="category")
        plotly_layout(fig, "Monthly Return Comparison")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(
            filtered_panel,
            x="ticker",
            y="cumulative_return",
            color="year",
            barmode="group",
            color_discrete_sequence=["#1E40AF", "#3B82F6"],
        )
        plotly_layout(fig, "Annual Cumulative Return by Firm")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Monthly Return Table")
    monthly_table = filtered_monthly[
        ["ticker", "month_name", "monthly_return", "avg_price"]
    ].copy()
    monthly_table["monthly_return"] = monthly_table["monthly_return"].map(pct_format)
    monthly_table["avg_price"] = monthly_table["avg_price"].map(lambda v: f"{v:,.2f}")
    st.dataframe(monthly_table, width="stretch", hide_index=True)

with risk_tab:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.line(
            filtered_daily.dropna(subset=["rolling_20d_volatility"]),
            x="date",
            y="rolling_20d_volatility",
            color="ticker",
            color_discrete_map=TICKER_COLORS,
        )
        plotly_layout(fig, "20-Day Rolling Annualized Volatility")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.line(
            filtered_daily,
            x="date",
            y="drawdown",
            color="ticker",
            color_discrete_map=TICKER_COLORS,
        )
        plotly_layout(fig, "Drawdown Trajectory")
        st.plotly_chart(fig, width="stretch")

    subplot = make_subplots(
        rows=1,
        cols=len(selected_companies),
        subplot_titles=selected_companies,
        shared_yaxes=True,
    )
    for index, ticker in enumerate(selected_companies, start=1):
        subset = filtered_daily[filtered_daily["ticker"] == ticker]
        subplot.add_trace(
            go.Histogram(
                x=subset["daily_return"],
                name=ticker,
                marker_color=TICKER_COLORS[ticker],
                opacity=0.8,
                nbinsx=30,
                showlegend=False,
            ),
            row=1,
            col=index,
        )
    subplot.update_layout(
        title="Daily Return Distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.65)",
        margin=dict(l=28, r=24, t=56, b=36),
        font=dict(family="Fira Sans"),
    )
    st.plotly_chart(subplot, width="stretch")

with fundamentals_tab:
    latest_year = 2024 if selected_year == "All" else selected_year
    fundamentals_view = filtered_fundamentals[filtered_fundamentals["year"] == latest_year]
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.bar(
            fundamentals_view,
            x="ticker",
            y="sales",
            color="ticker",
            color_discrete_map=TICKER_COLORS,
        )
        plotly_layout(fig, f"Sales Comparison ({latest_year})", show_legend=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(
            fundamentals_view,
            x="ticker",
            y="net_income",
            color="ticker",
            color_discrete_map=TICKER_COLORS,
        )
        plotly_layout(fig, f"Net Income Comparison ({latest_year})", show_legend=False)
        st.plotly_chart(fig, width="stretch")

    ratio_table = filtered_panel[
        [
            "year",
            "ticker",
            "roa",
            "net_profit_margin",
            "leverage_ratio",
            "asset_turnover",
        ]
    ].copy()
    ratio_table["roa"] = ratio_table["roa"].map(pct_format)
    ratio_table["net_profit_margin"] = ratio_table["net_profit_margin"].map(pct_format)
    ratio_table["leverage_ratio"] = ratio_table["leverage_ratio"].map(lambda v: f"{v:.2f}")
    ratio_table["asset_turnover"] = ratio_table["asset_turnover"].map(lambda v: f"{v:.2f}")
    st.markdown("### Fundamental Ratio Table")
    st.dataframe(ratio_table, width="stretch", hide_index=True)

with insight_tab:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        fig = px.scatter(
            filtered_panel,
            x="roa",
            y="cumulative_return",
            color="ticker",
            symbol="year",
            size="sales",
            color_discrete_map=TICKER_COLORS,
            hover_data=["net_profit_margin", "leverage_ratio"],
        )
        plotly_layout(fig, "Annual Return vs Return on Assets")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.scatter(
            filtered_panel,
            x="net_profit_margin",
            y="cumulative_return",
            color="ticker",
            symbol="year",
            size="total_assets",
            color_discrete_map=TICKER_COLORS,
            hover_data=["roa", "leverage_ratio"],
        )
        plotly_layout(fig, "Annual Return vs Net Profit Margin")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Analytical Notes")
    st.write(
        """
        - This project compares four representative firms rather than the full S&P 500 universe.
        - Daily market data are aligned to annual accounting data by calendar year for teaching purposes.
        - The dashboard supports exploratory interpretation, not forecasting or causal inference.
        - JPMorgan required an extra cleaning step because the raw fundamentals file included duplicate year entries under different `indfmt` categories.
        """
    )

st.caption(
    "English-only academic analytics product built with Python, Streamlit, Plotly, and local skill-guided design references."
)
