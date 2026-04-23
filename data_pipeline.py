from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_STOCK_PATH = Path("data/data.csv")
DEFAULT_FUNDAMENTALS_PATH = Path("data/data2.csv")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_FIGURES_DIR = Path("reports/figures")

TICKER_ORDER = ["AAPL", "MSFT", "JPM", "WMT"]
COMPANY_LABELS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "JPM": "JPMorgan Chase",
    "WMT": "Walmart",
}
TICKER_COLORS = {
    "AAPL": "#1E40AF",
    "MSFT": "#3B82F6",
    "JPM": "#0F766E",
    "WMT": "#F59E0B",
}


@dataclass
class OutputBundle:
    daily_stock: pd.DataFrame
    monthly_returns: pd.DataFrame
    fundamentals: pd.DataFrame
    company_year: pd.DataFrame


def _max_drawdown(returns: pd.Series) -> float:
    wealth_index = (1 + returns.fillna(0)).cumprod()
    peaks = wealth_index.cummax()
    drawdown = wealth_index / peaks - 1
    return float(drawdown.min())


def clean_stock_data(path: Path | str = DEFAULT_STOCK_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["PRC"] = pd.to_numeric(df["PRC"], errors="coerce")
    df["RET"] = pd.to_numeric(df["RET"], errors="coerce")
    df = df.rename(
        columns={
            "TICKER": "ticker",
            "COMNAM": "company_name",
            "PERMNO": "permno",
            "PRC": "price",
            "RET": "daily_return",
        }
    )
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["month_name"] = df["date"].dt.strftime("%b %Y")
    df["normalized_price"] = (
        df.groupby("ticker")["price"].transform(lambda s: s / s.iloc[0] * 100)
    )
    df["cumulative_return_index"] = (
        df.groupby("ticker")["daily_return"].transform(lambda s: (1 + s).cumprod())
    )
    df["drawdown"] = (
        df.groupby("ticker")["cumulative_return_index"].transform(
            lambda s: s / s.cummax() - 1
        )
    )
    df["rolling_20d_volatility"] = (
        df.groupby("ticker")["daily_return"]
        .transform(lambda s: s.rolling(20).std(ddof=0) * np.sqrt(252))
    )
    return df


def clean_fundamentals_data(path: Path | str = DEFAULT_FUNDAMENTALS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[
        (df["costat"] == "A")
        & (df["curcd"] == "USD")
        & (df["datafmt"] == "STD")
        & (df["consol"] == "C")
        & (df["indfmt"] == "INDL")
    ].copy()

    numeric_cols = ["fyr", "at", "lt", "ib", "ni", "sale"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["datadate"] = pd.to_datetime(df["datadate"])
    df = df.rename(
        columns={
            "tic": "ticker",
            "conm": "company_name",
            "gvkey": "gvkey",
            "datadate": "fiscal_date",
            "at": "total_assets",
            "lt": "total_liabilities",
            "ib": "income_before_extras",
            "ni": "net_income",
            "sale": "sales",
            "fyr": "fiscal_year_end_month",
        }
    )
    df["year"] = df["fiscal_date"].dt.year
    df["leverage_ratio"] = df["total_liabilities"] / df["total_assets"]
    df["roa"] = df["net_income"] / df["total_assets"]
    df["net_profit_margin"] = df["net_income"] / df["sales"]
    df["asset_turnover"] = df["sales"] / df["total_assets"]
    df["company_short"] = df["ticker"].map(COMPANY_LABELS)
    df = df.sort_values(["ticker", "fiscal_date"]).reset_index(drop=True)
    return df[
        [
            "ticker",
            "company_name",
            "company_short",
            "fiscal_date",
            "year",
            "gvkey",
            "fiscal_year_end_month",
            "total_assets",
            "total_liabilities",
            "income_before_extras",
            "net_income",
            "sales",
            "leverage_ratio",
            "roa",
            "net_profit_margin",
            "asset_turnover",
        ]
    ]


def build_company_year_panel(stock_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
    grouped = stock_df.groupby(["ticker", "year"], sort=True)
    summary = (
        grouped.agg(
            company_name=("company_name", "first"),
            trading_days=("date", "size"),
            start_price=("price", "first"),
            end_price=("price", "last"),
            cumulative_return=("daily_return", lambda s: (1 + s).prod() - 1),
            average_daily_return=("daily_return", "mean"),
            daily_return_volatility=("daily_return", lambda s: s.std(ddof=0)),
            positive_return_ratio=("daily_return", lambda s: (s > 0).mean()),
            best_day_return=("daily_return", "max"),
            worst_day_return=("daily_return", "min"),
        )
        .reset_index()
    )
    max_drawdown = grouped["daily_return"].apply(_max_drawdown).reset_index(name="max_drawdown")
    summary = summary.merge(max_drawdown, on=["ticker", "year"], how="left")
    summary["annualized_volatility"] = summary["daily_return_volatility"] * np.sqrt(252)
    summary["company_short"] = summary["ticker"].map(COMPANY_LABELS)

    merged = summary.merge(
        fundamentals_df,
        on=["ticker", "year"],
        how="left",
        suffixes=("_stock", "_fundamentals"),
    )
    merged["return_vs_roa_gap"] = merged["cumulative_return"] - merged["roa"]
    merged["return_rank"] = merged.groupby("year")["cumulative_return"].rank(
        ascending=False, method="dense"
    )
    merged["roa_rank"] = merged.groupby("year")["roa"].rank(
        ascending=False, method="dense"
    )
    merged = merged.sort_values(["year", "ticker"]).reset_index(drop=True)
    return merged


def build_monthly_returns(stock_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        stock_df.groupby(["ticker", "month"], as_index=False)
        .agg(
            company_name=("company_name", "first"),
            year=("year", "first"),
            month_name=("month_name", "first"),
            monthly_return=("daily_return", lambda s: (1 + s).prod() - 1),
            avg_price=("price", "mean"),
        )
        .sort_values(["ticker", "month"])
        .reset_index(drop=True)
    )
    return monthly


def build_outputs(
    stock_path: Path | str = DEFAULT_STOCK_PATH,
    fundamentals_path: Path | str = DEFAULT_FUNDAMENTALS_PATH,
) -> OutputBundle:
    daily_stock = clean_stock_data(stock_path)
    fundamentals = clean_fundamentals_data(fundamentals_path)
    company_year = build_company_year_panel(daily_stock, fundamentals)
    monthly_returns = build_monthly_returns(daily_stock)
    return OutputBundle(
        daily_stock=daily_stock,
        monthly_returns=monthly_returns,
        fundamentals=fundamentals,
        company_year=company_year,
    )


def build_daily_stock_table(stock_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "year",
        "month",
        "ticker",
        "company_name",
        "permno",
        "price",
        "daily_return",
        "normalized_price",
        "cumulative_return_index",
        "drawdown",
        "rolling_20d_volatility",
    ]
    return stock_df[columns].copy()


def _style_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linestyle="--")


def build_visual_assets(bundle: OutputBundle, figures_dir: Path | str = DEFAULT_FIGURES_DIR) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    daily = bundle.daily_stock
    panel = bundle.company_year
    monthly = bundle.monthly_returns
    fundamentals = bundle.fundamentals

    fig, ax = plt.subplots(figsize=(12, 6))
    for ticker in TICKER_ORDER:
        subset = daily[daily["ticker"] == ticker]
        ax.plot(
            subset["date"],
            subset["normalized_price"],
            label=ticker,
            linewidth=2.2,
            color=TICKER_COLORS[ticker],
        )
    _style_axes(ax, "Normalized Stock Price Index (Base = 100)", "Index Level")
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(figures_dir / "normalized_price_index.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(
        data=panel,
        x="ticker",
        y="cumulative_return",
        hue="year",
        palette=["#1E40AF", "#3B82F6"],
        ax=axes[0],
    )
    _style_axes(axes[0], "Annual Cumulative Return by Company", "Cumulative Return")
    axes[0].legend(title="Year", frameon=False)

    sns.barplot(
        data=panel,
        x="ticker",
        y="annualized_volatility",
        hue="year",
        palette=["#0F766E", "#F59E0B"],
        ax=axes[1],
    )
    _style_axes(axes[1], "Annualized Volatility by Company", "Annualized Volatility")
    axes[1].legend(title="Year", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "annual_return_and_volatility.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    latest_months = monthly[monthly["year"] == 2024].copy()
    latest_months["month_dt"] = pd.to_datetime(latest_months["month"] + "-01")
    for ticker in TICKER_ORDER:
        subset = latest_months[latest_months["ticker"] == ticker]
        axes[0].plot(
            subset["month_dt"],
            subset["monthly_return"],
            marker="o",
            linewidth=2,
            label=ticker,
            color=TICKER_COLORS[ticker],
        )
    _style_axes(axes[0], "Monthly Return Trend in 2024", "Monthly Return")
    axes[0].legend(frameon=False, ncol=2)

    sns.barplot(
        data=panel,
        x="ticker",
        y="max_drawdown",
        hue="year",
        palette=["#1E40AF", "#94A3B8"],
        ax=axes[1],
    )
    _style_axes(axes[1], "Maximum Drawdown by Company", "Max Drawdown")
    axes[1].legend(title="Year", frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "monthly_returns_and_drawdown.png", dpi=180)
    plt.close(fig)

    latest_fundamentals = fundamentals[fundamentals["year"] == 2024].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.barplot(
        data=latest_fundamentals,
        x="ticker",
        y="sales",
        hue="ticker",
        palette=TICKER_COLORS,
        legend=False,
        ax=axes[0],
    )
    _style_axes(axes[0], "2024 Sales by Company", "Sales")
    sns.barplot(
        data=latest_fundamentals,
        x="ticker",
        y="roa",
        hue="ticker",
        palette=TICKER_COLORS,
        legend=False,
        ax=axes[1],
    )
    _style_axes(axes[1], "2024 Return on Assets by Company", "ROA")
    fig.tight_layout()
    fig.savefig(figures_dir / "fundamentals_2024.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(
        data=panel,
        x="roa",
        y="cumulative_return",
        hue="ticker",
        style="year",
        s=150,
        palette=TICKER_COLORS,
        ax=axes[0],
    )
    _style_axes(axes[0], "Return vs ROA", "Cumulative Return")
    axes[0].legend(frameon=False)
    sns.scatterplot(
        data=panel,
        x="net_profit_margin",
        y="cumulative_return",
        hue="ticker",
        style="year",
        s=150,
        palette=TICKER_COLORS,
        ax=axes[1],
    )
    _style_axes(axes[1], "Return vs Net Profit Margin", "Cumulative Return")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "returns_vs_fundamentals.png", dpi=180)
    plt.close(fig)


def export_project_artifacts(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    figures_dir: Path | str = DEFAULT_FIGURES_DIR,
    stock_path: Path | str = DEFAULT_STOCK_PATH,
    fundamentals_path: Path | str = DEFAULT_FUNDAMENTALS_PATH,
) -> OutputBundle:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_outputs(stock_path=stock_path, fundamentals_path=fundamentals_path)
    build_daily_stock_table(bundle.daily_stock).to_csv(
        processed_dir / "daily_stock_clean.csv", index=False
    )
    bundle.monthly_returns.to_csv(processed_dir / "monthly_returns.csv", index=False)
    bundle.fundamentals.to_csv(processed_dir / "fundamentals_clean.csv", index=False)
    bundle.company_year.to_csv(processed_dir / "company_year_panel.csv", index=False)
    build_visual_assets(bundle, figures_dir=figures_dir)
    return bundle


def load_project_outputs(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> Dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    outputs = {
        "daily_stock": pd.read_csv(processed_dir / "daily_stock_clean.csv", parse_dates=["date"]),
        "monthly_returns": pd.read_csv(processed_dir / "monthly_returns.csv"),
        "fundamentals": pd.read_csv(
            processed_dir / "fundamentals_clean.csv", parse_dates=["fiscal_date"]
        ),
        "company_year": pd.read_csv(processed_dir / "company_year_panel.csv"),
    }
    return outputs
