# S&P 500 Representative Stocks Performance and Financial Fundamentals Analysis

## Project Overview

This project builds an English-only small data product for an academic audience using Python, daily stock market data, and annual financial fundamentals. The analysis focuses on four representative S&P 500 companies: Apple (`AAPL`), Microsoft (`MSFT`), JPMorgan Chase (`JPM`), and Walmart (`WMT`).

The project answers one central question:

> How do four representative S&P 500 firms differ in stock performance, risk profile, and financial fundamentals during 2023 and 2024?

The final output is not just raw code. It includes a reproducible Python workflow, processed analytical datasets, figures, notebooks, and an interactive Streamlit dashboard.

## Dataset Scope

The raw files come from the broader **S&P 500 Stock Performance & Financial Fundamental Analysis** dataset, but this project only uses a focused subset:

- `data/data.csv`: daily stock price and return data for AAPL, MSFT, JPM, and WMT from **2023-01-03** to **2024-12-31**
- `data/data2.csv`: annual financial fundamentals for the same four firms for **2023** and **2024**

This is a four-company teaching sample, not a full-market S&P 500 study.

## Why This Project Fits the Assignment

This submission directly satisfies the general assignment requirements:

- It uses a business- and finance-related dataset.
- It performs substantial Python work: cleaning, transformation, descriptive analytics, visualization, and data product creation.
- It defines a clear question and a clear audience.
- It delivers an end product that communicates value beyond code.
- It keeps the analysis focused and understandable instead of artificially overcomplicating the task.

## Analytical Workflow

### 1. Data Cleaning

- Parse daily dates and fiscal dates
- Convert price, return, asset, liability, sales, and income fields to numeric types
- Remove duplicate and incomplete fundamentals rows for JPM by keeping the `INDL` records and excluding the `FS` duplicates

### 2. Feature Engineering

For stock performance:

- normalized price index
- compounded cumulative return
- average daily return
- annualized volatility
- maximum drawdown
- positive-return day ratio
- monthly return summary

For fundamentals:

- total assets
- total liabilities
- net income
- sales
- leverage ratio
- return on assets
- net profit margin
- asset turnover

### 3. Integration

The project aggregates daily stock information to the company-year level and merges it with annual fundamentals using `ticker + year`.

This alignment is intentionally simple and transparent for coursework. Because fiscal years do not always match calendar years exactly, the README, dashboard, and reflection report all state this limitation clearly.

## Project Structure

```text
.
├── app.py
├── data
│   ├── data.csv
│   ├── data2.csv
│   └── processed
├── notebooks
├── reports
│   ├── figures
│   ├── presentation_script.md
│   ├── reflection_report.md
│   └── skill_profiles
├── references
├── scripts
└── src
    └── sp500_insights
```

## Dashboard Sections

The Streamlit dashboard is structured into five tabs:

1. `Overview`
2. `Stock Performance`
3. `Risk Profile`
4. `Financial Fundamentals`
5. `Insights and Limitations`

The interface supports:

- company multi-select
- year filter
- view switch between price trend and return performance

## Key Findings

The specific numerical results are generated directly from the processed data, but the expected analytical storyline is:

- Microsoft was the strongest return performer in 2023.
- Walmart delivered the strongest cumulative return in 2024.
- Walmart showed the lowest volatility profile in the four-company sample.
- Differences in profitability, leverage, and asset efficiency help contextualize why stock outcomes diverged across firms.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate processed datasets and figures

```bash
python scripts/build_project_assets.py
```

### 3. Optionally generate raw data profiles using the local CSV skill

```bash
python scripts/run_skill_profiles.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

## Local Skill Usage

This project intentionally uses the local `G:\\tmp\\skill` resources:

- `csv-data-summarizer-claude-skill`: initial raw CSV profiling
- `ui-ux-pro-max-skill`: dashboard color, typography, and layout direction
- `cookiecutter-data-science`: project structure reference

## Limitations

- The project covers only four firms, not the full S&P 500.
- The fundamentals file contains only two annual observations per firm.
- Fiscal-year alignment is simplified to calendar year merging.
- The project is descriptive and exploratory, not predictive.

## Future Extensions

- Add a market benchmark such as the S&P 500 index
- Expand to more firms and more years
- Add richer fundamentals such as EBITDA, cash flow, or valuation ratios
- Test whether cross-sectional relationships remain stable with a larger sample
