# %% [markdown]
# # Strategy-by-Strategy Quantitative Portfolio Analysis (Dynasto Baseline + Enhancements)
#
# This notebook-ready Python pipeline reproduces the baseline methodology from
# `AnalisisPortafolioDynasto.ipynb` for each strategy CSV separately, then extends it with:
#
# 1. **Monte Carlo stress testing** on historical trade profits.
# 2. **Monthly returns heatmaps** (seasonality view).
# 3. **Annualized monthly return analysis** using the same fixed-capital logic used in the baseline notebook.
#
# ---
#
# ## Mathematical notes
#
# Let:
# - \( E_t \): equity after trade \( t \)
# - \( P_t \): trade profit for trade \( t \)
# - \( E_0 \): initial capital (default: 10,000)
#
# Then:
# - **Trade profit extraction** (baseline-consistent):
#   \[
#   P_t = E_t - E_{t-1}, \quad P_1 = E_1 - E_0
#   \]
# - **Equity reconstruction**:
#   \[
#   \hat{E}_t = E_0 + \sum_{i=1}^{t} P_i
#   \]
# - **Drawdown**:
#   \[
#   DD_t = \frac{E_t - \max_{i \le t}(E_i)}{\max_{i \le t}(E_i)}
#   \]
# - **Maximum Drawdown**:
#   \[
#   \mathrm{MaxDD} = \min_t DD_t
#   \]
#
# **Monte Carlo**:
# - Keep the exact empirical set of historical trade profits.
# - Randomly permute order \( N \) times (sequence risk stress test).
# - Build \( N \) equity paths with cumulative sums.
# - Estimate confidence envelopes (e.g., 5th/50th/95th percentiles) and worst simulated drawdown.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="darkgrid", context="talk")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150


@dataclass
class AnalysisConfig:
    initial_capital: float = 10000.0
    monte_carlo_sims: int = 10000
    monte_carlo_seed: int = 69
    risk_of_ruin_dd_pct: float = -50.0
    monthly_resample_freq: str = "ME"


# %%
def _find_header_row(file_path: Path) -> Tuple[int, str]:
    """Replicate baseline logic: detect where headers start + encoding fallback."""
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            with file_path.open("r", encoding=enc, errors="ignore") as f:
                for i, line in enumerate(f):
                    if (
                        "Hora de apertura" in line
                        or "Open Time" in line
                        or "<DATE>" in line
                    ):
                        return i, enc
        except Exception:
            continue
    return 0, "utf-8"


def _parse_mt5_concatenated_single_column(
    raw_series: pd.Series,
) -> pd.DataFrame:
    """
    Parse MT5-style rows that were exported without visible delimiters, e.g.
    '2021.01.04 18:3010024.709975.370.0000'
    """
    pattern = re.compile(
        r"^(?P<DATE>\d{4}\.\d{2}\.\d{2}\s\d{2}:\d{2})"
        r"(?P<BALANCE>-?\d+\.\d{2})"
        r"(?P<EQUITY>-?\d+\.\d{2})"
        r"(?P<DEPOSIT_LOAD>-?\d+\.\d{4})$"
    )

    parsed_records: List[Dict[str, str]] = []
    for value in raw_series.astype(str):
        line = value.strip()
        if not line or line.startswith("<DATE>"):
            continue
        m = pattern.match(line)
        if m is None:
            continue
        parsed_records.append(m.groupdict())

    if not parsed_records:
        raise ValueError(
            "Could not parse MT5 concatenated format from single-column CSV."
        )
    return pd.DataFrame(parsed_records)


def load_strategy_csv(file_path: Path) -> pd.DataFrame:
    """
    Baseline-consistent ingestion and cleaning:
    - Header row detection
    - Encoding fallback
    - Drop empty columns
    - Robust fallback parser for single concatenated column exports
    """
    header_row, enc = _find_header_row(file_path)

    # First try baseline-style tab-separated import.
    read_attempts = [
        {"sep": "\t", "encoding": enc},
        {"sep": ",", "encoding": enc},
        {"sep": None, "encoding": enc, "engine": "python"},
    ]

    last_error = None
    df = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(file_path, skiprows=header_row, **kwargs)
            df = df.dropna(axis=1, how="all")
            if not df.empty:
                break
        except Exception as e:
            last_error = e
            continue

    if df is None or df.empty:
        raise ValueError(
            f"Failed to load {file_path.name}. Last error: {last_error}"
        )

    # If export is malformed into one giant column, parse with regex.
    if "<DATE>" not in df.columns and df.shape[1] == 1:
        df = _parse_mt5_concatenated_single_column(df.iloc[:, 0])

    # Normalize columns from baseline style.
    rename_map = {
        "<DATE>": "DATE",
        "<BALANCE>": "BALANCE",
        "<EQUITY>": "EQUITY",
        "<DEPOSIT LOAD>": "DEPOSIT_LOAD",
    }
    df = df.rename(columns=rename_map)

    required = {"DATE", "BALANCE"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{file_path.name} missing required columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Exact baseline date parsing intent.
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["BALANCE"] = (
        df["BALANCE"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["BALANCE"] = pd.to_numeric(df["BALANCE"], errors="coerce")

    if "EQUITY" in df.columns:
        df["EQUITY"] = (
            df["EQUITY"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["EQUITY"] = pd.to_numeric(df["EQUITY"], errors="coerce")

    if "DEPOSIT_LOAD" in df.columns:
        df["DEPOSIT_LOAD"] = (
            df["DEPOSIT_LOAD"]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["DEPOSIT_LOAD"] = pd.to_numeric(df["DEPOSIT_LOAD"], errors="coerce")

    df = df.dropna(subset=["DATE", "BALANCE"]).sort_values("DATE").reset_index(drop=True)
    return df


# %%
def build_trade_series(
    df: pd.DataFrame, initial_capital: float
) -> pd.DataFrame:
    """Replicate baseline trade profit extraction + cumulative equity and drawdown."""
    out = df.copy()

    # Baseline logic: Trade_Profit from BALANCE diff and custom first row.
    out["Trade_Profit"] = out["BALANCE"].diff()
    out.loc[out.index[0], "Trade_Profit"] = out.loc[out.index[0], "BALANCE"] - initial_capital

    out["Equity"] = initial_capital + out["Trade_Profit"].cumsum()
    out["Peak_Equity"] = out["Equity"].cummax()
    out["Drawdown_USD"] = out["Equity"] - out["Peak_Equity"]
    out["Drawdown_Pct"] = (out["Drawdown_USD"] / out["Peak_Equity"]) * 100.0

    # Trade-by-trade return (for extra diagnostics, not replacing Trade_Profit logic).
    equity_prev = out["Equity"].shift(1).fillna(initial_capital)
    out["Trade_Return_Pct"] = np.where(
        equity_prev != 0, (out["Trade_Profit"] / equity_prev) * 100.0, np.nan
    )
    return out


def compute_core_metrics(
    trades: pd.DataFrame, initial_capital: float
) -> Dict[str, float]:
    gross_profit = trades.loc[trades["Trade_Profit"] > 0, "Trade_Profit"].sum()
    gross_loss = -trades.loc[trades["Trade_Profit"] < 0, "Trade_Profit"].sum()
    total_return_usd = trades["Equity"].iloc[-1] - initial_capital
    total_return_pct = (total_return_usd / initial_capital) * 100.0
    max_dd_pct = trades["Drawdown_Pct"].min()
    max_dd_usd = trades["Drawdown_USD"].min()
    wins = (trades["Trade_Profit"] > 0).sum()
    total = len(trades)
    win_rate = (wins / total) * 100.0 if total else np.nan
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.inf
    recovery_factor = (
        total_return_usd / abs(max_dd_usd) if abs(max_dd_usd) > 0 else np.inf
    )

    return {
        "Initial_Capital": initial_capital,
        "Final_Equity": trades["Equity"].iloc[-1],
        "Total_Return_USD": total_return_usd,
        "Total_Return_Pct": total_return_pct,
        "Maximum_Drawdown_Pct": max_dd_pct,
        "Maximum_Drawdown_USD": max_dd_usd,
        "Win_Rate_Pct": win_rate,
        "Profit_Factor": profit_factor,
        "Recovery_Factor": recovery_factor,
        "Number_of_Trades": total,
    }


# %%
def run_monte_carlo(
    trades: pd.DataFrame,
    cfg: AnalysisConfig,
) -> Dict[str, np.ndarray | float | int]:
    """
    Monte Carlo sequence-risk simulation:
    same empirical profits, randomized order, many trajectories.
    """
    rng = np.random.default_rng(cfg.monte_carlo_seed)
    profits = trades["Trade_Profit"].to_numpy(dtype=float)
    n_trades = profits.size
    n_sims = cfg.monte_carlo_sims

    # Vectorized permutation index generation (each row is one random permutation).
    random_keys = rng.random((n_sims, n_trades))
    perm_idx = np.argsort(random_keys, axis=1)
    sim_matrix = profits[perm_idx]

    equity_matrix = cfg.initial_capital + np.cumsum(sim_matrix, axis=1)
    peaks = np.maximum.accumulate(equity_matrix, axis=1)
    drawdowns = ((equity_matrix - peaks) / peaks) * 100.0
    max_dds = drawdowns.min(axis=1)

    lower = np.percentile(equity_matrix, 5, axis=0)
    median = np.percentile(equity_matrix, 50, axis=0)
    upper = np.percentile(equity_matrix, 95, axis=0)

    final_capitals = equity_matrix[:, -1]
    prob_profit = (final_capitals > cfg.initial_capital).mean() * 100.0
    risk_of_ruin = (max_dds <= cfg.risk_of_ruin_dd_pct).mean() * 100.0

    idx_worst_dd = int(np.argmin(max_dds))
    idx_best_final = int(np.argmax(final_capitals))

    return {
        "equity_matrix": equity_matrix,
        "max_dds": max_dds,
        "lower_bound": lower,
        "median_bound": median,
        "upper_bound": upper,
        "probability_profit_pct": prob_profit,
        "risk_of_ruin_pct": risk_of_ruin,
        "worst_dd_pct": max_dds[idx_worst_dd],
        "best_final_capital": final_capitals[idx_best_final],
        "median_final_capital": np.median(final_capitals),
        "idx_worst_dd": idx_worst_dd,
        "idx_best_final": idx_best_final,
    }


# %%
def compute_monthly_tables(
    trades: pd.DataFrame,
    initial_capital: float,
    freq: str = "ME",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      1) Dynamic-capital monthly table (for heatmap seasonality)
      2) Fixed-capital monthly table (baseline final-section style)
    """
    temp = trades.set_index("DATE").sort_index()

    monthly = temp.resample(freq).agg(
        Profit_Monthly_USD=("Trade_Profit", "sum"),
        Max_DD_Monthly_Pct=("Drawdown_Pct", "min"),
        Max_DD_Monthly_USD=("Drawdown_USD", "min"),
        Equity_End=("Equity", "last"),
    )

    monthly["Equity_Start"] = monthly["Equity_End"].shift(1).fillna(initial_capital)
    monthly["Return_Monthly_Pct_Dynamic"] = (
        monthly["Profit_Monthly_USD"] / monthly["Equity_Start"]
    ) * 100.0

    monthly = monthly.dropna(subset=["Profit_Monthly_USD"]).reset_index()
    monthly["Year"] = monthly["DATE"].dt.year
    monthly["Month"] = monthly["DATE"].dt.month
    monthly["Month_Name"] = monthly["DATE"].dt.strftime("%b")

    # Fixed-capital annualized monthly analysis (baseline notebook ending logic).
    fixed = monthly.copy()
    fixed["Return_Monthly_Pct_Fixed"] = (
        fixed["Profit_Monthly_USD"] / initial_capital
    ) * 100.0
    fixed["DD_Pct_Fixed"] = (fixed["Max_DD_Monthly_USD"] / initial_capital) * 100.0

    fixed = fixed.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Return_Monthly_Pct_Fixed", "DD_Pct_Fixed"]
    )

    return monthly, fixed


# %%
def plot_equity_and_underwater(
    trades: pd.DataFrame, strategy_name: str, output_dir: Path
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    axes[0].plot(trades["DATE"], trades["Equity"], color="#1f77b4", linewidth=2.2)
    axes[0].set_title(f"{strategy_name} | Equity Curve", fontsize=16, weight="bold")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(
        trades["DATE"],
        trades["Drawdown_Pct"],
        0,
        color="#d62728",
        alpha=0.35,
        linewidth=0,
    )
    axes[1].plot(trades["DATE"], trades["Drawdown_Pct"], color="#d62728", linewidth=1.2)
    axes[1].set_title("Underwater (Drawdown %)", fontsize=13)
    axes[1].set_ylabel("DD (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_equity_underwater.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_monte_carlo(
    trades: pd.DataFrame,
    mc: Dict[str, np.ndarray | float | int],
    strategy_name: str,
    output_dir: Path,
    n_paths_to_draw: int = 120,
) -> None:
    equity_matrix = mc["equity_matrix"]  # type: ignore[index]
    lower = mc["lower_bound"]  # type: ignore[index]
    median = mc["median_bound"]  # type: ignore[index]
    upper = mc["upper_bound"]  # type: ignore[index]
    idx_worst = mc["idx_worst_dd"]  # type: ignore[index]

    x = np.arange(equity_matrix.shape[1])
    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw a subset of paths for readability.
    step = max(1, equity_matrix.shape[0] // n_paths_to_draw)
    sample_idx = np.arange(0, equity_matrix.shape[0], step)
    for i in sample_idx:
        ax.plot(x, equity_matrix[i], color="#7f8c8d", alpha=0.08, linewidth=0.7)

    ax.fill_between(x, lower, upper, color="#00bcd4", alpha=0.22, label="90% Confidence Zone")
    ax.plot(x, median, color="#17becf", linewidth=2.2, label="Median Path (P50)")
    ax.plot(x, equity_matrix[idx_worst], color="#e74c3c", linewidth=1.8, label="Worst DD Path")
    ax.plot(x, trades["Equity"].to_numpy(), color="white", linewidth=2.4, label="Historical Path")

    title = (
        f"{strategy_name} | Monte Carlo ({equity_matrix.shape[0]:,} simulations)\n"
        f"P(Profit): {mc['probability_profit_pct']:.1f}% | "
        f"Risk of Ruin (DD <= -50%): {mc['risk_of_ruin_pct']:.2f}% | "
        f"Worst Sim DD: {mc['worst_dd_pct']:.2f}%"
    )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Equity (USD)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_monte_carlo.png", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_monthly_heatmaps(
    monthly_dynamic: pd.DataFrame,
    monthly_fixed: pd.DataFrame,
    strategy_name: str,
    output_dir: Path,
) -> None:
    heat_dynamic = monthly_dynamic.pivot(
        index="Year",
        columns="Month",
        values="Return_Monthly_Pct_Dynamic",
    )
    heat_fixed = monthly_fixed.pivot(
        index="Year",
        columns="Month",
        values="Return_Monthly_Pct_Fixed",
    )
    heat_dynamic = heat_dynamic.reindex(columns=list(range(1, 13)))
    heat_fixed = heat_fixed.reindex(columns=list(range(1, 13)))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    sns.heatmap(
        heat_dynamic,
        ax=axes[0],
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="black",
        cbar_kws={"label": "Monthly Return (%)"},
    )
    axes[0].set_title(
        f"{strategy_name} | Monthly Return Heatmap (Dynamic Capital)",
        fontsize=14,
        weight="bold",
    )
    axes[0].set_ylabel("Year")
    axes[0].set_xlabel("")
    axes[0].set_xticklabels(month_labels, rotation=0)

    sns.heatmap(
        heat_fixed,
        ax=axes[1],
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="black",
        cbar_kws={"label": "Monthly Return (%) vs fixed 10,000 USD"},
    )
    axes[1].set_title(
        f"{strategy_name} | Monthly Return Heatmap (Fixed Base = 10,000 USD)",
        fontsize=14,
        weight="bold",
    )
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Year")
    axes[1].set_xticklabels(month_labels, rotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / f"{strategy_name}_monthly_heatmaps.png", bbox_inches="tight")
    heat_fixed.to_csv(
        output_dir / f"{strategy_name}_monthly_heatmap_fixed_base_10000.csv",
        index=True,
    )
    plt.show()
    plt.close(fig)


def plot_annualized_monthly_analysis(
    monthly_fixed: pd.DataFrame, strategy_name: str, output_dir: Path
) -> None:
    """
    Improved clarity chart:
    - Only monthly histogram (USD result).
    - Labels per month with USD and % result.
    - DD is kept on fixed 10K base and summarized as mean/worst values in subtitle.
    """
    years = sorted(monthly_fixed["Year"].unique())
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    for year in years:
        ydf = monthly_fixed[monthly_fixed["Year"] == year].copy()
        ydf["Month_Name"] = pd.Categorical(ydf["Month_Name"], categories=month_order, ordered=True)
        ydf = ydf.sort_values("Month_Name")
        if ydf.empty:
            continue

        x = np.arange(len(ydf))
        month_labels = ydf["Month_Name"].astype(str).to_list()
        profit_usd = ydf["Profit_Monthly_USD"].to_numpy()
        ret_pct_fixed = ydf["Return_Monthly_Pct_Fixed"].to_numpy()
        dd_pct_fixed = ydf["DD_Pct_Fixed"].to_numpy()
        dd_usd = ydf["Max_DD_Monthly_USD"].to_numpy()

        mean_profit_usd = float(np.mean(profit_usd))
        mean_ret_pct = float(np.mean(ret_pct_fixed))
        mean_dd_pct = float(np.mean(dd_pct_fixed))
        worst_dd_pct = float(np.min(dd_pct_fixed))
        worst_dd_usd = float(np.min(dd_usd))
        annual_profit = float(np.sum(profit_usd))

        colors = np.where(profit_usd >= 0, "#2ecc71", "#e74c3c")

        fig, ax1 = plt.subplots(figsize=(15, 7))
        bars = ax1.bar(
            x,
            profit_usd,
            color=colors,
            alpha=0.9,
            edgecolor="#dfe6e9",
            linewidth=0.8,
            label="Monthly Profit (USD)",
        )
        ax1.axhline(0, color="#bdc3c7", linewidth=1.0)
        ax1.set_ylabel("Monthly Result (USD)", color="#ecf0f1", fontsize=12, weight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(month_labels, fontsize=11, weight="bold")
        ax1.grid(True, axis="y", alpha=0.22, linestyle="--")

        max_abs = max(abs(float(np.max(profit_usd))), abs(float(np.min(profit_usd))), 1.0)
        y_margin = max_abs * 0.28
        ax1.set_ylim(float(np.min(profit_usd)) - y_margin, float(np.max(profit_usd)) + y_margin)

        for i, b in enumerate(bars):
            month_label = f"USD {profit_usd[i]:+,.0f}\n({ret_pct_fixed[i]:+.2f}%)"
            y = b.get_height()
            y_offset = max_abs * 0.03 if y >= 0 else -(max_abs * 0.03)
            ax1.text(
                b.get_x() + b.get_width() / 2.0,
                y + y_offset,
                month_label,
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9.5,
                fontweight="bold",
                color="#ecf0f1",
            )

        plt.title(
            f"{strategy_name} | Annualized Monthly Analysis - {year}\n"
            f"Annual Profit: USD {annual_profit:,.2f} | Mean Return: {mean_ret_pct:+.2f}% | "
            f"Mean DD (fixed 10K): {mean_dd_pct:.2f}% | Worst Monthly DD: {worst_dd_pct:.2f}% (USD {worst_dd_usd:,.2f})",
            fontsize=13,
            weight="bold",
        )

        ax1.legend(loc="upper left", frameon=True)

        fig.tight_layout()
        fig.savefig(
            output_dir / f"{strategy_name}_annualized_monthly_{year}.png",
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)


# %% [markdown]
# ## Orchestration
#
# `run_full_analysis` scans all CSV files in a folder, processes each strategy independently,
# computes metrics, runs Monte Carlo, and generates all requested visual artifacts.

# %%
def analyze_one_strategy(
    csv_path: Path,
    output_root: Path,
    cfg: AnalysisConfig,
) -> Dict[str, float]:
    strategy_name = csv_path.stem
    strategy_out = output_root / strategy_name
    strategy_out.mkdir(parents=True, exist_ok=True)

    raw = load_strategy_csv(csv_path)
    trades = build_trade_series(raw, cfg.initial_capital)
    metrics = compute_core_metrics(trades, cfg.initial_capital)
    mc = run_monte_carlo(trades, cfg)
    monthly_dynamic, monthly_fixed = compute_monthly_tables(
        trades, cfg.initial_capital, cfg.monthly_resample_freq
    )

    plot_equity_and_underwater(trades, strategy_name, strategy_out)
    plot_monte_carlo(trades, mc, strategy_name, strategy_out)
    plot_monthly_heatmaps(monthly_dynamic, monthly_fixed, strategy_name, strategy_out)
    plot_annualized_monthly_analysis(monthly_fixed, strategy_name, strategy_out)

    # Persist useful artifacts.
    trades.to_csv(strategy_out / f"{strategy_name}_processed_trades.csv", index=False)
    monthly_dynamic.to_csv(strategy_out / f"{strategy_name}_monthly_dynamic.csv", index=False)
    monthly_fixed.to_csv(strategy_out / f"{strategy_name}_monthly_fixed.csv", index=False)
    pd.DataFrame([metrics]).to_csv(strategy_out / f"{strategy_name}_metrics.csv", index=False)

    return metrics


def run_full_analysis(
    data_dir: str | Path,
    output_dir: str | Path = "analysis_outputs",
    cfg: AnalysisConfig = AnalysisConfig(),
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    summary_rows = []
    for csv_path in csv_files:
        print(f"\nProcessing strategy: {csv_path.stem}")
        metrics = analyze_one_strategy(csv_path, output_dir, cfg)
        metrics["Strategy"] = csv_path.stem
        summary_rows.append(metrics)

    summary = pd.DataFrame(summary_rows)
    summary = summary[
        [
            "Strategy",
            "Initial_Capital",
            "Final_Equity",
            "Total_Return_USD",
            "Total_Return_Pct",
            "Maximum_Drawdown_Pct",
            "Maximum_Drawdown_USD",
            "Win_Rate_Pct",
            "Profit_Factor",
            "Recovery_Factor",
            "Number_of_Trades",
        ]
    ].sort_values("Total_Return_Pct", ascending=False)

    summary.to_csv(output_dir / "all_strategies_summary.csv", index=False)
    return summary


# %% [markdown]
# ## Execution cell
# Update `DATA_DIRECTORY` if needed, then run.

# %%
DATA_DIRECTORY = Path(r"c:\Users\juana\Documents\claude_sandbox\historico_trades")
OUTPUT_DIRECTORY = Path(r"c:\Users\juana\Documents\claude_sandbox\historico_trades\analysis_outputs")

config = AnalysisConfig(
    initial_capital=10000.0,
    monte_carlo_sims=10000,
    monte_carlo_seed=69,
    risk_of_ruin_dd_pct=-50.0,
)

if __name__ == "__main__":
    summary_df = run_full_analysis(DATA_DIRECTORY, OUTPUT_DIRECTORY, config)
    print(summary_df.to_string(index=False))
