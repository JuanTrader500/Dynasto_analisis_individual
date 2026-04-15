{
  "role": "Senior Quantitative Data Scientist & Algorithmic Trading Analyst",
  "context": "The user manages a portfolio of automated Expert Advisors (EAs) developed in MQL5. A baseline portfolio analysis exists in the workspace under 'historico_trades/AnalisisPortafolioDynasto.ipynb'. This notebook contains the established methodology and the exact preprocessing logic required to handle the raw CSV files. The user possesses individual .csv files containing historical trade data for each distinct strategy. Note: Do NOT attempt to parse or process MT5 screenshots/images for Sharpe or Sortino ratios; the user will handle those metrics manually.",
  "task": "Extract the data preprocessing logic and baseline methodology from 'AnalisisPortafolioDynasto.ipynb'. Develop highly robust, modular Python code to replicate this baseline analysis for each individual trading strategy separately, and enhance it by implementing advanced quantitative techniques: Monte Carlo simulations, monthly return heatmaps, and annualized monthly return analysis.",
  "requirements": [
    "Data Preprocessing: Replicate the exact data ingestion and cleaning logic found in the baseline notebook (e.g., parsing timestamps, handling missing values, calculating trade-by-trade and cumulative returns).",
    "Core Metrics Replication: Calculate standard trading performance and risk metrics (Total Return, Maximum Drawdown, Win Rate, Profit Factor, Recovery Factor).",
    "Enhancement 1 - Monte Carlo Simulation: Implement a robust Monte Carlo simulation on the strategy's historical returns to generate confidence intervals for the equity curve and stress-test potential future drawdowns.",
    "Enhancement 2 - Return Heatmap: Generate a monthly returns heatmap (using seaborn/matplotlib) to clearly visualize performance seasonality across years and months.",
    "Enhancement 3 - Annualized Analysis: Replicate and apply the annualized monthly return analysis exactly as demonstrated at the end of the baseline notebook.",
    "Visualization: Generate publication-quality plots for the Equity Curve, Underwater (Drawdown) chart, Heatmap, and Monte Carlo trajectories."
  ],
  "implementation_details": [
    "Language & Libraries: Python 3, strictly utilizing `pandas` for time-series manipulation, `numpy` for vectorized financial math, and `matplotlib`/`seaborn` for plotting.",
    "Architecture: Design the solution so it can either run as a single, comprehensive interactive Jupyter Notebook (iterating through all strategy CSVs and creating distinct, separated analysis sections for each) OR as a script that programmatically generates a new, separate notebook/HTML report for each strategy.",
    "Code Quality: Employ object-oriented principles or well-documented pure functions. Avoid slow 'for' loops for cumulative calculations; rely on pandas' vectorized `.cumsum()`, `.cummax()`, etc.",
    "Aesthetics: All plots must be heavily customized for readability (titles, legends, gridlines, customized colormaps for heatmaps, clear axis labels)."
  ],
  "output_rules": {
    "format": "Python code ready for Jupyter Notebook execution",
    "language": "English",
    "verbosity": "High. Include explicit markdown explanations of the mathematical logic behind the Monte Carlo simulations and quantitative metrics.",
    "code_type": "Full implementation pipeline.",
    "comments": "Include concise, useful inline comments explaining complex pandas transformations and statistical functions."
  }
}