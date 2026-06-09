"""User-facing company valuation lookup.

This CLI combines the completed Step 1 -> Step 2 -> Step 3 workflow outputs:
- identify the company and industry;
- show the selected valuation multiple and why it was selected;
- use the historical panel model prediction already generated in Step 3;
- calculate fair value, compare with market cap, and print the valuation signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = str(_SCRIPT_DIR / "outputs/calculated_feature_database/step3_market_implied_multiple_predictions.csv")
DEFAULT_METRICS = str(_SCRIPT_DIR / "outputs/calculated_feature_database/step3_model_metrics.csv")

REQUIRED_COLUMNS = [
    "panel_id",
    "order_book_id",
    "symbol",
    "quarter",
    "info_date",
    "valuation_date",
    "first_industry_name",
    "selected_multiple",
    "selection_reason",
    "selected_base_column",
    "peer_symbols",
    "actual_selected_multiple",
    "model_predicted_fair_multiple",
    "peer_median_multiple",
    "peer_multiple_count_used",
    "peer_blend_weight",
    "final_fair_multiple",
    "predicted_fair_multiple",
    "selected_base_value",
    "fair_value_formula",
    "fair_equity_value",
    "actual_market_cap",
    "upside_downside",
    "valuation_signal",
    "close_price",
    "model_training_rows",
    "model_method",
]

OPTIONAL_DEBUG_COLUMNS = [
    "fair_price",
    "business_similarity_source",
    "ebitda_value",
    "actual_enterprise_value_proxy",
    "vendor_enterprise_value",
    "actual_enterprise_value",
    "enterprise_value_basis",
    "enterprise_value_to_equity_adjustment",
    "actual_value_from_selected_multiple",
    "actual_equity_value_from_selected_multiple",
    "fair_enterprise_value",
    "net_debt",
    "interest_bearing_debt",
    "cash_equivalent_value",
    "actual_ev_to_ebitda_recomputed",
    "predicted_ev_to_ebitda_recomputed",
    "ev_to_ebitda_basis_gap_pct",
    "ev_to_ebitda_basis_warning",
    "peer_similarity_mean",
    "multiple_clip_applied",
    "fair_to_market_cap",
    "multiple_divergence_ratio",
    "valuation_confidence",
    "valuation_sanity_flag",
    "pe_ratio_ttm",
    "pb_ratio",
    "ps_ratio_ttm",
    "ev_to_ebitda",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Look up a company's fair multiple valuation result.")
    parser.add_argument("company", nargs="?", help="Company name, partial name, or order_book_id, e.g. 平安银行 or 000001.XSHE.")
    parser.add_argument("--quarter", default="latest", help="Quarter to use, e.g. 2025q4. Defaults to latest available.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Step 3 prediction CSV path.")
    parser.add_argument("--metrics", default=DEFAULT_METRICS, help="Step 3 model metrics CSV path.")
    parser.add_argument("--top-peers", type=int, default=5, help="Number of comparable peers to print.")
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Step 3 predictions not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction file is missing required columns: {missing}")
    return frame


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def quarter_sort_value(quarter: object) -> int:
    value = str(quarter).lower().strip()
    if "q" not in value:
        return -1
    year, qtr = value.split("q", 1)
    try:
        return int(year) * 10 + int(qtr)
    except ValueError:
        return -1


def normalize_text(value: object) -> str:
    return str(value).strip().lower()


def resolve_company(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    normalized_query = normalize_text(query)
    order_ids = frame["order_book_id"].astype(str).str.lower()
    symbols = frame["symbol"].astype(str).str.lower()

    exact = frame[(order_ids == normalized_query) | (symbols == normalized_query)]
    if not exact.empty:
        return exact

    contains = frame[order_ids.str.contains(normalized_query, regex=False) | symbols.str.contains(normalized_query, regex=False)]
    unique_companies = contains[["order_book_id", "symbol"]].drop_duplicates()
    if len(unique_companies) == 1:
        return contains
    if contains.empty:
        raise LookupError(f"No company matched: {query}")

    examples = unique_companies.head(20).to_string(index=False)
    raise LookupError(
        "Multiple companies matched. Please use an exact symbol or order_book_id.\n"
        f"Matches:\n{examples}"
    )


def select_quarter(company_rows: pd.DataFrame, quarter: str) -> pd.Series:
    rows = company_rows.copy()
    rows["quarter_sort"] = rows["quarter"].map(quarter_sort_value)
    if quarter.lower() != "latest":
        selected = rows[rows["quarter"].astype(str).str.lower() == quarter.lower()]
        if selected.empty:
            available = ", ".join(sorted(rows["quarter"].astype(str).unique(), key=quarter_sort_value))
            raise LookupError(f"Quarter {quarter} is not available for this company. Available: {available}")
        rows = selected
    return rows.sort_values(["quarter_sort", "valuation_date"]).iloc[-1]


def to_float(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def format_number(value: object, suffix: str = "") -> str:
    number = to_float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:,.4f}{suffix}"


def format_integer(value: object) -> str:
    number = to_float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{int(number):,}"


def format_currency(value: object) -> str:
    number = to_float(value)
    if not np.isfinite(number):
        return "N/A"
    absolute = abs(number)
    if absolute >= 1e12:
        return f"¥{number / 1e12:,.3f}T"
    if absolute >= 1e9:
        return f"¥{number / 1e9:,.3f}B"
    if absolute >= 1e6:
        return f"¥{number / 1e6:,.3f}M"
    return f"¥{number:,.2f}"


def format_percent(value: object) -> str:
    number = to_float(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number * 100:,.2f}%"


def format_bool(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return "yes" if str(value).strip().lower() in {"true", "1", "yes"} else "no"


def metric_for_multiple(metrics: pd.DataFrame, multiple: str) -> dict[str, object]:
    if metrics.empty or "selected_multiple" not in metrics.columns:
        return {}
    row = metrics[metrics["selected_multiple"] == multiple]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def peer_preview(peer_symbols: object, top_n: int) -> str:
    if pd.isna(peer_symbols):
        return "N/A"
    peers = [peer for peer in str(peer_symbols).split(";") if peer]
    if not peers:
        return "N/A"
    return ", ".join(peers[:top_n])


def row_value(row: pd.Series, column: str) -> object:
    if column not in row.index:
        return np.nan
    return row[column]


def secondary_market_multiple_lines(row: pd.Series) -> list[str]:
    return [
        "Secondary market multiple checks",
        f"P/E TTM: {format_number(row_value(row, 'pe_ratio_ttm'))}",
        f"P/B: {format_number(row_value(row, 'pb_ratio'))}",
        f"P/S TTM: {format_number(row_value(row, 'ps_ratio_ttm'))}",
        f"EV/EBITDA: {format_number(row_value(row, 'ev_to_ebitda'))}",
    ]


def ev_ebitda_debug_lines(row: pd.Series) -> list[str]:
    if str(row["selected_multiple"]) != "EV/EBITDA":
        return []
    warning_raw = row_value(row, "ev_to_ebitda_basis_warning")
    warning = str(warning_raw).strip() if pd.notna(warning_raw) else ""
    lines = [
        "",
        "EV/EBITDA calculation check",
        f"EBITDA used by project: {format_currency(row_value(row, 'ebitda_value'))}",
        f"Interest-bearing debt proxy: {format_currency(row_value(row, 'interest_bearing_debt'))}",
        f"Cash: {format_currency(row_value(row, 'cash_equivalent_value'))}",
        f"Net debt proxy: {format_currency(row_value(row, 'net_debt'))}",
        f"Project EV proxy = market cap + net debt: {format_currency(row_value(row, 'actual_enterprise_value_proxy'))}",
        f"Vendor enterprise value: {format_currency(row_value(row, 'vendor_enterprise_value'))}",
        f"EV basis used: {row_value(row, 'enterprise_value_basis')}",
        f"Actual EV used: {format_currency(row_value(row, 'actual_enterprise_value'))}",
        f"Equity bridge adjustment: {format_currency(row_value(row, 'enterprise_value_to_equity_adjustment'))}",
        f"Actual equity after EV bridge: {format_currency(row_value(row, 'actual_equity_value_from_selected_multiple'))}",
        f"Actual EV/EBITDA recomputed from EV basis: {format_number(row_value(row, 'actual_ev_to_ebitda_recomputed'))}",
        f"Predicted fair EV: {format_currency(row_value(row, 'fair_enterprise_value'))}",
        f"Predicted EV/EBITDA recomputed: {format_number(row_value(row, 'predicted_ev_to_ebitda_recomputed'))}",
    ]
    if warning:
        lines.append(f"Warning: {warning}")
    return lines


def valuation_quality_lines(row: pd.Series) -> list[str]:
    lines = [
        "",
        "Valuation quality checks",
        f"Peer similarity score: {format_percent(row_value(row, 'peer_similarity_mean'))}",
        f"Peer range clipping applied: {format_bool(row_value(row, 'multiple_clip_applied'))}",
        f"Fair value / market cap: {format_number(row_value(row, 'fair_to_market_cap'))}x",
        f"Market/fair multiple divergence: {format_number(row_value(row, 'multiple_divergence_ratio'))}x",
        f"Confidence: {str(row_value(row, 'valuation_confidence')).upper()}",
        f"Sanity flag: {row_value(row, 'valuation_sanity_flag')}",
    ]
    return lines


def signal_sentence(signal: object) -> str:
    value = str(signal).strip().lower()
    if value == "undervalued":
        return "The company appears undervalued versus the final fair value."
    if value == "overvalued":
        return "The company appears overvalued versus the final fair value."
    return "The valuation signal is unavailable because one or more market comparison fields are missing."


def build_report(row: pd.Series, metrics: pd.DataFrame, top_peers: int) -> str:
    selected_multiple = str(row["selected_multiple"])
    metric = metric_for_multiple(metrics, selected_multiple)
    metric_line = "Model metrics: N/A"
    if metric:
        metric_line = (
            "Model metrics: "
            f"trainable rows={int(to_float(metric.get('trainable_rows')))}, "
            f"split={metric.get('split_method', 'N/A')}, "
            f"test R²={format_number(metric.get('test_r2_log'))}, "
            f"test MAE(log)={format_number(metric.get('test_mae_log'))}"
        )

    lines = [
        "Company valuation result",
        "=" * 24,
        f"Company: {row['symbol']} ({row['order_book_id']})",
        f"Industry: {row['first_industry_name']}",
        f"Quarter used: {row['quarter']} | info_date: {row['info_date']} | valuation_date: {row['valuation_date']}",
        "",
        "Step 2: selected valuation multiple",
        f"Selected multiple: {selected_multiple}",
        f"Selection reason: {row['selection_reason']}",
        f"Comparable peers used by Step 1: {peer_preview(row['peer_symbols'], top_peers)}",
        "",
        "Step 3: fair multiple prediction",
        f"Actual market multiple: {format_number(row['actual_selected_multiple'])}",
        f"Model-only fair multiple: {format_number(row['model_predicted_fair_multiple'])}",
        f"Peer median multiple: {format_number(row['peer_median_multiple'])} "
        f"(usable peers={format_integer(row['peer_multiple_count_used'])}, "
        f"blend weight={format_percent(row['peer_blend_weight'])})",
        f"Final blended fair multiple: {format_number(row['final_fair_multiple'])}",
        f"Valuation base used: {row['selected_base_column']} = {format_currency(row['selected_base_value'])}",
        f"Formula applied: {row['fair_value_formula']}",
        f"Model method: {row['model_method']}",
        metric_line,
        "",
        *secondary_market_multiple_lines(row),
        *ev_ebitda_debug_lines(row),
        *valuation_quality_lines(row),
        "",
        "Fair value comparison (SGD — primary linear model)",
        f"Close price: {format_number(row['close_price'])}",
        f"Fair price (derived, SGD): {format_number(row_value(row, 'fair_price'))}",
        f"Actual market cap: {format_currency(row['actual_market_cap'])}",
        f"Upside/downside: {format_percent(row['upside_downside'])}",
        f"Valuation signal: {str(row['valuation_signal']).upper()}",
        signal_sentence(row["valuation_signal"]),
        "",
        "Fair value comparison (HGB — secondary non-linear model)",
        f"Fair price (derived, HGB): {format_number(row_value(row, 'fair_price_hgb'))}",
        f"HGB upside/downside: {format_percent(row_value(row, 'upside_downside_hgb'))}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    query = args.company or input("Enter company name or order_book_id: ").strip()
    if not query:
        print("Company name or order_book_id is required.", file=sys.stderr)
        return 2

    try:
        predictions = load_predictions(Path(args.predictions))
        metrics = load_metrics(Path(args.metrics))
        company_rows = resolve_company(predictions, query)
        selected_row = select_quarter(company_rows, args.quarter)
    except (FileNotFoundError, ValueError, LookupError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(build_report(selected_row, metrics, args.top_peers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
