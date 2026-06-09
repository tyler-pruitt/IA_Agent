from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

DEFAULT_PREDICTIONS = "outputs/calculated_feature_database/step3_market_implied_multiple_predictions.csv"
DEFAULT_METRICS = "outputs/calculated_feature_database/step3_model_metrics.csv"

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
    "fair_price_hgb",
    "upside_downside_hgb",
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
    "valuation_confidence",
    "valuation_sanity_flag",
    "pe_ratio_ttm",
    "pb_ratio",
    "ps_ratio_ttm",
    "ev_to_ebitda",
]


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
    return [
        "",
        "Valuation quality checks",
        f"Peer similarity score: {format_percent(row_value(row, 'peer_similarity_mean'))}",
        f"Peer range clipping applied: {format_bool(row_value(row, 'multiple_clip_applied'))}",
        f"Fair value / market cap: {format_number(row_value(row, 'fair_to_market_cap'))}x",
        f"Confidence: {str(row_value(row, 'valuation_confidence')).upper()}",
        f"Sanity flag: {row_value(row, 'valuation_sanity_flag')}",
    ]


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
        "Fair value comparison",
        f"Close price: {format_number(row['close_price'])}",
        f"Fair equity value: {format_currency(row['fair_equity_value'])}",
        f"Actual market cap: {format_currency(row['actual_market_cap'])}",
        f"Upside/downside: {format_percent(row['upside_downside'])}",
        f"Valuation signal: {str(row['valuation_signal']).upper()}",
        signal_sentence(row["valuation_signal"]),
    ]
    return "\n".join(lines)



CASHFLOW_NOTEBOOK_SOURCE = "Final_predict_cashflow.ipynb"
CASHFLOW_COMPANY_ID = "601318.XSHG"
CASHFLOW_COMPANY_NAME = "中国平安"
CASHFLOW_NOTEBOOK_PREDICTIONS = [
    {
        "model": "Random Forest",
        "r2": -0.11738465219886018,
        "mae": 1.0537357591382552,
        "cash_flows": [7.64370518, 9.16119058, 10.36025751, 10.46907515, 5.82264521],
    },
    {
        "model": "Gradient Boosting Regression",
        "r2": -0.04255939762594736,
        "mae": 1.0453515429194349,
        "cash_flows": [21.13978714, 26.50799444, 23.41340999, 22.88117037, 1.62783564],
    },
    {
        "model": "LSTM",
        "r2": -1.312308,
        "mae": 1.618468,
        "cash_flows": [16.954739, 6.320756, -10.627827, 2.6037467, -6.24342108450203],
    },
]


def finite_float(value: object) -> float | None:
    number = to_float(value)
    return float(number) if np.isfinite(number) else None


def terminal_value_gordon(cf_final_year: float, discount_rate: float, growth_rate: float) -> float:
    if discount_rate <= growth_rate:
        raise ValueError("discount_rate must be greater than growth_rate.")
    return cf_final_year * (1 + growth_rate) / (discount_rate - growth_rate)


def dcf_valuation(cash_flows: list[float], discount_rate: float, terminal_value: float | None = None) -> float:
    cash_flow_array = np.array(cash_flows, dtype=float)
    years = np.arange(1, len(cash_flow_array) + 1)
    present_value = float((cash_flow_array / (1 + discount_rate) ** years).sum())
    if terminal_value is not None:
        present_value += float(terminal_value / (1 + discount_rate) ** len(cash_flow_array))
    return present_value


def cashflow_payload(discount_rate: float = 0.20, growth_rate: float = 0.15) -> dict[str, object]:
    if discount_rate <= growth_rate:
        raise ValueError("discount_rate must be greater than growth_rate.")

    models = []
    for item in CASHFLOW_NOTEBOOK_PREDICTIONS:
        cash_flows = item["cash_flows"]
        terminal_value = terminal_value_gordon(cash_flows[-1], discount_rate, growth_rate)
        dcf_value = dcf_valuation([cash_flows], discount_rate, terminal_value)
        models.append(
            {
                "model": item["model"],
                "r2": item["r2"],
                "mae": item["mae"],
                "cash_flows": cash_flows,
                "terminal_value": terminal_value,
                "dcf_value": dcf_value,
            }
        )

    return {
        "company_id": CASHFLOW_COMPANY_ID,
        "company_name": CASHFLOW_COMPANY_NAME,
        "source_notebook": CASHFLOW_NOTEBOOK_SOURCE,
        "discount_rate": discount_rate,
        "growth_rate": growth_rate,
        "note": "Uses saved notebook prediction outputs and notebook DCF formulas; TensorFlow is not required for this page.",
        "models": models,
    }


def lookup_payload(
    company: str,
    quarter: str = "latest",
    predictions_path: Path | str = DEFAULT_PREDICTIONS,
    metrics_path: Path | str = DEFAULT_METRICS,
    top_peers: int = 5,
) -> dict[str, object]:
    query = company.strip()
    if not query:
        raise ValueError("Company name or order_book_id is required.")

    predictions = load_predictions(ROOT / predictions_path if isinstance(predictions_path, str) else Path(predictions_path))
    metrics = load_metrics(ROOT / metrics_path if isinstance(metrics_path, str) else Path(metrics_path))
    company_rows = resolve_company(predictions, query)
    row = select_quarter(company_rows, quarter)
    selected_multiple = str(row["selected_multiple"])
    metric = metric_for_multiple(metrics, selected_multiple)

    return {
        "order_book_id": str(row["order_book_id"]),
        "symbol": str(row["symbol"]),
        "industry": str(row["first_industry_name"]),
        "quarter": str(row["quarter"]),
        "info_date": str(row["info_date"]),
        "valuation_date": str(row["valuation_date"]),
        "selected_multiple": selected_multiple,
        "selection_reason": str(row["selection_reason"]),
        "selected_base_column": str(row["selected_base_column"]),
        "selected_base_value": finite_float(row["selected_base_value"]),
        "selected_base_value_formatted": format_currency(row["selected_base_value"]),
        "actual_selected_multiple": finite_float(row["actual_selected_multiple"]),
        "actual_selected_multiple_formatted": format_number(row["actual_selected_multiple"]),
        "model_predicted_fair_multiple": finite_float(row["model_predicted_fair_multiple"]),
        "model_predicted_fair_multiple_formatted": format_number(row["model_predicted_fair_multiple"]),
        "peer_median_multiple": finite_float(row["peer_median_multiple"]),
        "peer_median_multiple_formatted": format_number(row["peer_median_multiple"]),
        "peer_multiple_count_used": finite_float(row["peer_multiple_count_used"]),
        "peer_blend_weight": finite_float(row["peer_blend_weight"]),
        "peer_blend_weight_formatted": format_percent(row["peer_blend_weight"]),
        "peer_similarity_mean": finite_float(row_value(row, "peer_similarity_mean")),
        "peer_similarity_mean_formatted": format_percent(row_value(row, "peer_similarity_mean")),
        "multiple_clip_applied": format_bool(row_value(row, "multiple_clip_applied")),
        "peer_symbols": peer_preview(row["peer_symbols"], top_peers),
        "final_fair_multiple": finite_float(row["final_fair_multiple"]),
        "final_fair_multiple_formatted": format_number(row["final_fair_multiple"]),
        "fair_value_formula": str(row["fair_value_formula"]),
        "fair_equity_value": finite_float(row["fair_equity_value"]),
        "fair_equity_value_formatted": format_currency(row["fair_equity_value"]),
        "fair_price": finite_float(row_value(row, "fair_price")),
        "fair_price_formatted": format_number(row_value(row, "fair_price")),
        "fair_price_hgb": finite_float(row_value(row, "fair_price_hgb")),
        "fair_price_hgb_formatted": format_number(row_value(row, "fair_price_hgb")),
        "upside_downside_hgb": finite_float(row_value(row, "upside_downside_hgb")),
        "upside_downside_hgb_formatted": format_percent(row_value(row, "upside_downside_hgb")),
        "actual_market_cap": finite_float(row["actual_market_cap"]),
        "actual_market_cap_formatted": format_currency(row["actual_market_cap"]),
        "upside_downside": finite_float(row["upside_downside"]),
        "upside_downside_formatted": format_percent(row["upside_downside"]),
        "fair_to_market_cap": finite_float(row_value(row, "fair_to_market_cap")),
        "fair_to_market_cap_formatted": format_number(row_value(row, "fair_to_market_cap"), "x"),
        "valuation_confidence": str(row_value(row, "valuation_confidence")),
        "valuation_sanity_flag": str(row_value(row, "valuation_sanity_flag")),
        "valuation_signal": str(row["valuation_signal"]),
        "signal_sentence": signal_sentence(row["valuation_signal"]),
        "close_price": finite_float(row["close_price"]),
        "close_price_formatted": format_number(row["close_price"]),
        "model_method": str(row["model_method"]),
        "model_training_rows": finite_float(row["model_training_rows"]),
        "metric_trainable_rows": finite_float(metric.get("trainable_rows")) if metric else None,
        "metric_split_method": metric.get("split_method") if metric else None,
        "metric_test_r2_log": finite_float(metric.get("test_r2_log")) if metric else None,
        "metric_test_mae_log": finite_float(metric.get("test_mae_log")) if metric else None,
        "text_report": build_report(row, metrics, top_peers),
        "secondary_multiples": {
            "pe_ratio_ttm": format_number(row_value(row, "pe_ratio_ttm")),
            "pb_ratio": format_number(row_value(row, "pb_ratio")),
            "ps_ratio_ttm": format_number(row_value(row, "ps_ratio_ttm")),
            "ev_to_ebitda": format_number(row_value(row, "ev_to_ebitda")),
        },
    }


class ValuationRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/valuation":
            self.handle_valuation(parsed.query)
            return
        if parsed.path == "/api/cashflow":
            self.handle_cashflow(parsed.query)
            return
        if parsed.path == "/":
            self.path = "/valuation_agent_dashboard.html"
        super().do_GET()

    def do_HEAD(self) -> None:
        if urlparse(self.path).path == "/":
            self.path = "/valuation_agent_dashboard.html"
        super().do_HEAD()

    def handle_valuation(self, query_string: str) -> None:
        params = parse_qs(query_string)
        company = params.get("company", [""])[0]
        quarter = params.get("quarter", ["latest"])[0]
        try:
            payload = lookup_payload(company, quarter)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, 400)
        except LookupError as exc:
            self.write_json({"error": str(exc)}, 404)
        except FileNotFoundError as exc:
            self.write_json({"error": str(exc)}, 500)
        else:
            self.write_json(payload, 200)

    def handle_cashflow(self, query_string: str) -> None:
        params = parse_qs(query_string)
        try:
            discount_rate = float(params.get("discount_rate", ["0.20"])[0])
            growth_rate = float(params.get("growth_rate", ["0.15"])[0])
            payload = cashflow_payload(discount_rate, growth_rate)
        except ValueError as exc:
            self.write_json({"error": str(exc)}, 400)
        else:
            self.write_json(payload, 200)

    def write_json(self, payload: dict[str, object], status: int) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the valuation dashboard and real lookup API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def _free_port(host: str, port: int) -> None:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=5,
        )
        for token in result.stdout.strip().split():
            pid = int(token)
            if pid != os.getpid():
                os.kill(pid, 9)
                print(f"Killed stale process {pid} on port {port}")
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
        pass


def main() -> None:
    args = parse_args()
    handler = partial(ValuationRequestHandler, directory=str(ROOT))

    _free_port(args.host, args.port)

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving valuation dashboard at http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
