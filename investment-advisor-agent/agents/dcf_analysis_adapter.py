"""Adapter for the local DCF workflow implemented in top-level program.py."""

from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from program import DISCOUNT_RATE, TERMINAL_GROWTH, FinancialAnalysisSystem


def create_dcf_system(model_dir: str | None = None) -> FinancialAnalysisSystem:
    """Create the DCF system with a stable model path."""
    resolved_model_dir = model_dir or os.path.join(PROJECT_ROOT, "model")
    return FinancialAnalysisSystem(model_dir=resolved_model_dir)


def run_dcf_analysis(
    system: FinancialAnalysisSystem,
    company_name: str,
    discount_rate: float = DISCOUNT_RATE,
    terminal_growth: float = TERMINAL_GROWTH,
    generate_llm_report: bool = False,
) -> dict[str, Any]:
    """Run the DCF pipeline and normalize results for UI rendering."""
    system.dcf_valuator.discount_rate = discount_rate
    system.dcf_valuator.terminal_growth = terminal_growth

    logs = io.StringIO()
    with contextlib.redirect_stdout(logs):
        results, report, chat_context = system.analyze_company(
            company_name,
            generate_llm_report=generate_llm_report,
        )

    return {
        "results": _to_jsonable(results),
        "report": report,
        "chat_context": _to_jsonable(chat_context),
        "logs": logs.getvalue(),
        "assumptions": {
            "discount_rate": discount_rate,
            "terminal_growth": terminal_growth,
            "generate_llm_report": generate_llm_report,
        },
        "tables": _build_display_tables(results),
    }


def chat_with_dcf_analysis(
    system: FinancialAnalysisSystem,
    chat_context: list[dict[str, Any]],
    user_message: str,
) -> str:
    """Ask a follow-up question using program.py's LLMAnalyzer chat method."""
    return system.llm_analyzer.chat(chat_context.copy(), user_message)


def _build_display_tables(results: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cash_flow = results.get("cash_flow_predictions", {})
    earnings = results.get("earnings_predictions", {})
    health = results.get("health_assessment", {})
    moat = results.get("moat_analysis", {})

    prediction_table = []
    cash_ensemble = cash_flow.get("ensemble", [])
    earnings_ensemble = earnings.get("ensemble", [])
    for idx in range(max(len(cash_ensemble), len(earnings_ensemble))):
        prediction_table.append(
            {
                "年份": f"Year {idx + 1}",
                "FCF/Share 预测": _safe_round(_pick_sequence(cash_ensemble, idx)),
                "EPS 预测": _safe_round(_pick_sequence(earnings_ensemble, idx)),
            }
        )

    health_table = []
    for key, value in health.get("metrics", {}).items():
        if not isinstance(value, dict):
            continue
        row = {"维度": _labelize(key)}
        row.update({k: _safe_round(v) for k, v in value.items()})
        health_table.append(row)

    moat_table = []
    for group, values in moat.get("indicators", {}).items():
        if not isinstance(values, dict):
            continue
        for metric, value in values.items():
            moat_table.append(
                {
                    "类别": _labelize(group),
                    "指标": _labelize(metric),
                    "数值": _safe_round(value),
                }
            )

    return {
        "prediction_table": prediction_table,
        "health_table": health_table,
        "moat_table": moat_table,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        records = value.reset_index().where(pd.notnull(value.reset_index()), None).to_dict(orient="records")
        return [_to_jsonable(record) for record in records]
    if isinstance(value, pd.Series):
        return _to_jsonable(value.where(pd.notnull(value), None).to_dict())
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        if isinstance(item, float) and pd.isna(item):
            return None
        return item
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _pick_sequence(values: Any, index: int) -> Any:
    try:
        return values[index]
    except (IndexError, TypeError, KeyError):
        return None


def _safe_round(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return round(float(value), 4)
    return value


def _labelize(value: str) -> str:
    return str(value or "").replace("_", " ").title()
