"""Adapter for Relative_prediction/valuation_api_server.py.

The original file remains unchanged. This module imports its public helpers and
normalizes their outputs for the Streamlit investment-advisor frontend.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Relative_prediction.valuation_api_server import (  # noqa: E402
    DEFAULT_METRICS_FILENAME,
    DEFAULT_PREDICTIONS_FILENAME,
    load_metrics,
    load_predictions,
    lookup_payload,
    resolve_artifact_path,
)


def get_relative_valuation(company: str, quarter: str = "latest", top_peers: int = 5) -> dict[str, Any]:
    """Return one company's fair-multiple valuation payload."""
    return _jsonable(lookup_payload(company=company, quarter=quarter, top_peers=top_peers))


def get_model_metrics() -> list[dict[str, Any]]:
    """Load Step 3 model metrics from the same artifact resolution rules."""
    path = resolve_artifact_path(None, DEFAULT_METRICS_FILENAME, "Step 3 model metrics", required=False)
    frame = load_metrics(path)
    if frame.empty:
        return []
    columns = [
        "selected_multiple",
        "trainable_rows",
        "test_r2_log",
        "test_mae_log",
        "model_method",
        "final_prediction_rows",
    ]
    existing = [column for column in columns if column in frame.columns]
    return _jsonable(frame.loc[:, existing].to_dict(orient="records"))


def get_prediction_artifact_stats() -> dict[str, Any]:
    """Summarize the Step 3 prediction artifact for dashboard overview cards."""
    path = resolve_artifact_path(None, DEFAULT_PREDICTIONS_FILENAME, "Step 3 predictions")
    frame = load_predictions(path)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "companies": int(frame["order_book_id"].nunique()) if "order_book_id" in frame.columns else 0,
        "quarters": int(frame["quarter"].nunique()) if "quarter" in frame.columns else 0,
        "signals": int(frame["valuation_signal"].notna().sum()) if "valuation_signal" in frame.columns else 0,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
