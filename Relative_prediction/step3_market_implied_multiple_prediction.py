"""Step 3: train market-implied multiple models and estimate model-implied value.

This module is downstream of the connected Step 1 -> Step 2 workflow. It uses
Step 2's selected-multiple output as the canonical universe, joins the RQData
market-label export, trains one supervised log-multiple model per selected
multiple, and writes model-implied multiple / market-implied value estimates.

Important semantics: this is not an intrinsic DCF model. The target labels are
observed market multiples, so the output should be read as market-implied value
or predicted market multiple, not fundamental fair value.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_START_QUARTER = "2020q1"
DEFAULT_END_QUARTER = "2025q4"
DEFAULT_STEP2_INPUT = "outputs/calculated_feature_database/step2_selected_multiples.csv"
DEFAULT_MARKET_LABELS = ""
DEFAULT_RAW_DB_DIR = "outputs/raw_rqdatac_database"
DEFAULT_OUTPUT_DIR = "outputs/calculated_feature_database"

MULTIPLE_CONFIG = {
    "P/E": {"label": "pe_ratio_ttm", "base": "net_profit_ttm", "fair_value_kind": "equity"},
    "P/B": {"label": "pb_ratio", "base": "equity_base", "fair_value_kind": "equity"},
    "P/S": {"label": "ps_ratio_ttm", "base": "revenue_base", "fair_value_kind": "equity"},
    "EV/EBITDA": {"label": "ev_to_ebitda", "base": "ebitda_value", "fair_value_kind": "enterprise"},
}

FEATURE_COLUMNS = [
    "roe",
    "roa",
    "net_margin",
    "operating_margin",
    "gross_margin",
    "ebitda_margin",
    "debt_ratio",
    "debt_to_equity",
    "current_ratio",
    "cash_ratio",
    "asset_turnover",
    "fixed_asset_ratio",
    "inventory_to_assets",
    "receivables_to_revenue",
    "inventory_turnover_proxy",
    "ocf_margin",
    "fcff_margin",
    "fcfe_margin",
    "cash_conversion",
    "da_to_ebit",
    "net_debt_to_ebitda",
    "cash_to_assets",
    "log_total_assets",
    "log_revenue",
    "log_equity",
    "listed_age_years",
    "peer_weighted_log_multiple",
    "peer_median_log_multiple",
    "peer_multiple_log_iqr",
    "peer_multiple_count_used",
    "peer_similarity_mean",
    "multiple_selection_confidence",
]

CATEGORICAL_COLUMNS = ["first_industry_name", "selection_rule", "selection_method"]

VENDOR_ENTERPRISE_VALUE_COLUMNS = [
    "vendor_enterprise_value",
    "enterprise_value",
    "market_enterprise_value",
    "total_enterprise_value",
    "ev",
]

MIN_SAME_MULTIPLE_PEERS = 3
EV_EBITDA_BASIS_MISMATCH_RATIO = 0.25
MODERATE_MULTIPLE_DIVERGENCE_RATIO = 2.0
EXTREME_MULTIPLE_DIVERGENCE_RATIO = 3.0

REQUIRED_STEP2_COLUMNS = [
    "panel_id",
    "order_book_id",
    "symbol",
    "first_industry_name",
    "selected_multiple",
    "selection_rule",
    "selected_label_column",
    "selected_base_column",
    "net_profit_ttm",
    "equity_base",
    "revenue_base",
    "ebitda_value",
    "interest_bearing_debt",
    "cash_equivalent_value",
    "peer_order_book_ids",
    "peer_count",
    "selection_method",
]

REQUIRED_LABEL_COLUMNS = [
    "order_book_id",
    "valuation_date",
    "close_price",
    "total_market_cap",
    "pe_ratio_ttm",
    "pb_ratio",
    "ps_ratio_ttm",
    "ev_to_ebitda",
]

OPTIONAL_LABEL_COLUMNS = ["vendor_enterprise_value"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 3 market-implied multiple prediction.")
    parser.add_argument("--start-quarter", default=DEFAULT_START_QUARTER, help="Start quarter used by the market-label artifact suffix.")
    parser.add_argument("--end-quarter", default=DEFAULT_END_QUARTER, help="End quarter used by the market-label artifact suffix.")
    parser.add_argument(
        "--step2-input",
        default=DEFAULT_STEP2_INPUT,
        help="Canonical Step 2 output. Do not pass Step 1 or raw financial data here.",
    )
    parser.add_argument(
        "--market-labels",
        default=DEFAULT_MARKET_LABELS,
        help="RQData market labels fetched after info_date.",
    )
    parser.add_argument("--raw-db-dir", default=DEFAULT_RAW_DB_DIR, help="Directory containing raw market label artifacts.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for Step 3 outputs.")
    parser.add_argument(
        "--write-legacy-fair-filename",
        action="store_true",
        help="Also write the backward-compatible step3_fair_multiple_predictions.csv filename.",
    )
    parser.add_argument(
        "--output-scope",
        choices=["scored_rows", "full_panel"],
        default="scored_rows",
        help="Write only rows with a model-implied prediction by default; use full_panel for every historical row.",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="Validation split for model metrics.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument(
        "--prediction-mode",
        choices=["out_of_sample_by_time", "fit_all_rows"],
        default="out_of_sample_by_time",
        help=(
            "Prediction mode for final outputs. out_of_sample_by_time trains each quarter only on earlier "
            "quarters to avoid in-sample fitted values. fit_all_rows preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--prediction-scope",
        choices=["latest_quarter", "all_quarters_expanding"],
        default="latest_quarter",
        help=(
            "Rows to produce final out-of-sample predictions for. latest_quarter is recommended for "
            "investment signals and trains only on earlier quarters. all_quarters_expanding produces a "
            "historical expanding-window backtest but is slower."
        ),
    )
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=30,
        help="Minimum trainable rows required before fitting an HGB model.",
    )
    return parser.parse_args()


def load_csv(path: Path, required_columns: list[str], name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    return frame


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def first_positive_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").where(lambda series: series > 0)
        result = result.where(result.notna(), values)
    return result


def choose_label_join_keys(step2: pd.DataFrame, labels: pd.DataFrame) -> list[str]:
    """Require a time-aware join key for panel data.

    The previous order_book_id-only fallback is intentionally removed because it
    can join the wrong quarter when a company has multiple panel rows.
    """
    if "panel_id" in step2.columns and "panel_id" in labels.columns:
        return ["panel_id"]
    if all(column in step2.columns for column in ["order_book_id", "quarter"]) and all(
        column in labels.columns for column in ["order_book_id", "quarter"]
    ):
        return ["order_book_id", "quarter"]
    raise ValueError(
        "Market labels must contain either panel_id or both order_book_id and quarter. "
        "Refusing to join on order_book_id alone because this is panel data."
    )


def build_training_frame(step2: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    join_keys = choose_label_join_keys(step2, labels)

    if step2[join_keys].duplicated().any():
        duplicates = step2.loc[step2[join_keys].duplicated(), join_keys].head(10).to_dict("records")
        raise ValueError(f"Step 2 input must be one row per {join_keys}. Duplicate examples: {duplicates}")
    if labels[join_keys].duplicated().any():
        duplicates = labels.loc[labels[join_keys].duplicated(), join_keys].head(10).to_dict("records")
        raise ValueError(f"Market labels must be one row per {join_keys}. Duplicate examples: {duplicates}")

    label_columns = list(dict.fromkeys(join_keys + REQUIRED_LABEL_COLUMNS + OPTIONAL_LABEL_COLUMNS))
    label_columns = [column for column in label_columns if column in labels.columns]

    joined = step2.merge(
        labels[label_columns],
        on=join_keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_market"),
    )
    joined["market_label_missing"] = joined["valuation_date"].isna()
    joined["market_label_join_keys"] = "+".join(join_keys)
    return joined

def make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    estimator: object,
    use_scaler: bool = False,
) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def make_hgb_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return make_pipeline(
        numeric_features,
        categorical_features,
        HistGradientBoostingRegressor(loss="squared_error", max_iter=500, random_state=42),
        use_scaler=False,
    )


def make_sgd_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return make_pipeline(
        numeric_features,
        categorical_features,
        SGDRegressor(loss="squared_error", penalty="l2", max_iter=1000, random_state=42),
        use_scaler=True,
    )


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def winsorize_positive_labels(labels: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if labels.notna().sum() < 20:
        return labels
    lower_bound = labels.quantile(lower)
    upper_bound = labels.quantile(upper)
    return labels.clip(lower=lower_bound, upper=upper_bound)


def parse_peer_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [peer_id.strip() for peer_id in str(value).split(";") if peer_id.strip()]


def scalar_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_float_list(value: object) -> list[float]:
    if value is None or pd.isna(value):
        return []
    values = []
    for part in str(value).split(";"):
        if not part.strip():
            continue
        try:
            values.append(float(part))
        except ValueError:
            values.append(float("nan"))
    return values


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float(np.median(values))
    sorted_order = np.argsort(values[finite])
    sorted_values = values[finite][sorted_order]
    sorted_weights = weights[finite][sorted_order]
    cumulative_weight = np.cumsum(sorted_weights)
    cutoff = sorted_weights.sum() / 2.0
    return float(sorted_values[np.searchsorted(cumulative_weight, cutoff, side="left")])


def peer_industry_factor(selection_method: object) -> float:
    method = str(selection_method)
    if method.startswith("same_quarter_first_second_third_industry"):
        return 1.0
    if method.startswith("fallback_same_quarter_first_second_industry"):
        return 0.75
    if method.startswith("fallback_same_quarter_first_industry"):
        return 0.50
    return 0.0


def peer_count_factor(usable_peer_count: int) -> float:
    if usable_peer_count >= 8:
        return 0.50
    if usable_peer_count >= 5:
        return 0.35
    if usable_peer_count >= 3:
        return 0.20
    return 0.0


def peer_dispersion_factor(log_iqr: float) -> float:
    if not np.isfinite(log_iqr):
        return 0.0
    if log_iqr <= 1.00:
        return 1.0
    if log_iqr <= 1.50:
        return 0.75
    if log_iqr <= 2.00:
        return 0.50
    return 0.25


def peer_similarity_factor(peer_similarity: float) -> float:
    if not np.isfinite(peer_similarity):
        return 1.0
    if peer_similarity >= 0.85:
        return 1.0
    if peer_similarity >= 0.70:
        return 0.85
    if peer_similarity >= 0.55:
        return 0.70
    return 0.50


def peer_blend_weight(usable_peer_count: int, log_iqr: float, selection_method: object, max_weight: float = 0.60) -> float:
    weight = (
        peer_count_factor(usable_peer_count)
        * peer_industry_factor(selection_method)
        * peer_dispersion_factor(log_iqr)
    )
    return float(min(max_weight, weight))


def build_peer_lookup(frame: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        key = (str(row["order_book_id"]), str(row.get("quarter", "")).lower())
        lookup[key] = row
    return lookup


def add_peer_anchor_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    n_rows = len(result)
    peer_median_log = np.full(n_rows, np.nan, dtype="float64")
    peer_median_multiple = np.full(n_rows, np.nan, dtype="float64")
    peer_weighted_log = np.full(n_rows, np.nan, dtype="float64")
    peer_weighted_multiple = np.full(n_rows, np.nan, dtype="float64")
    peer_count_used = np.zeros(n_rows, dtype="int64")
    peer_same_multiple_count_used = np.zeros(n_rows, dtype="int64")
    peer_log_iqr_values = np.full(n_rows, np.nan, dtype="float64")
    peer_clip_lower_values = np.full(n_rows, np.nan, dtype="float64")
    peer_clip_upper_values = np.full(n_rows, np.nan, dtype="float64")
    peer_anchor_purity = np.full(n_rows, "", dtype=object)

    if "peer_similarity_mean" in result.columns:
        result["peer_similarity_mean"] = pd.to_numeric(result["peer_similarity_mean"], errors="coerce")
    else:
        result["peer_similarity_mean"] = np.nan
    result["multiple_selection_confidence"] = pd.to_numeric(
        result.get("multiple_selection_confidence", pd.Series(0.60, index=result.index)),
        errors="coerce",
    ).fillna(0.60).clip(0.40, 0.95)

    order_ids = result["order_book_id"].astype(str).to_numpy()
    quarters = result.get("quarter", pd.Series("", index=result.index)).astype(str).str.lower().to_numpy()
    selected_multiples = result["selected_multiple"].astype(str).to_numpy()
    peer_id_strings = result.get("peer_order_book_ids", pd.Series("", index=result.index)).fillna("").astype(str).to_numpy()
    peer_score_strings = result.get("peer_similarity_scores", pd.Series("", index=result.index)).fillna("").astype(str).to_numpy()
    market_missing = result.get("market_label_missing", pd.Series(False, index=result.index)).fillna(False).astype(bool).to_numpy()

    peer_lookup = {(order_id, quarter): idx for idx, (order_id, quarter) in enumerate(zip(order_ids, quarters))}
    numeric_columns = set()
    for config in MULTIPLE_CONFIG.values():
        numeric_columns.add(config["label"])
        numeric_columns.add(config["base"])
    numeric_values = {
        column: pd.to_numeric(result[column], errors="coerce").to_numpy(dtype="float64")
        for column in numeric_columns
        if column in result.columns
    }

    for idx in range(n_rows):
        config = MULTIPLE_CONFIG.get(selected_multiples[idx])
        if config is None:
            continue
        label_column = config["label"]
        base_column = config["base"]
        if label_column not in numeric_values or base_column not in numeric_values:
            continue

        peer_ids = [peer_id.strip() for peer_id in peer_id_strings[idx].split(";") if peer_id.strip()]
        if not peer_ids:
            continue
        peer_similarity_scores = parse_float_list(peer_score_strings[idx])

        peer_candidates: list[tuple[float, float, bool]] = []
        quarter = quarters[idx]
        label_values = numeric_values[label_column]
        base_values = numeric_values[base_column]
        for peer_position, peer_id in enumerate(peer_ids):
            peer_idx = peer_lookup.get((peer_id, quarter))
            if peer_idx is None or market_missing[peer_idx]:
                continue
            peer_multiple = label_values[peer_idx]
            peer_base = base_values[peer_idx]
            if np.isfinite(peer_multiple) and peer_multiple > 0 and np.isfinite(peer_base) and peer_base > 0:
                if peer_position < len(peer_similarity_scores):
                    score = peer_similarity_scores[peer_position]
                else:
                    score = float("nan")
                weight = float(score) if np.isfinite(score) and score > 0 else 1.0
                same_multiple = selected_multiples[peer_idx] == selected_multiples[idx]
                peer_candidates.append((float(np.log(peer_multiple)), weight, same_multiple))

        if not peer_candidates:
            continue

        same_multiple_candidates = [candidate for candidate in peer_candidates if candidate[2]]
        if len(same_multiple_candidates) >= MIN_SAME_MULTIPLE_PEERS:
            selected_candidates = same_multiple_candidates
            peer_anchor_purity[idx] = "same_multiple"
        else:
            selected_candidates = peer_candidates
            peer_anchor_purity[idx] = "mixed_multiple"

        peer_logs = np.asarray([candidate[0] for candidate in selected_candidates], dtype="float64")
        peer_weights_for_anchor = np.asarray([candidate[1] for candidate in selected_candidates], dtype="float64")
        median_log = float(np.median(peer_logs))
        weighted_log = weighted_median(peer_logs, peer_weights_for_anchor)
        log_iqr = float(np.quantile(peer_logs, 0.75) - np.quantile(peer_logs, 0.25))

        if len(peer_logs) >= 5:
            lower = float(np.quantile(peer_logs, 0.10))
            upper = float(np.quantile(peer_logs, 0.90))
            margin = max(0.25, min(0.75, log_iqr * 0.50)) if np.isfinite(log_iqr) else 0.25
            clip_lower = lower - margin
            clip_upper = upper + margin
        else:
            clip_lower = float("nan")
            clip_upper = float("nan")

        peer_median_log[idx] = median_log
        peer_median_multiple[idx] = float(np.exp(median_log))
        peer_weighted_log[idx] = weighted_log
        peer_weighted_multiple[idx] = float(np.exp(weighted_log))
        peer_count_used[idx] = int(len(peer_logs))
        peer_same_multiple_count_used[idx] = int(len(same_multiple_candidates))
        peer_log_iqr_values[idx] = log_iqr
        peer_clip_lower_values[idx] = clip_lower
        peer_clip_upper_values[idx] = clip_upper

    result["peer_median_log_multiple"] = peer_median_log
    result["peer_median_multiple"] = peer_median_multiple
    result["peer_weighted_log_multiple"] = peer_weighted_log
    result["peer_weighted_multiple"] = peer_weighted_multiple
    result["peer_multiple_count_used"] = peer_count_used
    result["peer_same_multiple_count_used"] = peer_same_multiple_count_used
    result["peer_anchor_purity"] = peer_anchor_purity
    result["peer_multiple_log_iqr"] = peer_log_iqr_values
    result["peer_range_clip_lower_log_multiple"] = peer_clip_lower_values
    result["peer_range_clip_upper_log_multiple"] = peer_clip_upper_values
    return result


def _blend_model_predictions(
    result: pd.DataFrame,
    n_rows: int,
    model_log_input: str,
    model_method_input: str,
    output_suffix: str,
    peer_weighted_log: np.ndarray,
    peer_count_values: np.ndarray,
    clip_lower_values: np.ndarray,
    clip_upper_values: np.ndarray,
    peer_similarity_values: np.ndarray,
    peer_log_iqr_values: np.ndarray,
    selection_methods: np.ndarray,
    multiple_selection_confidence: np.ndarray,
    is_primary: bool = False,
) -> pd.DataFrame:
    model_log = pd.to_numeric(result[model_log_input], errors="coerce").to_numpy(dtype="float64")
    model_multiple = np.exp(model_log)
    result[f"model_predicted_fair_multiple{output_suffix}"] = model_multiple

    blended_log_values = model_log.copy()
    pre_clip_log_values = model_log.copy()
    final_multiple_values = model_multiple.copy()
    final_methods = result[model_method_input].astype(str).to_numpy(copy=True)
    if is_primary:
        peer_weights = np.zeros(n_rows, dtype="float64")
        clip_applied_values = np.zeros(n_rows, dtype="bool")

    for idx in range(n_rows):
        if str(final_methods[idx]).startswith("not_scored_historical_row_latest_quarter_scope"):
            continue
        if not np.isfinite(peer_weighted_log[idx]) or peer_count_values[idx] <= 0:
            continue

        max_peer_weight = 0.75 * multiple_selection_confidence[idx]
        weight = peer_blend_weight(
            int(peer_count_values[idx]),
            float(peer_log_iqr_values[idx]),
            selection_methods[idx],
            max_weight=max_peer_weight,
        )
        weight *= peer_similarity_factor(peer_similarity_values[idx])
        weight = float(min(max_peer_weight, weight))
        current_model_log = model_log[idx]

        if np.isfinite(current_model_log) and weight > 0:
            blended_log = (1 - weight) * current_model_log + weight * peer_weighted_log[idx]
            final_method = "linear_weighted_peer_log_blend"
        elif np.isfinite(current_model_log):
            blended_log = current_model_log
            final_method = final_methods[idx] if final_methods[idx] else "sgd_log_market_multiple"
        else:
            blended_log = peer_weighted_log[idx]
            weight = 1.0
            final_method = "weighted_peer_log_fallback"

        pre_clip_log = blended_log
        clip_applied = False
        if np.isfinite(clip_lower_values[idx]) and np.isfinite(clip_upper_values[idx]) and np.isfinite(blended_log):
            clipped_log = float(np.clip(blended_log, clip_lower_values[idx], clip_upper_values[idx]))
            clip_applied = not np.isclose(clipped_log, blended_log)
            blended_log = clipped_log
            if clip_applied:
                final_method = f"{final_method}_peer_range_clipped"

        if is_primary:
            peer_weights[idx] = weight
            clip_applied_values[idx] = clip_applied
        pre_clip_log_values[idx] = pre_clip_log
        blended_log_values[idx] = blended_log
        final_multiple_values[idx] = float(np.exp(blended_log)) if np.isfinite(blended_log) else np.nan
        final_methods[idx] = final_method

    result[f"blended_log_fair_multiple{output_suffix}"] = blended_log_values
    result[f"final_fair_multiple{output_suffix}"] = final_multiple_values
    result[f"final_multiple_method{output_suffix}"] = final_methods
    result[f"predicted_fair_multiple{output_suffix}"] = result[f"final_fair_multiple{output_suffix}"]
    if is_primary:
        result["peer_blend_weight"] = peer_weights
        result["multiple_clip_applied"] = clip_applied_values
    return result


def apply_peer_blend(frame: pd.DataFrame) -> pd.DataFrame:
    peer_anchor_columns = [
        "peer_median_log_multiple",
        "peer_median_multiple",
        "peer_weighted_log_multiple",
        "peer_weighted_multiple",
        "peer_multiple_count_used",
        "peer_multiple_log_iqr",
        "peer_range_clip_lower_log_multiple",
        "peer_range_clip_upper_log_multiple",
    ]
    if all(column in frame.columns for column in peer_anchor_columns):
        result = frame.copy()
    else:
        result = add_peer_anchor_features(frame)
    n_rows = len(result)

    peer_similarity_values = pd.to_numeric(result["peer_similarity_mean"], errors="coerce").to_numpy(dtype="float64")
    peer_count_values = pd.to_numeric(result["peer_multiple_count_used"], errors="coerce").fillna(0).to_numpy(dtype="int64")
    peer_log_iqr_values = pd.to_numeric(result["peer_multiple_log_iqr"], errors="coerce").to_numpy(dtype="float64")
    peer_weighted_log = pd.to_numeric(result["peer_weighted_log_multiple"], errors="coerce").to_numpy(dtype="float64")
    clip_lower_values = pd.to_numeric(result["peer_range_clip_lower_log_multiple"], errors="coerce").to_numpy(dtype="float64")
    clip_upper_values = pd.to_numeric(result["peer_range_clip_upper_log_multiple"], errors="coerce").to_numpy(dtype="float64")
    selection_methods = result.get("selection_method", pd.Series("", index=result.index)).astype(str).to_numpy()
    multiple_selection_confidence = pd.to_numeric(
        result.get("multiple_selection_confidence", pd.Series(0.60, index=result.index)),
        errors="coerce",
    ).fillna(0.60).clip(0.40, 0.95).to_numpy(dtype="float64")

    shared_kwargs = dict(
        n_rows=n_rows,
        peer_weighted_log=peer_weighted_log,
        peer_count_values=peer_count_values,
        clip_lower_values=clip_lower_values,
        clip_upper_values=clip_upper_values,
        peer_similarity_values=peer_similarity_values,
        peer_log_iqr_values=peer_log_iqr_values,
        selection_methods=selection_methods,
        multiple_selection_confidence=multiple_selection_confidence,
    )

    # Backward compat: if dual-model columns absent, create from unsuffixed columns
    if "predicted_log_fair_multiple_sgd" not in result.columns and "predicted_log_fair_multiple" in result.columns:
        result["predicted_log_fair_multiple_sgd"] = result["predicted_log_fair_multiple"]
    if "model_method_sgd" not in result.columns and "model_method" in result.columns:
        result["model_method_sgd"] = result["model_method"]
    if "predicted_log_fair_multiple_hgb" not in result.columns and "predicted_log_fair_multiple" in result.columns:
        result["predicted_log_fair_multiple_hgb"] = result["predicted_log_fair_multiple"]
    if "model_method_hgb" not in result.columns and "model_method" in result.columns:
        result["model_method_hgb"] = result["model_method"]

    model_configs_for_blend: list[tuple[str, str, str]] = []
    if "predicted_log_fair_multiple_sgd" in result.columns:
        model_configs_for_blend.append(("predicted_log_fair_multiple_sgd", "model_method_sgd", ""))
    if "predicted_log_fair_multiple_hgb" in result.columns:
        model_configs_for_blend.append(("predicted_log_fair_multiple_hgb", "model_method_hgb", "_hgb"))

    for idx_config, (log_col, method_col, suffix) in enumerate(model_configs_for_blend):
        result = _blend_model_predictions(
            result, model_log_input=log_col,
            model_method_input=method_col, output_suffix=suffix,
            is_primary=(idx_config == 0), **shared_kwargs,
        )

    if model_configs_for_blend:
        result["model_predicted_log_fair_multiple"] = result[model_configs_for_blend[0][0]]
        result["predicted_fair_multiple"] = result["final_fair_multiple"]
    return result

def quarter_sort_value(quarter: object) -> int:
    value = str(quarter).lower().strip()
    if "q" not in value:
        return -1
    year, qtr = value.split("q", 1)
    try:
        return int(year) * 10 + int(qtr)
    except ValueError:
        return -1


def split_train_test_by_time(
    trainable: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if "quarter" in trainable.columns and trainable["quarter"].nunique() >= 4:
        quarters = sorted(trainable["quarter"].dropna().unique(), key=quarter_sort_value)
        test_quarter_count = max(1, int(np.ceil(len(quarters) * test_size)))
        test_quarters = set(quarters[-test_quarter_count:])
        test_mask = trainable["quarter"].isin(test_quarters)
        if test_mask.any() and (~test_mask).any():
            return (
                trainable.loc[~test_mask],
                trainable.loc[test_mask],
                y.loc[~test_mask],
                y.loc[test_mask],
                "time_based_latest_quarters",
            )
    train_data, test_data, y_train, y_test = train_test_split(
        trainable,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return train_data, test_data, y_train, y_test, "random_row_split"



def fit_predict_log_multiple(
    train_data: pd.DataFrame,
    y_train: pd.Series,
    predict_data: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    pipeline_builder: Callable = make_sgd_pipeline,
) -> np.ndarray:
    pipeline = pipeline_builder(numeric_features, categorical_features)
    feature_columns = numeric_features + categorical_features
    pipeline.fit(train_data[feature_columns], y_train)
    return pipeline.predict(predict_data[feature_columns])


def predict_out_of_sample_by_time(
    subset: pd.DataFrame,
    y_by_index: pd.Series,
    trainable_mask: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    min_train_rows: int,
    prediction_scope: str,
    pipeline_builder: Callable = make_sgd_pipeline,
    model_label: str = "sgd",
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Predict each quarter using only labels from earlier quarters.

    Returns predicted log multiple, method, warning, and per-row training counts,
    all indexed like subset.
    """
    predictions = pd.Series(np.nan, index=subset.index, dtype="float64")
    methods = pd.Series("", index=subset.index, dtype="object")
    warnings = pd.Series("", index=subset.index, dtype="object")
    training_counts = pd.Series(0, index=subset.index, dtype="int64")

    if "quarter" not in subset.columns:
        warnings.loc[:] = "quarter column unavailable; cannot produce time-based out-of-sample predictions"
        return predictions, methods, warnings, training_counts

    quarter_values = subset["quarter"].map(quarter_sort_value)
    if quarter_values.lt(0).all():
        warnings.loc[:] = "quarter values are not parseable; cannot produce time-based out-of-sample predictions"
        return predictions, methods, warnings, training_counts

    quarters = sorted(subset.loc[quarter_values.ge(0), "quarter"].dropna().unique(), key=quarter_sort_value)
    if prediction_scope == "latest_quarter" and quarters:
        quarters_to_predict = [quarters[-1]]
        skipped_mask = ~subset["quarter"].eq(quarters[-1])
        methods.loc[skipped_mask] = "not_scored_historical_row_latest_quarter_scope"
        warnings.loc[skipped_mask] = "historical row not scored because prediction_scope=latest_quarter"
    else:
        quarters_to_predict = quarters

    for quarter in quarters_to_predict:
        quarter_value = quarter_sort_value(quarter)
        target_mask = subset["quarter"].eq(quarter)
        historical_mask = trainable_mask & quarter_values.lt(quarter_value) & y_by_index.notna()
        train_data = subset.loc[historical_mask]
        y_train = y_by_index.loc[historical_mask]
        training_counts.loc[target_mask] = int(len(train_data))

        if len(train_data) >= min_train_rows:
            predictions.loc[target_mask] = fit_predict_log_multiple(
                train_data,
                y_train,
                subset.loc[target_mask],
                numeric_features,
                categorical_features,
                pipeline_builder=pipeline_builder,
            )
            methods.loc[target_mask] = f"{model_label}_log_market_multiple_past_quarters"
        elif len(train_data) > 0:
            predictions.loc[target_mask] = float(y_train.median())
            methods.loc[target_mask] = "past_median_log_market_multiple_fallback"
            warnings.loc[target_mask] = (
                f"fewer than {min_train_rows} prior-quarter trainable rows; used prior median log multiple"
            )
        else:
            methods.loc[target_mask] = "unavailable_no_prior_quarter_labels"
            warnings.loc[target_mask] = "no prior-quarter labels available; prediction left missing to avoid look-ahead leakage"

    return predictions, methods, warnings, training_counts


def add_market_implied_aliases(result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    alias_pairs = {
        "model_predicted_log_market_multiple": "model_predicted_log_fair_multiple",
        "model_predicted_market_multiple": "model_predicted_fair_multiple",
        "peer_blended_log_market_multiple": "blended_log_fair_multiple",
        "predicted_log_market_multiple": "predicted_log_fair_multiple",
        "predicted_market_multiple": "predicted_fair_multiple",
        "final_market_implied_multiple": "final_fair_multiple",
        "market_implied_enterprise_value": "fair_enterprise_value",
        "market_implied_equity_value": "fair_equity_value",
        "market_implied_to_market_cap": "fair_to_market_cap",
        "market_implied_log_gap": "fair_to_market_log_gap",
        "market_implied_upside_downside": "upside_downside",
        "market_implied_signal": "valuation_signal",
        "market_implied_confidence": "valuation_confidence",
        "market_implied_sanity_flag": "valuation_sanity_flag",
        "market_implied_fair_price": "fair_price",
        # HGB secondary model aliases
        "hgb_model_predicted_fair_multiple": "model_predicted_fair_multiple_hgb",
        "hgb_predicted_fair_multiple": "predicted_fair_multiple_hgb",
        "hgb_blended_log_fair_multiple": "blended_log_fair_multiple_hgb",
        "hgb_final_fair_multiple": "final_fair_multiple_hgb",
        "hgb_fair_enterprise_value": "fair_enterprise_value_hgb",
        "hgb_fair_equity_value": "fair_equity_value_hgb",
        "hgb_fair_price": "fair_price_hgb",
        "hgb_upside_downside": "upside_downside_hgb",
        "hgb_fair_to_market_cap": "fair_to_market_cap_hgb",
    }
    for new_column, old_column in alias_pairs.items():
        if old_column in result.columns:
            result[new_column] = result[old_column]
    result["valuation_semantics"] = (
        "market_implied_multiple_model; labels are observed market multiples, not intrinsic DCF fair value. "
        "Columns without suffix are SGD (primary linear model); *_hgb columns are HGB (secondary non-linear model)."
    )
    return result


def train_and_predict(
    frame: pd.DataFrame,
    test_size: float,
    random_state: int,
    prediction_mode: str,
    prediction_scope: str,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = add_peer_anchor_features(frame)
    result["actual_selected_multiple"] = np.nan
    result["actual_log_selected_multiple"] = np.nan
    result["predicted_log_fair_multiple"] = np.nan
    result["predicted_fair_multiple"] = np.nan
    result["model_training_rows"] = 0
    result["model_method"] = ""
    result["model_warning"] = ""
    result["prediction_mode"] = prediction_mode
    result["prediction_scope"] = prediction_scope

    metrics: list[dict[str, object]] = []
    available_numeric_features = [column for column in FEATURE_COLUMNS if column in result.columns]
    available_categorical_features = [column for column in CATEGORICAL_COLUMNS if column in result.columns]

    for multiple, config in MULTIPLE_CONFIG.items():
        label_column = config["label"]
        base_column = config["base"]
        mask = result["selected_multiple"] == multiple
        subset = result.loc[mask].copy()
        labels = numeric(subset, label_column)
        bases = numeric(subset, base_column)
        trainable_mask = labels.gt(0) & bases.gt(0) & ~subset["market_label_missing"]
        trainable = subset.loc[trainable_mask].copy()

        result.loc[mask, "actual_selected_multiple"] = labels.to_numpy()
        positive_label_mask = mask & numeric(result, label_column).gt(0)
        result.loc[positive_label_mask, "actual_log_selected_multiple"] = np.log(
            numeric(result.loc[positive_label_mask], label_column)
        )

        metric = {
            "selected_multiple": multiple,
            "total_rows": int(mask.sum()),
            "trainable_rows": int(len(trainable)),
            "positive_label_rows": int(labels.gt(0).sum()),
            "missing_label_rows": int(subset["market_label_missing"].sum()),
            "non_positive_label_rows": int(labels.notna().sum() - labels.gt(0).sum()),
            "model_method": "median_log_market_multiple_fallback",
            "split_method": "not_split",
            "train_mae_log": np.nan,
            "test_mae_log": np.nan,
            "test_r2_log": np.nan,
            "median_actual_multiple": np.nan,
            "prediction_mode": prediction_mode,
            "prediction_scope": prediction_scope,
            "min_train_rows": min_train_rows,
        }

        if len(trainable) == 0:
            result.loc[mask, "model_warning"] = "no positive labels available for this selected multiple"
            metrics.append(metric)
            continue

        numeric_features = [
            column for column in available_numeric_features if numeric(trainable, column).notna().sum() > 0
        ]
        categorical_features = [
            column for column in available_categorical_features if trainable[column].notna().sum() > 0
        ]
        raw_train_labels = numeric(trainable, label_column)
        winsorized_train_labels = winsorize_positive_labels(raw_train_labels)
        y = np.log(winsorized_train_labels)
        y_by_index = pd.Series(np.nan, index=subset.index, dtype="float64")
        y_by_index.loc[trainable.index] = y
        median_log = float(y.median())
        metric["median_actual_multiple"] = float(np.exp(median_log))
        metric["winsorized_label_min"] = float(winsorized_train_labels.min())
        metric["winsorized_label_max"] = float(winsorized_train_labels.max())
        metric["numeric_feature_count"] = len(numeric_features)
        metric["categorical_feature_count"] = len(categorical_features)
        metric["dropped_all_null_features"] = ";".join(
            column for column in available_numeric_features if column not in numeric_features
        )

        if len(trainable) >= min_train_rows:
            hgb_pipeline = make_hgb_pipeline(numeric_features, categorical_features)
            train_data, test_data, y_train, y_test, split_method = split_train_test_by_time(
                trainable, y, test_size=test_size, random_state=random_state
            )
            hgb_pipeline.fit(train_data[numeric_features + categorical_features], y_train)
            train_pred = hgb_pipeline.predict(train_data[numeric_features + categorical_features])
            test_pred = hgb_pipeline.predict(test_data[numeric_features + categorical_features])
            metric.update(
                {
                    "model_method": "hgb_log_market_multiple",
                    "split_method": split_method,
                    "train_mae_log": float(mean_absolute_error(y_train, train_pred)),
                    "test_mae_log": float(mean_absolute_error(y_test, test_pred)),
                    "test_r2_log": safe_r2(y_test.to_numpy(), test_pred),
                }
            )
        else:
            metric["model_method"] = "median_log_market_multiple_fallback"

        model_configs = [
            ("sgd", make_sgd_pipeline, True),   # (key, builder, is_primary)
            ("hgb", make_hgb_pipeline, False),
        ]

        for model_key, pipeline_builder, is_primary in model_configs:
            suffix = "" if is_primary else f"_{model_key}"

            if prediction_mode == "fit_all_rows":
                if len(trainable) >= min_train_rows:
                    all_pred = fit_predict_log_multiple(
                        trainable, y, subset,
                        numeric_features, categorical_features,
                        pipeline_builder=pipeline_builder,
                    )
                    result.loc[mask, f"predicted_log_fair_multiple{suffix}"] = all_pred
                    result.loc[mask, f"model_method{suffix}"] = f"{model_key}_log_market_multiple_fit_all_rows"
                    result.loc[mask, f"model_training_rows{suffix}"] = len(trainable)
                    result.loc[mask, f"model_warning{suffix}"] = ""
                else:
                    result.loc[mask, f"predicted_log_fair_multiple{suffix}"] = median_log
                    result.loc[mask, f"model_method{suffix}"] = f"{model_key}_log_market_multiple_fallback_fit_all_rows"
                    result.loc[mask, f"model_training_rows{suffix}"] = len(trainable)
                    result.loc[mask, f"model_warning{suffix}"] = f"fewer than {min_train_rows} trainable rows; used all-row median log multiple"
            else:
                predicted_log, methods, warnings_, training_counts = predict_out_of_sample_by_time(
                    subset, y_by_index, trainable_mask,
                    numeric_features, categorical_features,
                    min_train_rows, prediction_scope,
                    pipeline_builder=pipeline_builder,
                    model_label=model_key,
                )
                result.loc[mask, f"predicted_log_fair_multiple{suffix}"] = predicted_log
                result.loc[mask, f"model_method{suffix}"] = methods
                result.loc[mask, f"model_warning{suffix}"] = warnings_
                result.loc[mask, f"model_training_rows{suffix}"] = training_counts
                if is_primary:
                    metric["final_prediction_rows"] = int(predicted_log.notna().sum())
                    metric["not_scored_historical_rows"] = int(methods.eq("not_scored_historical_row_latest_quarter_scope").sum())
                    metric["no_prior_prediction_rows"] = int(methods.eq("unavailable_no_prior_quarter_labels").sum())
                    metric["past_median_fallback_rows"] = int(methods.eq("past_median_log_market_multiple_fallback").sum())
                    metric["past_model_rows"] = int(methods.str.contains(f"{model_key}_log_market_multiple_past_quarters").sum())

        metrics.append(metric)

    result["predicted_fair_multiple"] = np.exp(result["predicted_log_fair_multiple"])
    result = apply_peer_blend(result)
    return result, pd.DataFrame(metrics)

def calculate_fair_values(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["selected_base_value"] = np.nan
    result["fair_value_formula"] = ""
    result["actual_market_cap"] = numeric(result, "total_market_cap")
    result["net_debt"] = numeric(result, "interest_bearing_debt").fillna(0) - numeric(
        result, "cash_equivalent_value"
    ).fillna(0)
    result["actual_enterprise_value_proxy"] = result["actual_market_cap"] + result["net_debt"]
    result["vendor_enterprise_value"] = first_positive_numeric(result, VENDOR_ENTERPRISE_VALUE_COLUMNS)
    vendor_ev_available = result["vendor_enterprise_value"].notna()
    result["actual_enterprise_value"] = result["vendor_enterprise_value"].where(
        vendor_ev_available, result["actual_enterprise_value_proxy"]
    )
    result["enterprise_value_basis"] = np.where(vendor_ev_available, "vendor_enterprise_value", "project_ev_proxy")
    result["enterprise_value_to_equity_adjustment"] = result["actual_enterprise_value"] - result["actual_market_cap"]
    result["actual_ev_to_ebitda_recomputed"] = result["actual_enterprise_value"] / numeric(
        result, "ebitda_value"
    ).replace(0, np.nan)
    for suffix in ["", "_hgb"]:
        result[f"fair_enterprise_value{suffix}"] = np.nan
        result[f"fair_equity_value{suffix}"] = np.nan
        result[f"predicted_ev_to_ebitda_recomputed{suffix}"] = np.nan
    result["actual_value_from_selected_multiple"] = np.nan
    result["actual_equity_value_from_selected_multiple"] = np.nan

    for multiple, config in MULTIPLE_CONFIG.items():
        mask = result["selected_multiple"] == multiple
        base_values = numeric(result, config["base"])
        result.loc[mask, "selected_base_value"] = base_values.loc[mask]

        for suffix in ["", "_hgb"]:
            fair_multiple_col = f"predicted_fair_multiple{suffix}"
            if fair_multiple_col not in result.columns:
                continue

            fair_value = result.loc[mask, fair_multiple_col] * base_values.loc[mask]
            if config["fair_value_kind"] == "enterprise":
                actual_multiple = numeric(result, "actual_selected_multiple")
                recomputed_multiple = numeric(result, "actual_ev_to_ebitda_recomputed")
                basis_mismatch = (
                    mask
                    & vendor_ev_available
                    & actual_multiple.gt(0)
                    & recomputed_multiple.gt(0)
                    & ((recomputed_multiple / actual_multiple.replace(0, np.nan) - 1).abs() > EV_EBITDA_BASIS_MISMATCH_RATIO)
                )
                vendor_scale_mask = mask & vendor_ev_available & actual_multiple.gt(0) & ~basis_mismatch
                vendor_direct_mask = mask & vendor_ev_available & ~vendor_scale_mask
                formula_base = (
                    "enterprise_value = EBITDA * final blended EV/EBITDA; "
                    "equity_value = enterprise_value - project net debt"
                )
                formula_mismatch = (
                    "enterprise_value = EBITDA * final blended EV/EBITDA; "
                    "equity_value = enterprise_value - (actual enterprise value - market cap)"
                )
                formula_vendor = (
                    "enterprise_value = vendor enterprise value * final blended EV/EBITDA / actual vendor EV/EBITDA; "
                    "equity_value = enterprise_value - (actual enterprise value - market cap)"
                )
                ev_col = f"fair_enterprise_value{suffix}"
                eq_col = f"fair_equity_value{suffix}"
                ev_ebitda_col = f"predicted_ev_to_ebitda_recomputed{suffix}"
                result.loc[mask, ev_col] = fair_value
                result.loc[vendor_scale_mask, ev_col] = (
                    result.loc[vendor_scale_mask, "actual_enterprise_value"]
                    * result.loc[vendor_scale_mask, fair_multiple_col]
                    / numeric(result.loc[vendor_scale_mask], "actual_selected_multiple")
                )
                result.loc[mask, eq_col] = result.loc[mask, ev_col] - result.loc[
                    mask, "enterprise_value_to_equity_adjustment"
                ]
                result.loc[mask, ev_ebitda_col] = result.loc[mask, ev_col] / base_values.loc[mask]
                if suffix == "":
                    result.loc[mask, "fair_value_formula"] = formula_base
                    result.loc[vendor_direct_mask, "fair_value_formula"] = formula_mismatch
                    result.loc[vendor_scale_mask, "fair_value_formula"] = formula_vendor
                    result.loc[mask, "actual_value_from_selected_multiple"] = result.loc[mask, "actual_enterprise_value"]
                    result.loc[mask, "actual_equity_value_from_selected_multiple"] = (
                        result.loc[mask, "actual_value_from_selected_multiple"]
                        - result.loc[mask, "enterprise_value_to_equity_adjustment"]
                    )
            else:
                result.loc[mask, f"fair_equity_value{suffix}"] = fair_value
                if suffix == "":
                    result.loc[mask, "fair_value_formula"] = f"equity_value = {config['base']} * final blended {multiple}"
                    result.loc[mask, "actual_value_from_selected_multiple"] = result.loc[
                        mask, "actual_selected_multiple"
                    ] * base_values.loc[mask]
                    result.loc[mask, "actual_equity_value_from_selected_multiple"] = result.loc[
                        mask, "actual_value_from_selected_multiple"
                    ]

    result["ev_to_ebitda_basis_gap"] = result["actual_ev_to_ebitda_recomputed"] - numeric(result, "ev_to_ebitda")
    result["ev_to_ebitda_basis_gap_pct"] = result["ev_to_ebitda_basis_gap"] / numeric(result, "ev_to_ebitda").replace(
        0, np.nan
    )
    ev_selected = result["selected_multiple"] == "EV/EBITDA"
    wide_ev_gap = result["ev_to_ebitda_basis_gap_pct"].abs() > 0.25
    result["ev_to_ebitda_basis_warning"] = np.select(
        [
            ev_selected & wide_ev_gap & vendor_ev_available,
            ev_selected & wide_ev_gap & ~vendor_ev_available,
        ],
        [
            "Vendor enterprise value is used for the equity bridge, but project EBITDA implies an EV/EBITDA materially different from the RQData EV/EBITDA label; using direct EBITDA-based fair EV instead of vendor-scaled EV/EBITDA.",
            "Vendor enterprise value is unavailable; using project EV proxy, whose EV/EBITDA differs materially from the RQData EV/EBITDA label.",
        ],
        default="",
    )

    # Primary (SGD) upside / fair_price for principal valuation signal
    result["upside_downside"] = result["fair_equity_value"] / result["actual_market_cap"] - 1
    result["fair_to_market_cap"] = result["fair_equity_value"] / result["actual_market_cap"]
    close_price = pd.to_numeric(result["close_price"], errors="coerce")
    fair_price = close_price * result["fair_equity_value"] / result["actual_market_cap"].replace(0, np.nan)
    result["fair_price"] = fair_price.where(np.isfinite(fair_price))

    # Secondary (HGB) upside / fair_price
    result["upside_downside_hgb"] = result["fair_equity_value_hgb"] / result["actual_market_cap"] - 1
    result["fair_to_market_cap_hgb"] = result["fair_equity_value_hgb"] / result["actual_market_cap"]
    fair_price_hgb = close_price * result["fair_equity_value_hgb"] / result["actual_market_cap"].replace(0, np.nan)
    result["fair_price_hgb"] = fair_price_hgb.where(np.isfinite(fair_price_hgb))

    result["fair_to_market_log_gap"] = np.log(result["fair_to_market_cap"].where(result["fair_to_market_cap"] > 0))
    actual_multiple = numeric(result, "actual_selected_multiple")
    final_multiple = numeric(result, "predicted_fair_multiple")
    result["actual_to_fair_multiple_ratio"] = actual_multiple / final_multiple.replace(0, np.nan)
    result["fair_to_actual_multiple_ratio"] = final_multiple / actual_multiple.replace(0, np.nan)
    result["multiple_divergence_ratio"] = pd.concat(
        [result["actual_to_fair_multiple_ratio"], result["fair_to_actual_multiple_ratio"]],
        axis=1,
    ).max(axis=1, skipna=True)
    low_peer_support = pd.to_numeric(result["peer_multiple_count_used"], errors="coerce").fillna(0) < 5
    wide_peer_set = pd.to_numeric(result["peer_multiple_log_iqr"], errors="coerce") > 1.50
    if "peer_similarity_mean" in result.columns:
        peer_similarity = pd.to_numeric(result["peer_similarity_mean"], errors="coerce").fillna(1.0)
    else:
        peer_similarity = pd.Series(1.0, index=result.index)
    weak_similarity = peer_similarity < 0.55
    mixed_peer_anchor = result.get("peer_anchor_purity", pd.Series("", index=result.index)).eq("mixed_multiple")
    extreme_gap = (result["fair_to_market_cap"] > 10) | (result["fair_to_market_cap"] < 0.10)
    moderate_multiple_divergence = result["multiple_divergence_ratio"] >= MODERATE_MULTIPLE_DIVERGENCE_RATIO
    extreme_multiple_divergence = result["multiple_divergence_ratio"] >= EXTREME_MULTIPLE_DIVERGENCE_RATIO
    negative_fair_value = result["fair_equity_value"] < 0
    result["valuation_sanity_flag"] = np.select(
        [
            negative_fair_value,
            extreme_gap,
            extreme_multiple_divergence,
            wide_peer_set,
            low_peer_support,
            weak_similarity,
            mixed_peer_anchor,
        ],
        [
            "negative_fair_equity_value",
            "extreme_fair_to_market_gap",
            "structural_market_model_divergence",
            "wide_peer_multiple_dispersion",
            "low_peer_multiple_support",
            "weak_peer_business_similarity",
            "mixed_peer_multiple_anchor",
        ],
        default="ok",
    )
    result["valuation_confidence"] = np.select(
        [
            negative_fair_value | extreme_gap | wide_peer_set | extreme_multiple_divergence,
            low_peer_support | weak_similarity | mixed_peer_anchor | moderate_multiple_divergence,
        ],
        ["low", "medium"],
        default="high",
    )
    result["valuation_signal"] = np.select(
        [result["upside_downside"] > 0, result["upside_downside"] < 0],
        ["undervalued", "overvalued"],
        default="unavailable",
    )
    missing_prediction = result["predicted_fair_multiple"].isna() | result["fair_equity_value"].isna()
    result.loc[missing_prediction, "valuation_signal"] = "unavailable"
    result = add_market_implied_aliases(result)
    return result


def add_equity_value_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    for column in ["equity_value_rows", "log_equity_value_mae", "fair_to_market_log_gap_mae"]:
        result[column] = np.nan

    actual_market_cap = pd.to_numeric(predictions["actual_market_cap"], errors="coerce")
    fair_equity_value = pd.to_numeric(predictions["fair_equity_value"], errors="coerce")
    valid_equity = actual_market_cap.gt(0) & fair_equity_value.gt(0)

    for idx, row in result.iterrows():
        multiple = row["selected_multiple"]
        mask = predictions["selected_multiple"].eq(multiple) & valid_equity
        if not mask.any():
            continue
        log_gap = np.log(fair_equity_value.loc[mask]) - np.log(actual_market_cap.loc[mask])
        result.loc[idx, "equity_value_rows"] = int(mask.sum())
        result.loc[idx, "log_equity_value_mae"] = float(log_gap.abs().mean())
        result.loc[idx, "fair_to_market_log_gap_mae"] = float(log_gap.abs().mean())

    return result


def write_summary(output_dir: Path, predictions: pd.DataFrame, metrics: pd.DataFrame, step2_path: Path) -> None:
    lines = ["Step 3 Market-Implied Multiple Prediction Run Summary", ""]
    lines.append(f"Input: {step2_path}")
    lines.append("Input contract: Step 2 selected-multiple output is the canonical universe.")
    lines.append("Semantics: outputs are market-implied multiples/values trained on observed market labels; not intrinsic DCF fair values.")
    lines.append(f"Rows: {len(predictions)}")
    lines.append("")
    lines.append("Selected multiple counts:")
    for multiple, count in predictions["selected_multiple"].value_counts().items():
        lines.append(f"- {multiple}: {count}")
    lines.append("")
    lines.append("Model metrics:")
    for row in metrics.to_dict("records"):
        lines.append(
            "- {selected_multiple}: rows={total_rows}, trainable={trainable_rows}, method={model_method}, "
            "prediction_mode={prediction_mode}, prediction_scope={prediction_scope}, "
            "test_mae_log={test_mae_log}, test_r2_log={test_r2_log}, "
            "log_equity_value_mae={log_equity_value_mae}".format(**row)
        )
    lines.append("")
    lines.append(f"SGD fair multiple rows: {int(predictions['predicted_fair_multiple'].notna().sum())}")
    lines.append(f"HGB fair multiple rows: {int(predictions['predicted_fair_multiple_hgb'].notna().sum())}")
    lines.append(f"Peer-anchor rows: {int(predictions['peer_median_multiple'].notna().sum())}")
    lines.append(f"Peer-blended rows: {int((predictions['peer_blend_weight'] > 0).sum())}")
    lines.append(f"Peer range clipped rows: {int(predictions['multiple_clip_applied'].sum())}")
    lines.append(f"Mean peer blend weight: {float(predictions['peer_blend_weight'].mean())}")
    lines.append(f"SGD fair equity value rows: {int(predictions['fair_equity_value'].notna().sum())}")
    lines.append(f"HGB fair equity value rows: {int(predictions['fair_equity_value_hgb'].notna().sum())}")
    valid_equity = pd.to_numeric(predictions["actual_market_cap"], errors="coerce").gt(0) & pd.to_numeric(
        predictions["fair_equity_value"], errors="coerce"
    ).gt(0)
    if valid_equity.any():
        equity_log_gap = np.log(pd.to_numeric(predictions.loc[valid_equity, "fair_equity_value"], errors="coerce")) - np.log(
            pd.to_numeric(predictions.loc[valid_equity, "actual_market_cap"], errors="coerce")
        )
        lines.append(f"Log equity value MAE: {float(equity_log_gap.abs().mean())}")
    lines.append(f"SGD upside/downside rows: {int(predictions['upside_downside'].notna().sum())}")
    lines.append(f"HGB upside/downside rows: {int(predictions['upside_downside_hgb'].notna().sum())}")
    lines.append(f"SGD fair price rows: {int(predictions['fair_price'].notna().sum())}")
    lines.append(f"HGB fair price rows: {int(predictions['fair_price_hgb'].notna().sum())}")
    lines.append("")
    lines.append("Valuation signal counts:")
    for signal, count in predictions["valuation_signal"].value_counts().items():
        lines.append(f"- {signal}: {count}")
    lines.append("")
    lines.append("Valuation confidence counts:")
    for confidence, count in predictions["valuation_confidence"].value_counts().items():
        lines.append(f"- {confidence}: {count}")
    lines.append("")
    lines.append("Valuation sanity flag counts:")
    for flag, count in predictions["valuation_sanity_flag"].value_counts().items():
        lines.append(f"- {flag}: {count}")
    (output_dir / "step3_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    step2_path = Path(args.step2_input)
    suffix = f"{args.start_quarter}_{args.end_quarter}"
    market_path = Path(args.market_labels) if args.market_labels else Path(args.raw_db_dir) / f"raw_market_labels_{suffix}.csv"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step2 = load_csv(step2_path, REQUIRED_STEP2_COLUMNS, "Step 2 selected-multiple output")
    labels = load_csv(market_path, REQUIRED_LABEL_COLUMNS, "Market label file")
    joined = build_training_frame(step2, labels)
    predictions, metrics = train_and_predict(
        joined,
        test_size=args.test_size,
        random_state=args.random_state,
        prediction_mode=args.prediction_mode,
        prediction_scope=args.prediction_scope,
        min_train_rows=args.min_train_rows,
    )
    predictions = calculate_fair_values(predictions)
    if args.output_scope == "scored_rows":
        output_predictions = predictions.loc[predictions["predicted_fair_multiple"].notna()].copy()
    else:
        output_predictions = predictions
    metrics = add_equity_value_metrics(metrics, output_predictions)

    output_predictions.to_csv(output_dir / "step3_market_implied_multiple_predictions.csv", index=False, encoding="utf-8-sig")
    if args.write_legacy_fair_filename:
        # Backward-compatible legacy filename. Columns now include explicit market_implied_* aliases.
        output_predictions.to_csv(output_dir / "step3_fair_multiple_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output_dir / "step3_model_metrics.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir, output_predictions, metrics, step2_path)

    print(f"input_rows={len(step2)}")
    print(f"joined_rows={len(joined)}")
    print(f"prediction_rows={len(output_predictions)}")
    print(f"output_scope={args.output_scope}")
    print("model_metrics=")
    print(metrics.to_string(index=False))
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
