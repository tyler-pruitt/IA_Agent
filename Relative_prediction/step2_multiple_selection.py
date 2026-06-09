"""Step 2: select the valuation multiple for each Step 1 company.

Improved policy: asset-heavy, financial, and real-estate balance-sheet businesses
are routed to P/B before the broad loss-making/high-growth P/S rule. Stable
profitable operating businesses now have an explicit P/E quality rule instead
of reaching P/E only as a broad fallback.

This script is intentionally downstream of Step 1. It consumes only
`outputs/calculated_feature_database/step1_to_step2_input.csv` so the cleaned universe,
clusters, and peer lists remain identical across the workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = "outputs/calculated_feature_database/step1_to_step2_input.csv"
DEFAULT_OUTPUT_DIR = "outputs/calculated_feature_database"

REQUIRED_COLUMNS = [
    "panel_id",
    "order_book_id",
    "symbol",
    "first_industry_name",
    "net_profit_ttm",
    "revenue_base",
    "equity_base",
    "ebitda_value",
    "debt_ratio",
    "fixed_asset_ratio",
    "da_to_ebit",
    "is_financial_industry",
    "pe_applicable",
    "pb_applicable",
    "ps_applicable",
    "ev_ebitda_applicable",
    "kmeans_cluster",
    "peer_order_book_ids",
]

MULTIPLE_TO_LABEL = {
    "P/E": "pe_ratio_ttm",
    "P/B": "pb_ratio",
    "P/S": "ps_ratio_ttm",
    "EV/EBITDA": "ev_to_ebitda",
}

MULTIPLE_TO_BASE = {
    "P/E": "net_profit_ttm",
    "P/B": "equity_base",
    "P/S": "revenue_base",
    "EV/EBITDA": "ebitda_value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 2 rule-based multiple selection.")
    parser.add_argument(
        "--step1-input",
        default=DEFAULT_INPUT,
        help="Canonical Step 1 handoff file. Do not pass the raw financial CSV here.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Step 2 output files.",
    )
    parser.add_argument(
        "--fixed-asset-threshold",
        type=float,
        default=0.60,
        help="P/B rule threshold for fixed assets / total assets.",
    )
    parser.add_argument(
        "--da-to-ebit-threshold",
        type=float,
        default=0.30,
        help="EV/EBITDA rule threshold for depreciation and amortization / EBIT.",
    )
    parser.add_argument(
        "--debt-ratio-threshold",
        type=float,
        default=0.70,
        help="EV/EBITDA rule threshold for total liabilities / total assets.",
    )
    parser.add_argument(
        "--pe-min-net-margin",
        type=float,
        default=0.03,
        help="P/E quality rule threshold for net profit / revenue.",
    )
    parser.add_argument(
        "--pe-min-roe",
        type=float,
        default=0.00,
        help="P/E quality rule threshold for ROE.",
    )
    parser.add_argument(
        "--pe-max-debt-ratio",
        type=float,
        default=0.70,
        help="P/E quality rule threshold for total liabilities / total assets.",
    )
    parser.add_argument(
        "--pe-max-da-to-ebit",
        type=float,
        default=0.30,
        help="P/E quality rule threshold for depreciation and amortization / EBIT.",
    )
    return parser.parse_args()


def load_step1_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Step 1 handoff file not found: {path}")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Step 1 handoff is missing required columns: {missing}")
    if frame["panel_id"].duplicated().any():
        duplicates = frame.loc[frame["panel_id"].duplicated(), "panel_id"].head(10).tolist()
        raise ValueError(f"Step 1 handoff must be one row per panel_id. Duplicate examples: {duplicates}")
    return frame


def bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def optional_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def choose_multiple(row: pd.Series) -> tuple[str, str, str]:
    """Choose one valuation multiple using explicit business-model priorities.

    Priority:
    1. P/B for financial, real-estate, and balance-sheet-heavy companies.
    2. EV/EBITDA for D&A-heavy or debt-heavy companies when EBITDA is usable.
    3. P/E for stable profitable operating companies with acceptable earnings quality.
    4. P/S for loss-making or hyper-growth operating companies.
    5. Fallback to any usable base.
    """
    if row["rule_pb_real_estate_or_asset_balance_sheet"] and row["pb_applicable_bool"]:
        return "P/B", "rule_1_pb_asset_anchor", row["rule_pb_asset_reason"]

    if row["rule_pb_financial_or_asset_heavy"] and row["pb_applicable_bool"]:
        return "P/B", "rule_1_pb_financial_or_asset_heavy", row["rule_pb_reason"]

    if row["rule_ev_ebitda_capital_structure"]:
        if row["ev_ebitda_applicable_bool"]:
            return "EV/EBITDA", "rule_2_ev_ebitda", row["rule_ev_reason"]
        if row["pb_applicable_bool"] and (
            row["rule_pb_real_estate_or_asset_balance_sheet"] or row["rule_pb_financial_or_asset_heavy"]
        ):
            return "P/B", "fallback_pb_from_ev", row["rule_ev_reason"] + "; EBITDA not positive, asset anchor fallback to P/B"
        if row["rule_pe_mature_profitable"] and row["pe_applicable_bool"]:
            return "P/E", "fallback_pe_from_ev", row["rule_ev_reason"] + "; EBITDA not positive, P/E quality rule remains satisfied"
        if row["pe_applicable_bool"]:
            return "P/E", "fallback_pe_from_ev", row["rule_ev_reason"] + "; EBITDA not positive, fallback to P/E"
        if row["ps_applicable_bool"]:
            return "P/S", "fallback_ps_from_ev", row["rule_ev_reason"] + "; EBITDA not positive, fallback to P/S"

    if row["rule_pe_mature_profitable"] and row["pe_applicable_bool"]:
        return "P/E", "rule_3_pe_mature_profitable", row["rule_pe_reason"]

    if row["rule_ps_profit_or_growth"] and row["ps_applicable_bool"]:
        return "P/S", "rule_4_ps_operating_growth_or_loss", row["rule_ps_reason"]

    if row["pe_applicable_bool"]:
        return "P/E", "fallback_pe_positive_earnings", "positive earnings base remains usable, but full P/E quality rule was not satisfied"
    if row["ev_ebitda_applicable_bool"]:
        return "EV/EBITDA", "fallback_ev_ebitda_positive_base", "positive EBITDA remains usable, but capital-structure rule was not triggered"
    if row["pb_applicable_bool"]:
        return "P/B", "fallback_pb", "positive book equity remains usable"
    if row["ps_applicable_bool"]:
        return "P/S", "fallback_ps", "revenue remains usable but higher-priority bases were unavailable"
    return "UNASSIGNED", "no_applicable_multiple", "no supported valuation base is positive or available"


def assign_selection_subtypes_and_confidence(result: pd.DataFrame, debt_ratio: pd.Series, roe: pd.Series) -> pd.DataFrame:
    result = result.copy()
    selected_multiple = result["selected_multiple"].astype(str)
    selection_rule = result["selection_rule"].astype(str)

    pb_rule = selected_multiple.eq("P/B")
    pe_rule = selected_multiple.eq("P/E")
    ps_rule = selected_multiple.eq("P/S")
    ev_rule = selected_multiple.eq("EV/EBITDA")

    ocf_margin = optional_numeric_series(result, "ocf_margin")
    net_debt_to_ebitda = optional_numeric_series(result, "net_debt_to_ebitda")
    ebitda_margin = optional_numeric_series(result, "ebitda_margin")

    result["pb_selection_subtype"] = np.select(
        [
            selection_rule.str.startswith("fallback_pb"),
            pb_rule & result["is_financial_industry_bool"],
            pb_rule & result["rule_real_estate_industry"],
            pb_rule & result["rule_inventory_asset_heavy"] & debt_ratio.gt(0.60),
            pb_rule & result["rule_fixed_asset_heavy"],
            pb_rule & roe.notna() & roe.le(0.03),
        ],
        [
            "pb_fallback",
            "pb_financial",
            "pb_real_estate_asset_anchor",
            "pb_inventory_levered_asset_anchor",
            "pb_fixed_asset_heavy",
            "pb_low_roe_or_distressed",
        ],
        default="",
    )

    result["pe_selection_subtype"] = np.select(
        [
            selection_rule.eq("rule_3_pe_mature_profitable"),
            pe_rule & result["rule_pe_positive_earnings"] & ~result["rule_pe_margin_quality"],
            pe_rule & result["rule_pe_positive_earnings"] & ~result["rule_pe_reasonable_debt"],
            pe_rule & result["rule_pe_positive_earnings"] & ~result["rule_pe_reasonable_da_burden"],
            selection_rule.str.startswith("fallback_pe") & result["rule_pe_mature_profitable"],
            selection_rule.str.startswith("fallback_pe"),
        ],
        [
            "pe_mature_quality",
            "pe_thin_margin_profitable",
            "pe_levered_profitable",
            "pe_da_burdened_profitable",
            "pe_quality_fallback_from_ev",
            "pe_fallback_positive_earnings",
        ],
        default="",
    )

    result["ps_selection_subtype"] = np.select(
        [
            selection_rule.str.startswith("fallback_ps"),
            ps_rule & result["rule_net_profit_non_positive_or_missing"] & result["rule_revenue_growth_gt_50"],
            ps_rule & result["rule_revenue_growth_gt_50"],
            ps_rule & result["rule_net_profit_non_positive_or_missing"],
        ],
        [
            "ps_fallback",
            "ps_loss_making_and_high_growth",
            "ps_high_growth",
            "ps_loss_making",
        ],
        default="",
    )

    result["ev_selection_subtype"] = np.select(
        [
            selection_rule.str.startswith("fallback_ev"),
            ev_rule & (ocf_margin.lt(0.00) | net_debt_to_ebitda.gt(5.0) | ebitda_margin.lt(0.05)),
            ev_rule & result["rule_da_heavy"] & result["rule_debt_heavy"],
            ev_rule & result["rule_da_heavy"] & ~result["rule_debt_heavy"],
            ev_rule & result["rule_debt_heavy"] & ~result["rule_da_heavy"],
            ev_rule & ebitda_margin.ge(0.10) & ocf_margin.ge(0.03) & (net_debt_to_ebitda.le(4.0) | net_debt_to_ebitda.isna()),
        ],
        [
            "ev_fallback",
            "ev_weak_cash_conversion",
            "ev_da_and_debt_heavy",
            "ev_da_heavy",
            "ev_debt_heavy",
            "ev_quality_operating_cashflow",
        ],
        default="",
    )

    result["multiple_selection_confidence"] = np.select(
        [
            result["pb_selection_subtype"].eq("pb_financial"),
            result["pb_selection_subtype"].eq("pb_real_estate_asset_anchor"),
            result["pb_selection_subtype"].eq("pb_inventory_levered_asset_anchor"),
            result["pb_selection_subtype"].eq("pb_fixed_asset_heavy"),
            result["pb_selection_subtype"].eq("pb_low_roe_or_distressed"),
            result["pb_selection_subtype"].eq("pb_fallback"),
            result["pe_selection_subtype"].eq("pe_mature_quality"),
            result["pe_selection_subtype"].eq("pe_quality_fallback_from_ev"),
            result["pe_selection_subtype"].eq("pe_thin_margin_profitable"),
            result["pe_selection_subtype"].eq("pe_levered_profitable"),
            result["pe_selection_subtype"].eq("pe_da_burdened_profitable"),
            result["pe_selection_subtype"].eq("pe_fallback_positive_earnings"),
            result["ev_selection_subtype"].eq("ev_quality_operating_cashflow"),
            result["ev_selection_subtype"].eq("ev_da_heavy"),
            result["ev_selection_subtype"].eq("ev_debt_heavy"),
            result["ev_selection_subtype"].eq("ev_da_and_debt_heavy"),
            result["ev_selection_subtype"].eq("ev_weak_cash_conversion"),
            result["ev_selection_subtype"].eq("ev_fallback"),
            result["ps_selection_subtype"].eq("ps_loss_making"),
            result["ps_selection_subtype"].eq("ps_high_growth"),
            result["ps_selection_subtype"].eq("ps_loss_making_and_high_growth"),
            result["ps_selection_subtype"].eq("ps_fallback"),
            selection_rule.str.startswith("fallback"),
        ],
        [
            0.90, 0.85, 0.80, 0.80, 0.65, 0.55,
            0.85, 0.65, 0.70, 0.65, 0.60, 0.55,
            0.85, 0.80, 0.75, 0.70, 0.60, 0.55,
            0.65, 0.60, 0.55, 0.45,
            0.50,
        ],
        default=0.60,
    )
    result["multiple_selection_confidence"] = result["multiple_selection_confidence"].clip(0.40, 0.95)
    return result


def apply_selection_rules(
    frame: pd.DataFrame,
    fixed_asset_threshold: float,
    da_to_ebit_threshold: float,
    debt_ratio_threshold: float,
    pe_min_net_margin: float,
    pe_min_roe: float,
    pe_max_debt_ratio: float,
    pe_max_da_to_ebit: float,
) -> pd.DataFrame:
    result = frame.copy()

    net_profit = numeric_series(result, "net_profit_ttm")
    roe = optional_numeric_series(result, "roe")
    net_margin = optional_numeric_series(result, "net_margin")
    fixed_asset_ratio = numeric_series(result, "fixed_asset_ratio")
    inventory_to_assets = numeric_series(result, "inventory_to_assets") if "inventory_to_assets" in result.columns else pd.Series(np.nan, index=result.index)
    da_to_ebit = numeric_series(result, "da_to_ebit")
    debt_ratio = numeric_series(result, "debt_ratio")
    industry_text = result[[column for column in ["first_industry_name", "second_industry_name", "third_industry_name"] if column in result.columns]].astype(str).agg(" ".join, axis=1)

    result["pe_applicable_bool"] = bool_series(result, "pe_applicable")
    result["pb_applicable_bool"] = bool_series(result, "pb_applicable")
    result["ps_applicable_bool"] = bool_series(result, "ps_applicable")
    result["ev_ebitda_applicable_bool"] = bool_series(result, "ev_ebitda_applicable")
    result["is_financial_industry_bool"] = bool_series(result, "is_financial_industry")

    if "revenue_growth" in result.columns:
        revenue_growth = numeric_series(result, "revenue_growth")
        result["rule_revenue_growth_gt_50"] = revenue_growth > 0.50
        result["rule_revenue_growth_available"] = revenue_growth.notna()
    else:
        result["rule_revenue_growth_gt_50"] = False
        result["rule_revenue_growth_available"] = False

    result["rule_net_profit_non_positive_or_missing"] = ~result["pe_applicable_bool"] | net_profit.isna()
    result["rule_ps_profit_or_growth"] = (
        result["rule_net_profit_non_positive_or_missing"] | result["rule_revenue_growth_gt_50"]
    )
    result["rule_fixed_asset_heavy"] = fixed_asset_ratio > fixed_asset_threshold
    result["rule_inventory_asset_heavy"] = inventory_to_assets > 0.40
    result["rule_real_estate_industry"] = industry_text.str.contains("房地产|地产|物业", na=False)
    result["rule_pb_real_estate_or_asset_balance_sheet"] = result["rule_real_estate_industry"] | (
        result["rule_inventory_asset_heavy"] & (debt_ratio > 0.60)
    )
    result["rule_pb_financial_or_asset_heavy"] = (
        result["is_financial_industry_bool"] | result["rule_fixed_asset_heavy"]
    )
    result["rule_da_heavy"] = da_to_ebit > da_to_ebit_threshold
    result["rule_debt_heavy"] = debt_ratio > debt_ratio_threshold
    result["rule_ev_ebitda_capital_structure"] = result["rule_da_heavy"] | result["rule_debt_heavy"]

    result["rule_pe_positive_earnings"] = result["pe_applicable_bool"] & net_profit.gt(0)
    result["rule_pe_positive_roe"] = roe.gt(pe_min_roe)
    result["rule_pe_margin_quality"] = net_margin.ge(pe_min_net_margin)
    result["rule_pe_reasonable_debt"] = debt_ratio.le(pe_max_debt_ratio) | debt_ratio.isna()
    result["rule_pe_reasonable_da_burden"] = da_to_ebit.le(pe_max_da_to_ebit) | da_to_ebit.isna()
    result["rule_pe_not_asset_or_financial_special_case"] = ~(
        result["rule_pb_real_estate_or_asset_balance_sheet"] | result["rule_pb_financial_or_asset_heavy"]
    )
    result["rule_pe_mature_profitable"] = (
        result["rule_pe_positive_earnings"]
        & result["rule_pe_positive_roe"]
        & result["rule_pe_margin_quality"]
        & result["rule_pe_reasonable_debt"]
        & result["rule_pe_reasonable_da_burden"]
        & result["rule_pe_not_asset_or_financial_special_case"]
    )

    result["rule_ps_reason"] = np.select(
        [
            result["rule_net_profit_non_positive_or_missing"] & result["rule_revenue_growth_gt_50"],
            result["rule_net_profit_non_positive_or_missing"] & ~result["rule_revenue_growth_available"],
            result["rule_net_profit_non_positive_or_missing"],
            result["rule_revenue_growth_gt_50"],
        ],
        [
            "net profit non-positive or missing; revenue growth > 50%",
            "net profit non-positive or missing; revenue growth unavailable for this row",
            "net profit non-positive or missing",
            "revenue growth > 50%",
        ],
        default="P/S rule triggered",
    )
    result["rule_pb_reason"] = np.where(
        result["is_financial_industry_bool"] & result["rule_fixed_asset_heavy"],
        "financial industry and fixed assets / total assets > 60%",
        np.where(
            result["is_financial_industry_bool"],
            "financial industry",
            "fixed assets / total assets > 60%",
        ),
    )
    result["rule_pb_asset_reason"] = np.where(
        result["rule_real_estate_industry"] & result["rule_inventory_asset_heavy"],
        "real-estate/asset-balance-sheet business with high inventory assets; P/B is more stable than EV/EBITDA",
        np.where(
            result["rule_real_estate_industry"],
            "real-estate/asset-balance-sheet business; P/B is more stable than EV/EBITDA",
            "inventory-heavy and leveraged balance sheet; P/B selected before EV/EBITDA",
        ),
    )
    result["rule_ev_reason"] = np.where(
        result["rule_da_heavy"] & result["rule_debt_heavy"],
        "depreciation and amortization / EBIT > 30% and debt ratio > 70%",
        np.where(
            result["rule_da_heavy"],
            "depreciation and amortization / EBIT > 30%",
            "debt ratio > 70%",
        ),
    )
    result["rule_pe_reason"] = (
        "positive earnings, positive ROE, net margin above threshold, moderate leverage, "
        "moderate D&A burden, and not a financial/real-estate/asset-heavy special case"
    )

    selections = result.apply(choose_multiple, axis=1, result_type="expand")
    selections.columns = ["selected_multiple", "selection_rule", "selection_reason"]
    result = pd.concat([result, selections], axis=1)
    result = assign_selection_subtypes_and_confidence(result, debt_ratio, roe)
    result["multiple_policy_version"] = "explicit_pe_quality_rule_v3"
    result["multiple_policy_note"] = (
        "P/B asset/financial rules are evaluated first; EV/EBITDA capital-structure rule is evaluated next; "
        "stable profitable operating businesses have an explicit P/E quality rule before broad P/S loss/growth fallback."
    )

    result["selected_label_column"] = result["selected_multiple"].map(MULTIPLE_TO_LABEL).fillna("")
    result["selected_base_column"] = result["selected_multiple"].map(MULTIPLE_TO_BASE).fillna("")
    result["selection_warning"] = ""
    fallback_from_ev_rule = result["selection_reason"].str.contains("EBITDA not positive", na=False)
    result.loc[fallback_from_ev_rule, "selection_warning"] = (
        "EV/EBITDA rule triggered but EBITDA is not positive, so fallback multiple was selected"
    )

    helper_columns = [
        "pe_applicable_bool",
        "pb_applicable_bool",
        "ps_applicable_bool",
        "ev_ebitda_applicable_bool",
        "is_financial_industry_bool",
        "rule_ps_reason",
        "rule_pb_reason",
        "rule_pb_asset_reason",
        "rule_ev_reason",
        "rule_pe_reason",
    ]
    return result.drop(columns=helper_columns)


def write_summary(output_dir: Path, selected: pd.DataFrame, source_path: Path) -> None:
    lines = ["Step 2 Multiple Selection Run Summary", ""]
    lines.append(f"Input: {source_path}")
    lines.append("Input contract: Step 1 canonical handoff; raw financial CSV is not reloaded here.")
    lines.append("Policy version: explicit_pe_quality_rule_v3")
    lines.append(f"Rows: {len(selected)}")
    lines.append("")
    lines.append("Selected multiple counts:")
    for multiple, count in selected["selected_multiple"].value_counts().items():
        lines.append(f"- {multiple}: {count}")
    lines.append("")
    lines.append("Selection rule counts:")
    for rule, count in selected["selection_rule"].value_counts().items():
        lines.append(f"- {rule}: {count}")
    lines.append("")
    lines.append("Applicability counts from Step 1:")
    for column in ["pe_applicable", "pb_applicable", "ps_applicable", "ev_ebitda_applicable"]:
        lines.append(f"- {column}: {int(bool_series(selected, column).sum())}")
    lines.append("")
    lines.append("Rule trigger counts:")
    for column in [
        "rule_net_profit_non_positive_or_missing",
        "rule_revenue_growth_gt_50",
        "rule_revenue_growth_available",
        "rule_fixed_asset_heavy",
        "rule_inventory_asset_heavy",
        "rule_real_estate_industry",
        "rule_pb_real_estate_or_asset_balance_sheet",
        "rule_pb_financial_or_asset_heavy",
        "rule_da_heavy",
        "rule_debt_heavy",
        "rule_ev_ebitda_capital_structure",
        "rule_pe_positive_earnings",
        "rule_pe_positive_roe",
        "rule_pe_margin_quality",
        "rule_pe_reasonable_debt",
        "rule_pe_reasonable_da_burden",
        "rule_pe_not_asset_or_financial_special_case",
        "rule_pe_mature_profitable",
    ]:
        lines.append(f"- {column}: {int(bool_series(selected, column).sum())}")
    lines.append("")
    lines.append("Selection subtype counts:")
    for column in [
        "pb_selection_subtype",
        "pe_selection_subtype",
        "ps_selection_subtype",
        "ev_selection_subtype",
    ]:
        if column in selected.columns:
            lines.append(f"{column}:")
            for subtype, count in selected[column].replace("", "none").value_counts().items():
                lines.append(f"- {subtype}: {count}")
    if "multiple_selection_confidence" in selected.columns:
        confidence = pd.to_numeric(selected["multiple_selection_confidence"], errors="coerce")
        lines.append("")
        lines.append("Multiple selection confidence:")
        lines.append(f"- mean: {float(confidence.mean())}")
        lines.append(f"- median: {float(confidence.median())}")
        lines.append(f"- min: {float(confidence.min())}")
        lines.append(f"- max: {float(confidence.max())}")
    warning_count = int(selected["selection_warning"].astype(bool).sum())
    lines.append("")
    lines.append(f"Rows with selection warnings: {warning_count}")
    lines.append("Revenue growth rule note: disabled unless a revenue_growth column exists in the Step 1 handoff.")
    (output_dir / "step2_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.step1_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step1 = load_step1_input(input_path)
    selected = apply_selection_rules(
        step1,
        fixed_asset_threshold=args.fixed_asset_threshold,
        da_to_ebit_threshold=args.da_to_ebit_threshold,
        debt_ratio_threshold=args.debt_ratio_threshold,
        pe_min_net_margin=args.pe_min_net_margin,
        pe_min_roe=args.pe_min_roe,
        pe_max_debt_ratio=args.pe_max_debt_ratio,
        pe_max_da_to_ebit=args.pe_max_da_to_ebit,
    )
    selected.to_csv(output_dir / "step2_selected_multiples.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir, selected, input_path)

    print(f"input_rows={len(step1)}")
    print(f"output_rows={len(selected)}")
    print("selected_multiple_counts=")
    print(selected["selected_multiple"].value_counts().to_string())
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
