"""Fetch RQData source data into the project database layout.

This script writes two workflow-ready database folders:

- raw_rqdatac_database: RQData/raw values only, preserving vendor factor names.
- calculated_feature_database: analysis-ready files derived by later pipeline steps.

The fetch output intentionally does not calculate valuation ratios such as ROE,
ROA, revenue_growth, or fair multiples. Those belong downstream in Step 1/2/3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from License import init_rqdatac


DEFAULT_START_QUARTER = "2020q1"
DEFAULT_END_QUARTER = "2025q4"
DEFAULT_RAW_DB_DIR = "outputs/raw_rqdatac_database"
DEFAULT_CALCULATED_DB_DIR = "outputs/calculated_feature_database"
DEFAULT_UNIVERSE_CSV = ""

PIT_FIELDS = [
    "operating_revenue",
    "net_profit",
    "total_assets",
    "total_liabilities",
    "equity_parent_company",
    # Income Statement items
    "gross_profit",
    "ebitda",
    "ebit",
    "cost_of_goods_sold",
    "depreciation_and_amortization",
    "interest_expense",
    "profit_before_tax",
    "income_tax",
    "r_n_d",
    "adjusted_net_profit",
    "return_on_equity_weighted_average",
    "net_profit_parent_company",
    "selling_expense",
    # Balance Sheet items
    "current_assets",
    "current_liabilities",
    "inventory",
    "net_accts_receivable",
    "short_term_loans",
    "long_term_loans",
    "net_fixed_assets",
    "goodwill",
    "intangible_assets",
    "cash_equivalent",
]
PIT_OUTPUT_FIELDS = [
    "operating_revenue",
    "net_profit",
    "pit_total_assets",
    "pit_total_liabilities",
    "pit_equity_parent_company",
    # Income Statement items (no factor collision — keep original names)
    "gross_profit",
    "ebitda",
    "ebit",
    "cost_of_goods_sold",
    "depreciation_and_amortization",
    "pit_interest_expense",
    "profit_before_tax",
    "income_tax",
    "r_n_d",
    "adjusted_net_profit",
    "return_on_equity_weighted_average",
    "net_profit_parent_company",
    "selling_expense",
    # Balance Sheet items — pit_ prefix to distinguish from any factor fields
    "pit_current_assets",
    "pit_current_liabilities",
    "pit_inventory",
    "pit_net_accts_receivable",
    "pit_short_term_loans",
    "pit_long_term_loans",
    "pit_net_fixed_assets",
    "pit_goodwill",
    "pit_intangible_assets",
    "pit_cash_equivalent",
]

RAW_FACTOR_FIELDS = [
    "operating_revenue_ttm_0",
    "net_profit_ttm_0",
    "operating_profitTTM",
    "ebitda_ttm",
    "ebit_ttm",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "equity_parent_company",
    "cash_equivalent",
    "net_operate_cashflowTTM",
    "fcff_ttm",
    "fcfe_ttm",
    "total_fixed_assets",
    "depreciation_and_amortization_ttm",
    "interest_expense",
    "bond_payable",
    "interest_bearing_debt",
]

MARKET_FACTOR_FIELDS = ["market_cap", "market_cap_3", "pe_ratio_ttm", "pb_ratio", "ps_ratio_ttm", "ev_to_ebitda"]
VENDOR_ENTERPRISE_VALUE_FACTOR_CANDIDATES = ["enterprise_value", "market_enterprise_value", "total_enterprise_value", "ev"]
PRICE_FIELDS = ["close", "volume", "total_turnover"]

RAW_UNIVERSE_COLUMNS = [
    "order_book_id",
    "symbol",
    "listed_date",
    "de_listed_date",
    "industry_date",
    "industry_source",
    "first_industry_code",
    "first_industry_name",
    "second_industry_code",
    "second_industry_name",
    "third_industry_code",
    "third_industry_name",
]

STEP1_FIELD_MAP = {
    "operating_revenue_ttm_0": "operating_revenueTTM",
    "net_profit_ttm_0": "net_profitTTM",
    "ebit_ttm": "ebitTTM",
    "fcff_ttm": "fcff",
    "fcfe_ttm": "fcfe",
}

MARKET_LABEL_RENAME = {
    "close": "close_price",
    "market_cap": "total_market_cap",
    "market_cap_3": "total_market_cap_alt",
    **{column: "vendor_enterprise_value" for column in VENDOR_ENTERPRISE_VALUE_FACTOR_CANDIDATES},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch RQData panel data into raw/calculated database folders.")
    parser.add_argument("--start-quarter", default=DEFAULT_START_QUARTER, help="Start reporting quarter, e.g. 2020q1.")
    parser.add_argument("--end-quarter", default=DEFAULT_END_QUARTER, help="End reporting quarter, e.g. 2025q4.")
    parser.add_argument("--raw-db-dir", default=DEFAULT_RAW_DB_DIR, help="Directory for raw RQData database files.")
    parser.add_argument(
        "--calculated-db-dir",
        default=DEFAULT_CALCULATED_DB_DIR,
        help="Directory for calculated feature database files produced downstream.",
    )
    parser.add_argument(
        "--universe-csv",
        default=DEFAULT_UNIVERSE_CSV,
        help="Optional fallback universe CSV. If omitted, the universe is fetched from rqdatac all_instruments.",
    )
    parser.add_argument("--industry-source", default="citics_2019", help="RQData industry source for classifications.")
    parser.add_argument("--industry-chunk-size", type=int, default=500, help="Stock chunk size for industry classification fetches.")
    parser.add_argument("--pit-chunk-size", type=int, default=300, help="Stock chunk size for PIT financial fetches.")
    parser.add_argument("--market-chunk-size", type=int, default=500, help="Stock chunk size for market/factor fetches.")
    parser.add_argument("--concept-chunk-size", type=int, default=500, help="Stock chunk size for concept tag fetches.")
    parser.add_argument(
        "--market-cache-dir",
        default="",
        help="Optional cache directory for per-valuation-date market/factor fetches. Defaults inside raw DB dir.",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing metadata, PIT, and market cache files.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print market fetch progress every N dates.")
    return parser.parse_args()


def quarter_range(start_quarter: str, end_quarter: str) -> list[str]:
    def parse(value: str) -> tuple[int, int]:
        year, quarter = value.lower().split("q", 1)
        return int(year), int(quarter)

    start_year, start_q = parse(start_quarter)
    end_year, end_q = parse(end_quarter)
    quarters: list[str] = []
    year, qtr = start_year, start_q
    while (year, qtr) <= (end_year, end_q):
        quarters.append(f"{year}q{qtr}")
        qtr += 1
        if qtr == 5:
            year += 1
            qtr = 1
    return quarters


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def factor_name_set(factor_names: object) -> set[str]:
    if isinstance(factor_names, pd.DataFrame):
        return set(factor_names.astype(str).to_numpy().ravel())
    if isinstance(factor_names, pd.Series):
        return set(factor_names.astype(str).tolist())
    if isinstance(factor_names, dict):
        return set(map(str, factor_names.keys())) | set(map(str, factor_names.values()))
    if isinstance(factor_names, (list, tuple, set, pd.Index)):
        return set(map(str, factor_names))
    return set()


def resolve_vendor_enterprise_value_fields(rq) -> list[str]:
    try:
        available_names = factor_name_set(rq.get_all_factor_names())
    except Exception as exc:
        print(f"vendor_enterprise_value_factor=unavailable reason={exc}")
        return []
    for candidate in VENDOR_ENTERPRISE_VALUE_FACTOR_CANDIDATES:
        if candidate in available_names:
            print(f"vendor_enterprise_value_factor={candidate}")
            return [candidate]
    print("vendor_enterprise_value_factor=not_found")
    return []


def normalize_instrument_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    rename_candidates = {"name": "symbol", "special_type": "type"}
    result = result.rename(columns={source: target for source, target in rename_candidates.items() if source in result.columns})
    for column in ["listed_date", "de_listed_date"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column].replace("0000-00-00", pd.NA), errors="coerce")
    if "symbol" not in result.columns and "order_book_id" in result.columns:
        result["symbol"] = result["order_book_id"]
    return result


def load_universe_from_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if "order_book_id" not in frame.columns:
        raise ValueError("Universe CSV is missing required column: order_book_id")
    return normalize_instrument_frame(frame).dropna(subset=["order_book_id"]).drop_duplicates("order_book_id")


def fetch_universe(
    rq,
    universe_csv: str,
    end_quarter: str,
    industry_source: str,
    industry_chunk_size: int,
) -> pd.DataFrame:
    if universe_csv:
        path = Path(universe_csv)
        if not path.exists():
            raise FileNotFoundError(f"Universe CSV not found: {path}")
        universe = load_universe_from_csv(path)
    else:
        instruments = rq.all_instruments(type="CS", market="cn")
        universe = normalize_instrument_frame(instruments)

    universe = universe.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id")
    ids = universe["order_book_id"].astype(str).tolist()
    industry_date = quarter_to_calendar_date(end_quarter)
    print(f"metadata_universe_ids={len(ids)} industry_date={industry_date.date()} industry_source={industry_source}")
    industry = fetch_industry_table(rq, ids, industry_date, industry_source, industry_chunk_size)
    return universe.merge(industry, on="order_book_id", how="left")


def fetch_industry_table(rq, ids: list[str], date: pd.Timestamp, source: str, chunk_size: int) -> pd.DataFrame:
    base = pd.DataFrame({"order_book_id": ids})
    base["industry_date"] = date.date().isoformat()
    base["industry_source"] = source
    for level, prefix in [(1, "first"), (2, "second"), (3, "third")]:
        parts: list[pd.DataFrame] = []
        for chunk_number, id_chunk in enumerate(chunks(ids, chunk_size), start=1):
            print(f"industry_level={level} chunk={chunk_number} ids={len(id_chunk)}")
            try:
                industry = rq.get_instrument_industry(id_chunk, source=source, level=level, date=date.date(), market="cn")
            except Exception as exc:  # rqdatac availability differs by installation/source.
                print(f"industry_level_{level}_chunk_{chunk_number}_warning={exc}")
                industry = None
            industry_frame = normalize_industry_result(industry, prefix)
            if not industry_frame.empty:
                parts.append(industry_frame)
        if parts:
            industry_frame = pd.concat(parts, ignore_index=True).drop_duplicates("order_book_id")
        else:
            industry_frame = pd.DataFrame(columns=["order_book_id", f"{prefix}_industry_code", f"{prefix}_industry_name"])
        base = base.merge(industry_frame, on="order_book_id", how="left")
    return base


def normalize_industry_result(industry: object, prefix: str) -> pd.DataFrame:
    if industry is None:
        return pd.DataFrame(columns=["order_book_id", f"{prefix}_industry_code", f"{prefix}_industry_name"])
    if isinstance(industry, pd.Series):
        frame = industry.rename(f"{prefix}_industry_name").reset_index()
    elif isinstance(industry, pd.DataFrame):
        frame = industry.reset_index() if "order_book_id" not in industry.columns else industry.copy()
    elif isinstance(industry, dict):
        frame = pd.DataFrame(list(industry.items()), columns=["order_book_id", f"{prefix}_industry_name"])
    else:
        return pd.DataFrame(columns=["order_book_id", f"{prefix}_industry_code", f"{prefix}_industry_name"])

    if "order_book_id" not in frame.columns:
        first_column = frame.columns[0]
        frame = frame.rename(columns={first_column: "order_book_id"})
    code_column = next((column for column in frame.columns if "code" in str(column).lower()), None)
    name_column = next((column for column in frame.columns if "name" in str(column).lower()), None)
    if name_column is None:
        candidates = [column for column in frame.columns if column != "order_book_id" and column != code_column]
        name_column = candidates[0] if candidates else None
    result = frame[["order_book_id"]].copy()
    result[f"{prefix}_industry_code"] = frame[code_column] if code_column else pd.NA
    result[f"{prefix}_industry_name"] = frame[name_column] if name_column else pd.NA
    return result.drop_duplicates("order_book_id")


def normalize_concept_result(concepts: object, fallback_ids: list[str]) -> pd.DataFrame:
    columns = ["order_book_id", "concept_name", "inclusion_date"]
    if concepts is None:
        return pd.DataFrame(columns=columns)
    if isinstance(concepts, pd.Series):
        frame = concepts.rename("concept_name").reset_index()
    elif isinstance(concepts, pd.DataFrame):
        frame = concepts.reset_index() if "order_book_id" not in concepts.columns else concepts.copy()
    elif isinstance(concepts, dict):
        frame = pd.DataFrame(concepts)
    else:
        return pd.DataFrame(columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    if "order_book_id" not in frame.columns:
        id_column = next((column for column in frame.columns if "order" in str(column).lower() or "stock" in str(column).lower()), None)
        if id_column is not None:
            frame = frame.rename(columns={id_column: "order_book_id"})
        elif len(fallback_ids) == 1:
            frame["order_book_id"] = fallback_ids[0]
    concept_column = next(
        (
            column
            for column in frame.columns
            if str(column).lower() in {"concept_name", "concept", "name", "concept_code", "concept_id"}
        ),
        None,
    )
    if concept_column is not None and concept_column != "concept_name":
        frame = frame.rename(columns={concept_column: "concept_name"})
    if "concept_name" not in frame.columns:
        return pd.DataFrame(columns=columns)
    if "inclusion_date" not in frame.columns:
        date_column = next((column for column in frame.columns if "date" in str(column).lower()), None)
        frame["inclusion_date"] = frame[date_column] if date_column is not None else pd.NaT

    result = frame[[column for column in columns if column in frame.columns]].copy()
    result["order_book_id"] = result["order_book_id"].astype(str)
    result["concept_name"] = result["concept_name"].astype(str).str.strip()
    result["inclusion_date"] = pd.to_datetime(result["inclusion_date"], errors="coerce")
    result = result.dropna(subset=["order_book_id"]).loc[result["concept_name"] != ""]
    return result.drop_duplicates(columns).reset_index(drop=True)


def fetch_concept_rows(rq, ids: list[str], chunk_size: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk_number, id_chunk in enumerate(chunks(ids, chunk_size), start=1):
        print(f"concept_chunk={chunk_number} ids={len(id_chunk)}")
        try:
            concepts = rq.get_stock_concept(id_chunk)
        except Exception as exc:
            print(f"concept_chunk_{chunk_number}_warning={exc}")
            continue
        concept_frame = normalize_concept_result(concepts, id_chunk)
        if not concept_frame.empty:
            parts.append(concept_frame)
    if not parts:
        return pd.DataFrame(columns=["order_book_id", "concept_name", "inclusion_date"])
    return pd.concat(parts, ignore_index=True).drop_duplicates().reset_index(drop=True)


def attach_historical_concept_tags(financial: pd.DataFrame, concepts: pd.DataFrame) -> pd.DataFrame:
    result = financial.copy()
    result["concept_tags"] = ""
    result["concept_tag_count"] = 0
    if concepts.empty or "order_book_id" not in concepts.columns or "concept_name" not in concepts.columns:
        return result

    concept_frame = concepts.copy()
    concept_frame["order_book_id"] = concept_frame["order_book_id"].astype(str)
    concept_frame["concept_name"] = concept_frame["concept_name"].astype(str).str.strip()
    concept_frame["inclusion_date"] = pd.to_datetime(concept_frame.get("inclusion_date", pd.NaT), errors="coerce")
    concept_frame = concept_frame.loc[concept_frame["concept_name"] != ""]
    grouped = {
        order_book_id: list(zip(group["inclusion_date"], group["concept_name"]))
        for order_book_id, group in concept_frame.sort_values(["order_book_id", "inclusion_date", "concept_name"]).groupby("order_book_id")
    }
    date_column = "factor_date" if "factor_date" in result.columns else "info_date"
    valuation_dates = pd.to_datetime(result[date_column], errors="coerce") if date_column in result.columns else pd.Series(pd.NaT, index=result.index)

    for index, row in result.iterrows():
        valuation_date = valuation_dates.loc[index]
        if pd.isna(valuation_date):
            continue
        active_tags: list[str] = []
        for inclusion_date, concept_name in grouped.get(str(row["order_book_id"]), []):
            if pd.notna(inclusion_date) and inclusion_date <= valuation_date:
                active_tags.append(concept_name)
        unique_tags = list(dict.fromkeys(active_tags))
        result.at[index, "concept_tags"] = ";".join(unique_tags)
        result.at[index, "concept_tag_count"] = len(unique_tags)
    return result


def quarter_to_calendar_date(quarter: str) -> pd.Timestamp:
    year_text, quarter_text = quarter.lower().split("q", 1)
    year = int(year_text)
    quarter_number = int(quarter_text)
    month_day = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}[quarter_number]
    return pd.Timestamp(f"{year}-{month_day}")


def fetch_pit_rows(rq, ids: list[str], start_quarter: str, end_quarter: str, chunk_size: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk_number, id_chunk in enumerate(chunks(ids, chunk_size), start=1):
        print(f"pit_chunk={chunk_number} ids={len(id_chunk)}")
        pit = rq.get_pit_financials_ex(
            id_chunk,
            fields=PIT_FIELDS,
            start_quarter=start_quarter,
            end_quarter=end_quarter,
            statements="all",
        )
        if pit is not None and len(pit):
            parts.append(pit.reset_index())
    if not parts:
        raise RuntimeError("No PIT financial rows returned from RQData.")
    pit_all = pd.concat(parts, ignore_index=True)
    pit_all["quarter"] = pit_all["quarter"].astype(str).str.lower()
    pit_all["info_date"] = pd.to_datetime(pit_all["info_date"], errors="coerce")
    pit_all["if_adjusted"] = pd.to_numeric(pit_all["if_adjusted"], errors="coerce").fillna(1).astype(int)
    pit_all = pit_all.dropna(subset=["order_book_id", "quarter", "info_date"])
    pit_all["adjusted_rank"] = (pit_all["if_adjusted"] != 0).astype(int)
    pit_all = pit_all.sort_values(["order_book_id", "quarter", "adjusted_rank", "info_date"])
    return pit_all.groupby(["order_book_id", "quarter"], as_index=False).first().drop(columns=["adjusted_rank"], errors="ignore")


def next_trading_dates(rq, dates: pd.Series) -> dict[pd.Timestamp, pd.Timestamp]:
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for date_value in sorted(pd.to_datetime(dates.dropna().unique())):
        next_date = rq.get_next_trading_date(pd.Timestamp(date_value).date(), n=1)
        if isinstance(next_date, list):
            next_date = next_date[0]
        mapping[pd.Timestamp(date_value)] = pd.Timestamp(next_date)
    return mapping


def fetch_market_for_dates(
    rq,
    panel: pd.DataFrame,
    chunk_size: int,
    cache_dir: Path,
    progress_every: int,
    refresh_cache: bool,
    vendor_enterprise_value_fields: list[str] | None = None,
) -> pd.DataFrame:
    factor_names = list(dict.fromkeys(RAW_FACTOR_FIELDS + MARKET_FACTOR_FIELDS + (vendor_enterprise_value_fields or [])))
    rows: list[pd.DataFrame] = []
    valuation_dates = sorted(panel["valuation_date"].dropna().unique())
    cache_dir.mkdir(parents=True, exist_ok=True)
    for date_number, valuation_date in enumerate(valuation_dates, start=1):
        valuation_timestamp = pd.Timestamp(valuation_date)
        date_panel = panel.loc[
            panel["valuation_date"] == valuation_date,
            ["panel_id", "order_book_id", "quarter", "info_date", "valuation_date"],
        ]
        ids = sorted(date_panel["order_book_id"].dropna().unique().tolist())
        cache_path = cache_dir / f"market_{valuation_timestamp:%Y%m%d}.csv"
        if cache_path.exists() and not refresh_cache:
            cached = pd.read_csv(cache_path, low_memory=False)
            if set(date_panel["panel_id"].astype(str)).issubset(set(cached.get("panel_id", pd.Series(dtype=str)).astype(str))):
                if should_print_progress(date_number, len(valuation_dates), progress_every, len(ids)):
                    print(f"market_date={date_number}/{len(valuation_dates)} date={valuation_timestamp.date()} ids={len(ids)} cache=hit")
                rows.append(cached)
                continue

        if should_print_progress(date_number, len(valuation_dates), progress_every, len(ids)):
            print(f"market_date={date_number}/{len(valuation_dates)} date={valuation_timestamp.date()} ids={len(ids)} cache=miss")
        date_parts: list[pd.DataFrame] = []
        for id_chunk in chunks(ids, chunk_size):
            factors = rq.get_factor(
                id_chunk,
                factor_names,
                start_date=valuation_timestamp.date(),
                end_date=valuation_timestamp.date(),
                expect_df=True,
            )
            prices = rq.get_price(
                id_chunk,
                start_date=valuation_timestamp.date(),
                end_date=valuation_timestamp.date(),
                frequency="1d",
                fields=PRICE_FIELDS,
                adjust_type="none",
                skip_suspended=True,
                expect_df=True,
            )
            factor_frame = factors.reset_index() if factors is not None and len(factors) else pd.DataFrame()
            price_frame = prices.reset_index() if prices is not None and len(prices) else pd.DataFrame()
            if factor_frame.empty and price_frame.empty:
                continue
            if factor_frame.empty:
                merged = price_frame
            elif price_frame.empty:
                merged = factor_frame
            else:
                merged = factor_frame.merge(price_frame, on=["order_book_id", "date"], how="left")
            date_parts.append(merged)
        if not date_parts:
            date_market = date_panel.copy()
            date_market.to_csv(cache_path, index=False, encoding="utf-8-sig")
            rows.append(date_market)
            continue
        date_market = pd.concat(date_parts, ignore_index=True)
        date_market["date"] = pd.to_datetime(date_market["date"])
        date_market = date_panel.merge(
            date_market,
            left_on=["order_book_id", "valuation_date"],
            right_on=["order_book_id", "date"],
            how="left",
        ).drop(columns=["date"], errors="ignore")
        date_market.to_csv(cache_path, index=False, encoding="utf-8-sig")
        rows.append(date_market)
    if not rows:
        raise RuntimeError("No market/factor panel rows returned from RQData.")
    return pd.concat(rows, ignore_index=True)


def should_print_progress(date_number: int, total_dates: int, progress_every: int, id_count: int) -> bool:
    if date_number in {1, total_dates}:
        return True
    if progress_every > 0 and date_number % progress_every == 0:
        return True
    return id_count >= 500


def build_raw_database(universe: pd.DataFrame, pit: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    base = pit.merge(universe, on="order_book_id", how="left")
    base["panel_id"] = base["order_book_id"].astype(str) + "__" + base["quarter"].astype(str)
    market = market.copy().rename(columns={"valuation_date": "factor_date"})
    market["price_date"] = market["factor_date"]
    raw = base.merge(
        market.drop(columns=["quarter", "info_date"], errors="ignore"),
        on=["panel_id", "order_book_id"],
        how="left",
        suffixes=("_pit", ""),
    )
    raw = raw.rename(
        columns={
            "total_assets_pit": "pit_total_assets",
            "total_liabilities_pit": "pit_total_liabilities",
            "equity_parent_company_pit": "pit_equity_parent_company",
            "cash_equivalent_pit": "pit_cash_equivalent",
            "interest_expense_pit": "pit_interest_expense",
        }
    )
    ordered = [
        "panel_id",
        *RAW_UNIVERSE_COLUMNS,
        "quarter",
        "info_date",
        "if_adjusted",
        *PIT_OUTPUT_FIELDS,
        "factor_date",
        *RAW_FACTOR_FIELDS,
        "price_date",
        *PRICE_FIELDS,
        *MARKET_FACTOR_FIELDS,
        *VENDOR_ENTERPRISE_VALUE_FACTOR_CANDIDATES,
    ]
    return raw[[column for column in ordered if column in raw.columns]]


def build_step1_financial_input(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy()
    result = result.rename(columns=STEP1_FIELD_MAP)
    # Order columns: core identifiers, renamed STEP1 features, raw factor fields,
    # and available PIT-sourced fields.
    columns = [
        "panel_id",
        "order_book_id",
        "symbol",
        "listed_date",
        "de_listed_date",
        "first_industry_code",
        "first_industry_name",
        "second_industry_code",
        "second_industry_name",
        "third_industry_code",
        "third_industry_name",
        "quarter",
        "info_date",
        "factor_date",
        *STEP1_FIELD_MAP.values(),
        "operating_profitTTM",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "equity_parent_company",
        "cash_equivalent",
        "net_operate_cashflowTTM",
        "ebitda_ttm",
        "depreciation_and_amortization_ttm",
        "total_fixed_assets",
        "interest_expense",
        "bond_payable",
        "interest_bearing_debt",
        # PIT-sourced balance-sheet fields (used by step1 for ratio computation)
        "pit_current_assets",
        "pit_current_liabilities",
        "pit_inventory",
        "pit_net_accts_receivable",
        "pit_short_term_loans",
        "pit_long_term_loans",
        "pit_net_fixed_assets",
        "pit_goodwill",
        "pit_intangible_assets",
        "pit_cash_equivalent",
        "pit_interest_expense",
        # PIT-sourced income-statement fields
        "gross_profit",
        "ebitda",
        "ebit",
        "cost_of_goods_sold",
        "depreciation_and_amortization",
        "profit_before_tax",
        "income_tax",
        "r_n_d",
        "adjusted_net_profit",
        "return_on_equity_weighted_average",
        "net_profit_parent_company",
        "selling_expense",
    ]
    return result[[column for column in columns if column in result.columns]]


def build_market_labels(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy().rename(columns={"factor_date": "valuation_date", **MARKET_LABEL_RENAME})
    columns = [
        "panel_id",
        "order_book_id",
        "quarter",
        "info_date",
        "valuation_date",
        "close_price",
        "volume",
        "total_turnover",
        "total_market_cap",
        "total_market_cap_alt",
        "vendor_enterprise_value",
        "pe_ratio_ttm",
        "pb_ratio",
        "ps_ratio_ttm",
        "ev_to_ebitda",
    ]
    return result[[column for column in columns if column in result.columns]]


def write_summary(
    raw_db_dir: Path,
    calculated_db_dir: Path,
    raw: pd.DataFrame,
    financial: pd.DataFrame,
    labels: pd.DataFrame,
    concepts: pd.DataFrame,
    start_quarter: str,
    end_quarter: str,
) -> None:
    lines = ["RQData Database Fetch Summary", ""]
    lines.append(f"Quarter range: {start_quarter} -> {end_quarter}")
    lines.append(f"Raw database rows: {len(raw)}")
    lines.append(f"Step 1 financial input rows: {len(financial)}")
    lines.append(f"Market label rows: {len(labels)}")
    lines.append(f"Concept tag rows: {len(concepts)}")
    lines.append(f"Companies: {raw['order_book_id'].nunique()}")
    lines.append(f"Companies with concept tags: {concepts['order_book_id'].nunique() if 'order_book_id' in concepts.columns else 0}")
    lines.append(f"Unique concept tags: {concepts['concept_name'].nunique() if 'concept_name' in concepts.columns else 0}")
    lines.append(f"Quarters: {raw['quarter'].nunique()}")
    lines.append(f"Raw database directory: {raw_db_dir}")
    lines.append(f"Calculated feature database directory: {calculated_db_dir}")
    lines.append("")
    lines.append("Raw financial non-null counts:")
    for column in ["operating_revenue_ttm_0", "net_profit_ttm_0", "ebitda_ttm", "total_assets", "equity_parent_company"]:
        if column in raw.columns:
            lines.append(f"- {column}: {int(raw[column].notna().sum())}")
    lines.append("")
    lines.append("Raw market non-null counts:")
    for column in ["close", "market_cap", "vendor_enterprise_value", "pe_ratio_ttm", "pb_ratio", "ps_ratio_ttm", "ev_to_ebitda"]:
        if column in raw.columns:
            lines.append(f"- {column}: {int(raw[column].notna().sum())}")
    (raw_db_dir / "rqdatac_database_fetch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_cached_frame(path: Path, parse_dates: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    for column in parse_dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    print(f"cache_loaded={path} rows={len(frame)}")
    return frame


def write_cached_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"cache_written={path} rows={len(frame)}")


def main() -> None:
    args = parse_args()
    rq = init_rqdatac()
    raw_db_dir = Path(args.raw_db_dir)
    calculated_db_dir = Path(args.calculated_db_dir)
    raw_db_dir.mkdir(parents=True, exist_ok=True)
    calculated_db_dir.mkdir(parents=True, exist_ok=True)

    quarters = quarter_range(args.start_quarter, args.end_quarter)
    suffix = f"{args.start_quarter}_{args.end_quarter}"
    universe_cache_path = raw_db_dir / f"metadata_universe_{suffix}.csv"
    pit_cache_path = raw_db_dir / f"pit_disclosures_{suffix}.csv"
    concept_cache_path = raw_db_dir / f"concept_tags_{suffix}.csv"

    universe = None if args.refresh_cache else read_cached_frame(universe_cache_path, ["listed_date", "de_listed_date"])
    if universe is None:
        universe = fetch_universe(rq, args.universe_csv, args.end_quarter, args.industry_source, args.industry_chunk_size)
        write_cached_frame(universe_cache_path, universe)
    ids = universe["order_book_id"].astype(str).tolist()
    print(f"universe_ids={len(ids)} quarters={len(quarters)}")

    concepts = None if args.refresh_cache else read_cached_frame(concept_cache_path, ["inclusion_date"])
    if concepts is None:
        concepts = fetch_concept_rows(rq, ids, args.concept_chunk_size)
        write_cached_frame(concept_cache_path, concepts)

    pit = None if args.refresh_cache else read_cached_frame(pit_cache_path, ["info_date", "valuation_date"])
    if pit is None:
        pit = fetch_pit_rows(rq, ids, args.start_quarter, args.end_quarter, args.pit_chunk_size)
        pit = pit.loc[pit["quarter"].isin(quarters)].copy()
        trading_map = next_trading_dates(rq, pit["info_date"])
        pit["valuation_date"] = pit["info_date"].map(trading_map)
        pit["panel_id"] = pit["order_book_id"].astype(str) + "__" + pit["quarter"].astype(str)
        write_cached_frame(pit_cache_path, pit)
    print(f"pit_selected_rows={len(pit)} unique_dates={pit['valuation_date'].nunique()}")

    market_cache_dir = Path(args.market_cache_dir) if args.market_cache_dir else raw_db_dir / f"market_fetch_cache_{suffix}"
    vendor_enterprise_value_fields = resolve_vendor_enterprise_value_fields(rq)
    market = fetch_market_for_dates(
        rq,
        pit,
        args.market_chunk_size,
        market_cache_dir,
        args.progress_every,
        args.refresh_cache,
        vendor_enterprise_value_fields,
    )
    raw = build_raw_database(universe, pit, market)
    financial = build_step1_financial_input(raw)
    financial = attach_historical_concept_tags(financial, concepts)
    labels = build_market_labels(raw)

    raw_path = raw_db_dir / f"raw_rqdatac_database_{suffix}.csv"
    financial_path = calculated_db_dir / f"calculated_feature_input_{suffix}.csv"
    labels_path = raw_db_dir / f"raw_market_labels_{suffix}.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    financial.to_csv(financial_path, index=False, encoding="utf-8-sig")
    labels.to_csv(labels_path, index=False, encoding="utf-8-sig")
    write_summary(raw_db_dir, calculated_db_dir, raw, financial, labels, concepts, args.start_quarter, args.end_quarter)

    print(f"raw_database_path={raw_path}")
    print(f"calculated_feature_input_path={financial_path}")
    print(f"market_labels_path={labels_path}")
    print(f"concept_tags_path={concept_cache_path}")
    print(f"raw_rows={len(raw)} financial_rows={len(financial)} labels_rows={len(labels)}")
    print("raw_financial_non_null_core=")
    print(raw[["operating_revenue_ttm_0", "net_profit_ttm_0", "ebitda_ttm", "total_assets"]].notna().sum().to_string())
    print("raw_market_non_null_core=")
    print(raw[["close", "market_cap", "pe_ratio_ttm", "pb_ratio", "ps_ratio_ttm", "ev_to_ebitda"]].notna().sum().to_string())
    if vendor_enterprise_value_fields:
        print("raw_vendor_enterprise_value_non_null=")
        print(raw[vendor_enterprise_value_fields].notna().sum().to_string())


if __name__ == "__main__":
    main()
