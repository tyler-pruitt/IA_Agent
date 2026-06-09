
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CORE_COLUMNS = [
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
]

DEFAULT_START_QUARTER = "2020q1"
DEFAULT_END_QUARTER = "2025q4"
DEFAULT_OUTPUT_DIR = "outputs/calculated_feature_database"
INDUSTRY_LEVEL_COLUMNS = ["first_industry_name", "second_industry_name", "third_industry_name"]
BUSINESS_TEXT_COLUMNS = [
    "business_description",
    "main_business",
    "business_scope",
    "company_profile",
    "主营业务",
    "经营范围",
    "业务描述",
]
CONCEPT_COLUMNS = ["concept_tags"]
FINANCIAL_STRUCTURE_FEATURES = [
    "roe",
    "roa",
    "net_margin",
    "operating_margin",
    "gross_margin",
    "ebitda_margin",
    "revenue_growth",
    "debt_ratio",
    "debt_to_equity",
    "current_ratio",
    "cash_ratio",
    "asset_turnover",
    "fixed_asset_ratio",
    "inventory_to_assets",
    "receivables_to_revenue",
    "ocf_margin",
    "fcff_margin",
    "fcfe_margin",
    "cash_conversion",
]
SCALE_FEATURES = ["log_total_assets", "log_revenue", "log_equity"]
BUSINESS_SIMILARITY_WEIGHT = 0.50
FINANCIAL_STRUCTURE_WEIGHT = 0.35
SCALE_SIMILARITY_WEIGHT = 0.15
TAXONOMY_ONLY_BUSINESS_SIMILARITY_WEIGHT = 0.35
TAXONOMY_ONLY_FINANCIAL_STRUCTURE_WEIGHT = 0.50

STEP2_DECISION_COLUMNS = [
    "is_financial_industry",
    "net_profit_ttm",
    "revenue_growth",
    "ebitda_value",
    "ebit_ttm",
    "operating_profit_ttm",
    "gross_profit_ttm",
    "total_assets_value",
    "total_liabilities_value",
    "cash_equivalent_value",
    "fixed_assets_base",
    "depreciation_amortization_value",
    "interest_bearing_debt",
    "pe_applicable",
    "pb_applicable",
    "ps_applicable",
    "ev_ebitda_applicable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 1 comparable-company outputs.")
    parser.add_argument("--start-quarter", default=DEFAULT_START_QUARTER, help="Start quarter used by the fetch artifact suffix.")
    parser.add_argument("--end-quarter", default=DEFAULT_END_QUARTER, help="End quarter used by the fetch artifact suffix.")
    parser.add_argument(
        "--financial-csv",
        default="",
        help="Analysis-ready financial input derived from the raw RQData database.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for Step 1 output files.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of peers per company.")
    parser.add_argument(
        "--min-listed-days",
        type=int,
        default=365,
        help="Minimum listing age for the clean universe.",
    )
    return parser.parse_args()


def to_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def winsorize(frame: pd.DataFrame, columns: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        series = result[column]
        if series.notna().sum() < 20:
            continue
        lo = series.quantile(lower)
        hi = series.quantile(upper)
        result[column] = series.clip(lower=lo, upper=hi)
    return result


def choose_first_positive(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    return primary.where(primary > 0, fallback)


def quarter_sort_value(quarter: object) -> int:
    value = str(quarter).lower().strip()
    if "q" not in value:
        return -1
    year, qtr = value.split("q", 1)
    try:
        return int(year) * 4 + int(qtr)
    except ValueError:
        return -1


def load_financials(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Financial CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def build_feature_frame(raw: pd.DataFrame, min_listed_days: int) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = raw.copy()
    frame["quarter"] = frame["quarter"].astype(str).str.lower()
    frame["panel_id"] = frame["order_book_id"].astype(str) + "__" + frame["quarter"]
    frame["info_date"] = pd.to_datetime(frame["info_date"], errors="coerce")
    frame["listed_date"] = pd.to_datetime(frame["listed_date"], errors="coerce")
    frame["de_listed_date"] = pd.to_datetime(
        frame["de_listed_date"].replace("0000-00-00", pd.NA), errors="coerce"
    )

    revenue = choose_first_positive(
        to_numeric(frame, "operating_revenueTTM"), to_numeric(frame, "total_operating_revenueTTM")
    )
    equity = choose_first_positive(to_numeric(frame, "equity_parent_company"), to_numeric(frame, "total_equity"))
    total_assets = to_numeric(frame, "total_assets")
    total_liabilities = to_numeric(frame, "total_liabilities")
    # Prefer PIT-sourced BS fields (available after fetch expansion), fall back to factor-sourced
    current_assets = to_numeric(frame, "pit_current_assets").fillna(to_numeric(frame, "current_assets"))
    current_liabilities = to_numeric(frame, "pit_current_liabilities").fillna(to_numeric(frame, "current_liabilities"))
    cash = to_numeric(frame, "cash_equivalent")
    net_profit = to_numeric(frame, "net_profitTTM")
    operating_profit = to_numeric(frame, "operating_profitTTM")
    gross_profit = to_numeric(frame, "gross_profit").fillna(to_numeric(frame, "gross_profitTTM"))
    ebitda = to_numeric(frame, "ebitda").fillna(to_numeric(frame, "ebitda_ttm"))
    ebit = to_numeric(frame, "ebitTTM")
    operating_cashflow = to_numeric(frame, "net_operate_cashflowTTM")
    fcff = to_numeric(frame, "fcff")
    fcfe = to_numeric(frame, "fcfe")
    inventory = to_numeric(frame, "pit_inventory").fillna(to_numeric(frame, "inventory"))
    receivables = to_numeric(frame, "pit_net_accts_receivable").fillna(to_numeric(frame, "net_accts_receivable"))
    operating_cost = to_numeric(frame, "cost_of_goods_sold").fillna(
        to_numeric(frame, "operating_cost").fillna(to_numeric(frame, "operating_costTTM"))
    )
    fixed_assets = choose_first_positive(
        to_numeric(frame, "total_fixed_assets"), to_numeric(frame, "pit_net_fixed_assets").fillna(to_numeric(frame, "net_fixed_assets"))
    )
    depreciation_amortization = to_numeric(frame, "depreciation_and_amortization").fillna(
        to_numeric(frame, "depreciation_and_amortization_ttm")
    )
    interest_expense = to_numeric(frame, "interest_expense")
    interest_debt = to_numeric(frame, "interest_bearing_debt")
    if interest_debt.isna().all():
        # Fallback for CSVs produced before fetch expansion
        interest_debt = (
            to_numeric(frame, "short_term_debt").fillna(0)
            + to_numeric(frame, "short_term_loans").fillna(0)
            + to_numeric(frame, "long_term_loans").fillna(0)
            + to_numeric(frame, "bond_payable").fillna(0)
        )
    else:
        interest_debt = interest_debt.fillna(0)

    frame["listed_age_days"] = (frame["info_date"] - frame["listed_date"]).dt.days
    frame["revenue_base"] = revenue
    frame["equity_base"] = equity
    revenue_history = frame[["order_book_id", "quarter"]].copy()
    revenue_history["quarter_sort"] = revenue_history["quarter"].map(quarter_sort_value)
    revenue_history["revenue_base"] = revenue
    prior_revenue = revenue_history.dropna(subset=["order_book_id", "revenue_base"]).copy()
    prior_revenue = prior_revenue.groupby(["order_book_id", "quarter_sort"], as_index=False)["revenue_base"].first()
    prior_revenue["quarter_sort"] = prior_revenue["quarter_sort"] + 4
    prior_revenue = prior_revenue.rename(columns={"revenue_base": "prior_year_revenue_base"})
    revenue_history = revenue_history.merge(prior_revenue, on=["order_book_id", "quarter_sort"], how="left")
    prior_year_revenue = pd.Series(revenue_history["prior_year_revenue_base"].to_numpy(), index=frame.index)
    frame["revenue_growth"] = safe_divide(revenue - prior_year_revenue, prior_year_revenue.where(prior_year_revenue > 0))
    frame["is_financial_industry"] = frame["first_industry_name"].astype(str).str.contains(
        "银行|非银行金融|综合金融|保险|证券", na=False
    )
    frame["net_profit_ttm"] = net_profit
    frame["ebitda_value"] = ebitda
    frame["ebit_ttm"] = ebit
    frame["operating_profit_ttm"] = operating_profit
    frame["gross_profit_ttm"] = gross_profit
    frame["total_assets_value"] = total_assets
    frame["total_liabilities_value"] = total_liabilities
    frame["cash_equivalent_value"] = cash
    frame["fixed_assets_base"] = fixed_assets
    frame["depreciation_amortization_value"] = depreciation_amortization
    frame["interest_bearing_debt"] = interest_debt
    frame["pe_applicable"] = net_profit > 0
    frame["pb_applicable"] = equity > 0
    frame["ps_applicable"] = revenue > 0
    frame["ev_ebitda_applicable"] = ebitda > 0
    frame["is_st_symbol"] = frame["symbol"].astype(str).str.contains("ST", case=False, na=False)
    frame["is_delisted_by_info_date"] = frame["de_listed_date"].notna() & (
        frame["de_listed_date"] <= frame["info_date"]
    )

    feature_data = {
        "roe": safe_divide(net_profit, equity),
        "roa": safe_divide(net_profit, total_assets),
        "net_margin": safe_divide(net_profit, revenue),
        "operating_margin": safe_divide(operating_profit, revenue),
        "gross_margin": safe_divide(gross_profit, revenue),
        "ebitda_margin": safe_divide(ebitda, revenue),
        "debt_ratio": safe_divide(total_liabilities, total_assets),
        "debt_to_equity": safe_divide(total_liabilities, equity),
        "current_ratio": safe_divide(current_assets, current_liabilities),
        "cash_ratio": safe_divide(cash, current_liabilities),
        "interest_coverage": safe_divide(ebit, interest_expense.where(interest_expense > 0)),
        "asset_turnover": safe_divide(revenue, total_assets),
        "fixed_asset_ratio": safe_divide(fixed_assets, total_assets),
        "inventory_to_assets": safe_divide(inventory, total_assets),
        "receivables_to_revenue": safe_divide(receivables, revenue),
        "inventory_turnover_proxy": safe_divide(operating_cost, inventory.where(inventory > 0)),
        "ocf_margin": safe_divide(operating_cashflow, revenue),
        "fcff_margin": safe_divide(fcff, revenue),
        "fcfe_margin": safe_divide(fcfe, revenue),
        "cash_conversion": safe_divide(operating_cashflow, net_profit.where(net_profit > 0)),
        "da_to_ebit": safe_divide(depreciation_amortization, ebit.where(ebit > 0)),
        "net_debt_to_ebitda": safe_divide(interest_debt - cash.fillna(0), ebitda.where(ebitda > 0)),
        "cash_to_assets": safe_divide(cash, total_assets),
        "log_total_assets": np.log1p(total_assets.where(total_assets > 0)),
        "log_revenue": np.log1p(revenue.where(revenue > 0)),
        "log_equity": np.log1p(equity.where(equity > 0)),
        "listed_age_years": frame["listed_age_days"] / 365.25,
    }
    feature_frame = pd.DataFrame(feature_data, index=frame.index)
    feature_columns = list(feature_frame.columns)
    frame = pd.concat([frame, feature_frame], axis=1)

    valid_mask = (
        frame["order_book_id"].notna()
        & frame["symbol"].notna()
        & frame["first_industry_name"].notna()
        & frame["info_date"].notna()
        & frame["listed_date"].notna()
        & (frame["listed_age_days"] >= min_listed_days)
        & ~frame["is_st_symbol"]
        & ~frame["is_delisted_by_info_date"]
        & (frame["revenue_base"] > 0)
        & (frame["equity_base"] > 0)
        & (total_assets > 0)
    )
    summary = {
        "raw_rows": int(len(frame)),
        "raw_companies": int(frame["order_book_id"].nunique()),
        "raw_quarters": int(frame["quarter"].nunique()),
        "missing_industry_rows": int(frame["first_industry_name"].isna().sum()),
        "st_symbol_rows": int(frame["is_st_symbol"].sum()),
        "delisted_by_info_date_rows": int(frame["is_delisted_by_info_date"].sum()),
        "short_listing_rows": int((frame["listed_age_days"] < min_listed_days).sum()),
        "non_positive_revenue_rows": int((frame["revenue_base"] <= 0).sum()),
        "non_positive_equity_rows": int((frame["equity_base"] <= 0).sum()),
        "clean_rows": int(valid_mask.sum()),
        "clean_companies": int(frame.loc[valid_mask, "order_book_id"].nunique()),
        "clean_quarters": int(frame.loc[valid_mask, "quarter"].nunique()),
    }

    available_business_text_columns = [column for column in BUSINESS_TEXT_COLUMNS if column in frame.columns]
    available_concept_columns = [column for column in CONCEPT_COLUMNS if column in frame.columns]
    clean = frame.loc[
        valid_mask,
        CORE_COLUMNS
        + available_business_text_columns
        + available_concept_columns
        + ["listed_age_days", "revenue_base", "equity_base"]
        + STEP2_DECISION_COLUMNS
        + feature_columns,
    ].copy()
    clean = winsorize(clean, feature_columns)
    usable_feature_columns = [column for column in feature_columns if clean[column].notna().sum() >= 20]
    dropped_feature_columns = [column for column in feature_columns if column not in usable_feature_columns]
    clean = clean[
        CORE_COLUMNS
        + available_business_text_columns
        + available_concept_columns
        + ["listed_age_days", "revenue_base", "equity_base"]
        + STEP2_DECISION_COLUMNS
        + usable_feature_columns
    ]
    clean = clean.reset_index(drop=True)
    summary["dropped_sparse_feature_count"] = len(dropped_feature_columns)
    summary["clean_rows_with_concept_tags"] = int(clean.get("concept_tags", pd.Series(dtype=object)).map(non_empty_text).ne("").sum())
    clean.attrs["feature_columns"] = usable_feature_columns
    clean.attrs["dropped_feature_columns"] = dropped_feature_columns
    return clean, summary


def build_model_matrix(clean: pd.DataFrame, numeric_features: list[str]) -> tuple[np.ndarray, ColumnTransformer]:
    categorical_features = [column for column in INDUSTRY_LEVEL_COLUMNS if column in clean.columns]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("industry", categorical_pipeline, categorical_features),
        ]
    )
    matrix = transformer.fit_transform(clean)
    return matrix, transformer


def dense_array(matrix: np.ndarray) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def assign_clusters(clean: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    dense_matrix = dense_array(matrix)
    assignments = clean[CORE_COLUMNS + ["listed_age_days"]].copy()
    assignments["kmeans_cluster"] = -1
    assignments["dbscan_cluster"] = -1
    assignments["dbscan_eps"] = np.nan

    for quarter, quarter_index in assignments.groupby("quarter", sort=True).groups.items():
        indices = np.array(list(quarter_index), dtype=int)
        quarter_matrix = dense_matrix[indices]
        n_rows = len(indices)
        if n_rows == 0:
            continue
        if n_rows == 1:
            assignments.loc[indices, "kmeans_cluster"] = 0
            continue
        kmeans_clusters = min(n_rows, min(80, max(8, int(round(np.sqrt(n_rows))))))
        kmeans = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init=20)
        assignments.loc[indices, "kmeans_cluster"] = kmeans.fit_predict(quarter_matrix)

        neighbor_count = min(10, n_rows - 1)
        if neighbor_count >= 2:
            neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(quarter_matrix)
            distances, _ = neighbors.kneighbors(quarter_matrix)
            eps = float(np.nanpercentile(distances[:, -1], 75)) if len(distances) else 0.5
            eps = max(eps, 0.5)
            dbscan = DBSCAN(eps=eps, min_samples=min(8, n_rows))
            assignments.loc[indices, "dbscan_cluster"] = dbscan.fit_predict(quarter_matrix)
            assignments.loc[indices, "dbscan_eps"] = eps
    return assignments


def non_empty_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "<na>"} else text


def same_industry_level(group_meta: pd.DataFrame, target: pd.Series, column: str) -> pd.Series:
    target_value = non_empty_text(target[column])
    if not target_value:
        return pd.Series(False, index=group_meta.index)
    return group_meta[column].map(non_empty_text) == target_value


def clean_business_text(value: object) -> str:
    text = non_empty_text(value)
    for boilerplate in ["一般项目", "许可项目", "主营业务", "经营范围", "包括", "从事"]:
        text = text.replace(boilerplate, " ")
    return " ".join(text.replace("；", " ").replace("，", " ").replace("。", " ").split()).lower()


def text_token_set(value: object) -> set[str]:
    text = clean_business_text(value)
    if not text:
        return set()
    ascii_tokens = {token for token in text.split() if len(token) >= 2}
    chinese_bigrams = {text[index : index + 2] for index in range(max(0, len(text) - 1)) if text[index : index + 2].strip()}
    return ascii_tokens | chinese_bigrams


def jaccard_similarity(left: object, right: object) -> float:
    left_tokens = text_token_set(left)
    right_tokens = text_token_set(right)
    if not left_tokens or not right_tokens:
        return float("nan")
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def concept_token_set(value: object) -> set[str]:
    text = non_empty_text(value)
    if not text:
        return set()
    separators = [";", "|", ",", "，"]
    for separator in separators[1:]:
        text = text.replace(separator, separators[0])
    return {token.strip().lower() for token in text.split(separators[0]) if token.strip()}


def concept_tfidf_document(value: object) -> str:
    return " ".join(sorted(token.replace(" ", "_") for token in concept_token_set(value)))


def taxonomy_business_similarity(
    same_first_industry: pd.Series,
    same_second_industry: pd.Series,
    same_third_industry: pd.Series,
) -> pd.Series:
    score = pd.Series(0.0, index=same_first_industry.index)
    score.loc[same_first_industry] = 0.50
    score.loc[same_first_industry & same_second_industry] = 0.75
    score.loc[same_first_industry & same_second_industry & same_third_industry] = 1.00
    return score


def optional_text_business_similarity(group_meta: pd.DataFrame, target: pd.Series) -> pd.Series:
    available_columns = [column for column in BUSINESS_TEXT_COLUMNS if column in group_meta.columns]
    if not available_columns:
        return pd.Series(np.nan, index=group_meta.index, dtype="float64")
    target_text = " ".join(clean_business_text(target[column]) for column in available_columns)
    if not target_text:
        return pd.Series(np.nan, index=group_meta.index, dtype="float64")
    scores = []
    for _, row in group_meta.iterrows():
        peer_text = " ".join(clean_business_text(row[column]) for column in available_columns)
        scores.append(jaccard_similarity(target_text, peer_text))
    return pd.Series(scores, index=group_meta.index, dtype="float64")


def concept_tfidf_documents(group_meta: pd.DataFrame) -> list[str]:
    available_columns = [column for column in CONCEPT_COLUMNS if column in group_meta.columns]
    if not available_columns:
        return []
    documents = []
    for _, row in group_meta.iterrows():
        peer_tags = ";".join(non_empty_text(row[column]) for column in available_columns)
        documents.append(concept_tfidf_document(peer_tags))
    return documents


def concept_tfidf_similarity_matrix(group_meta: pd.DataFrame) -> np.ndarray:
    documents = concept_tfidf_documents(group_meta)
    similarities = np.full((len(group_meta), len(group_meta)), np.nan, dtype="float64")
    if not documents or not any(documents):
        return similarities
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\S+\b", lowercase=False, smooth_idf=False)
    matrix = vectorizer.fit_transform(documents)
    scores = np.clip(cosine_similarity(matrix), 0.0, 1.0)
    non_empty = np.array([bool(document) for document in documents], dtype=bool)
    valid_pairs = np.outer(non_empty, non_empty)
    similarities[valid_pairs] = scores[valid_pairs]
    return similarities


def distance_to_similarity(distances: np.ndarray, feature_count: int) -> np.ndarray:
    if feature_count <= 0:
        return np.zeros_like(distances, dtype="float64")
    return np.exp(-distances / np.sqrt(feature_count))


def choose_similarity_weights(text_similarity: pd.Series, concept_similarity: pd.Series | None = None) -> tuple[float, float, float, str]:
    """Use full business weight only when real business text is available.

    If current artifacts do not include business-description text, business similarity is
    only an industry-taxonomy proxy. In that case the business component is downweighted
    and financial-structure similarity is upweighted to avoid overclaiming business fit.
    """
    if text_similarity.notna().any():
        return (
            BUSINESS_SIMILARITY_WEIGHT,
            FINANCIAL_STRUCTURE_WEIGHT,
            SCALE_SIMILARITY_WEIGHT,
            "text_business_similarity_with_taxonomy_fallback",
        )
    if concept_similarity is not None and concept_similarity.notna().any():
        return (
            BUSINESS_SIMILARITY_WEIGHT,
            FINANCIAL_STRUCTURE_WEIGHT,
            SCALE_SIMILARITY_WEIGHT,
            "concept_tag_business_similarity_with_taxonomy_fallback",
        )
    return (
        TAXONOMY_ONLY_BUSINESS_SIMILARITY_WEIGHT,
        TAXONOMY_ONLY_FINANCIAL_STRUCTURE_WEIGHT,
        SCALE_SIMILARITY_WEIGHT,
        "taxonomy_only_business_similarity_downweighted",
    )


def choose_peer_candidate_mask(
    same_full_industry: pd.Series,
    same_first_second_industry: pd.Series,
    same_first_industry: pd.Series,
    same_kmeans: pd.Series,
    not_self: pd.Series,
    top_n: int,
) -> tuple[pd.Series, str]:
    full_industry = same_full_industry & not_self
    if full_industry.any():
        full_industry_kmeans = full_industry & same_kmeans
        if full_industry_kmeans.sum() >= top_n:
            return full_industry_kmeans, "same_quarter_first_second_third_industry_and_kmeans"
        return full_industry, "same_quarter_first_second_third_industry"

    first_second_industry = same_first_second_industry & not_self
    if first_second_industry.any():
        first_second_industry_kmeans = first_second_industry & same_kmeans
        if first_second_industry_kmeans.sum() >= top_n:
            return first_second_industry_kmeans, "fallback_same_quarter_first_second_industry_and_kmeans"
        return first_second_industry, "fallback_same_quarter_first_second_industry"

    first_industry = same_first_industry & not_self
    if first_industry.any():
        first_industry_kmeans = first_industry & same_kmeans
        if first_industry_kmeans.sum() >= top_n:
            return first_industry_kmeans, "fallback_same_quarter_first_industry_and_kmeans"
        return first_industry, "fallback_same_quarter_first_industry"

    return not_self, "fallback_same_quarter_global_nearest_financials"


def build_peer_tables(
    clean: pd.DataFrame, assignments: pd.DataFrame, numeric_features: list[str], top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    financial_features = [column for column in FINANCIAL_STRUCTURE_FEATURES if column in numeric_features]
    scale_features = [column for column in SCALE_FEATURES if column in numeric_features]
    if not financial_features:
        financial_features = [column for column in numeric_features if column not in scale_features]
    if not scale_features:
        scale_features = [column for column in numeric_features if column.startswith("log_")]

    financial_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    financial_matrix = financial_pipeline.fit_transform(clean[financial_features])
    if scale_features:
        scale_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        scale_matrix = scale_pipeline.fit_transform(clean[scale_features])
    else:
        scale_matrix = np.zeros((len(clean), 1), dtype="float64")

    optional_meta_columns = [column for column in BUSINESS_TEXT_COLUMNS + CONCEPT_COLUMNS if column in clean.columns]
    meta = assignments[
        [
            "panel_id",
            "order_book_id",
            "symbol",
            "quarter",
            *INDUSTRY_LEVEL_COLUMNS,
            "kmeans_cluster",
            "dbscan_cluster",
        ]
    ].reset_index(drop=True)
    for column in optional_meta_columns:
        meta[column] = clean[column].reset_index(drop=True)
    records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for quarter, quarter_indices in meta.groupby("quarter", sort=True).groups.items():
        group_indices = np.array(list(quarter_indices), dtype=int)
        group_meta = meta.iloc[group_indices].reset_index(drop=True)
        group_financial_matrix = financial_matrix[group_indices]
        group_scale_matrix = scale_matrix[group_indices]
        financial_distances = pairwise_distances(group_financial_matrix, metric="euclidean")
        scale_distances = pairwise_distances(group_scale_matrix, metric="euclidean")
        financial_similarities = distance_to_similarity(financial_distances, len(financial_features))
        scale_similarities = distance_to_similarity(scale_distances, max(1, len(scale_features)))
        concept_similarities = concept_tfidf_similarity_matrix(group_meta)

        for local_idx, target in group_meta.iterrows():
            same_first_industry = same_industry_level(group_meta, target, "first_industry_name")
            same_second_industry = same_industry_level(group_meta, target, "second_industry_name")
            same_third_industry = same_industry_level(group_meta, target, "third_industry_name")
            same_first_second_industry = same_first_industry & same_second_industry
            same_full_industry = same_first_second_industry & same_third_industry
            taxonomy_similarity = taxonomy_business_similarity(
                same_first_industry, same_second_industry, same_third_industry
            )
            text_similarity = optional_text_business_similarity(group_meta, target)
            concept_similarity = pd.Series(concept_similarities[local_idx], index=group_meta.index, dtype="float64")
            same_kmeans = group_meta["kmeans_cluster"] == target["kmeans_cluster"]
            same_dbscan = (group_meta["dbscan_cluster"] == target["dbscan_cluster"]) & (target["dbscan_cluster"] != -1)
            not_self = group_meta.index != local_idx
            if text_similarity.notna().any():
                business_similarity = text_similarity.fillna(taxonomy_similarity)
            elif concept_similarity.notna().any():
                business_similarity = ((taxonomy_similarity + concept_similarity) / 2).where(
                    concept_similarity.notna(), taxonomy_similarity
                )
            else:
                business_similarity = taxonomy_similarity
            business_weight, financial_weight, scale_weight, business_similarity_source = choose_similarity_weights(
                text_similarity.loc[not_self], concept_similarity.loc[not_self]
            )

            candidate_mask, method = choose_peer_candidate_mask(
                same_full_industry=same_full_industry,
                same_first_second_industry=same_first_second_industry,
                same_first_industry=same_first_industry,
                same_kmeans=same_kmeans,
                not_self=not_self,
                top_n=top_n,
            )

            candidate_indices = np.flatnonzero(candidate_mask.to_numpy())
            combined_similarity = (
                business_weight * business_similarity.to_numpy(dtype="float64")
                + financial_weight * financial_similarities[local_idx]
                + scale_weight * scale_similarities[local_idx]
            )
            ranked = candidate_indices[np.argsort(-combined_similarity[candidate_indices])][:top_n]
            peer_ids: list[str] = []
            peer_symbols: list[str] = []
            peer_similarity_scores: list[float] = []

            for rank, peer_local_idx in enumerate(ranked, start=1):
                peer = group_meta.iloc[peer_local_idx]
                peer_ids.append(str(peer["order_book_id"]))
                peer_symbols.append(str(peer["symbol"]))
                peer_similarity_scores.append(float(combined_similarity[peer_local_idx]))
                records.append(
                    {
                        "target_order_book_id": target["order_book_id"],
                        "target_panel_id": target["panel_id"],
                        "target_symbol": target["symbol"],
                        "target_quarter": target["quarter"],
                        "target_industry": target["first_industry_name"],
                        "target_second_industry": target["second_industry_name"],
                        "target_third_industry": target["third_industry_name"],
                        "peer_rank": rank,
                        "peer_panel_id": peer["panel_id"],
                        "peer_order_book_id": peer["order_book_id"],
                        "peer_symbol": peer["symbol"],
                        "peer_quarter": peer["quarter"],
                        "peer_industry": peer["first_industry_name"],
                        "peer_second_industry": peer["second_industry_name"],
                        "peer_third_industry": peer["third_industry_name"],
                        "distance": financial_distances[local_idx, peer_local_idx],
                        "business_similarity": float(business_similarity.iloc[peer_local_idx]),
                        "financial_structure_similarity": float(financial_similarities[local_idx, peer_local_idx]),
                        "scale_similarity": float(scale_similarities[local_idx, peer_local_idx]),
                        "combined_similarity": float(combined_similarity[peer_local_idx]),
                        "business_similarity_source": business_similarity_source,
                        "business_similarity_weight": business_weight,
                        "financial_structure_weight": financial_weight,
                        "scale_similarity_weight": scale_weight,
                        "same_quarter": True,
                        "same_industry": bool(same_full_industry.iloc[peer_local_idx]),
                        "same_first_industry": bool(same_first_industry.iloc[peer_local_idx]),
                        "same_second_industry": bool(same_second_industry.iloc[peer_local_idx]),
                        "same_third_industry": bool(same_third_industry.iloc[peer_local_idx]),
                        "same_kmeans_cluster": bool(same_kmeans.iloc[peer_local_idx]),
                        "same_dbscan_cluster": bool(same_dbscan.iloc[peer_local_idx]),
                        "selection_method": method,
                    }
                )

            summary_records.append(
                {
                    "panel_id": target["panel_id"],
                    "order_book_id": target["order_book_id"],
                    "symbol": target["symbol"],
                    "quarter": target["quarter"],
                    "first_industry_name": target["first_industry_name"],
                    "second_industry_name": target["second_industry_name"],
                    "third_industry_name": target["third_industry_name"],
                    "kmeans_cluster": target["kmeans_cluster"],
                    "dbscan_cluster": target["dbscan_cluster"],
                    "peer_count": len(peer_ids),
                    "selection_method": method,
                    "peer_similarity_mean": float(np.mean(peer_similarity_scores)) if peer_similarity_scores else np.nan,
                    "peer_similarity_min": float(np.min(peer_similarity_scores)) if peer_similarity_scores else np.nan,
                    "business_similarity_source": business_similarity_source,
                    "business_similarity_weight": business_weight,
                    "financial_structure_weight": financial_weight,
                    "scale_similarity_weight": scale_weight,
                    "peer_order_book_ids": ";".join(peer_ids),
                    "peer_similarity_scores": ";".join(f"{score:.12g}" for score in peer_similarity_scores),
                    "peer_symbols": ";".join(peer_symbols),
                }
            )

    return pd.DataFrame(records), pd.DataFrame(summary_records)


def write_summary(
    output_dir: Path,
    filter_summary: dict[str, int],
    clean: pd.DataFrame,
    assignments: pd.DataFrame,
    peer_summary: pd.DataFrame,
    numeric_features: list[str],
) -> None:
    dropped_features = clean.attrs.get("dropped_feature_columns", [])
    lines = ["Step 1 Comparable Companies Run Summary", ""]
    lines.append("Universe filtering:")
    for key, value in filter_summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            f"Feature count: {len(numeric_features)}",
            "Features: " + ", ".join(numeric_features),
            "Dropped sparse features: " + (", ".join(dropped_features) if dropped_features else "none"),
            "Clustering inputs: financial features plus first/second/third industry one-hot encoding; no market-cap or valuation-label inputs.",
            "Peer selection: candidates still come from same-quarter industry tiers, but ranking uses weighted similarity.",
            "Business similarity uses text overlap when business-description text is supplied. If only industry taxonomy is available, the business proxy is downweighted to 35% and financial-structure similarity is upweighted to 50%.",
            "Concept tags from RQData concept_tags are treated as business evidence with TF-IDF cosine similarity when available, with inclusion-date filtering applied by the fetch step.",
            "Step 2 handoff: step1_to_step2_input.csv is the canonical input for rule-based multiple selection.",
            "Step 2 must not reload and re-filter the raw financial CSV independently.",
            "",
            f"K-means clusters: {assignments['kmeans_cluster'].nunique()}",
            f"DBSCAN clusters excluding noise: {assignments.loc[assignments['dbscan_cluster'] != -1, 'dbscan_cluster'].nunique()}",
            f"DBSCAN noise rows: {(assignments['dbscan_cluster'] == -1).sum()}",
            "",
            f"Peer summary rows: {len(peer_summary)}",
            f"Minimum peer count: {peer_summary['peer_count'].min()}",
            f"Median peer count: {peer_summary['peer_count'].median()}",
            f"Maximum peer count: {peer_summary['peer_count'].max()}",
            "",
            "Selection method counts:",
        ]
    )
    for method, count in peer_summary["selection_method"].value_counts().items():
        lines.append(f"- {method}: {count}")
    lines.extend(
        [
            "",
            f"Mean peer similarity: {float(peer_summary['peer_similarity_mean'].mean())}",
            f"Median peer similarity: {float(peer_summary['peer_similarity_mean'].median())}",
        ]
    )
    lines.extend(["", "Largest industries in clean universe:"])
    for industry, count in clean["first_industry_name"].value_counts().head(20).items():
        lines.append(f"- {industry}: {count}")
    (output_dir / "step1_run_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    suffix = f"{args.start_quarter}_{args.end_quarter}"
    input_path = Path(args.financial_csv) if args.financial_csv else output_dir / f"calculated_feature_input_{suffix}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_financials(input_path)
    clean, filter_summary = build_feature_frame(raw, args.min_listed_days)
    numeric_features = clean.attrs["feature_columns"]
    matrix, _ = build_model_matrix(clean, numeric_features)
    assignments = assign_clusters(clean, matrix)
    peer_pairs, peer_summary = build_peer_tables(clean, assignments, numeric_features, args.top_n)
    step2_input = clean.merge(
        peer_summary[
            [
                "panel_id",
                "kmeans_cluster",
                "dbscan_cluster",
                "peer_count",
                "selection_method",
                "peer_similarity_mean",
                "peer_similarity_min",
                "business_similarity_source",
                "business_similarity_weight",
                "financial_structure_weight",
                "scale_similarity_weight",
                "peer_order_book_ids",
                "peer_similarity_scores",
                "peer_symbols",
            ]
        ],
        on="panel_id",
        how="left",
        validate="one_to_one",
    )

    clean.to_csv(output_dir / "step1_clean_universe.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(output_dir / "step1_cluster_assignments.csv", index=False, encoding="utf-8-sig")
    peer_pairs.to_csv(output_dir / "step1_comparable_pairs.csv", index=False, encoding="utf-8-sig")
    peer_summary.to_csv(output_dir / "step1_comparable_summary.csv", index=False, encoding="utf-8-sig")
    step2_input.to_csv(output_dir / "step1_to_step2_input.csv", index=False, encoding="utf-8-sig")
    write_summary(output_dir, filter_summary, clean, assignments, peer_summary, numeric_features)

    print(f"clean_universe_rows={len(clean)}")
    print(f"kmeans_clusters={assignments['kmeans_cluster'].nunique()}")
    print(f"dbscan_clusters_ex_noise={assignments.loc[assignments['dbscan_cluster'] != -1, 'dbscan_cluster'].nunique()}")
    print(f"peer_summary_rows={len(peer_summary)}")
    print(f"step2_input_rows={len(step2_input)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
