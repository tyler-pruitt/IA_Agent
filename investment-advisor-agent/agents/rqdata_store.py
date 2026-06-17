"""RQData 离线数据适配层。

该模块只读 `outputs/raw_rqdatac_database` 中的结构化 CSV，把 RQData
面板数据转换为 agent 层可复用的 payload。它不替代 AKShare/MCP 的实时数据，
而是作为离线、PIT、行业和概念标签补充源。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import RQDATA_STORE_DIR, RQDATA_TECHNICAL_STORE_DIR

logger = logging.getLogger(__name__)


RAW_DATABASE_FILE = "raw_rqdatac_database_2020q1_2025q4.csv"
MARKET_LABELS_FILE = "raw_market_labels_2020q1_2025q4.csv"
PIT_DISCLOSURES_FILE = "pit_disclosures_2020q1_2025q4.csv"
CONCEPT_TAGS_FILE = "concept_tags_2020q1_2025q4.csv"
METADATA_FILE = "metadata_universe_2020q1_2025q4.csv"
CONSENSUS_FILE = "concensus.csv"
SECTOR_FILE = "sector.csv"
PRICE_FILE = "price.csv"
TURNOVER_FILE = "turnover.csv"
INFLOW_OUTFLOW_FILE = "inflow_outflow.csv"

TECHNICAL_KEY_INDICATORS = [
    "MA5",
    "MA10",
    "MA20",
    "MA30",
    "MA60",
    "EMA12",
    "EMA26",
    "MACD_DIFF",
    "MACD_DEA",
    "MACD_HIST",
    "KDJ_K",
    "KDJ_D",
    "KDJ_J",
    "RSI6",
    "RSI10",
    "BOLL",
    "BOLL_UP",
    "BOLL_DOWN",
    "ATR",
    "OBV",
    "VOL5",
    "VOL10",
    "VOL20",
    "WR",
]

PREFERRED_COLUMNS = {
    "metadata": [
        "order_book_id",
        "symbol",
        "trading_code",
        "exchange",
        "status",
        "listed_date",
        "de_listed_date",
        "sector_code_name",
        "industry_name",
        "province",
        "first_industry_name",
        "second_industry_name",
        "third_industry_name",
    ],
    "valuation": [
        "order_book_id",
        "quarter",
        "info_date",
        "valuation_date",
        "close_price",
        "close",
        "volume",
        "total_turnover",
        "total_market_cap",
        "market_cap",
        "pe_ratio_ttm",
        "pb_ratio",
        "ps_ratio_ttm",
        "ev_to_ebitda",
        "vendor_enterprise_value",
        "enterprise_value",
    ],
    "indicator": [
        "order_book_id",
        "quarter",
        "info_date",
        "factor_date",
        "operating_revenue_ttm_0",
        "net_profit_ttm_0",
        "operating_profitTTM",
        "ebitda_ttm",
        "ebit_ttm",
        "return_on_equity_weighted_average",
        "net_operate_cashflowTTM",
        "fcff_ttm",
        "fcfe_ttm",
    ],
    "balance": [
        "order_book_id",
        "quarter",
        "info_date",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "equity_parent_company",
        "cash_equivalent",
        "total_fixed_assets",
        "bond_payable",
        "interest_bearing_debt",
    ],
    "profit": [
        "order_book_id",
        "quarter",
        "info_date",
        "operating_revenue",
        "operating_revenue_ttm_0",
        "net_profit",
        "net_profit_ttm_0",
        "net_profit_parent_company",
        "gross_profit",
        "ebitda",
        "ebit",
        "profit_before_tax",
        "income_tax",
        "r_n_d",
        "selling_expense",
        "adjusted_net_profit",
    ],
    "cashflow": [
        "order_book_id",
        "quarter",
        "info_date",
        "cash_equivalent",
        "pit_cash_equivalent",
        "net_operate_cashflowTTM",
        "fcff_ttm",
        "fcfe_ttm",
    ],
    "pit": [
        "order_book_id",
        "quarter",
        "info_date",
        "valuation_date",
        "operating_revenue",
        "net_profit",
        "net_profit_parent_company",
        "total_assets",
        "total_liabilities",
        "equity_parent_company",
        "cash_equivalent",
        "return_on_equity_weighted_average",
        "rice_create_tm",
        "if_adjusted",
    ],
    "consensus": [
        "order_book_id",
        "date",
        "institute",
        "price_raw",
        "price_prd",
        "grd_coef",
        "grd_prd",
        "quarter_recommendation",
        "half_year_target_price",
        "one_year_target_price",
        "rice_create_tm",
        "create_tm",
    ],
    "sector": [
        "sector_code",
        "sector_name_cn",
        "stock_code",
    ],
    "price": [
        "order_book_id",
        "date",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "volume",
        "turnover_rate",
        "total_turnover",
        "limit_up",
        "limit_down",
        "num_trades",
    ],
    "inflow_outflow": [
        "order_book_id",
        "date",
        "buy_volume",
        "buy_value",
        "sell_volume",
        "sell_value",
        "net_volume",
        "net_value",
        "net_value_wan",
        "净流入",
    ],
}


class RQDataStore:
    """面向 agent 的 RQData CSV 查询器。"""

    def __init__(self, root_dir: str | None = None, technical_root_dir: str | None = None):
        self.root_dir = Path(root_dir or RQDATA_STORE_DIR).expanduser()
        self.technical_root_dir = Path(technical_root_dir or RQDATA_TECHNICAL_STORE_DIR).expanduser()
        self._frames: dict[str, pd.DataFrame] = {}
        self._load_counts: dict[str, int] = {}

    @property
    def available(self) -> bool:
        """判断离线数据目录是否具备核心文件。"""
        required = [RAW_DATABASE_FILE, METADATA_FILE, CONCEPT_TAGS_FILE]
        return self.root_dir.exists() and all((self.root_dir / name).exists() for name in required)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "root_dir": str(self.root_dir),
            "technical_root_dir": str(self.technical_root_dir),
            "available": self.available,
            "technical_available": self.technical_root_dir.exists(),
            "loaded_tables": sorted(self._frames.keys()),
            "load_counts": dict(self._load_counts),
        }

    def normalize_symbol(self, symbol: str) -> str:
        """把 `000001`、`SZ000001` 等输入规范为 RQData 的 order_book_id。"""
        text = str(symbol or "").strip().upper()
        if not text:
            return ""

        dotted = re.search(r"(\d{6})\.(XSHE|XSHG|XBSE)", text)
        if dotted:
            return f"{dotted.group(1)}.{dotted.group(2)}"

        prefixed = re.search(r"(SZ|SH|BJ)(\d{6})", text)
        if prefixed:
            exchange = {"SZ": "XSHE", "SH": "XSHG", "BJ": "XBSE"}[prefixed.group(1)]
            return f"{prefixed.group(2)}.{exchange}"

        match = re.search(r"(\d{6})", text)
        if not match:
            return text

        code = match.group(1)
        if code.startswith(("600", "601", "603", "605", "688", "689", "900")):
            return f"{code}.XSHG"
        if code.startswith(("430", "820", "830", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "920")):
            return f"{code}.XBSE"
        return f"{code}.XSHE"

    def get_metadata(self, symbol: str) -> dict[str, Any]:
        order_book_id = self.normalize_symbol(symbol)
        frame = self._table("metadata")
        if frame.empty or "order_book_id" not in frame.columns:
            return self._empty_payload(symbol, "metadata", "metadata_unavailable")

        rows = frame.loc[frame["order_book_id"] == order_book_id]
        data = _records(rows, PREFERRED_COLUMNS["metadata"], limit=1)
        payload = self._base_payload(symbol, order_book_id, "metadata")
        payload["data"] = data
        if data:
            payload["metadata"] = data[0]
        else:
            payload["source_warning"] = f"RQData 元数据未找到 {order_book_id}"
        return payload

    def get_concepts(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        order_book_id = self.normalize_symbol(symbol)
        frame = self._table("concepts")
        if frame.empty or "order_book_id" not in frame.columns:
            return self._empty_payload(symbol, "concept_tags", "concept_tags_unavailable")

        rows = frame.loc[frame["order_book_id"] == order_book_id]
        if "inclusion_date" in rows.columns:
            rows = rows.sort_values("inclusion_date", ascending=False, na_position="last")

        data = _records(rows, ["order_book_id", "concept_name", "inclusion_date"], limit=limit)
        payload = self._base_payload(symbol, order_book_id, "concept_tags")
        payload["data"] = data
        payload["concept_names"] = [item.get("concept_name") for item in data if item.get("concept_name")]
        if not data:
            payload["source_warning"] = f"RQData 概念标签未找到 {order_book_id}"
        return payload

    def get_market_labels(
        self,
        symbol: str,
        quarter: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        order_book_id = self.normalize_symbol(symbol)
        rows = self._symbol_rows("market_labels", order_book_id, quarter)
        data = _records(rows, PREFERRED_COLUMNS["valuation"], limit=limit)
        payload = self._base_payload(symbol, order_book_id, "market_labels")
        payload["quarter"] = _latest_quarter(rows)
        payload["data"] = data
        if not data:
            payload["source_warning"] = f"RQData 市场标签未找到 {order_book_id}"
        return payload

    def get_pit_disclosures(
        self,
        symbol: str,
        quarter: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        order_book_id = self.normalize_symbol(symbol)
        rows = self._symbol_rows("pit", order_book_id, quarter)
        data = _records(rows, PREFERRED_COLUMNS["pit"], limit=limit)
        payload = self._base_payload(symbol, order_book_id, "pit_disclosures")
        payload["quarter"] = _latest_quarter(rows)
        payload["data"] = data
        if not data:
            payload["source_warning"] = f"RQData PIT 披露数据未找到 {order_book_id}"
        return payload

    def get_fundamental(
        self,
        symbol: str,
        report_type: str = "all",
        quarter: str | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """返回与 MCP 基本面工具相近的结构化 payload。"""
        order_book_id = self.normalize_symbol(symbol)
        rows = self._symbol_rows("raw", order_book_id, quarter)
        market_rows = self._symbol_rows("market_labels", order_book_id, quarter)
        pit_rows = self._symbol_rows("pit", order_book_id, quarter)

        payload = self._base_payload(symbol, order_book_id, "fundamental")
        payload.update(
            {
                "report_type": report_type,
                "quarter": _latest_quarter(rows) or _latest_quarter(market_rows),
                "metadata": self.get_metadata(order_book_id).get("metadata", {}),
                "concept_tags": self.get_concepts(order_book_id).get("data", []),
            }
        )

        sections = {
            "valuation": _records(
                market_rows if not market_rows.empty else rows,
                PREFERRED_COLUMNS["valuation"],
                limit=limit,
            ),
            "financial_indicator": _records(rows, PREFERRED_COLUMNS["indicator"], limit=limit),
            "balance_sheet": _records(rows, PREFERRED_COLUMNS["balance"], limit=limit),
            "profit_sheet": _records(rows, PREFERRED_COLUMNS["profit"], limit=limit),
            "cash_flow": _records(rows, PREFERRED_COLUMNS["cashflow"], limit=limit),
            "pit_disclosures": _records(pit_rows, PREFERRED_COLUMNS["pit"], limit=limit),
        }

        if report_type == "indicator":
            payload["financial_indicator"] = sections["financial_indicator"]
        elif report_type == "valuation":
            payload["valuation"] = sections["valuation"]
        elif report_type == "balance":
            payload["balance_sheet"] = sections["balance_sheet"]
        elif report_type == "profit":
            payload["profit_sheet"] = sections["profit_sheet"]
        elif report_type == "cashflow":
            payload["cash_flow"] = sections["cash_flow"]
        elif report_type == "pit":
            payload["pit_disclosures"] = sections["pit_disclosures"]
        else:
            payload.update(sections)

        if rows.empty and market_rows.empty and pit_rows.empty:
            payload["source_warning"] = f"RQData 基本面数据未找到 {order_book_id}"
        return payload

    def get_industry_scorecard(
        self,
        symbol: str,
        quarter: str | None = None,
        min_peer_count: int = 8,
        peer_limit: int = 20,
    ) -> dict[str, Any]:
        """计算股票在所属行业内的相对分位和排名。"""
        order_book_id = self.normalize_symbol(symbol)
        frame = self._table("raw")
        if frame.empty or "order_book_id" not in frame.columns:
            return self._empty_payload(symbol, "industry_scorecard", "raw_database_unavailable")

        symbol_rows = self._symbol_rows("raw", order_book_id, quarter)
        if symbol_rows.empty:
            return self._empty_payload(symbol, "industry_scorecard", f"symbol_not_found:{order_book_id}")

        target = symbol_rows.iloc[0]
        target_quarter = str(quarter or target.get("quarter") or "")
        quarter_rows = frame
        if target_quarter and "quarter" in frame.columns:
            quarter_rows = frame.loc[frame["quarter"].astype(str).str.lower() == target_quarter.lower()]

        industry_level, industry_name, peers = self._select_peer_group(
            quarter_rows,
            target,
            min_peer_count=min_peer_count,
        )
        if peers.empty:
            return self._empty_payload(symbol, "industry_scorecard", "industry_peers_unavailable")

        scored = _score_peer_frame(peers)
        target_scored = scored.loc[scored["order_book_id"] == order_book_id]
        if target_scored.empty:
            return self._empty_payload(symbol, "industry_scorecard", "target_score_unavailable")

        target_score = target_scored.iloc[0]
        scored = scored.sort_values("overall_score", ascending=False, na_position="last")
        scored["industry_rank"] = range(1, len(scored) + 1)
        target_rank_row = scored.loc[scored["order_book_id"] == order_book_id].iloc[0]

        payload = self._base_payload(symbol, order_book_id, "industry_scorecard")
        payload.update(
            {
                "quarter": target_quarter,
                "industry_level": industry_level,
                "industry_name": industry_name,
                "peer_count": int(len(scored)),
                "industry_rank": int(target_rank_row["industry_rank"]),
                "overall_score": _safe_float(target_score.get("overall_score")),
                "overall_percentile": _safe_float(target_score.get("overall_percentile")),
                "metrics": _metric_payload(target_score, scored),
                "peer_table": _peer_table_payload(scored.head(peer_limit)),
                "methodology": {
                    "position": "结果层: 基于 RQData PIT 面板计算行业相对位置。",
                    "input": "输入层: 最新报告期的财务、估值、行业分类和市场标签。",
                    "calculation": "计算层: 高优指标取行业分位，低优估值指标取反向分位，综合分为可用指标均值。",
                    "quality_check": "质量复核: 样本不足、缺失值或负估值会降低可用指标数。",
                },
                "data_quality": _scorecard_quality_notes(scored, target_score),
            }
        )
        return payload

    def get_technical_indicators(
        self,
        symbol: str,
        date: str | None = None,
    ) -> dict[str, Any]:
        """读取按交易日缓存的 RQData 技术指标 parquet/csv。"""
        order_book_id = self.normalize_symbol(symbol)
        path = self._technical_file(date)
        payload = self._base_payload(symbol, order_book_id, "technical_indicators")
        payload.update(
            {
                "source": "rqdatac_technical_parquet",
                "source_note": "来自 outputs/raw_rqdatac_technical_indicators 的米筐技术指标缓存。",
                "data": [],
                "key_indicators": {},
                "indicator_count": 0,
                "non_null_indicator_count": 0,
            }
        )

        if path is None:
            payload["error"] = "technical_indicator_cache_unavailable"
            payload["source_warning"] = f"未找到技术指标缓存目录或文件: {self.technical_root_dir}"
            return payload

        try:
            frame = _read_table_file(path)
        except Exception as exc:
            logger.warning("[RQDataStore] 技术指标读取失败: %s", exc)
            payload["error"] = "technical_indicator_read_failed"
            payload["source_warning"] = f"技术指标缓存读取失败: {path}"
            payload["cache_path"] = str(path)
            return payload

        if frame.empty or "order_book_id" not in frame.columns:
            payload["error"] = "technical_indicator_empty"
            payload["source_warning"] = f"技术指标缓存为空或缺少 order_book_id: {path}"
            payload["cache_path"] = str(path)
            return payload

        frame = frame.copy()
        frame["order_book_id"] = frame["order_book_id"].astype(str).str.upper()
        rows = frame.loc[frame["order_book_id"] == order_book_id]
        if rows.empty:
            payload["error"] = f"symbol_not_found:{order_book_id}"
            payload["source_warning"] = f"技术指标缓存未找到 {order_book_id}"
            payload["cache_path"] = str(path)
            return payload

        row = rows.iloc[0]
        cleaned = _clean_record(row.to_dict())
        indicator_columns = [
            column
            for column in frame.columns
            if column not in {"order_book_id", "date"} and column in cleaned
        ]
        key_indicators = {
            column: cleaned[column]
            for column in TECHNICAL_KEY_INDICATORS
            if column in cleaned and cleaned[column] is not None
        }
        non_null_count = sum(cleaned.get(column) is not None for column in indicator_columns)

        payload.update(
            {
                "date": _date_text(cleaned.get("date")) or _date_from_technical_path(path),
                "cache_path": str(path),
                "data": [cleaned],
                "key_indicators": key_indicators,
                "indicator_count": len(indicator_columns),
                "non_null_indicator_count": int(non_null_count),
            }
        )
        if non_null_count == 0:
            payload["source_warning"] = f"技术指标缓存存在 {order_book_id}，但指标列均为空。"
        return payload

    def get_technical_indicator_timeseries(
        self,
        symbol: str,
        limit: int = 30,
        indicators: list[str] | None = None,
    ) -> dict[str, Any]:
        """读取近一段时间的 RQData 技术指标序列。"""
        order_book_id = self.normalize_symbol(symbol)
        payload = self._base_payload(symbol, order_book_id, "technical_indicator_timeseries")
        selected_indicators = indicators or TECHNICAL_KEY_INDICATORS
        payload.update(
            {
                "source": "rqdatac_technical_parquet",
                "source_note": "来自 outputs/raw_rqdatac_technical_indicators 的多日米筐技术指标缓存。",
                "data": [],
                "indicators": selected_indicators,
            }
        )

        paths = self._recent_technical_files(limit=limit)
        if not paths:
            payload["error"] = "technical_indicator_cache_unavailable"
            payload["source_warning"] = f"未找到技术指标缓存目录或文件: {self.technical_root_dir}"
            return payload

        records = []
        read_errors = []
        wanted_columns = ["order_book_id", "date"] + selected_indicators
        for path in paths:
            try:
                frame = _read_table_file(path)
            except Exception as exc:
                read_errors.append(f"{path.name}: {exc}")
                continue
            if frame.empty or "order_book_id" not in frame.columns:
                continue

            columns = [column for column in wanted_columns if column in frame.columns]
            if "order_book_id" not in columns:
                continue
            rows = frame.loc[frame["order_book_id"].astype(str).str.upper() == order_book_id, columns]
            if rows.empty:
                continue

            row = rows.iloc[0].copy()
            if "date" not in row or pd.isna(row.get("date")):
                row["date"] = _date_from_technical_path(path)
            records.append(_clean_record(row.to_dict()))

        if read_errors:
            payload["read_errors"] = read_errors[:5]
        if not records:
            payload["error"] = f"symbol_not_found:{order_book_id}"
            payload["source_warning"] = f"技术指标时序缓存未找到 {order_book_id}"
            return payload

        records = sorted(records, key=lambda item: _date_text(item.get("date")))
        payload["data"] = records[-limit:]
        payload["record_count"] = len(payload["data"])
        payload["start_date"] = _date_text(payload["data"][0].get("date"))
        payload["end_date"] = _date_text(payload["data"][-1].get("date"))
        payload["trend_summary"] = _technical_timeseries_summary(payload["data"])
        return payload

    def get_consensus(
        self,
        symbol: str,
        limit: int = 30,
    ) -> dict[str, Any]:
        """读取分析师一致预期目标价和评级数据。"""
        order_book_id = self.normalize_symbol(symbol)
        frame = self._table("consensus")
        payload = self._base_payload(symbol, order_book_id, "analyst_consensus")
        payload.update(
            {
                "source_note": "来自 outputs/raw_rqdatac_database/concensus.csv 的分析师一致预期数据。",
                "data": [],
                "summary": {},
            }
        )

        if frame.empty or "order_book_id" not in frame.columns:
            payload["error"] = "consensus_unavailable"
            payload["source_warning"] = f"一致预期数据不可用: {self.root_dir / CONSENSUS_FILE}"
            return payload

        rows = frame.loc[frame["order_book_id"] == order_book_id].copy()
        if rows.empty:
            payload["error"] = f"symbol_not_found:{order_book_id}"
            payload["source_warning"] = f"一致预期未找到 {order_book_id}"
            return payload

        for column in [
            "price_raw",
            "grd_coef",
            "quarter_recommendation",
            "half_year_target_price",
            "one_year_target_price",
        ]:
            if column in rows.columns:
                rows[column] = pd.to_numeric(rows[column], errors="coerce")
        for column in ["date", "rice_create_tm", "create_tm"]:
            if column in rows.columns:
                rows[column] = pd.to_datetime(rows[column], errors="coerce")

        sort_columns = [column for column in ["date", "rice_create_tm", "create_tm"] if column in rows.columns]
        if sort_columns:
            rows = rows.sort_values(sort_columns, ascending=False, na_position="last")

        latest_by_institute = rows
        if "institute" in rows.columns:
            latest_by_institute = rows.drop_duplicates("institute", keep="first")

        close_price = self._latest_close_price(order_book_id)
        target_candidates = _target_price_series(latest_by_institute)
        target_mean = _safe_float(target_candidates.mean()) if not target_candidates.dropna().empty else None
        upside = None
        if target_mean is not None and close_price and close_price > 0:
            upside = target_mean / close_price - 1

        rating_values = pd.Series(dtype="float64")
        for column in ["quarter_recommendation", "grd_coef"]:
            if column in latest_by_institute.columns:
                rating_values = pd.to_numeric(latest_by_institute[column], errors="coerce").dropna()
                if not rating_values.empty:
                    break
        rating_mean = _safe_float(rating_values.mean()) if not rating_values.empty else None

        summary = {
            "latest_date": _date_text(rows.iloc[0].get("date")) if not rows.empty else "",
            "record_count": int(len(rows)),
            "institute_count": int(latest_by_institute["institute"].nunique()) if "institute" in latest_by_institute.columns else int(len(latest_by_institute)),
            "target_price_mean": target_mean,
            "raw_target_price_mean": _safe_float(pd.to_numeric(latest_by_institute.get("price_raw"), errors="coerce").mean()) if "price_raw" in latest_by_institute.columns else None,
            "half_year_target_price_mean": _safe_float(pd.to_numeric(latest_by_institute.get("half_year_target_price"), errors="coerce").mean()) if "half_year_target_price" in latest_by_institute.columns else None,
            "one_year_target_price_mean": _safe_float(pd.to_numeric(latest_by_institute.get("one_year_target_price"), errors="coerce").mean()) if "one_year_target_price" in latest_by_institute.columns else None,
            "rating_coef_mean": rating_mean,
            "rating_label": _rating_label(rating_mean),
            "close_price": close_price,
            "target_price_upside": _safe_float(upside),
            "rating_distribution": _rating_distribution(latest_by_institute),
        }

        payload["summary"] = summary
        payload["data"] = _records(rows, PREFERRED_COLUMNS["consensus"], limit=limit)
        payload["latest_by_institute"] = _records(latest_by_institute, PREFERRED_COLUMNS["consensus"], limit=limit)
        return payload

    def get_sector_constituents(
        self,
        name: str = "",
        limit: int = 50,
    ) -> dict[str, Any]:
        """读取本地 sector.csv 的行业成分股映射，作为实时板块接口失败时的兜底。"""
        frame = self._table("sector")
        payload = {
            "dataset": "rqdata_sector_constituents",
            "source": "rqdatac_offline",
            "source_note": "来自 outputs/raw_rqdatac_database/sector.csv 的 RQData 行业映射，仅包含行业归属和成分股，不包含实时涨跌幅。",
            "sector": name,
            "data": [],
        }
        if frame.empty:
            payload["error"] = "sector_mapping_unavailable"
            return payload

        rows = frame.copy()
        query = str(name or "").strip()
        if query:
            text = query.lower()
            matched = rows.loc[
                rows.get("sector_name_cn", pd.Series(dtype=str)).astype(str).str.contains(query, case=False, na=False)
                | rows.get("sector_code", pd.Series(dtype=str)).astype(str).str.lower().str.contains(text, na=False)
            ]
            if not matched.empty:
                rows = matched
            else:
                payload["source_warning"] = f"本地 sector.csv 未匹配到行业: {query}"
                rows = rows.iloc[0:0]

        sector_names = sorted(rows["sector_name_cn"].dropna().astype(str).unique().tolist()) if "sector_name_cn" in rows.columns else []
        payload.update(
            {
                "sector_names": sector_names,
                "constituent_count": int(rows["stock_code"].nunique()) if "stock_code" in rows.columns else int(len(rows)),
                "data": _records(rows, PREFERRED_COLUMNS["sector"], limit=limit),
            }
        )
        return payload

    def get_price_volume(
        self,
        symbol: str,
        period: str = "daily",
        start_date: str = "",
        end_date: str = "",
        limit: int = 120,
    ) -> dict[str, Any]:
        """读取本地 price.csv 日线 OHLCV。"""
        order_book_id = self.normalize_symbol(symbol)
        payload = self._base_payload(symbol, order_book_id, "price_volume")
        payload.update(
            {
                "source_note": "来自 outputs/raw_rqdatac_database/price.csv 的本地 RQData 日线 OHLCV。",
                "period": period,
                "adjust": "rqdata_raw",
                "data": [],
            }
        )
        if period != "daily":
            payload["error"] = f"unsupported_period:{period}"
            payload["source_warning"] = "本地 price.csv 仅提供日线数据。"
            return payload

        frame = self._table("price")
        if frame.empty or "order_book_id" not in frame.columns:
            payload["error"] = "price_unavailable"
            payload["source_warning"] = f"本地 price.csv 不可用: {self.root_dir / PRICE_FILE}"
            return payload

        rows = frame.loc[frame["order_book_id"] == order_book_id].copy()
        if rows.empty:
            payload["error"] = f"symbol_not_found:{order_book_id}"
            payload["source_warning"] = f"本地 price.csv 未找到 {order_book_id}"
            return payload

        if "date" in rows.columns:
            rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
            if start_date:
                start = pd.to_datetime(start_date, errors="coerce")
                if pd.notna(start):
                    rows = rows.loc[rows["date"] >= start]
            if end_date:
                end = pd.to_datetime(end_date, errors="coerce")
                if pd.notna(end):
                    rows = rows.loc[rows["date"] <= end]
            rows = rows.sort_values("date", ascending=True, na_position="last")

        rows = self._attach_turnover(rows, order_book_id)

        payload["data"] = _records(rows.tail(limit), PREFERRED_COLUMNS["price"], limit=limit)
        payload["record_count"] = int(len(rows))
        if payload["data"]:
            payload["start_date"] = payload["data"][0].get("date")
            payload["end_date"] = payload["data"][-1].get("date")
        return payload

    def get_capital_flow(
        self,
        symbol: str,
        scope: str = "individual",
        limit: int = 60,
    ) -> dict[str, Any]:
        """读取本地 inflow_outflow.csv 个股资金流。"""
        order_book_id = self.normalize_symbol(symbol)
        payload = self._base_payload(symbol, order_book_id, "capital_flow")
        payload.update(
            {
                "source_note": "来自 outputs/raw_rqdatac_database/inflow_outflow.csv 的本地 RQData 个股资金流。",
                "scope": scope,
                "data": [],
            }
        )
        if scope != "individual":
            payload["error"] = f"unsupported_scope:{scope}"
            payload["source_warning"] = "本地 inflow_outflow.csv 仅提供个股资金流。"
            return payload

        frame = self._table("inflow_outflow")
        if frame.empty or "order_book_id" not in frame.columns:
            payload["error"] = "capital_flow_unavailable"
            payload["source_warning"] = f"本地 inflow_outflow.csv 不可用: {self.root_dir / INFLOW_OUTFLOW_FILE}"
            return payload

        rows = frame.loc[frame["order_book_id"] == order_book_id].copy()
        if rows.empty:
            payload["error"] = f"symbol_not_found:{order_book_id}"
            payload["source_warning"] = f"本地 inflow_outflow.csv 未找到 {order_book_id}"
            return payload

        for column in ["buy_volume", "buy_value", "sell_volume", "sell_value"]:
            if column in rows.columns:
                rows[column] = pd.to_numeric(rows[column], errors="coerce")
        if {"buy_volume", "sell_volume"}.issubset(rows.columns):
            rows["net_volume"] = rows["buy_volume"] - rows["sell_volume"]
        if {"buy_value", "sell_value"}.issubset(rows.columns):
            rows["net_value"] = rows["buy_value"] - rows["sell_value"]
            rows["net_value_wan"] = rows["net_value"] / 10000
            rows["净流入"] = rows["net_value_wan"].map(lambda value: f"{value:.2f}万元" if pd.notna(value) else None)

        if "date" in rows.columns:
            rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
            rows = rows.sort_values("date", ascending=False, na_position="last")

        payload["data"] = _records(rows, PREFERRED_COLUMNS["inflow_outflow"], limit=limit)
        payload["record_count"] = int(len(rows))
        if payload["data"]:
            payload["latest_date"] = payload["data"][0].get("date")
            payload["latest_net_value_wan"] = payload["data"][0].get("net_value_wan")
        return payload

    def _attach_turnover(self, price_rows: pd.DataFrame, order_book_id: str) -> pd.DataFrame:
        if price_rows.empty or "date" not in price_rows.columns:
            return price_rows

        turnover = self._table("turnover")
        if turnover.empty or not {"order_book_id", "tradedate", "today"}.issubset(turnover.columns):
            return price_rows

        matched = turnover.loc[turnover["order_book_id"] == order_book_id, ["tradedate", "today"]].copy()
        if matched.empty:
            return price_rows

        rows = price_rows.copy()
        rows["_merge_date"] = pd.to_datetime(rows["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        matched["_merge_date"] = pd.to_datetime(matched["tradedate"], errors="coerce").dt.strftime("%Y-%m-%d")
        matched["turnover_rate"] = pd.to_numeric(matched["today"], errors="coerce")
        rows = rows.merge(matched[["_merge_date", "turnover_rate"]], on="_merge_date", how="left")
        return rows.drop(columns=["_merge_date"])

    def _select_peer_group(
        self,
        quarter_rows: pd.DataFrame,
        target: pd.Series,
        min_peer_count: int,
    ) -> tuple[str, str, pd.DataFrame]:
        industry_columns = [
            ("third_industry_name", "三级行业"),
            ("second_industry_name", "二级行业"),
            ("first_industry_name", "一级行业"),
        ]
        fallback = pd.DataFrame()
        fallback_label = ""
        fallback_name = ""

        for column, label in industry_columns:
            if column not in quarter_rows.columns:
                continue
            industry_name = target.get(column)
            if not industry_name or pd.isna(industry_name):
                continue
            peers = quarter_rows.loc[quarter_rows[column] == industry_name].copy()
            if fallback.empty:
                fallback = peers
                fallback_label = label
                fallback_name = str(industry_name)
            if len(peers) >= min_peer_count:
                return label, str(industry_name), peers

        if not fallback.empty:
            return fallback_label, fallback_name, fallback
        return "", "", pd.DataFrame()

    def _symbol_rows(
        self,
        table: str,
        order_book_id: str,
        quarter: str | None = None,
    ) -> pd.DataFrame:
        frame = self._table(table)
        if frame.empty or "order_book_id" not in frame.columns:
            return pd.DataFrame()

        rows = frame.loc[frame["order_book_id"] == order_book_id]
        if quarter and "quarter" in rows.columns:
            rows = rows.loc[rows["quarter"].astype(str).str.lower() == quarter.lower()]
        return _sort_quarter_rows(rows)

    def _table(self, name: str) -> pd.DataFrame:
        if name in self._frames:
            return self._frames[name]

        file_map = {
            "raw": RAW_DATABASE_FILE,
            "market_labels": MARKET_LABELS_FILE,
            "pit": PIT_DISCLOSURES_FILE,
            "concepts": CONCEPT_TAGS_FILE,
            "metadata": METADATA_FILE,
            "consensus": CONSENSUS_FILE,
            "sector": SECTOR_FILE,
            "price": PRICE_FILE,
            "turnover": TURNOVER_FILE,
            "inflow_outflow": INFLOW_OUTFLOW_FILE,
        }
        path = self.root_dir / file_map[name]
        if not path.exists():
            logger.warning("[RQDataStore] 文件不存在: %s", path)
            frame = pd.DataFrame()
        else:
            frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            if "order_book_id" in frame.columns:
                frame["order_book_id"] = frame["order_book_id"].astype(str).str.upper()
            if "quarter" in frame.columns:
                frame["quarter"] = frame["quarter"].astype(str).str.lower()

        self._frames[name] = frame
        self._load_counts[name] = self._load_counts.get(name, 0) + 1
        return frame

    def _technical_file(self, date: str | None = None) -> Path | None:
        candidates = self._technical_files()
        if not candidates:
            return None

        if date:
            digits = re.sub(r"\D", "", str(date))
            if len(digits) >= 8:
                target = digits[:8]
                dated = [path for path in candidates if f"technical_{target}" in path.name]
                if dated:
                    return sorted(dated, key=lambda path: (path.suffix == ".parquet", path.stat().st_mtime), reverse=True)[0]
                return None

        return sorted(
            candidates,
            key=lambda path: (_date_from_technical_path(path), path.suffix == ".parquet", path.stat().st_mtime),
            reverse=True,
        )[0]

    def _technical_files(self, limit: int | None = None) -> list[Path]:
        if not self.technical_root_dir.exists():
            return []

        candidates = list(self.technical_root_dir.glob("technical_fetch_cache_*_*/technical_*.parquet"))
        candidates.extend(self.technical_root_dir.glob("technical_fetch_cache_*_*/technical_*.csv"))
        ordered = sorted(
            candidates,
            key=lambda path: (_date_from_technical_path(path), path.suffix == ".parquet", path.stat().st_mtime),
        )
        if limit is not None:
            return ordered[-limit:]
        return ordered

    def _recent_technical_files(self, limit: int = 30, calendar_days: int = 45) -> list[Path]:
        candidates = self._technical_files()
        if not candidates:
            return []

        dated = []
        for path in candidates:
            date_value = pd.to_datetime(_date_from_technical_path(path), errors="coerce")
            if pd.notna(date_value):
                dated.append((date_value, path))
        if not dated:
            return candidates[-limit:]

        latest_date = max(date_value for date_value, _ in dated)
        window_start = latest_date - pd.Timedelta(days=calendar_days)
        window_paths = [path for date_value, path in dated if date_value >= window_start]
        return window_paths[-limit:]

    def _latest_close_price(self, order_book_id: str) -> float | None:
        rows = self._symbol_rows("market_labels", order_book_id)
        if rows.empty:
            rows = self._symbol_rows("raw", order_book_id)
        if rows.empty:
            return None
        for column in ["close_price", "close"]:
            if column not in rows.columns:
                continue
            values = pd.to_numeric(rows[column], errors="coerce").dropna()
            if not values.empty:
                return _safe_float(values.iloc[0])
        return None

    def _base_payload(self, symbol: str, order_book_id: str, dataset: str) -> dict[str, Any]:
        return {
            "symbol": _plain_symbol(symbol or order_book_id),
            "order_book_id": order_book_id,
            "dataset": dataset,
            "source": "rqdatac_offline",
            "source_note": "来自 outputs/raw_rqdatac_database 的离线 RQData CSV。",
        }

    def _empty_payload(self, symbol: str, dataset: str, error: str) -> dict[str, Any]:
        order_book_id = self.normalize_symbol(symbol)
        payload = self._base_payload(symbol, order_book_id, dataset)
        payload["data"] = []
        payload["error"] = error
        return payload


def _plain_symbol(symbol: str) -> str:
    match = re.search(r"(\d{6})", str(symbol or ""))
    return match.group(1) if match else str(symbol or "")


def _records(frame: pd.DataFrame, preferred_columns: list[str], limit: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    columns = [column for column in preferred_columns if column in frame.columns]
    if not columns:
        columns = list(frame.columns)
    rows = frame.loc[:, columns].head(limit)
    return [_clean_record(record) for record in rows.to_dict(orient="records")]


def _read_table_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in record.items():
        if value is None:
            cleaned[key] = None
            continue
        try:
            if pd.isna(value):
                cleaned[key] = None
                continue
        except (TypeError, ValueError):
            pass
        if hasattr(value, "isoformat"):
            cleaned[key] = value.isoformat()
        else:
            cleaned[key] = value
    return cleaned


def _target_price_series(frame: pd.DataFrame) -> pd.Series:
    target_columns = [
        "one_year_target_price",
        "half_year_target_price",
        "price_raw",
    ]
    if frame.empty:
        return pd.Series(dtype="float64")
    values = []
    for _, row in frame.iterrows():
        target = None
        for column in target_columns:
            if column not in frame.columns:
                continue
            candidate = _safe_float(row.get(column))
            if candidate is not None and candidate > 0:
                target = candidate
                break
        values.append(target)
    return pd.Series(values, dtype="float64")


def _rating_label(value: float | None) -> str:
    if value is None:
        return "暂无评级"
    if value <= 1.3:
        return "强力买入"
    if value <= 2.3:
        return "买入"
    if value <= 3.3:
        return "观望"
    if value <= 4.3:
        return "适度减持"
    return "卖出"


def _rating_distribution(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    rating_columns = [column for column in ["quarter_recommendation", "grd_coef"] if column in frame.columns]
    if not rating_columns:
        return {}
    values = pd.to_numeric(frame[rating_columns[0]], errors="coerce").dropna()
    distribution: dict[str, int] = {}
    for value in values:
        label = _rating_label(_safe_float(value))
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def _date_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    match = re.search(r"(\d{4})[-/]?(\d{2})[-/]?(\d{2})", text)
    if not match:
        return text
    return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"


def _date_from_technical_path(path: Path) -> str:
    match = re.search(r"technical_(\d{8})", path.name)
    if not match:
        return ""
    digits = match.group(1)
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


def _score_peer_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if {"market_cap", "net_operate_cashflowTTM"}.issubset(rows.columns):
        market_cap = pd.to_numeric(rows["market_cap"], errors="coerce")
        cashflow = pd.to_numeric(rows["net_operate_cashflowTTM"], errors="coerce")
        rows["pcf_ratio"] = market_cap / cashflow.replace(0, pd.NA)

    metric_specs = {
        "pe_ratio_ttm": ("pe_score", False),
        "pb_ratio": ("pb_score", False),
        "ps_ratio_ttm": ("ps_score", False),
        "pcf_ratio": ("pcf_score", False),
        "return_on_equity_weighted_average": ("roe_score", True),
        "net_profit_ttm_0": ("profit_ttm_score", True),
        "operating_revenue_ttm_0": ("revenue_ttm_score", True),
        "market_cap": ("market_cap_score", True),
        "total_turnover": ("turnover_score", True),
    }

    score_columns = []
    for metric, (score_column, higher_is_better) in metric_specs.items():
        if metric not in rows.columns:
            continue
        values = pd.to_numeric(rows[metric], errors="coerce")
        if metric in {"pe_ratio_ttm", "pb_ratio", "ps_ratio_ttm", "pcf_ratio"}:
            values = values.where(values > 0)
        rows[score_column] = _percentile_score(values, higher_is_better)
        score_columns.append(score_column)

    if {"net_operate_cashflowTTM", "net_profit_ttm_0"}.issubset(rows.columns):
        cashflow = pd.to_numeric(rows["net_operate_cashflowTTM"], errors="coerce")
        profit = pd.to_numeric(rows["net_profit_ttm_0"], errors="coerce").abs()
        quality = cashflow / profit.where(profit > 0)
        rows["cashflow_quality_score"] = _percentile_score(quality, True)
        rows["cashflow_quality"] = quality
        score_columns.append("cashflow_quality_score")

    if score_columns:
        rows["available_metric_count"] = rows[score_columns].notna().sum(axis=1)
        rows["overall_score"] = rows[score_columns].mean(axis=1).round(1)
        rows["overall_percentile"] = _percentile_score(rows["overall_score"], True).round(1)
    else:
        rows["available_metric_count"] = 0
        rows["overall_score"] = pd.NA
        rows["overall_percentile"] = pd.NA
    return rows


def _percentile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=values.index, dtype="Float64")
    ranked = values.rank(method="average", ascending=higher_is_better, pct=True) * 100
    return ranked.round(1)


def _metric_payload(target: pd.Series, peers: pd.DataFrame) -> list[dict[str, Any]]:
    metric_specs = [
        ("pe_ratio_ttm", "PE(TTM)", "pe_score", "低于同业更优"),
        ("pb_ratio", "PB", "pb_score", "低于同业更优"),
        ("ps_ratio_ttm", "PS(TTM)", "ps_score", "低于同业更优"),
        ("pcf_ratio", "P/CF", "pcf_score", "低于同业更优"),
        ("return_on_equity_weighted_average", "ROE", "roe_score", "高于同业更优"),
        ("net_profit_ttm_0", "净利润TTM", "profit_ttm_score", "高于同业更优"),
        ("operating_revenue_ttm_0", "营收TTM", "revenue_ttm_score", "高于同业更优"),
        ("market_cap", "市值", "market_cap_score", "高于同业更优"),
        ("total_turnover", "成交额", "turnover_score", "高于同业更优"),
        ("cashflow_quality", "经营现金流/净利润", "cashflow_quality_score", "高于同业更优"),
    ]
    metrics = []
    for value_column, label, score_column, direction in metric_specs:
        if score_column not in target:
            continue
        score = _safe_float(target.get(score_column))
        if score is None:
            continue
        rank = None
        if score_column in peers.columns:
            ordered = peers.sort_values(score_column, ascending=False, na_position="last")
            matches = ordered.index[ordered["order_book_id"] == target.get("order_book_id")].tolist()
            if matches:
                rank = ordered.index.tolist().index(matches[0]) + 1
        metrics.append(
            {
                "metric": value_column,
                "label": label,
                "value": _safe_float(target.get(value_column)),
                "score": score,
                "rank": rank,
                "peer_count": int(peers[score_column].notna().sum()) if score_column in peers.columns else 0,
                "direction": direction,
            }
        )
    return metrics


def _peer_table_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "industry_rank",
        "order_book_id",
        "symbol",
        "overall_score",
        "overall_percentile",
        "available_metric_count",
        "pe_ratio_ttm",
        "pb_ratio",
        "return_on_equity_weighted_average",
        "net_profit_ttm_0",
        "market_cap",
        "total_turnover",
    ]
    return _records(frame, columns, limit=len(frame))


def _scorecard_quality_notes(peers: pd.DataFrame, target: pd.Series) -> list[str]:
    notes = []
    metric_count = target.get("available_metric_count", 0)
    if metric_count < 4:
        notes.append(f"目标股票可用指标数较少: {metric_count}，行业综合分可靠性下降。")
    if len(peers) < 8:
        notes.append(f"行业样本数较少: {len(peers)}，排名可能不稳定。")
    if "pe_ratio_ttm" in target and pd.notna(target.get("pe_ratio_ttm")) and target.get("pe_ratio_ttm") <= 0:
        notes.append("PE(TTM) 为负或零，估值分位未纳入综合计算。")
    return notes


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _technical_timeseries_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    latest = records[-1]
    first = records[0]
    ma5 = _safe_float(latest.get("MA5"))
    ma20 = _safe_float(latest.get("MA20"))
    ma60 = _safe_float(latest.get("MA60"))
    macd_hist = _safe_float(latest.get("MACD_HIST"))
    macd_first = _safe_float(first.get("MACD_HIST"))
    rsi6 = _safe_float(latest.get("RSI6"))
    kdj_j = _safe_float(latest.get("KDJ_J"))

    if ma5 is not None and ma20 is not None and ma60 is not None:
        if ma5 > ma20 > ma60:
            trend_signal = "短中期均线多头排列"
            trend_bias = "偏强"
        elif ma5 < ma20 < ma60:
            trend_signal = "短中期均线空头排列"
            trend_bias = "偏弱"
        elif ma5 > ma20:
            trend_signal = "短线站上中期均线"
            trend_bias = "修复"
        else:
            trend_signal = "短线仍弱于中期均线"
            trend_bias = "谨慎"
    else:
        trend_signal = "均线数据不足"
        trend_bias = "中性"

    momentum_signal = "MACD 动能待确认"
    if macd_hist is not None:
        if macd_hist > 0 and (macd_first is None or macd_hist >= macd_first):
            momentum_signal = "MACD 柱体为正且动能改善"
        elif macd_hist > 0:
            momentum_signal = "MACD 柱体为正但动能收敛"
        elif macd_first is not None and macd_hist > macd_first:
            momentum_signal = "MACD 负值收窄，弱修复"
        else:
            momentum_signal = "MACD 动能偏弱"

    risk_signal = "摆动指标处于常规区间"
    if rsi6 is not None and rsi6 >= 80:
        risk_signal = "RSI6 偏高，短线追涨风险上升"
    elif rsi6 is not None and rsi6 <= 20:
        risk_signal = "RSI6 偏低，存在超卖修复观察点"
    elif kdj_j is not None and kdj_j >= 100:
        risk_signal = "KDJ-J 高位，注意短线回落压力"
    elif kdj_j is not None and kdj_j <= 0:
        risk_signal = "KDJ-J 低位，关注止跌信号"

    return {
        "trend_bias": trend_bias,
        "trend_signal": trend_signal,
        "momentum_signal": momentum_signal,
        "risk_signal": risk_signal,
        "latest_date": _date_text(latest.get("date")),
        "window_days": len(records),
        "operation_hint": _technical_operation_hint(trend_bias, momentum_signal, risk_signal),
    }


def _technical_operation_hint(trend_bias: str, momentum_signal: str, risk_signal: str) -> str:
    if trend_bias == "偏强" and "改善" in momentum_signal and "风险" not in risk_signal:
        return "趋势与动能共振偏强，可关注回踩均线后的低吸机会。"
    if trend_bias in {"偏弱", "谨慎"} and "偏弱" in momentum_signal:
        return "趋势和动能尚未修复，操作上更适合等待放量企稳或均线重新拐头。"
    if "超卖" in risk_signal or "低位" in risk_signal:
        return "短线存在修复观察点，但需要成交量和资金流同步确认。"
    if "追涨风险" in risk_signal or "高位" in risk_signal:
        return "短线指标偏热，宜避免追高，关注回撤后的承接强度。"
    return "当前信号偏中性，建议结合资金流与估值安全边际确认操作节奏。"


def _sort_quarter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "quarter" not in frame.columns:
        return frame

    rows = frame.copy()
    rows["_quarter_sort"] = rows["quarter"].map(_quarter_key)
    sort_columns = ["_quarter_sort"]
    ascending = [False]
    if "info_date" in rows.columns:
        sort_columns.append("info_date")
        ascending.append(False)
    rows = rows.sort_values(sort_columns, ascending=ascending, na_position="last")
    return rows.drop(columns=["_quarter_sort"])


def _quarter_key(value: Any) -> tuple[int, int]:
    match = re.search(r"(\d{4})q([1-4])", str(value or "").lower())
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def _latest_quarter(frame: pd.DataFrame) -> str:
    if frame.empty or "quarter" not in frame.columns:
        return ""
    value = frame.iloc[0].get("quarter", "")
    return str(value) if value is not None else ""
