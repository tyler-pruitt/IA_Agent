"""
宏观经济工具 — 封装 AKShare 宏观数据相关函数

提供两个 MCP Tool:
  1. macro_china_overview — 中国宏观经济全景
  2. macro_global_interest — 全球主要央行利率
"""

import akshare as ak
import pandas as pd

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter

_DATE_COLUMNS = ("日期", "date", "月份", "TRADE_DATE")
_EMPTY_STRINGS = {"", "nan", "none", "null", "NaN", "None", "--", "-"}


def _safe_call(limiter: RateLimiter, func, *args, **kwargs) -> pd.DataFrame:
    """带限流的安全调用"""
    func_name = func.__name__
    if not limiter.acquire(func_name, timeout=15.0):
        raise TimeoutError(f"限流超时: {func_name} (数据源: {limiter.get_source(func_name)})")
    return func(*args, **kwargs)


def _sort_by_date(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """按常见日期字段排序。"""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            out_df = df.copy()
            out_df["_sort_date"] = pd.to_datetime(out_df[col], errors="coerce")
            out_df.sort_values("_sort_date", ascending=ascending, na_position="last", inplace=True)
            return out_df.drop(columns=["_sort_date"])
    return df


def _df_to_records(df: pd.DataFrame, max_rows: int = 500, latest_window: bool = False) -> list:
    """DataFrame 转为 record 列表"""
    if df is None or df.empty:
        return []
    if latest_window:
        df = _sort_by_date(df, ascending=True).tail(max_rows)
    if len(df) > max_rows:
        df = df.head(max_rows)
    # 将 date 类型转为字符串，避免 JSON 序列化问题
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if hasattr(x, "strftime") else x)
    df = df.fillna("")
    return df.to_dict(orient="records")


def _is_valid_value(value) -> bool:
    """过滤宏观日历中尚未公布的空值、NaN 和占位符。"""
    if pd.isna(value):
        return False
    if isinstance(value, str) and value.strip() in _EMPTY_STRINGS:
        return False
    return True


def _detect_date_col(df: pd.DataFrame, date_col: str = None) -> str | None:
    if date_col:
        return date_col
    for col in _DATE_COLUMNS:
        if col in df.columns:
            return col
    for col in df.columns:
        if "日期" in str(col) or "date" in str(col).lower():
            return col
    return None


def _detect_value_col(df: pd.DataFrame, value_col: str = None) -> str | None:
    if value_col:
        return value_col
    priority_cols = ("今值", "综合PMI", "制造业PMI", "服务业PMI", "value", "close")
    for col in priority_cols:
        if col in df.columns:
            return col
    for col in df.columns:
        if any(token in str(col) for token in ("值", "率", "增速", "余额")):
            return col
    return df.columns[-1] if len(df.columns) > 1 else None


def _freshness_note(latest_date: str, label: str) -> str:
    """说明最新有效值是否明显滞后。"""
    if not latest_date:
        return ""
    parsed = pd.to_datetime(latest_date, errors="coerce")
    if pd.isna(parsed):
        return ""
    age_days = (pd.Timestamp.today().normalize() - parsed.normalize()).days
    if age_days > 120:
        return f"{label} 最新有效值日期为 {latest_date}，数据源可能滞后或该指标低频发布"
    return f"{label} 最新有效值日期为 {latest_date}"


def _extract_latest(df: pd.DataFrame, value_col: str = None, date_col: str = None) -> dict:
    """从宏观数据 DataFrame 提取最新有效值，跳过尚未公布的 NaN。"""
    if df is None or df.empty:
        return {"value": None, "date": None, "value_col": value_col, "status": "empty"}

    date_col = _detect_date_col(df, date_col)
    value_col = _detect_value_col(df, value_col)
    if not value_col or value_col not in df.columns:
        return {"value": None, "date": None, "value_col": value_col, "status": "value_column_missing"}

    out_df = _sort_by_date(df, ascending=True)
    valid_df = out_df[out_df[value_col].apply(_is_valid_value)]
    if valid_df.empty:
        latest = out_df.iloc[-1]
        return {
            "value": None,
            "date": str(latest.get(date_col, "")) if date_col else "",
            "value_col": value_col,
            "status": "no_valid_actual_value",
        }

    latest = valid_df.iloc[-1]
    return {
        "value": str(latest.get(value_col, "")),
        "date": str(latest.get(date_col, "")) if date_col else "",
        "value_col": value_col,
        "status": "ok",
    }


# ──────────────────────────────────────────────
# Tool 1: macro_china_overview
# ──────────────────────────────────────────────

def macro_china_overview(
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    中国宏观经济全景

    包含指标:
      - GDP 增速
      - CPI 同比/环比
      - PPI 同比
      - PMI (官方 + 财新)
      - M2 同比
      - 贸易差额
      - 工业增加值
      - 失业率

    :return: dict 包含各宏观指标最新值
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {}

    macro_calls = [
        ("GDP增速", "macro_china_gdp_yearly", lambda: _safe_call(limiter, ak.macro_china_gdp_yearly), "今值"),
        ("CPI同比", "macro_china_cpi_yearly", lambda: _safe_call(limiter, ak.macro_china_cpi_yearly), "今值"),
        ("CPI环比", "macro_china_cpi_monthly", lambda: _safe_call(limiter, ak.macro_china_cpi_monthly), "今值"),
        ("PPI同比", "macro_china_ppi_yearly", lambda: _safe_call(limiter, ak.macro_china_ppi_yearly), "今值"),
        ("PMI", "macro_china_pmi_yearly", lambda: _safe_call(limiter, ak.macro_china_pmi_yearly), "今值"),
        ("财新PMI", "index_pmi_com_cx", lambda: _safe_call(limiter, ak.index_pmi_com_cx), "综合PMI"),
        ("M2同比", "macro_china_m2_yearly", lambda: _safe_call(limiter, ak.macro_china_m2_yearly), "今值"),
        ("出口同比", "macro_china_exports_yoy", lambda: _safe_call(limiter, ak.macro_china_exports_yoy), "今值"),
        ("进口同比", "macro_china_imports_yoy", lambda: _safe_call(limiter, ak.macro_china_imports_yoy), "今值"),
        ("贸易差额", "macro_china_trade_balance", lambda: _safe_call(limiter, ak.macro_china_trade_balance), "今值"),
        ("工业增加值", "macro_china_industrial_production_yoy", lambda: _safe_call(limiter, ak.macro_china_industrial_production_yoy), "今值"),
        ("失业率", "macro_china_urban_unemployment", lambda: _safe_call(limiter, ak.macro_china_urban_unemployment), "value"),
    ]

    for name, func_key, call_fn, value_col in macro_calls:
        try:
            cached = cache.get("macro", func_key, ())
            if cached is not None:
                df = cached
            else:
                df = call_fn()
                cache.set("macro", func_key, (), data=df)
            latest = _extract_latest(df, value_col=value_col)
            result[name] = {
                "latest": latest,
                "history": _df_to_records(df, max_rows=10, latest_window=True),
                "note": _freshness_note(latest.get("date", ""), name),
            }
        except Exception as e:
            result[name] = {"error": str(e)}

    # 中国利率
    try:
        cached = cache.get("macro", "macro_bank_china_interest_rate", ())
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.macro_bank_china_interest_rate)
            cache.set("macro", "macro_bank_china_interest_rate", (), data=df)
        latest = _extract_latest(df, value_col="今值")
        result["中国利率"] = {
            "latest": latest,
            "history": _df_to_records(df, max_rows=10, latest_window=True),
            "note": (
                _freshness_note(latest.get("date", ""), "中国央行决议报告")
                + "；该接口口径可能不是当前 LPR/政策利率主口径"
            ),
        }
    except Exception as e:
        result["中国利率"] = {"error": str(e)}

    # 中国 LPR
    try:
        cached = cache.get("macro", "macro_china_lpr", ())
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.macro_china_lpr)
            cache.set("macro", "macro_china_lpr", (), data=df)
        latest = _extract_latest(df, value_col="LPR1Y", date_col="TRADE_DATE")
        result["中国LPR"] = {
            "latest": latest,
            "history": _df_to_records(df, max_rows=10, latest_window=True),
            "note": _freshness_note(latest.get("date", ""), "中国1年期LPR"),
        }
    except Exception as e:
        result["中国LPR"] = {"error": str(e)}

    return result


# ──────────────────────────────────────────────
# Tool 2: macro_global_interest
# ──────────────────────────────────────────────

def macro_global_interest(
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    全球主要央行利率

    包含: 美国、欧元区、中国、日本、英国、澳大利亚、瑞士、新西兰、俄罗斯、印度、巴西

    :return: dict 包含各国利率
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {}

    interest_calls = [
        ("美国", "macro_bank_usa_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_usa_interest_rate)),
        ("欧元区", "macro_bank_euro_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_euro_interest_rate)),
        ("中国", "macro_bank_china_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_china_interest_rate)),
        ("日本", "macro_bank_japan_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_japan_interest_rate)),
        ("英国", "macro_bank_english_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_english_interest_rate)),
        ("澳大利亚", "macro_bank_australia_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_australia_interest_rate)),
        ("瑞士", "macro_bank_switzerland_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_switzerland_interest_rate)),
        ("新西兰", "macro_bank_newzealand_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_newzealand_interest_rate)),
        ("俄罗斯", "macro_bank_russia_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_russia_interest_rate)),
        ("印度", "macro_bank_india_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_india_interest_rate)),
        ("巴西", "macro_bank_brazil_interest_rate", lambda: _safe_call(limiter, ak.macro_bank_brazil_interest_rate)),
    ]

    for name, func_key, call_fn in interest_calls:
        try:
            cached = cache.get("macro", func_key, ())
            if cached is not None:
                df = cached
            else:
                df = call_fn()
                cache.set("macro", func_key, (), data=df)
            latest = _extract_latest(df, value_col="今值")
            result[name] = {
                "latest": latest,
                "note": _freshness_note(latest.get("date", ""), f"{name}利率"),
            }
        except Exception as e:
            result[name] = {"error": str(e)}

    return result
