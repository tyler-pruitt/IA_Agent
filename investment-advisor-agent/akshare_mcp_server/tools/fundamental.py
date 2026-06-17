"""
基本面数据工具 — 封装 AKShare 基本面相关函数

提供三个 MCP Tool:
  1. stock_fundamental — 个股基本面全景 (三表+估值+指标)
  2. stock_profit_forecast — 分析师盈利预测
  3. stock_earnings_preview — 业绩预告/快报
"""

import akshare as ak
import pandas as pd

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter

_DATE_COLUMNS = (
    "数据日期",
    "REPORT_DATE",
    "报告日期",
    "公告日期",
    "NOTICE_DATE",
    "UPDATE_DATE",
    "最新公告日期",
)


def _safe_call(limiter: RateLimiter, func, *args, **kwargs) -> pd.DataFrame:
    """带限流的安全调用"""
    func_name = func.__name__
    if not limiter.acquire(func_name, timeout=15.0):
        raise TimeoutError(f"限流超时: {func_name} (数据源: {limiter.get_source(func_name)})")
    return func(*args, **kwargs)


def _is_a_stock_code(symbol: str) -> bool:
    """判断是否为 6 位 A 股股票代码。"""
    return symbol.isdigit() and len(symbol) == 6


def _market_prefix_code(symbol: str) -> str:
    """转换为东方财富三大报表使用的市场前缀代码，如 SZ000001。"""
    if symbol.startswith(("SH", "SZ", "BJ")):
        return symbol
    if symbol.startswith("6"):
        return f"SH{symbol}"
    if symbol.startswith(("4", "8")):
        return f"BJ{symbol}"
    return f"SZ{symbol}"


def _market_suffix_code(symbol: str) -> str:
    """转换为东方财富财务指标使用的 SECUCODE，如 000001.SZ。"""
    if "." in symbol:
        return symbol
    if symbol.startswith(("SH", "SZ", "BJ")) and len(symbol) == 8:
        return f"{symbol[2:]}.{symbol[:2]}"
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    return f"{symbol}.SZ"


def _sort_latest_first(df: pd.DataFrame) -> pd.DataFrame:
    """按常见日期字段倒序排列，确保截取样本时优先保留最新数据。"""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            out_df = df.copy()
            out_df["_sort_date"] = pd.to_datetime(out_df[col], errors="coerce")
            out_df.sort_values("_sort_date", ascending=False, na_position="last", inplace=True)
            return out_df.drop(columns=["_sort_date"])
    return df


def _filter_by_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """从全市场 DataFrame 中过滤目标股票。"""
    if df is None or df.empty:
        return pd.DataFrame()

    for col in ("股票代码", "代码", "SECURITY_CODE", "SECUCODE"):
        if col in df.columns:
            stock_codes = df[col].astype(str).str.extract(r"(\d{6})", expand=False)
            return df[stock_codes == symbol]
    return pd.DataFrame()


def _df_to_records(df: pd.DataFrame, max_rows: int = 500, latest_first: bool = False) -> list:
    """DataFrame 转为 record 列表，控制输出大小"""
    if df is None or df.empty:
        return []
    if latest_first:
        df = _sort_latest_first(df)
    if len(df) > max_rows:
        df = df.head(max_rows)
    # 将 date 类型转为字符串，避免 JSON 序列化问题
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if hasattr(x, "strftime") else x)
    df = df.fillna("")
    return df.to_dict(orient="records")


# ──────────────────────────────────────────────
# Tool 1: stock_fundamental
# ──────────────────────────────────────────────

def stock_fundamental(
    symbol: str,
    report_type: str = "all",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    获取个股基本面全景数据

    :param symbol: 股票代码，如 "000001"
    :param report_type: 返回类型
        - "all": 全部 (估值+指标+三表摘要)
        - "valuation": 估值分析 (PE/PB/PEG/PS/PCF)
        - "indicator": 财务分析指标 (ROE/毛利率等)
        - "balance": 资产负债表
        - "profit": 利润表
        - "cashflow": 现金流量表
    :return: dict 包含请求的基本面数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"symbol": symbol, "report_type": report_type}

    try:
        if report_type in ("all", "valuation"):
            cached = cache.get("fundamental", "stock_value_em", (symbol,))
            if cached is not None:
                result["valuation"] = _df_to_records(cached, max_rows=30, latest_first=True)
            else:
                df = _safe_call(limiter, ak.stock_value_em, symbol=symbol)
                cache.set("fundamental", "stock_value_em", (symbol,), data=df)
                result["valuation"] = _df_to_records(df, max_rows=30, latest_first=True)
    except Exception as e:
        result["valuation_error"] = str(e)

    try:
        if report_type in ("all", "indicator"):
            indicator_symbol = _market_suffix_code(symbol)
            cached = cache.get("fundamental", "stock_financial_analysis_indicator_em", (indicator_symbol,))
            if cached is not None:
                result["financial_indicator"] = _df_to_records(cached, max_rows=10, latest_first=True)
            else:
                df = _safe_call(limiter, ak.stock_financial_analysis_indicator_em, symbol=indicator_symbol)
                cache.set("fundamental", "stock_financial_analysis_indicator_em", (indicator_symbol,), data=df)
                result["financial_indicator"] = _df_to_records(df, max_rows=10, latest_first=True)
            if not result["financial_indicator"]:
                result["indicator_note"] = f"未返回 {indicator_symbol} 的主要财务指标"
    except Exception as e:
        result["indicator_error"] = (
            f"{e}; 已按东方财富 SECUCODE 格式查询，实际参数为 {_market_suffix_code(symbol)}"
        )

    try:
        if report_type in ("all", "balance"):
            symbol_prefix = _market_prefix_code(symbol)
            cached = cache.get("fundamental", "stock_balance_sheet_by_report_em", (symbol_prefix,))
            if cached is not None:
                result["balance_sheet"] = _df_to_records(cached, max_rows=5, latest_first=True)
            else:
                df = _safe_call(limiter, ak.stock_balance_sheet_by_report_em, symbol=symbol_prefix)
                cache.set("fundamental", "stock_balance_sheet_by_report_em", (symbol_prefix,), data=df)
                result["balance_sheet"] = _df_to_records(df, max_rows=5, latest_first=True)
    except Exception as e:
        result["balance_error"] = str(e)

    try:
        if report_type in ("all", "profit"):
            symbol_prefix = _market_prefix_code(symbol)
            cached = cache.get("fundamental", "stock_profit_sheet_by_report_em", (symbol_prefix,))
            if cached is not None:
                result["profit_sheet"] = _df_to_records(cached, max_rows=5, latest_first=True)
            else:
                df = _safe_call(limiter, ak.stock_profit_sheet_by_report_em, symbol=symbol_prefix)
                cache.set("fundamental", "stock_profit_sheet_by_report_em", (symbol_prefix,), data=df)
                result["profit_sheet"] = _df_to_records(df, max_rows=5, latest_first=True)
    except Exception as e:
        result["profit_error"] = str(e)

    try:
        if report_type in ("all", "cashflow"):
            symbol_prefix = _market_prefix_code(symbol)
            cached = cache.get("fundamental", "stock_cash_flow_sheet_by_report_em", (symbol_prefix,))
            if cached is not None:
                result["cash_flow"] = _df_to_records(cached, max_rows=5, latest_first=True)
            else:
                df = _safe_call(limiter, ak.stock_cash_flow_sheet_by_report_em, symbol=symbol_prefix)
                cache.set("fundamental", "stock_cash_flow_sheet_by_report_em", (symbol_prefix,), data=df)
                result["cash_flow"] = _df_to_records(df, max_rows=5, latest_first=True)
    except Exception as e:
        result["cashflow_error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 2: stock_profit_forecast
# ──────────────────────────────────────────────

def stock_profit_forecast(
    symbol: str,
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    获取分析师盈利预测

    :param symbol: 股票代码，如 "000001"
    :return: dict 包含盈利预测数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"symbol": symbol}

    try:
        is_stock_code = _is_a_stock_code(symbol)
        query_symbol = "__all__" if is_stock_code else symbol
        cached = cache.get("fundamental", "stock_profit_forecast_em", (query_symbol,))
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_profit_forecast_em, symbol="" if is_stock_code else symbol)
            cache.set("fundamental", "stock_profit_forecast_em", (query_symbol,), data=df)

        if is_stock_code:
            filtered = _filter_by_symbol(df, symbol)
            result["forecast"] = _df_to_records(filtered, max_rows=10)
            result["query_scope"] = "all_market_filtered_by_stock_code"
            result["source_note"] = (
                "ak.stock_profit_forecast_em 不支持直接按个股代码查询，"
                "MCP 已先获取全市场盈利预测后按股票代码过滤"
            )
            if filtered.empty:
                result["note"] = f"全市场盈利预测中未找到 {symbol}，可能该股暂无有效机构一致预期"
        else:
            result["forecast"] = _df_to_records(df, max_rows=10)
            result["query_scope"] = "industry_board"
    except Exception as e:
        result["error"] = (
            f"{e}; 若传入的是股票代码，MCP 会尝试全市场盈利预测后按代码过滤"
        )

    return result


# ──────────────────────────────────────────────
# Tool 3: stock_earnings_preview
# ──────────────────────────────────────────────

def stock_earnings_preview(
    date: str,
    preview_type: str = "yjyg",
    symbol: str = "",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    获取业绩预告/快报数据

    :param date: 报告期，如 "20240331"
    :param preview_type: 类型
        - "yjyg": 业绩预告
        - "yjkb": 业绩快报
        - "yjbb": 业绩报表
        
    :return: dict 包含业绩数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"date": date, "preview_type": preview_type}
    if symbol:
        result["symbol"] = symbol

    try:
        if preview_type == "yjyg":
            cached = cache.get("fundamental", "stock_yjyg_em", (date,))
            if cached is not None:
                df = cached
            else:
                df = _safe_call(limiter, ak.stock_yjyg_em, date=date)
                cache.set("fundamental", "stock_yjyg_em", (date,), data=df)
        elif preview_type == "yjkb":
            cached = cache.get("fundamental", "stock_yjkb_em", (date,))
            if cached is not None:
                df = cached
            else:
                df = _safe_call(limiter, ak.stock_yjkb_em, date=date)
                cache.set("fundamental", "stock_yjkb_em", (date,), data=df)
        elif preview_type == "yjbb":
            cached = cache.get("fundamental", "stock_yjbb_em", (date,))
            if cached is not None:
                df = cached
            else:
                df = _safe_call(limiter, ak.stock_yjbb_em, date=date)
                cache.set("fundamental", "stock_yjbb_em", (date,), data=df)
        else:
            result["error"] = f"不支持的 preview_type: {preview_type}"
            return result

        if symbol:
            filtered = _filter_by_symbol(df, symbol)
            result["data"] = _df_to_records(filtered, max_rows=20, latest_first=True)
            if filtered.empty:
                result["note"] = f"{date} 报告期未找到 {symbol} 对应业绩数据"
        else:
            result["data"] = _df_to_records(df, max_rows=100, latest_first=True)
    except Exception as e:
        result["error"] = str(e)

    return result
