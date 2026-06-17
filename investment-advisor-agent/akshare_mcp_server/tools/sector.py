"""
行业/概念板块工具 — 封装 AKShare 行业概念相关函数

提供两个 MCP Tool:
  1. stock_sector_analysis — 行业/概念板块分析
  2. stock_market_valuation — 市场整体估值 (PE/PB/巴菲特指标)
"""

import akshare as ak
import pandas as pd

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter

_DATE_COLUMNS = ("日期", "date", "交易日", "报告日")


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


# ──────────────────────────────────────────────
# Tool 1: stock_sector_analysis
# ──────────────────────────────────────────────

def stock_sector_analysis(
    sector_type: str = "industry",
    name: str = "",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    行业/概念板块分析

    :param sector_type: 板块类型
        - "industry": 行业板块
        - "concept": 概念板块
        - "list": 列出所有板块名称 (不需要 name)
    :param name: 板块名称 (sector_type 非 list 时必填)，如 "小金属", "数据要素"
    :return: dict 包含板块数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"sector_type": sector_type}

    try:
        if sector_type == "list":
            # 列出所有行业板块
            try:
                cached = cache.get("sector", "stock_board_industry_name_em", ())
                if cached is not None:
                    result["industry"] = _df_to_records(cached, max_rows=100)
                else:
                    df = _safe_call(limiter, ak.stock_board_industry_name_em)
                    cache.set("sector", "stock_board_industry_name_em", (), data=df)
                    result["industry"] = _df_to_records(df, max_rows=100)
            except Exception as e:
                result["industry_error"] = str(e)

            # 列出所有概念板块
            try:
                cached = cache.get("sector", "stock_board_concept_name_em", ())
                if cached is not None:
                    result["concept"] = _df_to_records(cached, max_rows=100)
                else:
                    df = _safe_call(limiter, ak.stock_board_concept_name_em)
                    cache.set("sector", "stock_board_concept_name_em", (), data=df)
                    result["concept"] = _df_to_records(df, max_rows=100)
            except Exception as e:
                result["concept_error"] = str(e)

            return result

        if not name:
            return {"error": f"sector_type={sector_type} 时需要提供板块名称 name，或使用 sector_type=list 查看所有板块"}

        if sector_type == "industry":
            # 行业板块行情
            try:
                cached = cache.get("sector", "stock_board_industry_spot_em", (name,))
                if cached is not None:
                    result["spot"] = _df_to_records(cached, max_rows=5)
                else:
                    df = _safe_call(limiter, ak.stock_board_industry_spot_em, symbol=name)
                    cache.set("sector", "stock_board_industry_spot_em", (name,), data=df)
                    result["spot"] = _df_to_records(df, max_rows=5)
            except Exception as e:
                result["spot_error"] = str(e)

            # 行业板块历史K线
            try:
                cached = cache.get("sector", "stock_board_industry_hist_em", (name,))
                if cached is not None:
                    result["hist"] = _df_to_records(cached, max_rows=60)
                else:
                    df = _safe_call(limiter, ak.stock_board_industry_hist_em, symbol=name, period="日k", start_date="", end_date="", adjust="")
                    cache.set("sector", "stock_board_industry_hist_em", (name,), data=df)
                    result["hist"] = _df_to_records(df, max_rows=60)
            except Exception as e:
                result["hist_error"] = str(e)

            # 行业板块成分股
            try:
                cached = cache.get("sector", "stock_board_industry_cons_em", (name,))
                if cached is not None:
                    result["constituents"] = _df_to_records(cached, max_rows=20)
                else:
                    df = _safe_call(limiter, ak.stock_board_industry_cons_em, symbol=name)
                    cache.set("sector", "stock_board_industry_cons_em", (name,), data=df)
                    result["constituents"] = _df_to_records(df, max_rows=20)
            except Exception as e:
                result["cons_error"] = str(e)

        elif sector_type == "concept":
            # 概念板块行情
            try:
                cached = cache.get("sector", "stock_board_concept_spot_em", (name,))
                if cached is not None:
                    result["spot"] = _df_to_records(cached, max_rows=5)
                else:
                    df = _safe_call(limiter, ak.stock_board_concept_spot_em, symbol=name)
                    cache.set("sector", "stock_board_concept_spot_em", (name,), data=df)
                    result["spot"] = _df_to_records(df, max_rows=5)
            except Exception as e:
                result["spot_error"] = str(e)

            # 概念板块历史K线
            try:
                cached = cache.get("sector", "stock_board_concept_hist_em", (name,))
                if cached is not None:
                    result["hist"] = _df_to_records(cached, max_rows=60)
                else:
                    df = _safe_call(limiter, ak.stock_board_concept_hist_em, symbol=name, period="日k", start_date="", end_date="", adjust="")
                    cache.set("sector", "stock_board_concept_hist_em", (name,), data=df)
                    result["hist"] = _df_to_records(df, max_rows=60)
            except Exception as e:
                result["hist_error"] = str(e)

            # 概念板块成分股
            try:
                cached = cache.get("sector", "stock_board_concept_cons_em", (name,))
                if cached is not None:
                    result["constituents"] = _df_to_records(cached, max_rows=20)
                else:
                    df = _safe_call(limiter, ak.stock_board_concept_cons_em, symbol=name)
                    cache.set("sector", "stock_board_concept_cons_em", (name,), data=df)
                    result["constituents"] = _df_to_records(df, max_rows=20)
            except Exception as e:
                result["cons_error"] = str(e)

        else:
            result["error"] = f"不支持的 sector_type: {sector_type}"

    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 2: stock_market_valuation
# ──────────────────────────────────────────────

def stock_market_valuation(
    index: str = "沪深300",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    市场整体估值指标

    :param index: 指数名称，如 "沪深300", "上证50", "深证"
    :return: dict 包含市场PE/PB及巴菲特指标
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"index": index}

    # 指数PE
    try:
        cached = cache.get("sector", "stock_index_pe_lg", (index,))
        if cached is not None:
            result["index_pe"] = _df_to_records(cached, max_rows=10, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_index_pe_lg, symbol=index)
            cache.set("sector", "stock_index_pe_lg", (index,), data=df)
            result["index_pe"] = _df_to_records(df, max_rows=10, latest_window=True)
    except Exception as e:
        result["pe_error"] = str(e)

    # 指数PB
    try:
        cached = cache.get("sector", "stock_index_pb_lg", (index,))
        if cached is not None:
            result["index_pb"] = _df_to_records(cached, max_rows=10, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_index_pb_lg, symbol=index)
            cache.set("sector", "stock_index_pb_lg", (index,), data=df)
            result["index_pb"] = _df_to_records(df, max_rows=10, latest_window=True)
    except Exception as e:
        result["pb_error"] = str(e)

    # 巴菲特指标
    try:
        cached = cache.get("sector", "stock_buffett_index_lg", ())
        if cached is not None:
            result["buffett_index"] = _df_to_records(cached, max_rows=5, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_buffett_index_lg)
            cache.set("sector", "stock_buffett_index_lg", (), data=df)
            result["buffett_index"] = _df_to_records(df, max_rows=5, latest_window=True)
    except Exception as e:
        result["buffett_error"] = str(e)

    # 全A股PE/PB
    try:
        cached = cache.get("sector", "stock_a_all_pb", ())
        if cached is not None:
            result["a_all_pb"] = _df_to_records(cached, max_rows=5, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_a_all_pb)
            cache.set("sector", "stock_a_all_pb", (), data=df)
            result["a_all_pb"] = _df_to_records(df, max_rows=5, latest_window=True)
    except Exception as e:
        result["all_pb_error"] = str(e)

    return result
