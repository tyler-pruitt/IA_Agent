"""
量价数据工具 — 封装 AKShare 量价/资金流/龙虎榜相关函数

提供四个 MCP Tool:
  1. stock_price_volume — 日/周/月/分钟K线
  2. stock_capital_flow — 资金流向 (个股/板块/市场/北向)
  3. stock_lhb — 龙虎榜数据
  4. stock_margin_detail — 融资融券
"""

import akshare as ak
import pandas as pd

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter

_DATE_COLUMNS = (
    "日期",
    "date",
    "最近上榜日",
    "TRADE_DATE",
    "STATISTICS_DATE",
)


def _safe_call(limiter: RateLimiter, func, *args, **kwargs) -> pd.DataFrame:
    """带限流的安全调用"""
    func_name = func.__name__
    if not limiter.acquire(func_name, timeout=15.0):
        raise TimeoutError(f"限流超时: {func_name} (数据源: {limiter.get_source(func_name)})")
    return func(*args, **kwargs)


def _split_symbol(symbol: str) -> tuple[str, str]:
    """拆分股票代码和交易所标识，兼容 000001 / SZ000001 / 000001.SZ。"""
    raw_symbol = symbol.upper()
    if "." in raw_symbol:
        code, market = raw_symbol.split(".", 1)
        return code, market.lower()
    if raw_symbol.startswith(("SH", "SZ", "BJ")):
        return raw_symbol[2:], raw_symbol[:2].lower()
    if raw_symbol.startswith("6"):
        return raw_symbol, "sh"
    if raw_symbol.startswith(("4", "8")):
        return raw_symbol, "bj"
    return raw_symbol, "sz"


def _tx_symbol(symbol: str) -> str:
    """转换为腾讯历史行情接口使用的代码，如 sz000001。"""
    code, market = _split_symbol(symbol)
    return f"{market}{code}"


def _sort_by_date(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """按常见日期字段排序。"""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            out_df = df.copy()
            out_df["_sort_date"] = pd.to_datetime(out_df[col], errors="coerce")
            out_df.sort_values("_sort_date", ascending=ascending, na_position="last", inplace=True)
            return out_df.drop(columns=["_sort_date"])
    return df


def _normalize_tx_hist(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把腾讯历史行情字段转换为东方财富 K 线字段，便于下游统一消费。"""
    if df is None or df.empty:
        return pd.DataFrame()

    code, _ = _split_symbol(symbol)
    out_df = pd.DataFrame()
    out_df["日期"] = df["date"]
    out_df["股票代码"] = code
    out_df["开盘"] = df["open"]
    out_df["收盘"] = df["close"]
    out_df["最高"] = df["high"]
    out_df["最低"] = df["low"]
    # 腾讯接口第 6 列为成交量，AKShare 当前命名为 amount。
    out_df["成交量"] = df["amount"]
    out_df["成交额"] = ""
    out_df["振幅"] = (
        (out_df["最高"] - out_df["最低"]) / out_df["收盘"].shift(1) * 100
    ).round(2)
    out_df["涨跌幅"] = out_df["收盘"].pct_change().mul(100).round(2)
    out_df["涨跌额"] = out_df["收盘"].diff().round(2)
    out_df["换手率"] = ""
    return out_df


def _filter_by_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """从排名类全市场数据中过滤目标股票。"""
    if df is None or df.empty:
        return pd.DataFrame()

    code, _ = _split_symbol(symbol)
    for col in ("代码", "股票代码", "SECURITY_CODE", "SECUCODE"):
        if col in df.columns:
            stock_codes = df[col].astype(str).str.extract(r"(\d{6})", expand=False)
            return df[stock_codes == code]
    return pd.DataFrame()


def _df_to_records(df: pd.DataFrame, max_rows: int = 500, latest_window: bool = False) -> list:
    """DataFrame 转为 record 列表"""
    if df is None or df.empty:
        return []
    if latest_window:
        # 先取最新窗口，再倒序展示，确保 UI 和 Agent prompt 都是最新数据在前。
        df = _sort_by_date(df, ascending=True).tail(max_rows)
        df = _sort_by_date(df, ascending=False)
    elif len(df) > max_rows:
        df = df.head(max_rows)
    # 将 date 类型转为字符串，避免 JSON 序列化问题
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: str(x) if hasattr(x, "strftime") else x)
    df = df.fillna("")
    return df.to_dict(orient="records")


# ──────────────────────────────────────────────
# Tool 1: stock_price_volume
# ──────────────────────────────────────────────

PERIOD_MAP = {
    "daily": "daily",
    "weekly": "weekly",
    "monthly": "monthly",
}


def stock_price_volume(
    symbol: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    获取个股K线量价数据

    :param symbol: 股票代码，如 "000001"
    :param period: 周期 daily | weekly | monthly
    :param start_date: 起始日期 "20240101" (可选)
    :param end_date: 结束日期 "20241231" (可选)
    :param adjust: 复权类型 qfq(前复权) | hfq(后复权) | ""(不复权)
    :return: dict 包含K线数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    code, market = _split_symbol(symbol)
    result = {"symbol": code, "period": period, "market": market}

    try:
        if period in PERIOD_MAP:
            cache_args = (symbol, period, start_date, end_date, adjust)
            cached = cache.get("quote", "stock_zh_a_hist", cache_args)
            df = cached if cached is not None else None
            if df is not None and not df.empty:
                result["source"] = "eastmoney.stock_zh_a_hist"

            primary_error = None
            if df is None or df.empty:
                try:
                    df = _safe_call(
                        limiter, ak.stock_zh_a_hist,
                        symbol=code, period=period,
                        start_date=start_date, end_date=end_date,
                        adjust=adjust,
                    )
                    cache.set("quote", "stock_zh_a_hist", cache_args, data=df)
                    result["source"] = "eastmoney.stock_zh_a_hist"
                except Exception as e:
                    primary_error = str(e)

            if (df is None or df.empty) and period == "daily":
                fallback_args = (_tx_symbol(code), start_date or "19000101", end_date or "20500101", adjust)
                fallback_cached = cache.get("quote", "stock_zh_a_hist_tx", fallback_args)
                try:
                    if fallback_cached is not None and not fallback_cached.empty:
                        fallback_df = fallback_cached
                    else:
                        fallback_df = _safe_call(
                            limiter,
                            ak.stock_zh_a_hist_tx,
                            symbol=_tx_symbol(code),
                            start_date=start_date or "19000101",
                            end_date=end_date or "20500101",
                            adjust=adjust,
                            timeout=15,
                        )
                        cache.set("quote", "stock_zh_a_hist_tx", fallback_args, data=fallback_df)
                    df = _normalize_tx_hist(fallback_df, code)
                    result["source"] = "tencent.stock_zh_a_hist_tx"
                    result["source_warning"] = (
                        "东方财富 K 线接口不可用或返回空，已切换到腾讯历史行情备用源"
                    )
                    if primary_error:
                        result["primary_source_error"] = primary_error
                except Exception as fallback_error:
                    if primary_error:
                        result["error"] = (
                            f"东方财富 K 线失败: {primary_error}; "
                            f"腾讯备用 K 线也失败: {fallback_error}"
                        )
                    else:
                        result["error"] = str(fallback_error)
                    result["data"] = []
                    return result

            result["sort_order"] = "date_desc"
            result["data"] = _df_to_records(df, max_rows=300, latest_window=True)
            if not result["data"] and primary_error:
                result["error"] = primary_error
        else:
            result["error"] = f"不支持的 period: {period}，请使用 daily/weekly/monthly"
    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 2: stock_capital_flow
# ──────────────────────────────────────────────

def stock_capital_flow(
    symbol: str = "",
    scope: str = "individual",
    indicator: str = "今日",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    资金流向分析

    :param symbol: 股票代码 (scope=individual 时必填)
    :param scope: 范围
        - "individual": 个股资金流
        - "sector": 板块资金排名
        - "market": 大盘资金流
        - "north": 北向资金
    :param indicator: 时间维度 (今日/3日/5日/10日)，scope=sector 时使用
    :return: dict 包含资金流数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"scope": scope}

    try:
        if scope == "individual":
            if not symbol:
                return {"error": "个股资金流需要提供 symbol"}
            code, market = _split_symbol(symbol)
            result.update({"symbol": code, "market": market})
            cached = cache.get("quote", "stock_individual_fund_flow", (symbol,))
            df = cached if cached is not None else None
            primary_error = None
            if df is None or df.empty:
                try:
                    df = _safe_call(limiter, ak.stock_individual_fund_flow, stock=code, market=market)
                    cache.set("quote", "stock_individual_fund_flow", (symbol,), data=df)
                    result["source"] = "eastmoney.stock_individual_fund_flow"
                except Exception as e:
                    primary_error = str(e)
            else:
                result["source"] = "eastmoney.stock_individual_fund_flow"

            if df is None or df.empty:
                try:
                    rank_cached = cache.get("quote", "stock_individual_fund_flow_rank", ("今日",))
                    if rank_cached is not None and not rank_cached.empty:
                        rank_df = rank_cached
                    else:
                        rank_df = _safe_call(limiter, ak.stock_individual_fund_flow_rank, indicator="今日")
                        cache.set("quote", "stock_individual_fund_flow_rank", ("今日",), data=rank_df)
                    df = _filter_by_symbol(rank_df, code)
                    result["source"] = "eastmoney.stock_individual_fund_flow_rank"
                    result["source_warning"] = (
                        "个股历史资金流接口不可用或返回空，已用今日资金流排名按股票代码过滤兜底"
                    )
                    if primary_error:
                        result["primary_source_error"] = primary_error
                except Exception as fallback_error:
                    if primary_error:
                        result["error"] = (
                            f"个股历史资金流失败: {primary_error}; "
                            f"今日资金流排名兜底也失败: {fallback_error}"
                        )
                    else:
                        result["error"] = str(fallback_error)
                    result["data"] = []
                    return result

            result["data"] = _df_to_records(df, max_rows=30, latest_window=True)
            if not result["data"] and primary_error:
                result["error"] = primary_error

        elif scope == "sector":
            cached = cache.get("quote", "stock_sector_fund_flow_rank", (indicator,))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=50)
            else:
                df = _safe_call(limiter, ak.stock_sector_fund_flow_rank, indicator=indicator, sector_type="行业资金流")
                cache.set("quote", "stock_sector_fund_flow_rank", (indicator,), data=df)
                result["data"] = _df_to_records(df, max_rows=50)

        elif scope == "market":
            cached = cache.get("quote", "stock_market_fund_flow", ())
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=10, latest_window=True)
            else:
                df = _safe_call(limiter, ak.stock_market_fund_flow)
                cache.set("quote", "stock_market_fund_flow", (), data=df)
                result["data"] = _df_to_records(df, max_rows=10, latest_window=True)

        elif scope == "north":
            cached = cache.get("quote", "stock_hsgt_hist_em", ("北向资金",))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=30, latest_window=True)
            else:
                df = _safe_call(limiter, ak.stock_hsgt_hist_em, symbol="北向资金")
                cache.set("quote", "stock_hsgt_hist_em", ("北向资金",), data=df)
                result["data"] = _df_to_records(df, max_rows=30, latest_window=True)

        else:
            result["error"] = f"不支持的 scope: {scope}"

    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 3: stock_lhb
# ──────────────────────────────────────────────

def stock_lhb(
    date: str = "",
    symbol: str = "",
    detail_type: str = "detail",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    龙虎榜数据

    :param date: 日期 "20241008" (detail_type=detail 时必填)
    :param symbol: 股票代码 (detail_type=stock_statistic 时可选)
    :param detail_type:
        - "detail": 龙虎榜明细
        - "stock_statistic": 个股上榜统计
        - "institution": 机构买卖统计
    :return: dict 包含龙虎榜数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"detail_type": detail_type}

    try:
        if detail_type == "detail":
            if not date:
                return {"error": "龙虎榜明细需要提供 date"}
            cached = cache.get("quote", "stock_lhb_detail_em", (date,))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=50, latest_window=True)
            else:
                df = _safe_call(limiter, ak.stock_lhb_detail_em, start_date=date, end_date=date)
                cache.set("quote", "stock_lhb_detail_em", (date,), data=df)
                result["data"] = _df_to_records(df, max_rows=50, latest_window=True)

        elif detail_type == "stock_statistic":
            cached = cache.get("quote", "stock_lhb_stock_statistic_em", ("近一月",))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=50)
            else:
                df = _safe_call(limiter, ak.stock_lhb_stock_statistic_em, symbol="近一月")
                cache.set("quote", "stock_lhb_stock_statistic_em", ("近一月",), data=df)
                result["data"] = _df_to_records(df, max_rows=50)

        elif detail_type == "institution":
            cached = cache.get("quote", "stock_lhb_jgmmtj_em", ("近一月",))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=50)
            else:
                df = _safe_call(limiter, ak.stock_lhb_jgmmtj_em, symbol="近一月")
                cache.set("quote", "stock_lhb_jgmmtj_em", ("近一月",), data=df)
                result["data"] = _df_to_records(df, max_rows=50)

        else:
            result["error"] = f"不支持的 detail_type: {detail_type}"

    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 4: stock_margin_detail
# ──────────────────────────────────────────────

def stock_margin_detail(
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    融资融券数据

    :return: dict 包含融资融券数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {}

    try:
        cached = cache.get("quote", "stock_margin_account_info", ())
        if cached is not None:
            result["data"] = _df_to_records(cached, max_rows=10, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_margin_account_info)
            cache.set("quote", "stock_margin_account_info", (), data=df)
            result["data"] = _df_to_records(df, max_rows=10, latest_window=True)
    except Exception as e:
        result["error"] = str(e)

    return result
