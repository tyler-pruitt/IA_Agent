"""
舆情/热度工具 — 封装 AKShare 舆情、热度、新闻相关函数

提供三个 MCP Tool:
  1. stock_sentiment — 综合舆情热度 (人气排名+百度热搜+微博舆情+千股千评)
  2. stock_news — 新闻与公告
  3. stock_market_emotion — 市场整体情绪指标
"""

from datetime import date
from io import StringIO

import akshare as ak
import pandas as pd
import requests
from bs4 import BeautifulSoup

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter

_DATE_COLUMNS = (
    "交易日",
    "交易日期",
    "日期",
    "时间",
    "发布时间",
    "date",
)


def _safe_call(limiter: RateLimiter, func, *args, **kwargs) -> pd.DataFrame:
    """带限流的安全调用"""
    func_name = func.__name__
    if not limiter.acquire(func_name, timeout=15.0):
        raise TimeoutError(f"限流超时: {func_name} (数据源: {limiter.get_source(func_name)})")
    return func(*args, **kwargs)


def _market_symbol(symbol: str) -> str:
    """转换为东方财富人气榜使用的带市场代码，如 SZ000001。"""
    raw_symbol = symbol.upper()
    if raw_symbol.startswith(("SH", "SZ", "BJ")):
        return raw_symbol
    if "." in raw_symbol:
        code, market = raw_symbol.split(".", 1)
        return f"{market}{code}"
    if raw_symbol.startswith("6"):
        return f"SH{raw_symbol}"
    if raw_symbol.startswith(("4", "8")):
        return f"BJ{raw_symbol}"
    return f"SZ{raw_symbol}"


def _plain_symbol(symbol: str) -> str:
    """提取 6 位股票代码。"""
    raw_symbol = symbol.upper()
    if raw_symbol.startswith(("SH", "SZ", "BJ")):
        return raw_symbol[2:]
    if "." in raw_symbol:
        return raw_symbol.split(".", 1)[0]
    return raw_symbol


def _sort_by_date(df: pd.DataFrame, ascending: bool = True) -> pd.DataFrame:
    """按常见日期字段排序。"""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            out_df = df.copy()
            out_df["_sort_date"] = pd.to_datetime(out_df[col], errors="coerce")
            out_df.sort_values("_sort_date", ascending=ascending, na_position="last", inplace=True)
            return out_df.drop(columns=["_sort_date"])
    return df


def _latest_date(df: pd.DataFrame) -> str:
    """提取数据中的最新日期字符串。"""
    for col in _DATE_COLUMNS:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                return str(parsed.max().date())
    return ""


def _freshness_note(df: pd.DataFrame, label: str, max_age_days: int = 30) -> str:
    """生成数据时效说明，帮助 UI 和 LLM 判断是否滞后。"""
    latest = _latest_date(df)
    if not latest:
        return ""
    latest_date = pd.to_datetime(latest, errors="coerce")
    if pd.isna(latest_date):
        return ""
    age_days = (pd.Timestamp(date.today()) - latest_date).days
    if age_days > max_age_days:
        return f"{label} 最新日期为 {latest}，距离当前日期超过 {max_age_days} 天，需按滞后数据处理"
    return f"{label} 最新日期为 {latest}"


def _filter_by_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """兼容不同字段名，从全市场数据中过滤目标股票。"""
    if df is None or df.empty:
        return pd.DataFrame()
    target = _plain_symbol(symbol)
    for col in ("股票代码", "代码", "证券代码", "SECURITY_CODE", "SECUCODE"):
        if col in df.columns:
            stock_codes = df[col].astype(str).str.extract(r"(\d{6})", expand=False)
            return df[stock_codes == target]
    return pd.DataFrame()


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


def _stock_buffett_index_lg_safe() -> pd.DataFrame:
    """兼容乐咕巴菲特指标字段变化，缺失分位数字段时保留主数据。"""
    from akshare.stock_feature.stock_a_indicator import get_cookie_csrf, get_token_lg

    token = get_token_lg()
    url = "https://legulegu.com/api/stockdata/marketcap-gdp/get-marketcap-gdp"
    r = requests.get(
        url,
        params={"token": token},
        **get_cookie_csrf(url="https://legulegu.com/stockdata/marketcap-gdp"),
    )
    data_json = r.json()
    temp_df = pd.DataFrame(data_json.get("data", []))
    if temp_df.empty:
        return pd.DataFrame()

    temp_df.rename(
        columns={
            "marketCap": "总市值",
            "gdp": "GDP",
            "close": "收盘价",
            "date": "日期",
            "quantileInAllHistory": "总历史分位数",
            "quantileInRecent10Years": "近十年分位数",
        },
        inplace=True,
    )
    for col in ("日期", "收盘价", "总市值", "GDP", "近十年分位数", "总历史分位数"):
        if col not in temp_df.columns:
            temp_df[col] = ""
    temp_df = temp_df[["日期", "收盘价", "总市值", "GDP", "近十年分位数", "总历史分位数"]]
    temp_df["日期"] = pd.to_datetime(temp_df["日期"], utc=True, errors="coerce").dt.date
    for col in ("收盘价", "总市值", "GDP", "近十年分位数", "总历史分位数"):
        temp_df[col] = pd.to_numeric(temp_df[col], errors="coerce")
    return temp_df


def _stock_market_activity_legu_safe() -> pd.DataFrame:
    """兼容乐咕市场活跃度页面结构变化。"""
    from akshare.utils.cons import headers

    url = "https://legulegu.com/stockdata/market-activity"
    r = requests.get(url, headers=headers)
    tables = pd.read_html(StringIO(r.text))
    if not tables:
        return pd.DataFrame()

    temp_df = tables[0]
    chunks = []
    for start in (0, 2, 4):
        chunk = temp_df.iloc[:, start : start + 2].copy()
        if chunk.shape[1] == 2:
            chunk.columns = ["item", "value"]
            chunks.append(chunk)
    out_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["item", "value"])
    out_df.dropna(how="all", axis=0, inplace=True)

    soup = BeautifulSoup(r.text, features="lxml")
    current_index = soup.find(name="div", attrs={"class": "current-index"})
    if current_index and current_index.text:
        parts = [item.strip() for item in current_index.text.split("：")]
        if len(parts) >= 2:
            out_df = pd.concat(
                [out_df, pd.DataFrame([{"item": parts[0], "value": parts[1]}])],
                ignore_index=True,
            )

    current_data = soup.find(name="div", attrs={"class": "current-data"})
    if current_data and current_data.text:
        out_df = pd.concat(
            [out_df, pd.DataFrame([{"item": "统计日期", "value": current_data.text.strip()}])],
            ignore_index=True,
        )
    return out_df.reset_index(drop=True)


# ──────────────────────────────────────────────
# Tool 1: stock_sentiment
# ──────────────────────────────────────────────

def stock_sentiment(
    symbol: str,
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    个股综合舆情热度分析

    聚合数据源:
      - 东方财富人气排名 + 关键词
      - 百度股市热搜
      - 微博舆情报告
      - 千股千评 (综合得分/关注指数/参与意愿/机构参与度)

    :param symbol: 股票代码，如 "000001"
    :return: dict 包含多维舆情数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"symbol": symbol}

    # 东方财富人气排名
    try:
        cached = cache.get("sentiment", "stock_hot_rank_em", ())
        if cached is not None:
            rank_df = cached
        else:
            rank_df = _safe_call(limiter, ak.stock_hot_rank_em)
            cache.set("sentiment", "stock_hot_rank_em", (), data=rank_df)
        # 筛选目标个股，AKShare 该接口当前字段为“代码”，旧版本可能为“股票代码”
        matched = _filter_by_symbol(rank_df, symbol)
        if not matched.empty:
            result["hot_rank"] = _df_to_records(matched, max_rows=1)
        else:
            result["hot_rank"] = "未进入人气排名"
    except Exception as e:
        result["hot_rank_error"] = str(e)

    # 个股热门关键词
    try:
        symbol_prefix = _market_symbol(symbol)
        cached = cache.get("sentiment", "stock_hot_keyword_em", (symbol_prefix,))
        if cached is not None:
            result["hot_keywords"] = _df_to_records(cached, max_rows=10, latest_window=True)
        else:
            df = _safe_call(limiter, ak.stock_hot_keyword_em, symbol=symbol_prefix)
            cache.set("sentiment", "stock_hot_keyword_em", (symbol_prefix,), data=df)
            result["hot_keywords"] = _df_to_records(df, max_rows=10, latest_window=True)
    except Exception as e:
        result["hot_keywords_error"] = str(e)

    # 千股千评 — 综合评价
    try:
        cached = cache.get("sentiment", "stock_comment_detail_zhpj_lspf_em", (symbol,))
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_comment_detail_zhpj_lspf_em, symbol=symbol)
            cache.set("sentiment", "stock_comment_detail_zhpj_lspf_em", (symbol,), data=df)
        result["comment_score_history"] = _df_to_records(df, max_rows=15, latest_window=True)
        result["comment_score_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "千股千评历史评分")
        if note:
            result["comment_score_note"] = note
    except Exception as e:
        result["comment_score_error"] = str(e)

    # 用户关注指数
    try:
        cached = cache.get("sentiment", "stock_comment_detail_scrd_focus_em", (symbol,))
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_comment_detail_scrd_focus_em, symbol=symbol)
            cache.set("sentiment", "stock_comment_detail_scrd_focus_em", (symbol,), data=df)
        result["user_focus_index"] = _df_to_records(df, max_rows=15, latest_window=True)
        result["user_focus_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "用户关注指数")
        if note:
            result["user_focus_note"] = note
    except Exception as e:
        result["focus_error"] = str(e)

    # 市场参与意愿
    try:
        cached = cache.get("sentiment", "stock_comment_detail_scrd_desire_em", (symbol,))
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_comment_detail_scrd_desire_em, symbol=symbol)
            cache.set("sentiment", "stock_comment_detail_scrd_desire_em", (symbol,), data=df)
        result["participation_desire"] = _df_to_records(df, max_rows=15, latest_window=True)
        result["participation_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "市场参与意愿")
        if note:
            result["participation_note"] = note
    except Exception as e:
        result["desire_error"] = str(e)

    # 机构参与度
    try:
        cached = cache.get("sentiment", "stock_comment_detail_zlkp_jgcyd_em", (symbol,))
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_comment_detail_zlkp_jgcyd_em, symbol=symbol)
            cache.set("sentiment", "stock_comment_detail_zlkp_jgcyd_em", (symbol,), data=df)
        result["institution_participation"] = _df_to_records(df, max_rows=15, latest_window=True)
        result["institution_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "机构参与度")
        if note:
            result["institution_note"] = note
    except Exception as e:
        result["institution_error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 2: stock_news
# ──────────────────────────────────────────────

def stock_news(
    symbol: str = "",
    scope: str = "individual",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    新闻与公告

    :param symbol: 股票代码 (scope=individual 时必填)
    :param scope:
        - "individual": 个股新闻 (东方财富)
        - "market": 财经要闻
        - "cctv": 央视新闻联播
    :return: dict 包含新闻数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {"scope": scope}

    try:
        if scope == "individual":
            if not symbol:
                return {"error": "个股新闻需要提供 symbol"}
            cached = cache.get("news", "stock_news_em", (symbol,))
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=15)
            else:
                df = _safe_call(limiter, ak.stock_news_em, symbol=symbol)
                cache.set("news", "stock_news_em", (symbol,), data=df)
                result["data"] = _df_to_records(df, max_rows=15)

        elif scope == "market":
            cached = cache.get("news", "stock_news_main_cx", ())
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=20)
            else:
                df = _safe_call(limiter, ak.stock_news_main_cx)
                cache.set("news", "stock_news_main_cx", (), data=df)
                result["data"] = _df_to_records(df, max_rows=20)

        elif scope == "cctv":
            cached = cache.get("news", "news_cctv", ())
            if cached is not None:
                result["data"] = _df_to_records(cached, max_rows=15)
            else:
                df = _safe_call(limiter, ak.news_cctv, date="")
                cache.set("news", "news_cctv", (), data=df)
                result["data"] = _df_to_records(df, max_rows=15)

        else:
            result["error"] = f"不支持的 scope: {scope}"

    except Exception as e:
        result["error"] = str(e)

    return result


# ──────────────────────────────────────────────
# Tool 3: stock_market_emotion
# ──────────────────────────────────────────────

def stock_market_emotion(
    date: str = "",
    cache: DataCache = None,
    limiter: RateLimiter = None,
) -> dict:
    """
    市场整体情绪指标

    聚合:
      - 千股千评 (全市场综合得分分布)
      - 拥挤度指标
      - 巴菲特指标
      - 涨停板池
      - 市场活跃度

    :param date: 日期 "20241008" (涨停池等需要)，可选
    :return: dict 包含市场情绪数据
    """
    if cache is None:
        cache = DataCache()
    if limiter is None:
        limiter = RateLimiter()

    result = {}

    # 千股千评 — 全市场概览 (取前50/后50)
    try:
        cached = cache.get("sentiment", "stock_comment_em", ())
        if cached is not None:
            comment_df = cached
        else:
            comment_df = _safe_call(limiter, ak.stock_comment_em)
            cache.set("sentiment", "stock_comment_em", (), data=comment_df)
        # 提取概要统计
        if "综合得分" in comment_df.columns:
            scores = comment_df["综合得分"]
            result["comment_summary"] = {
                "total_stocks": len(comment_df),
                "avg_score": round(float(scores.mean()), 1),
                "median_score": round(float(scores.median()), 1),
                "above_80": int((scores >= 80).sum()),
                "below_40": int((scores <= 40).sum()),
            }
        result["comment_top10"] = _df_to_records(comment_df.head(10), max_rows=10)
    except Exception as e:
        result["comment_error"] = str(e)

    # 拥挤度
    try:
        cached = cache.get("sentiment", "stock_a_congestion_lg", ())
        if cached is not None:
            df = cached
        else:
            df = _safe_call(limiter, ak.stock_a_congestion_lg)
            cache.set("sentiment", "stock_a_congestion_lg", (), data=df)
        result["congestion"] = _df_to_records(df, max_rows=5, latest_window=True)
        result["congestion_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "A股拥挤度")
        if note:
            result["congestion_note"] = note
    except Exception as e:
        result["congestion_error"] = str(e)

    # 巴菲特指标
    try:
        cached = cache.get("sentiment", "stock_buffett_index_lg", ())
        if cached is not None:
            df = cached
        else:
            try:
                df = _safe_call(limiter, ak.stock_buffett_index_lg)
            except KeyError:
                df = _stock_buffett_index_lg_safe()
                result["buffett_note"] = "乐咕巴菲特指标分位数字段缺失，已保留可用主字段"
            cache.set("sentiment", "stock_buffett_index_lg", (), data=df)
        result["buffett_index"] = _df_to_records(df, max_rows=5, latest_window=True)
        result["buffett_latest_date"] = _latest_date(df)
        note = _freshness_note(df, "巴菲特指标")
        if note:
            result["buffett_freshness_note"] = note
    except Exception as e:
        result["buffett_error"] = str(e)

    # 涨停板池 (需要日期)
    if date:
        try:
            cached = cache.get("sentiment", "stock_zt_pool_em", (date,))
            if cached is not None:
                zt_df = cached
            else:
                zt_df = _safe_call(limiter, ak.stock_zt_pool_em, date=date)
                cache.set("sentiment", "stock_zt_pool_em", (date,), data=zt_df)
            result["zt_pool"] = {
                "date": date,
                "count": len(zt_df),
                "top_stocks": _df_to_records(zt_df.head(10), max_rows=10),
            }
        except Exception as e:
            result["zt_pool_error"] = str(e)

    # 市场活跃度
    try:
        cached = cache.get("sentiment", "stock_market_activity_legu", ())
        if cached is not None:
            result["market_activity"] = _df_to_records(cached, max_rows=5)
        else:
            try:
                df = _safe_call(limiter, ak.stock_market_activity_legu)
            except AttributeError:
                df = _stock_market_activity_legu_safe()
                result["activity_note"] = "乐咕市场活跃度页面结构变化，已使用安全解析保留可用表格字段"
            cache.set("sentiment", "stock_market_activity_legu", (), data=df)
            result["market_activity"] = _df_to_records(df, max_rows=5)
    except Exception as e:
        result["activity_error"] = str(e)

    return result
