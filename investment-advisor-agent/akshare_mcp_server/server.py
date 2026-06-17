"""
AKShare MCP Server — 基于 MCP 协议为 AI Agent 提供金融数据工具

启动方式:
  stdio 模式:  python -m akshare_mcp_server
  SSE 模式:   python -m akshare_mcp_server --transport sse --port 8080

MCP 工具列表:
  1. stock_fundamental       — 个股基本面全景 (估值+指标+三表)
  2. stock_profit_forecast   — 分析师盈利预测
  3. stock_earnings_preview   — 业绩预告/快报
  4. stock_price_volume      — 日/周/月K线量价数据
  5. stock_capital_flow      — 资金流向 (个股/板块/市场/北向)
  6. stock_lhb               — 龙虎榜数据
  7. stock_margin_detail     — 融资融券
  8. stock_sentiment         — 个股综合舆情热度
  9. stock_news              — 新闻与公告
  10. stock_market_emotion   — 市场整体情绪指标
  11. macro_china_overview   — 中国宏观经济全景
  12. macro_global_interest  — 全球主要央行利率
  13. stock_sector_analysis  — 行业/概念板块分析
  14. stock_market_valuation — 市场整体估值指标
"""

import argparse
import json
import traceback

from mcp.server.fastmcp import FastMCP

from akshare_mcp_server.cache import DataCache
from akshare_mcp_server.rate_limiter import RateLimiter
from akshare_mcp_server.tools.fundamental import (
    stock_fundamental,
    stock_profit_forecast,
    stock_earnings_preview,
)
from akshare_mcp_server.tools.price_volume import (
    stock_price_volume,
    stock_capital_flow,
    stock_lhb,
    stock_margin_detail,
)
from akshare_mcp_server.tools.sentiment import (
    stock_sentiment,
    stock_news,
    stock_market_emotion,
)
from akshare_mcp_server.tools.macro import (
    macro_china_overview,
    macro_global_interest,
)
from akshare_mcp_server.tools.sector import (
    stock_sector_analysis,
    stock_market_valuation,
)

# ──────────────────────────────────────────────
# MCP Server 实例
# ──────────────────────────────────────────────

mcp = FastMCP(
    "AKShare Finance Data",
    instructions="""
你是金融数据分析助手，可以通过以下工具获取 A 股市场数据。

数据类别:
- 基本面: stock_fundamental, stock_profit_forecast, stock_earnings_preview
- 量价: stock_price_volume, stock_capital_flow, stock_lhb, stock_margin_detail
- 舆情: stock_sentiment, stock_news, stock_market_emotion
- 宏观: macro_china_overview, macro_global_interest
- 行业: stock_sector_analysis, stock_market_valuation

注意事项:
- 股票代码格式: 纯数字如 "000001"，无需加前缀 SH/SZ
- 日期格式: "20240331" (YYYYMMDD)
- 所有数据仅供参考，不构成投资建议
- 数据来源: 东方财富、同花顺、新浪财经、巨潮资讯、金十数据等
""".strip(),
)

# 全局缓存和限流实例
_cache = DataCache()
_limiter = RateLimiter()


# ──────────────────────────────────────────────
# 基本面工具
# ──────────────────────────────────────────────

@mcp.tool()
def get_stock_fundamental(symbol: str, report_type: str = "all") -> str:
    """
    获取个股基本面全景数据，包括估值(PE/PB/PEG/PS)、财务指标(ROE/毛利率)、三大报表。

    :param symbol: 股票代码，如 "000001"; MCP 会用全市场预测结果按代码过滤
    :param report_type: 返回类型 — all(全部) | valuation(估值) | indicator(指标) | balance(资产负债表) | profit(利润表) | cashflow(现金流量表)
    :return: JSON 格式的基本面数据
    """
    try:
        result = stock_fundamental(symbol, report_type, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_profit_forecast(symbol: str) -> str:
    """
    获取分析师盈利预测数据，包括一致预期EPS、目标价等。

    :param symbol: 股票代码，如 "000001"
    :return: JSON 格式的盈利预测数据
    """
    try:
        result = stock_profit_forecast(symbol, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_earnings_preview(date: str, preview_type: str = "yjyg", symbol: str = "") -> str:
    """
    获取业绩预告/快报/报表数据，用于检测业绩超预期信号。

    :param date: 报告期，如 "20240331"
    :param preview_type: 类型 — yjyg(业绩预告) | yjkb(业绩快报) | yjbb(业绩报表)
    :param symbol: 可选股票代码，如 "000001"; 传入后会先按目标股票过滤
    :return: JSON 格式的业绩数据
    """
    try:
        result = stock_earnings_preview(date, preview_type, symbol=symbol, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 量价工具
# ──────────────────────────────────────────────

@mcp.tool()
def get_stock_price_volume(
    symbol: str,
    period: str = "daily",
    start_date: str = "",
    end_date: str = "",
    adjust: str = "qfq",
) -> str:
    """
    获取个股K线量价数据（日/周/月K线），包含开盘价、收盘价、最高价、最低价、成交量、成交额等。

    :param symbol: 股票代码，如 "000001"
    :param period: 周期 — daily | weekly | monthly
    :param start_date: 起始日期 "20240101"（可选）
    :param end_date: 结束日期 "20241231"（可选）
    :param adjust: 复权类型 — qfq(前复权) | hfq(后复权) | 空(不复权)
    :return: JSON 格式的K线数据
    """
    try:
        result = stock_price_volume(symbol, period, start_date, end_date, adjust, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_capital_flow(symbol: str = "", scope: str = "individual", indicator: str = "今日") -> str:
    """
    获取资金流向数据，包括个股资金流、板块资金排名、大盘资金流、北向资金。

    :param symbol: 股票代码（scope=individual 时必填）
    :param scope: 范围 — individual(个股) | sector(板块) | market(大盘) | north(北向)
    :param indicator: 时间维度 — 今日 | 3日 | 5日 | 10日（scope=sector 时使用）
    :return: JSON 格式的资金流数据
    """
    try:
        result = stock_capital_flow(symbol, scope, indicator, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_lhb(date: str = "", symbol: str = "", detail_type: str = "detail") -> str:
    """
    获取龙虎榜数据，追踪游资和机构动向。

    :param date: 日期 "20241008"（detail_type=detail 时必填）
    :param symbol: 股票代码（可选）
    :param detail_type: 类型 — detail(明细) | stock_statistic(个股上榜统计) | institution(机构买卖统计)
    :return: JSON 格式的龙虎榜数据
    """
    try:
        result = stock_lhb(date, symbol, detail_type, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_margin_detail() -> str:
    """
    获取融资融券数据，反映市场杠杆情绪。

    :return: JSON 格式的融资融券数据
    """
    try:
        result = stock_margin_detail(cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 舆情工具
# ──────────────────────────────────────────────

@mcp.tool()
def get_stock_sentiment(symbol: str) -> str:
    """
    获取个股综合舆情热度，包括东方财富人气排名、热门关键词、千股千评得分、用户关注指数、市场参与意愿、机构参与度。

    :param symbol: 股票代码，如 "000001"
    :return: JSON 格式的舆情热度数据
    """
    try:
        result = stock_sentiment(symbol, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_news(symbol: str = "", scope: str = "individual") -> str:
    """
    获取新闻与公告，包括个股新闻、财经要闻、央视新闻联播。

    :param symbol: 股票代码（scope=individual 时必填）
    :param scope: 范围 — individual(个股新闻) | market(财经要闻) | cctv(央视新闻联播)
    :return: JSON 格式的新闻数据
    """
    try:
        result = stock_news(symbol, scope, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_market_emotion(date: str = "") -> str:
    """
    获取市场整体情绪指标，包括千股千评全市场概览、拥挤度、巴菲特指标、涨停板池、市场活跃度。

    :param date: 日期 "20241008"（可选，涨停池需要）
    :return: JSON 格式的市场情绪数据
    """
    try:
        result = stock_market_emotion(date, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 宏观工具
# ──────────────────────────────────────────────

@mcp.tool()
def get_macro_china_overview() -> str:
    """
    获取中国宏观经济全景数据，包括GDP、CPI、PPI、PMI(官方+财新)、M2、进出口、工业增加值、失业率、利率等。

    :return: JSON 格式的中国宏观数据
    """
    try:
        result = macro_china_overview(cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_macro_global_interest() -> str:
    """
    获取全球主要央行利率数据，包括美国、欧元区、中国、日本、英国、澳大利亚、瑞士、新西兰、俄罗斯、印度、巴西。

    :return: JSON 格式的全球利率数据
    """
    try:
        result = macro_global_interest(cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 行业/概念工具
# ──────────────────────────────────────────────

@mcp.tool()
def get_stock_sector_analysis(sector_type: str = "industry", name: str = "") -> str:
    """
    获取行业/概念板块分析数据，包括板块行情、历史K线、成分股。

    :param sector_type: 板块类型 — industry(行业) | concept(概念) | list(列出所有板块名称)
    :param name: 板块名称，如 "小金属"、"数据要素"（sector_type=list 时不需要）
    :return: JSON 格式的板块数据
    """
    try:
        result = stock_sector_analysis(sector_type, name, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


@mcp.tool()
def get_stock_market_valuation(index: str = "沪深300") -> str:
    """
    获取市场整体估值指标，包括指数PE/PB、巴菲特指标、全A股PB。

    :param index: 指数名称，如 "沪深300"、"上证50"、"深证"
    :return: JSON 格式的估值数据
    """
    try:
        result = stock_market_valuation(index, cache=_cache, limiter=_limiter)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()}, ensure_ascii=False)


# ──────────────────────────────────────────────
# 启动入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AKShare MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输协议: stdio (默认) 或 sse",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="SSE 模式端口 (默认 8080)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="缓存目录 (默认 ~/.cache/akshare_mcp/)",
    )
    args = parser.parse_args()

    # 重新初始化缓存 (支持自定义目录)
    global _cache
    if args.cache_dir:
        _cache = DataCache(cache_dir=args.cache_dir)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", port=args.port)


if __name__ == "__main__":
    main()
