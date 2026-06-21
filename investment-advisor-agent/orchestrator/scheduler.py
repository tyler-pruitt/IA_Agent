"""
调度器 — 编排子Agent的并行执行和结果汇聚

支持:
  - 单股分析: 并行执行已启用子Agent → 综合决策
  - 对比分析: 分别分析 → 对比输出
  - 板块分析: 行业数据 + 可选宏观 → 板块景气度
  - 宏观分析: 宏观 Agent 启用时执行，否则返回暂停说明
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from agents.data_provider import DataProvider
from agents.fundamental_agent import analyze_fundamental
from agents.technical_agent import analyze_technical
from agents.sentiment_agent import analyze_sentiment
from agents.macro_agent import analyze_macro
from agents.data_freshness import result_freshness_reason, result_is_fresh
from config.settings import MACRO_AGENT_DISABLED_REASON, MACRO_AGENT_ENABLED
from orchestrator.planner import parse_intent, IntentType
from orchestrator.decision_agent import generate_recommendation
from models import (
    UserProfile,
    FundamentalResult,
    TechnicalResult,
    SentimentResult,
    MacroResult,
    Recommendation,
)

logger = logging.getLogger(__name__)


def _resolve_include_macro(include_macro: Optional[bool]) -> bool:
    """默认按配置决定是否启用宏观 Agent，显式参数可覆盖。"""
    return MACRO_AGENT_ENABLED if include_macro is None else include_macro


def _run_fundamental(symbol: str, dp: DataProvider) -> FundamentalResult:
    """执行基本面分析"""
    return analyze_fundamental(symbol, dp)


def _run_technical(symbol: str, dp: DataProvider) -> TechnicalResult:
    """执行量价技术分析"""
    return analyze_technical(symbol, dp)


def _run_sentiment(symbol: str, dp: DataProvider) -> SentimentResult:
    """执行舆情景气分析"""
    return analyze_sentiment(symbol, dp)


def _run_macro(dp: DataProvider) -> MacroResult:
    """执行宏观景气分析"""
    return analyze_macro(dp)


def _collect_agent_results(
    symbol: str,
    dp: DataProvider,
    include_macro: bool = True,
) -> dict:
    """并行执行子 Agent 并收集结果。"""
    results = {
        "fundamental": None,
        "technical": None,
        "sentiment": None,
        "macro": None,
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_run_fundamental, symbol, dp): "fundamental",
            executor.submit(_run_technical, symbol, dp): "technical",
            executor.submit(_run_sentiment, symbol, dp): "sentiment",
        }
        if include_macro:
            futures[executor.submit(_run_macro, dp)] = "macro"

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                results[agent_name] = future.result()
                logger.info(f"[调度] {agent_name} Agent 完成")
            except Exception as e:
                logger.error(f"[调度] {agent_name} Agent 异常: {e}")

    return results


def _build_recommendation_from_results(
    symbol: str,
    profile: UserProfile,
    results: dict,
    dp: DataProvider,
) -> Recommendation:
    """根据子 Agent 结果生成最终建议。"""
    return generate_recommendation(
        symbol=symbol,
        profile=profile,
        fundamental_result=results.get("fundamental"),
        technical_result=results.get("technical"),
        sentiment_result=results.get("sentiment"),
        macro_result=results.get("macro"),
        dp=dp,
    )


def _serialize_agent_results(results: dict, include_macro: bool) -> dict:
    """序列化子 Agent 输出，供 UI 展示数据快照。"""
    serialized = {}
    for name, result in results.items():
        if result is not None:
            serialized[name] = result.model_dump()
        elif name == "macro" and not include_macro:
            serialized[name] = {
                "disabled": True,
                "message": MACRO_AGENT_DISABLED_REASON,
            }
        else:
            serialized[name] = None
    return serialized


def analyze_stock(
    symbol: str,
    profile: UserProfile = None,
    dp: DataProvider = None,
    include_macro: Optional[bool] = None,
) -> Recommendation:
    """
    对单只股票进行全面分析

    已启用子Agent并行执行 → 综合决策

    :param symbol: 股票代码
    :param profile: 投资者偏好
    :param dp: 数据提供者
    :param include_macro: 是否包含宏观分析；None 时使用 MACRO_AGENT_ENABLED
    :return: Recommendation 投资建议
    """
    if profile is None:
        profile = UserProfile()
    if dp is None:
        dp = DataProvider()

    include_macro = _resolve_include_macro(include_macro)
    logger.info(f"=== 开始分析 {symbol} (macro={'on' if include_macro else 'off'}) ===")

    results = _collect_agent_results(symbol, dp, include_macro)
    recommendation = _build_recommendation_from_results(symbol, profile, results, dp)

    logger.info(f"=== 分析完成 {symbol}: {recommendation.advice.value} (评分:{recommendation.total_score}) ===")
    return recommendation


def analyze_stock_with_details(
    symbol: str,
    profile: UserProfile = None,
    dp: DataProvider = None,
    include_macro: Optional[bool] = None,
) -> dict:
    """分析个股并返回推荐和子 Agent 数据快照。"""
    if profile is None:
        profile = UserProfile()
    if dp is None:
        dp = DataProvider()

    include_macro = _resolve_include_macro(include_macro)
    logger.info(f"=== 开始分析 {symbol} (macro={'on' if include_macro else 'off'}) ===")
    results = _collect_agent_results(symbol, dp, include_macro)
    recommendation = _build_recommendation_from_results(symbol, profile, results, dp)
    logger.info(f"=== 分析完成 {symbol}: {recommendation.advice.value} (评分:{recommendation.total_score}) ===")

    return {
        "recommendation": recommendation,
        "agent_results": _serialize_agent_results(results, include_macro),
    }


def analyze_comparison(
    symbol_a: str,
    symbol_b: str,
    profile: UserProfile = None,
    dp: DataProvider = None,
) -> dict:
    """
    对比分析两只股票

    :return: {"stock_a": Recommendation, "stock_b": Recommendation, "comparison": "对比总结"}
    """
    if profile is None:
        profile = UserProfile()
    if dp is None:
        dp = DataProvider()

    # 并行分析两只股票
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(analyze_stock, symbol_a, profile, dp, include_macro=False)
        future_b = executor.submit(analyze_stock, symbol_b, profile, dp, include_macro=False)

    rec_a = future_a.result()
    rec_b = future_b.result()

    # 对比总结
    comparison = _build_comparison(rec_a, rec_b)

    return {
        "stock_a": rec_a,
        "stock_b": rec_b,
        "comparison": comparison,
    }


def _build_comparison(rec_a: Recommendation, rec_b: Recommendation) -> str:
    """构建对比总结"""
    lines = []
    winner = "A" if rec_a.total_score >= rec_b.total_score else "B"

    lines.append(f"对比结果: {winner}股综合评分更高")
    lines.append(f"  A股({rec_a.symbol}): {rec_a.total_score}分 - {rec_a.advice.value}")
    lines.append(f"  B股({rec_b.symbol}): {rec_b.total_score}分 - {rec_b.advice.value}")

    # 各维度对比
    ordered_dims = ["基本面", "量价", "舆情", "宏观"]
    active_dims = [
        dim
        for dim in ordered_dims
        if dim in rec_a.dimension_scores or dim in rec_b.dimension_scores
    ]
    for dim in active_dims:
        score_a = rec_a.dimension_scores.get(dim)
        score_b = rec_b.dimension_scores.get(dim)
        if score_a is None or score_b is None:
            lines.append(f"  {dim}: A={score_a if score_a is not None else '—'} vs B={score_b if score_b is not None else '—'}")
            continue
        winner_dim = "A" if score_a >= score_b else "B"
        lines.append(f"  {dim}: A={score_a} vs B={score_b} → {winner_dim}占优")

    return "\n".join(lines)


def run_query(
    user_query: str,
    profile: UserProfile = None,
    dp: DataProvider = None,
) -> dict:
    """
    用户查询入口 — 解析意图 → 路由到对应分析流程

    :param user_query: 用户自然语言查询
    :param profile: 投资者偏好
    :param dp: 数据提供者
    :return: 分析结果 dict
    """
    if profile is None:
        profile = UserProfile()
    if dp is None:
        dp = DataProvider()

    # 1. 解析意图
    intent_result = parse_intent(user_query)
    intent = intent_result["intent"]

    logger.info(f"[调度] 用户查询: {user_query} → 意图: {intent}")

    # 2. 路由到对应分析
    if intent == "stock_analysis":
        symbols = intent_result.get("symbols", [])
        if not symbols:
            return {"error": "未识别到股票代码，请提供具体股票代码，如 000001"}
        symbol = symbols[0]
        result = analyze_stock_with_details(symbol, profile, dp)
        return {
            "type": "stock_analysis",
            "recommendation": result["recommendation"].model_dump(),
            "analysis_data": result["agent_results"],
        }

    elif intent == "comparison":
        symbols = intent_result.get("symbols", [])
        if len(symbols) < 2:
            return {"error": "对比分析需要两只股票代码"}
        result = analyze_comparison(symbols[0], symbols[1], profile, dp)
        return {
            "type": "comparison",
            "stock_a": result["stock_a"].model_dump(),
            "stock_b": result["stock_b"].model_dump(),
            "comparison": result["comparison"],
        }

    elif intent == "sector_analysis":
        sector = intent_result.get("sector", "")
        if not sector:
            sector = "新能源"  # 默认
        sector_data = dp.get_sector("industry", sector)
        offline_sector = getattr(
            dp,
            "get_rqdata_sector_constituents",
            lambda *_args, **_kwargs: {},
        )(sector, limit=50)
        macro_data = _run_macro(dp) if MACRO_AGENT_ENABLED else None
        macro_usable = result_is_fresh(macro_data)
        return {
            "type": "sector_analysis",
            "sector": sector,
            "sector_data": sector_data,
            "offline_sector": offline_sector,
            "macro_disabled": not macro_usable,
            "macro_score": macro_data.total_score if macro_usable else None,
            "macro_summary": (
                macro_data.economic_cycle.summary
                if macro_usable
                else result_freshness_reason(macro_data, "宏观") or MACRO_AGENT_DISABLED_REASON
            ),
        }

    elif intent == "macro_analysis":
        if not MACRO_AGENT_ENABLED:
            return {
                "type": "macro_analysis",
                "disabled": True,
                "message": MACRO_AGENT_DISABLED_REASON,
            }
        result = _run_macro(dp)
        if not result_is_fresh(result):
            return {
                "type": "macro_analysis",
                "disabled": True,
                "message": result_freshness_reason(result, "宏观"),
                "result": result.model_dump(),
            }
        return {"type": "macro_analysis", "result": result.model_dump()}

    elif intent == "screening":
        # 筛选模式: 先获取千股千评初筛
        market_data = dp.get_market_emotion()
        return {
            "type": "screening",
            "message": "筛选功能将在后续版本实现，当前可使用 stock_analysis 分析具体股票",
            "market_overview": market_data,
        }

    else:
        return {
            "type": "general",
            "message": "我可以帮你分析具体股票(如'分析比亚迪')、板块(如'新能源板块')、或宏观形势(如'经济怎么样')。请告诉我你想了解什么？",
        }
