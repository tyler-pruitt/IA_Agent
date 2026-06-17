"""
舆情景气分析 Agent

职责: 从市场情绪、新闻事件、社交热度、投资者互动四个维度评估
输入: 股票代码
输出: SentimentResult 结构化评分
"""

import json
import logging
from typing import Optional

from agents.data_provider import DataProvider
from agents.data_freshness import apply_freshness_policy, freshness_reason
from agents.data_snapshot import summarize_raw_data
from agents.llm_utils import call_llm_json
from config.agent_prompts.sentiment_prompt import SENTIMENT_SYSTEM_PROMPT, SENTIMENT_USER_PROMPT
from models import SentimentResult, DimensionScore

logger = logging.getLogger(__name__)


def _skipped_sentiment_result(symbol: str, raw_data: dict) -> SentimentResult:
    """数据时效未通过时返回结构化跳过结果。"""
    freshness = raw_data.get("freshness", {})
    reason = freshness_reason(freshness, "舆情数据时效未通过，暂不参与评分。")
    return SentimentResult(
        symbol=symbol,
        market_emotion=DimensionScore(dimension="市场情绪", score=50, summary=reason),
        social_heat=DimensionScore(dimension="社交热度", score=50, summary=reason),
        overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=50, summary=reason),
        news_events=[{"type": "freshness_disabled", "description": reason}],
        total_score=50,
        raw_data_summary=summarize_raw_data(raw_data),
    )


def _extract_errors(result: dict, label: str) -> list[str]:
    """提取 MCP 工具返回中的错误信息，方便传给 LLM 判断数据质量。"""
    errors = []
    for key, value in result.items():
        if key == "error" or key.endswith("_error"):
            errors.append(f"{label}.{key}: {value}")
    return errors


def fetch_sentiment_data(symbol: str, dp: DataProvider) -> dict:
    """获取舆情分析所需的所有原始数据"""
    data = {"errors": []}

    # 个股舆情
    try:
        result = dp.get_sentiment(symbol)
        data["errors"].extend(_extract_errors(result, "sentiment"))
        data["sentiment"] = result
    except Exception as e:
        logger.warning(f"获取个股舆情失败: {e}")
        data["errors"].append(f"sentiment.exception: {e}")
        data["sentiment"] = {}

    # 个股新闻
    try:
        result = dp.get_news(symbol, "individual")
        data["errors"].extend(_extract_errors(result, "news"))
        data["news"] = result
    except Exception as e:
        logger.warning(f"获取个股新闻失败: {e}")
        data["errors"].append(f"news.exception: {e}")
        data["news"] = {}

    # 市场整体情绪
    try:
        result = dp.get_market_emotion()
        data["errors"].extend(_extract_errors(result, "market_emotion"))
        data["market_emotion"] = result
    except Exception as e:
        logger.warning(f"获取市场情绪失败: {e}")
        data["errors"].append(f"market_emotion.exception: {e}")
        data["market_emotion"] = {}

    return data


def format_data_for_prompt(data: dict) -> dict:
    """将原始数据格式化为提示词文本"""
    return {
        "sentiment_data": json.dumps(data.get("sentiment", {}), ensure_ascii=False, default=str)[:3000],
        "news_data": json.dumps(data.get("news", {}), ensure_ascii=False, default=str)[:2000],
        "market_emotion_data": json.dumps(data.get("market_emotion", {}), ensure_ascii=False, default=str)[:2000],
        "data_quality_notes": json.dumps(data.get("errors", []), ensure_ascii=False, default=str)[:2000],
    }


def parse_sentiment_result(raw: dict, symbol: str) -> SentimentResult:
    """将 LLM 输出解析为 SentimentResult"""
    try:
        return SentimentResult(
            symbol=symbol,
            market_emotion=DimensionScore(**raw.get("market_emotion", {
                "dimension": "市场情绪", "score": 50, "summary": "数据不足",
            })),
            news_events=raw.get("news_events", []),
            social_heat=DimensionScore(**raw.get("social_heat", {
                "dimension": "社交热度", "score": 50, "summary": "数据不足",
            })),
            overall_market_emotion=DimensionScore(**raw.get("overall_market_emotion", {
                "dimension": "市场整体情绪", "score": 50, "summary": "数据不足",
            })),
            total_score=raw.get("total_score", 50),
            raw_data_summary=raw.get("raw_data_summary", {}),
        )
    except Exception as e:
        logger.error(f"解析舆情结果失败: {e}")
        return SentimentResult(
            symbol=symbol,
            market_emotion=DimensionScore(dimension="市场情绪", score=50, summary=f"解析失败: {e}"),
            social_heat=DimensionScore(dimension="社交热度", score=50, summary="解析失败"),
            overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=50, summary="解析失败"),
            total_score=50,
        )


def analyze_sentiment(
    symbol: str,
    dp: DataProvider = None,
) -> SentimentResult:
    """
    舆情景气分析 Agent 主入口

    :param symbol: 股票代码
    :param dp: 数据提供者
    :return: SentimentResult
    """
    if dp is None:
        dp = DataProvider()

    logger.info(f"[舆情Agent] 开始分析 {symbol}")

    # 1. 获取数据
    raw_data = apply_freshness_policy("sentiment", fetch_sentiment_data(symbol, dp))
    freshness = raw_data.get("freshness", {})
    if freshness and not freshness.get("usable", True):
        logger.warning(f"[舆情Agent] 数据时效未通过，跳过分析: {freshness.get('reason')}")
        return _skipped_sentiment_result(symbol, raw_data)

    formatted = format_data_for_prompt(raw_data)

    # 2. 构建 prompt
    user_prompt = SENTIMENT_USER_PROMPT.format(symbol=symbol, **formatted)

    # 3. 调用 LLM
    try:
        llm_output = call_llm_json(
            system_prompt=SENTIMENT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[舆情Agent] LLM 调用失败: {e}")
        return SentimentResult(
            symbol=symbol,
            market_emotion=DimensionScore(dimension="市场情绪", score=50, summary=f"LLM调用失败: {e}"),
            social_heat=DimensionScore(dimension="社交热度", score=50, summary="LLM调用失败"),
            overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=50, summary="LLM调用失败"),
            total_score=50,
            raw_data_summary=summarize_raw_data(raw_data),
        )

    # 4. 解析结果
    result = parse_sentiment_result(llm_output, symbol)
    result.raw_data_summary = summarize_raw_data(raw_data)
    logger.info(f"[舆情Agent] 分析完成: 综合得分={result.total_score}")

    return result
