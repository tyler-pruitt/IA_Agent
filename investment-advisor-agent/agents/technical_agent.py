"""
量价技术分析 Agent

职责: 从趋势、动量、资金流、盘口异动四个维度评估
输入: 股票代码
输出: TechnicalResult 结构化评分
"""

import json
import logging
from typing import Optional

from agents.data_provider import DataProvider
from agents.data_freshness import apply_freshness_policy, freshness_reason
from agents.data_snapshot import summarize_raw_data
from agents.llm_utils import call_llm_json
from config.agent_prompts.technical_prompt import TECHNICAL_SYSTEM_PROMPT, TECHNICAL_USER_PROMPT
from models import TechnicalResult, DimensionScore

logger = logging.getLogger(__name__)


def _skipped_technical_result(symbol: str, raw_data: dict) -> TechnicalResult:
    """数据时效未通过时返回结构化跳过结果。"""
    freshness = raw_data.get("freshness", {})
    reason = freshness_reason(freshness, "量价数据时效未通过，暂不参与评分。")
    return TechnicalResult(
        symbol=symbol,
        trend=DimensionScore(dimension="趋势判断", score=50, summary=reason),
        momentum=DimensionScore(dimension="动量信号", score=50, summary=reason),
        capital_flow=DimensionScore(dimension="资金流", score=50, summary=reason),
        anomalies=[{"type": "freshness_disabled", "description": reason, "significance": 8}],
        total_score=50,
        raw_data_summary=summarize_raw_data(raw_data),
    )


def _extract_errors(result: dict, label: str) -> list[str]:
    """提取 MCP 工具返回中的错误信息。"""
    errors = []
    for key, value in result.items():
        if key == "error" or key.endswith("_error"):
            errors.append(f"{label}.{key}: {value}")
    return errors


def fetch_technical_data(symbol: str, dp: DataProvider) -> dict:
    """获取量价技术分析所需的所有原始数据"""
    data = {"errors": []}

    # K线数据 (近30个交易日)
    try:
        result = dp.get_price_volume(symbol, "daily", start_date="", end_date="", adjust="qfq")
        data["errors"].extend(_extract_errors(result, "price"))
        data["price"] = result.get("data", [])
        data["price_meta"] = {key: value for key, value in result.items() if key != "data"}
    except Exception as e:
        logger.warning(f"获取K线数据失败: {e}")
        data["errors"].append(f"price.exception: {e}")
        data["price"] = []
        data["price_meta"] = {}

    # 资金流向
    try:
        result = dp.get_capital_flow(symbol, "individual")
        data["errors"].extend(_extract_errors(result, "capital_flow"))
        data["capital_flow"] = result
    except Exception as e:
        logger.warning(f"获取资金流失败: {e}")
        data["errors"].append(f"capital_flow.exception: {e}")
        data["capital_flow"] = {}

    # 北向资金
    try:
        result = dp.get_capital_flow(scope="north")
        data["errors"].extend(_extract_errors(result, "north_flow"))
        data["north_flow"] = result
    except Exception as e:
        logger.warning(f"获取北向资金失败: {e}")
        data["errors"].append(f"north_flow.exception: {e}")
        data["north_flow"] = {}

    # 龙虎榜
    try:
        result = dp.get_lhb(detail_type="stock_statistic")
        data["errors"].extend(_extract_errors(result, "lhb"))
        data["lhb"] = result
    except Exception as e:
        logger.warning(f"获取龙虎榜失败: {e}")
        data["errors"].append(f"lhb.exception: {e}")
        data["lhb"] = {}

    # 融资融券
    try:
        result = dp.get_margin()
        data["errors"].extend(_extract_errors(result, "margin"))
        data["margin"] = result
    except Exception as e:
        logger.warning(f"获取融资融券失败: {e}")
        data["errors"].append(f"margin.exception: {e}")
        data["margin"] = {}

    return data


def format_data_for_prompt(data: dict) -> dict:
    """将原始数据格式化为提示词中的文本"""
    return {
        "price_data": json.dumps(data.get("price", []), ensure_ascii=False, default=str)[:3000],
        "price_meta": json.dumps(data.get("price_meta", {}), ensure_ascii=False, default=str)[:1000],
        "capital_flow_data": json.dumps(data.get("capital_flow", {}), ensure_ascii=False, default=str)[:2000],
        "north_flow_data": json.dumps(data.get("north_flow", {}), ensure_ascii=False, default=str)[:1500],
        "lhb_data": json.dumps(data.get("lhb", {}), ensure_ascii=False, default=str)[:1000],
        "margin_data": json.dumps(data.get("margin", {}), ensure_ascii=False, default=str)[:1000],
        "data_quality_notes": json.dumps(data.get("errors", []), ensure_ascii=False, default=str)[:2000],
    }


def parse_technical_result(raw: dict, symbol: str) -> TechnicalResult:
    """将 LLM 输出解析为 TechnicalResult"""
    try:
        return TechnicalResult(
            symbol=symbol,
            trend=DimensionScore(**raw.get("trend", {
                "dimension": "趋势判断", "score": 50, "summary": "数据不足",
            })),
            momentum=DimensionScore(**raw.get("momentum", {
                "dimension": "动量信号", "score": 50, "summary": "数据不足",
            })),
            capital_flow=DimensionScore(**raw.get("capital_flow", {
                "dimension": "资金流", "score": 50, "summary": "数据不足",
            })),
            anomalies=raw.get("anomalies", []),
            total_score=raw.get("total_score", 50),
            raw_data_summary=raw.get("raw_data_summary", {}),
        )
    except Exception as e:
        logger.error(f"解析量价结果失败: {e}")
        return TechnicalResult(
            symbol=symbol,
            trend=DimensionScore(dimension="趋势判断", score=50, summary=f"解析失败: {e}"),
            momentum=DimensionScore(dimension="动量信号", score=50, summary="解析失败"),
            capital_flow=DimensionScore(dimension="资金流", score=50, summary="解析失败"),
            total_score=50,
        )


def analyze_technical(
    symbol: str,
    dp: DataProvider = None,
) -> TechnicalResult:
    """
    量价技术分析 Agent 主入口

    :param symbol: 股票代码
    :param dp: 数据提供者
    :return: TechnicalResult
    """
    if dp is None:
        dp = DataProvider()

    logger.info(f"[量价Agent] 开始分析 {symbol}")

    # 1. 获取数据
    raw_data = apply_freshness_policy("technical", fetch_technical_data(symbol, dp))
    freshness = raw_data.get("freshness", {})
    if freshness and not freshness.get("usable", True):
        logger.warning(f"[量价Agent] 数据时效未通过，跳过分析: {freshness.get('reason')}")
        return _skipped_technical_result(symbol, raw_data)

    formatted = format_data_for_prompt(raw_data)

    # 2. 构建 prompt
    user_prompt = TECHNICAL_USER_PROMPT.format(symbol=symbol, **formatted)

    # 3. 调用 LLM
    try:
        llm_output = call_llm_json(
            system_prompt=TECHNICAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[量价Agent] LLM 调用失败: {e}")
        return TechnicalResult(
            symbol=symbol,
            trend=DimensionScore(dimension="趋势判断", score=50, summary=f"LLM调用失败: {e}"),
            momentum=DimensionScore(dimension="动量信号", score=50, summary="LLM调用失败"),
            capital_flow=DimensionScore(dimension="资金流", score=50, summary="LLM调用失败"),
            total_score=50,
            raw_data_summary=summarize_raw_data(raw_data),
        )

    # 4. 解析结果
    result = parse_technical_result(llm_output, symbol)
    result.raw_data_summary = summarize_raw_data(raw_data)
    logger.info(f"[量价Agent] 分析完成: 综合得分={result.total_score}")

    return result
