"""
宏观景气分析 Agent

职责: 从经济周期、货币政策、产业景气三个维度评估
输入: 无需特定股票代码 (全局数据)
输出: MacroResult 结构化评分
"""

import json
import logging
from typing import Optional

from agents.data_provider import DataProvider
from agents.data_freshness import apply_freshness_policy, freshness_reason
from agents.data_snapshot import summarize_raw_data
from agents.llm_utils import call_llm_json
from config.agent_prompts.macro_prompt import MACRO_SYSTEM_PROMPT, MACRO_USER_PROMPT
from models import MacroResult, DimensionScore

logger = logging.getLogger(__name__)


def _skipped_macro_result(raw_data: dict) -> MacroResult:
    """数据时效未通过时返回结构化跳过结果。"""
    freshness = raw_data.get("freshness", {})
    reason = freshness_reason(freshness, "宏观数据时效未通过，暂不参与评分。")
    return MacroResult(
        economic_cycle=DimensionScore(dimension="经济周期", score=50, summary=reason),
        monetary_policy=DimensionScore(dimension="货币环境", score=50, summary=reason),
        industry_cycle=DimensionScore(dimension="产业景气", score=50, summary=reason),
        total_score=50,
        raw_data_summary=summarize_raw_data(raw_data),
    )


def _extract_quality_notes(result: dict, label: str) -> list[str]:
    """提取宏观 MCP 返回中的错误、无有效值和滞后说明。"""
    notes = []
    for key, value in result.items():
        prefix = f"{label}.{key}"
        if key == "error" or key.endswith("_error"):
            notes.append(f"{prefix}: {value}")
        elif isinstance(value, dict):
            if value.get("error"):
                notes.append(f"{prefix}.error: {value['error']}")
            latest = value.get("latest")
            if isinstance(latest, dict) and latest.get("status") not in (None, "ok"):
                notes.append(f"{prefix}.latest_status: {latest.get('status')} ({latest.get('value_col')})")
            if value.get("note"):
                notes.append(f"{prefix}.note: {value['note']}")
    return notes


def fetch_macro_data(dp: DataProvider) -> dict:
    """获取宏观分析所需的所有原始数据"""
    data = {"errors": []}

    # 中国宏观数据
    try:
        result = dp.get_macro_china()
        data["errors"].extend(_extract_quality_notes(result, "macro"))
        data["macro"] = result
    except Exception as e:
        logger.warning(f"获取中国宏观数据失败: {e}")
        data["errors"].append(f"macro.exception: {e}")
        data["macro"] = {}

    # 全球利率
    try:
        result = dp.get_global_interest()
        data["errors"].extend(_extract_quality_notes(result, "interest"))
        data["interest"] = result
    except Exception as e:
        logger.warning(f"获取全球利率失败: {e}")
        data["errors"].append(f"interest.exception: {e}")
        data["interest"] = {}

    # 市场估值
    try:
        result = dp.get_market_valuation()
        data["errors"].extend(_extract_quality_notes(result, "valuation"))
        data["valuation"] = result
    except Exception as e:
        logger.warning(f"获取市场估值失败: {e}")
        data["errors"].append(f"valuation.exception: {e}")
        data["valuation"] = {}

    return data


def format_data_for_prompt(data: dict) -> dict:
    """将原始数据格式化为提示词文本"""
    return {
        "macro_data": json.dumps(data.get("macro", {}), ensure_ascii=False, default=str)[:4000],
        "interest_data": json.dumps(data.get("interest", {}), ensure_ascii=False, default=str)[:1500],
        "valuation_data": json.dumps(data.get("valuation", {}), ensure_ascii=False, default=str)[:1500],
        "data_quality_notes": json.dumps(data.get("errors", []), ensure_ascii=False, default=str)[:2000],
    }


def parse_macro_result(raw: dict) -> MacroResult:
    """将 LLM 输出解析为 MacroResult"""
    try:
        return MacroResult(
            economic_cycle=DimensionScore(**raw.get("economic_cycle", {
                "dimension": "经济周期", "score": 50, "summary": "数据不足",
            })),
            monetary_policy=DimensionScore(**raw.get("monetary_policy", {
                "dimension": "货币环境", "score": 50, "summary": "数据不足",
            })),
            industry_cycle=DimensionScore(**raw.get("industry_cycle", {
                "dimension": "产业景气", "score": 50, "summary": "数据不足",
            })),
            total_score=raw.get("total_score", 50),
            raw_data_summary=raw.get("raw_data_summary", {}),
        )
    except Exception as e:
        logger.error(f"解析宏观结果失败: {e}")
        return MacroResult(
            economic_cycle=DimensionScore(dimension="经济周期", score=50, summary=f"解析失败: {e}"),
            monetary_policy=DimensionScore(dimension="货币环境", score=50, summary="解析失败"),
            industry_cycle=DimensionScore(dimension="产业景气", score=50, summary="解析失败"),
            total_score=50,
        )


def analyze_macro(
    dp: DataProvider = None,
) -> MacroResult:
    """
    宏观景气分析 Agent 主入口

    :param dp: 数据提供者
    :return: MacroResult
    """
    if dp is None:
        dp = DataProvider()

    logger.info("[宏观Agent] 开始分析")

    # 1. 获取数据
    raw_data = apply_freshness_policy("macro", fetch_macro_data(dp))
    freshness = raw_data.get("freshness", {})
    if freshness and not freshness.get("usable", True):
        logger.warning(f"[宏观Agent] 数据时效未通过，跳过分析: {freshness.get('reason')}")
        return _skipped_macro_result(raw_data)

    formatted = format_data_for_prompt(raw_data)

    # 2. 构建 prompt
    user_prompt = MACRO_USER_PROMPT.format(**formatted)

    # 3. 调用 LLM
    try:
        llm_output = call_llm_json(
            system_prompt=MACRO_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[宏观Agent] LLM 调用失败: {e}")
        return MacroResult(
            economic_cycle=DimensionScore(dimension="经济周期", score=50, summary=f"LLM调用失败: {e}"),
            monetary_policy=DimensionScore(dimension="货币环境", score=50, summary="LLM调用失败"),
            industry_cycle=DimensionScore(dimension="产业景气", score=50, summary="LLM调用失败"),
            total_score=50,
            raw_data_summary=summarize_raw_data(raw_data),
        )

    # 4. 解析结果
    result = parse_macro_result(llm_output)
    result.raw_data_summary = summarize_raw_data(raw_data)
    logger.info(f"[宏观Agent] 分析完成: 综合得分={result.total_score}")

    return result
