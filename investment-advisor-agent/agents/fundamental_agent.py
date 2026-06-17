"""
基本面分析 Agent

职责: 从财务健康度、估值合理性、盈利成长性三个维度评估
输入: 股票代码
输出: FundamentalResult 结构化评分
"""

import json
import logging
from datetime import date
from typing import Optional

from agents.data_provider import DataProvider
from agents.data_freshness import apply_freshness_policy, freshness_reason
from agents.data_snapshot import summarize_raw_data
from agents.llm_utils import call_llm_json
from config.agent_prompts.fundamental_prompt import FUNDAMENTAL_SYSTEM_PROMPT, FUNDAMENTAL_USER_PROMPT
from models import FundamentalResult, DimensionScore, Signal

logger = logging.getLogger(__name__)


def _skipped_fundamental_result(symbol: str, raw_data: dict) -> FundamentalResult:
    """数据时效未通过时返回结构化跳过结果，保留原始数据快照。"""
    freshness = raw_data.get("freshness", {})
    reason = freshness_reason(freshness, "基本面数据时效未通过，暂不参与评分。")
    return FundamentalResult(
        symbol=symbol,
        financial_health=DimensionScore(dimension="财务健康度", score=50, summary=reason),
        valuation=DimensionScore(dimension="估值合理性", score=50, summary=reason),
        growth=DimensionScore(dimension="盈利成长性", score=50, summary=reason),
        risk_warnings=[reason],
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


def _recent_report_dates(limit: int = 8) -> list[str]:
    """生成最近若干个财报报告期。"""
    today = date.today()
    quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
    periods = []
    for year in range(today.year, today.year - 3, -1):
        for month, day in reversed(quarter_ends):
            report_date = date(year, month, day)
            if report_date <= today:
                periods.append(report_date.strftime("%Y%m%d"))
            if len(periods) >= limit:
                return periods
    return periods


def fetch_fundamental_data(symbol: str, dp: DataProvider) -> dict:
    """
    获取基本面分析所需的所有原始数据

    :param symbol: 股票代码
    :param dp: 数据提供者
    :return: dict 包含各类基本面数据
    """
    data = {"errors": []}

    # 估值数据
    try:
        result = dp.get_fundamental(symbol, "valuation")
        data["errors"].extend(_extract_errors(result, "valuation"))
        data["valuation"] = result.get("valuation", [])
    except Exception as e:
        logger.warning(f"获取估值数据失败: {e}")
        data["errors"].append(f"valuation.exception: {e}")
        data["valuation"] = []

    # 财务指标
    try:
        result = dp.get_fundamental(symbol, "indicator")
        data["errors"].extend(_extract_errors(result, "indicator"))
        data["indicator"] = result.get("financial_indicator", [])
    except Exception as e:
        logger.warning(f"获取财务指标失败: {e}")
        data["errors"].append(f"indicator.exception: {e}")
        data["indicator"] = []

    # 三大报表
    try:
        result = dp.get_fundamental(symbol, "balance")
        data["errors"].extend(_extract_errors(result, "balance"))
        data["balance"] = result.get("balance_sheet", [])
    except Exception as e:
        logger.warning(f"获取资产负债表失败: {e}")
        data["errors"].append(f"balance.exception: {e}")
        data["balance"] = []

    try:
        result = dp.get_fundamental(symbol, "profit")
        data["errors"].extend(_extract_errors(result, "profit"))
        data["profit"] = result.get("profit_sheet", [])
    except Exception as e:
        logger.warning(f"获取利润表失败: {e}")
        data["errors"].append(f"profit.exception: {e}")
        data["profit"] = []

    try:
        result = dp.get_fundamental(symbol, "cashflow")
        data["errors"].extend(_extract_errors(result, "cashflow"))
        data["cashflow"] = result.get("cash_flow", [])
    except Exception as e:
        logger.warning(f"获取现金流量表失败: {e}")
        data["errors"].append(f"cashflow.exception: {e}")
        data["cashflow"] = []

    # 盈利预测
    try:
        result = dp.get_profit_forecast(symbol)
        data["errors"].extend(_extract_errors(result, "forecast"))
        data["forecast"] = result
    except Exception as e:
        logger.warning(f"获取盈利预测失败: {e}")
        data["errors"].append(f"forecast.exception: {e}")
        data["forecast"] = {}

    # 业绩预告
    earnings_matches = []
    checked_dates = []
    for report_date in _recent_report_dates():
        checked_dates.append(report_date)
        try:
            result = dp.get_earnings_preview(report_date, symbol=symbol)
            data["errors"].extend(_extract_errors(result, f"earnings[{report_date}]"))
            earnings_matches = result.get("data", [])
            if earnings_matches:
                data["earnings"] = {
                    "date": report_date,
                    "preview_type": result.get("preview_type", "yjyg"),
                    "data": earnings_matches,
                }
                break
        except Exception as e:
            logger.warning(f"获取业绩预告失败: {e}")
            data["errors"].append(f"earnings[{report_date}].exception: {e}")
    else:
        data["earnings"] = {
            "date_checked": checked_dates,
            "data": [],
            "note": f"最近报告期未找到 {symbol} 对应业绩预告，不能使用其他股票样本替代",
        }

    return data


def format_data_for_prompt(data: dict) -> dict:
    """将原始数据格式化为提示词中的文本"""
    return {
        "valuation_data": json.dumps(data.get("valuation", []), ensure_ascii=False, default=str)[:3000],
        "indicator_data": json.dumps(data.get("indicator", []), ensure_ascii=False, default=str)[:2000],
        "balance_data": json.dumps(data.get("balance", []), ensure_ascii=False, default=str)[:2000],
        "profit_data": json.dumps(data.get("profit", []), ensure_ascii=False, default=str)[:2000],
        "cashflow_data": json.dumps(data.get("cashflow", []), ensure_ascii=False, default=str)[:2000],
        "forecast_data": json.dumps(data.get("forecast", {}), ensure_ascii=False, default=str)[:1000],
        "earnings_data": json.dumps(data.get("earnings", {}), ensure_ascii=False, default=str)[:1000],
        "data_quality_notes": json.dumps(data.get("errors", []), ensure_ascii=False, default=str)[:2000],
    }


def parse_fundamental_result(raw: dict, symbol: str) -> FundamentalResult:
    """将 LLM 输出的 dict 解析为 FundamentalResult 模型"""
    try:
        return FundamentalResult(
            symbol=symbol,
            financial_health=DimensionScore(**raw.get("financial_health", {
                "dimension": "财务健康度", "score": 50, "summary": "数据不足，给予中性评分",
            })),
            valuation=DimensionScore(**raw.get("valuation", {
                "dimension": "估值合理性", "score": 50, "summary": "数据不足，给予中性评分",
            })),
            growth=DimensionScore(**raw.get("growth", {
                "dimension": "盈利成长性", "score": 50, "summary": "数据不足，给予中性评分",
            })),
            risk_warnings=raw.get("risk_warnings", []),
            total_score=raw.get("total_score", 50),
            raw_data_summary=raw.get("raw_data_summary", {}),
        )
    except Exception as e:
        logger.error(f"解析基本面结果失败: {e}, raw={raw}")
        return FundamentalResult(
            symbol=symbol,
            financial_health=DimensionScore(dimension="财务健康度", score=50, summary=f"解析失败: {e}"),
            valuation=DimensionScore(dimension="估值合理性", score=50, summary="解析失败"),
            growth=DimensionScore(dimension="盈利成长性", score=50, summary="解析失败"),
            total_score=50,
        )


def analyze_fundamental(
    symbol: str,
    dp: DataProvider = None,
) -> FundamentalResult:
    """
    基本面分析 Agent 主入口

    :param symbol: 股票代码
    :param dp: 数据提供者 (可选，默认创建新实例)
    :return: FundamentalResult 结构化评分
    """
    if dp is None:
        dp = DataProvider()

    logger.info(f"[基本面Agent] 开始分析 {symbol}")

    # 1. 获取数据
    raw_data = apply_freshness_policy("fundamental", fetch_fundamental_data(symbol, dp))
    freshness = raw_data.get("freshness", {})
    if freshness and not freshness.get("usable", True):
        logger.warning(f"[基本面Agent] 数据时效未通过，跳过分析: {freshness.get('reason')}")
        return _skipped_fundamental_result(symbol, raw_data)

    formatted = format_data_for_prompt(raw_data)

    # 2. 构建 prompt
    user_prompt = FUNDAMENTAL_USER_PROMPT.format(symbol=symbol, **formatted)

    # 3. 调用 LLM
    try:
        llm_output = call_llm_json(
            system_prompt=FUNDAMENTAL_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[基本面Agent] LLM 调用失败: {e}")
        return FundamentalResult(
            symbol=symbol,
            financial_health=DimensionScore(dimension="财务健康度", score=50, summary=f"LLM调用失败: {e}"),
            valuation=DimensionScore(dimension="估值合理性", score=50, summary="LLM调用失败"),
            growth=DimensionScore(dimension="盈利成长性", score=50, summary="LLM调用失败"),
            total_score=50,
            raw_data_summary=summarize_raw_data(raw_data),
        )

    # 4. 解析结果
    result = parse_fundamental_result(llm_output, symbol)
    result.raw_data_summary = summarize_raw_data(raw_data)
    logger.info(f"[基本面Agent] 分析完成: 综合得分={result.total_score}")

    return result
