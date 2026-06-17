"""
决策综合 Agent — 综合四个子Agent结果，生成投资建议

决策逻辑:
  1. 多维度加权打分 (按投资风格/周期/资金规模)
  2. 风险过滤 (质押/ST/流动性)
  3. 偏好匹配
  4. 输出投资建议
"""

import json
import logging

from agents.llm_utils import call_llm_json
from agents.data_provider import DataProvider
from agents.data_freshness import result_freshness_reason, result_is_fresh
from config.settings import MACRO_AGENT_DISABLED_REASON, MACRO_AGENT_ENABLED, get_effective_weights
from orchestrator.risk_filter import full_risk_and_preference_filter
from models import (
    AgentState,
    FundamentalResult,
    TechnicalResult,
    SentimentResult,
    MacroResult,
    UserProfile,
    Recommendation,
    AdviceLevel,
    Signal,
    SignalType,
    OperationStrategy,
)

logger = logging.getLogger(__name__)


DECISION_SYSTEM_PROMPT = """你是一位专业的投资顾问。你的任务是根据可用分析维度的结果，结合投资者的偏好，生成综合投资建议。

## 综合评分规则
各维度已按投资者偏好进行了加权计算，总得分已给出。

## 投资建议等级
- 85-100: 强烈推荐 — 多维度共振，风险可控
- 70-84: 积极关注 — 有明确逻辑支撑
- 55-69: 适度配置 — 有亮点但有隐忧
- 40-54: 谨慎观望 — 信号矛盾或风险偏高
- 0-39: 回避 — 风险显著或趋势恶化

## 输出要求
严格按以下JSON格式输出:

```json
{
  "name": "股票名称(根据数据推断或留空)",
  "total_score": <综合评分>,
  "advice": "强烈推荐/积极关注/适度配置/谨慎观望/回避",
  "position_suggestion": "仓位建议，如 5-10%",
  "core_logic": "1-3句核心逻辑，必须引用具体数据和维度",
  "risk_warnings": ["风险1", "风险2", ...],
  "operation": {
    "entry_condition": "入场条件描述",
    "stop_loss": "止损位描述",
    "target": "目标位描述",
    "holding_period": "建议持有周期"
  },
  "key_signals": [
    {"type": "利好/风险", "source": "来源", "description": "具体信号+数值", "significance": <0-10>}
  ]
}
```

## 重要规则
1. 综合评分必须基于提供的加权得分，不可随意调整
2. 核心逻辑必须引用至少2个维度的具体数据
3. 风险提示要包含已启用维度暴露的风险
4. 操作策略要与投资周期匹配
5. 必须包含免责声明
"""

DECISION_USER_PROMPT = """请根据以下分析结果，为股票 {symbol} 生成综合投资建议。

## 投资者偏好
- 风险承受: {risk_tolerance}
- 投资周期: {investment_horizon}
- 投资风格: {style}
- 资金规模: {capital_size}
- 偏好行业: {preferred_sectors}
- 回避行业: {avoid_sectors}

## 加权综合得分: {weighted_score}
## 参与评分维度: {active_dimensions}

### 基本面分析 (状态: {fundamental_status}, 得分: {fundamental_score})
{fundamental_detail}

### 量价技术分析 (状态: {technical_status}, 得分: {technical_score})
{technical_detail}

### 舆情景气分析 (状态: {sentiment_status}, 得分: {sentiment_score})
{sentiment_detail}

### 宏观景气分析 (状态: {macro_status}, 得分: {macro_score})
{macro_detail}

请严格按照系统提示中的JSON格式输出投资建议。"""


def compute_weighted_score(
    fundamental_score: float,
    technical_score: float,
    sentiment_score: float,
    macro_score: float,
    profile: UserProfile,
    include_macro: bool = True,
    active_dimensions: list[str] | tuple[str, ...] | set[str] | None = None,
) -> float:
    """
    按投资者偏好计算加权综合得分

    :param include_macro: False 时宏观权重置 0，其余维度重新归一
    :param active_dimensions: 指定实际参与评分的内部维度 key
    :return: 加权综合得分 0-100
    """
    weights = get_effective_weights(
        style=profile.style.value,
        horizon=profile.investment_horizon.value,
        include_macro=include_macro,
        active_dimensions=active_dimensions,
    )

    # 计算
    score_map = {
        "fundamental": fundamental_score,
        "technical": technical_score,
        "sentiment": sentiment_score,
        "macro": macro_score,
    }
    score = sum(score_map[key] * weights.get(key, 0) for key in score_map)

    return round(score, 1)


def _format_dimension_detail(result) -> str:
    """将子Agent结果格式化为决策prompt中的文本"""
    if result is None:
        return "未分析"

    lines = []
    if hasattr(result, "financial_health"):
        lines.append(f"  财务健康度: {result.financial_health.score} - {result.financial_health.summary}")
        for s in result.financial_health.signals[:3]:
            lines.append(f"    [{s.type.value}] {s.description}")
    if hasattr(result, "valuation"):
        lines.append(f"  估值合理性: {result.valuation.score} - {result.valuation.summary}")
        for s in result.valuation.signals[:3]:
            lines.append(f"    [{s.type.value}] {s.description}")
    if hasattr(result, "growth"):
        lines.append(f"  盈利成长性: {result.growth.score} - {result.growth.summary}")
        for s in result.growth.signals[:3]:
            lines.append(f"    [{s.type.value}] {s.description}")
    if hasattr(result, "trend"):
        lines.append(f"  趋势判断: {result.trend.score} - {result.trend.summary}")
    if hasattr(result, "momentum"):
        lines.append(f"  动量信号: {result.momentum.score} - {result.momentum.summary}")
    if hasattr(result, "capital_flow"):
        lines.append(f"  资金流: {result.capital_flow.score} - {result.capital_flow.summary}")
    if hasattr(result, "market_emotion"):
        lines.append(f"  市场情绪: {result.market_emotion.score} - {result.market_emotion.summary}")
    if hasattr(result, "social_heat"):
        lines.append(f"  社交热度: {result.social_heat.score} - {result.social_heat.summary}")
    if hasattr(result, "overall_market_emotion"):
        lines.append(f"  市场整体情绪: {result.overall_market_emotion.score} - {result.overall_market_emotion.summary}")
    if hasattr(result, "economic_cycle"):
        lines.append(f"  经济周期: {result.economic_cycle.score} - {result.economic_cycle.summary}")
    if hasattr(result, "monetary_policy"):
        lines.append(f"  货币环境: {result.monetary_policy.score} - {result.monetary_policy.summary}")
    if hasattr(result, "industry_cycle"):
        lines.append(f"  产业景气: {result.industry_cycle.score} - {result.industry_cycle.summary}")

    if hasattr(result, "risk_warnings") and result.risk_warnings:
        lines.append(f"  风险提示: {'; '.join(result.risk_warnings)}")
    if hasattr(result, "anomalies") and result.anomalies:
        for a in result.anomalies[:3]:
            lines.append(f"  异动: {a.get('description', '')} [{a.get('significance', '')}]")

    return "\n".join(lines) if lines else "分析数据不足"


def _dimension_status(result, label: str) -> tuple[bool, str, str]:
    """返回维度是否参与评分、状态文案和不可用原因。"""
    if result_is_fresh(result):
        return True, "启用", ""
    reason = result_freshness_reason(result, label)
    if "时效" in reason or "滞后" in reason:
        return False, "数据时效未通过", reason
    return False, "未返回", reason


def _dimension_detail_or_reason(result, enabled: bool, reason: str) -> str:
    if enabled:
        return _format_dimension_detail(result)
    return reason or "该维度未参与评分。"


def generate_recommendation(
    symbol: str,
    profile: UserProfile,
    fundamental_result: FundamentalResult = None,
    technical_result: TechnicalResult = None,
    sentiment_result: SentimentResult = None,
    macro_result: MacroResult = None,
    dp: DataProvider = None,
) -> Recommendation:
    """
    决策综合 Agent 主入口

    :param symbol: 股票代码
    :param profile: 投资者偏好
    :param fundamental_result: 基本面分析结果
    :param technical_result: 量价分析结果
    :param sentiment_result: 舆情分析结果
    :param macro_result: 宏观分析结果
    :param dp: 数据提供者
    :return: Recommendation 投资建议
    """
    logger.info(f"[决策Agent] 开始综合决策 {symbol}")

    # 1. 计算加权综合得分
    f_score = fundamental_result.total_score if fundamental_result else 50
    t_score = technical_result.total_score if technical_result else 50
    s_score = sentiment_result.total_score if sentiment_result else 50
    m_score = macro_result.total_score if macro_result else 0
    macro_unavailable_reason = (
        MACRO_AGENT_DISABLED_REASON
        if not MACRO_AGENT_ENABLED
        else "宏观 Agent 未返回结果，暂不参与个股评分。"
    )

    dimension_meta = [
        ("fundamental", "基本面", fundamental_result, f_score),
        ("technical", "量价", technical_result, t_score),
        ("sentiment", "舆情", sentiment_result, s_score),
        ("macro", "宏观", macro_result, m_score),
    ]
    dimension_states = {}
    active_dimension_keys = []
    active_dimension_scores = {}
    inactive_dimension_reasons = []

    for key, label, result, score in dimension_meta:
        enabled, status, reason = _dimension_status(result, label)
        if key == "macro" and result is None:
            enabled = False
            status = "已暂停" if not MACRO_AGENT_ENABLED else "未返回"
            reason = macro_unavailable_reason
        dimension_states[key] = {
            "label": label,
            "enabled": enabled,
            "status": status,
            "reason": reason,
            "score": score,
        }
        if enabled:
            active_dimension_keys.append(key)
            active_dimension_scores[label] = score
        else:
            inactive_dimension_reasons.append(reason)

    if active_dimension_keys:
        weighted_score = compute_weighted_score(
            f_score,
            t_score,
            s_score,
            m_score,
            profile,
            include_macro="macro" in active_dimension_keys,
            active_dimensions=active_dimension_keys,
        )
    else:
        weighted_score = 50.0
        inactive_dimension_reasons.append("所有维度均未通过数据时效校验，综合评分仅为中性占位。")

    # 2. 风险过滤 + 偏好匹配
    risk_warnings = []
    if dp is not None:
        weighted_score, risk_warnings = full_risk_and_preference_filter(
            symbol, weighted_score, profile, dp,
        )

    # 3. 确定建议等级
    if weighted_score >= 85:
        advice = AdviceLevel.STRONG_BUY
    elif weighted_score >= 70:
        advice = AdviceLevel.BUY
    elif weighted_score >= 55:
        advice = AdviceLevel.HOLD
    elif weighted_score >= 40:
        advice = AdviceLevel.WATCH
    else:
        advice = AdviceLevel.AVOID

    active_dimensions = [item["label"] for item in dimension_states.values() if item["enabled"]]
    active_dimensions_text = "、".join(active_dimensions) if active_dimensions else "无"

    # 3. 构建 prompt 调用 LLM 生成详细建议
    user_prompt = DECISION_USER_PROMPT.format(
        symbol=symbol,
        risk_tolerance=profile.risk_tolerance.value,
        investment_horizon=profile.investment_horizon.value,
        style=profile.style.value,
        capital_size=profile.capital_size.value,
        preferred_sectors=",".join(profile.sectors_preference) or "无",
        avoid_sectors=",".join(profile.avoid_sectors) or "无",
        weighted_score=weighted_score,
        active_dimensions=active_dimensions_text,
        fundamental_status=dimension_states["fundamental"]["status"],
        fundamental_score=f_score if dimension_states["fundamental"]["enabled"] else "不参与评分",
        fundamental_detail=_dimension_detail_or_reason(
            fundamental_result,
            dimension_states["fundamental"]["enabled"],
            dimension_states["fundamental"]["reason"],
        ),
        technical_status=dimension_states["technical"]["status"],
        technical_score=t_score if dimension_states["technical"]["enabled"] else "不参与评分",
        technical_detail=_dimension_detail_or_reason(
            technical_result,
            dimension_states["technical"]["enabled"],
            dimension_states["technical"]["reason"],
        ),
        sentiment_status=dimension_states["sentiment"]["status"],
        sentiment_score=s_score if dimension_states["sentiment"]["enabled"] else "不参与评分",
        sentiment_detail=_dimension_detail_or_reason(
            sentiment_result,
            dimension_states["sentiment"]["enabled"],
            dimension_states["sentiment"]["reason"],
        ),
        macro_status=dimension_states["macro"]["status"],
        macro_score=m_score if dimension_states["macro"]["enabled"] else "不参与评分",
        macro_detail=_dimension_detail_or_reason(
            macro_result,
            dimension_states["macro"]["enabled"],
            dimension_states["macro"]["reason"],
        ),
    )

    try:
        llm_output = call_llm_json(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
    except Exception as e:
        logger.error(f"[决策Agent] LLM调用失败: {e}, 使用规则回退")
        llm_output = _fallback_recommendation(
            symbol,
            weighted_score,
            advice,
            active_dimension_scores,
            [state["label"] for state in dimension_states.values() if not state["enabled"]],
        )

    # 4. 构建 Recommendation
    try:
        operation_data = llm_output.get("operation", {})
        key_signals = []
        for s in llm_output.get("key_signals", []):
            try:
                key_signals.append(Signal(**s))
            except Exception:
                pass

        recommendation = Recommendation(
            symbol=symbol,
            name=llm_output.get("name", ""),
            total_score=weighted_score,
            advice=advice,
            position_suggestion=llm_output.get("position_suggestion", _suggest_position(weighted_score, profile)),
            core_logic=llm_output.get("core_logic", f"综合评分{weighted_score}分"),
            risk_warnings=llm_output.get("risk_warnings", []) + risk_warnings + inactive_dimension_reasons,
            operation=OperationStrategy(
                entry_condition=operation_data.get("entry_condition", ""),
                stop_loss=operation_data.get("stop_loss", ""),
                target=operation_data.get("target", ""),
                holding_period=operation_data.get("holding_period", profile.investment_horizon.value),
            ),
            dimension_scores=active_dimension_scores,
            key_signals=key_signals,
        )
    except Exception as e:
        logger.error(f"[决策Agent] 构建建议失败: {e}")
        recommendation = Recommendation(
            symbol=symbol,
            total_score=weighted_score,
            advice=advice,
            core_logic=f"综合评分{weighted_score}分，{advice.value}",
            dimension_scores=active_dimension_scores,
            risk_warnings=inactive_dimension_reasons,
        )

    logger.info(f"[决策Agent] 决策完成: {symbol} 综合评分={weighted_score} 建议={advice.value}")
    return recommendation


def _suggest_position(score: float, profile: UserProfile) -> str:
    """根据评分和偏好建议仓位"""
    if score >= 85:
        max_pos = profile.max_position_per_stock
        return f"{max_pos * 0.6:.0f}-{max_pos:.0f}%"
    elif score >= 70:
        return f"{profile.max_position_per_stock * 0.3:.0f}-{profile.max_position_per_stock * 0.6:.0f}%"
    elif score >= 55:
        return f"{profile.max_position_per_stock * 0.1:.0f}-{profile.max_position_per_stock * 0.3:.0f}%"
    else:
        return "0% 或 观望"


def _fallback_recommendation(
    symbol,
    score,
    advice,
    dimension_scores: dict[str, float],
    inactive_dimensions: list[str],
) -> dict:
    """LLM 不可用时的规则回退"""
    active_text = "+".join(f"{label}{value}" for label, value in dimension_scores.items()) or "无有效维度"
    inactive_text = "+".join(f"{label}未参与" for label in inactive_dimensions)
    dimension_text = active_text if not inactive_text else f"{active_text}+{inactive_text}"

    return {
        "name": "",
        "total_score": score,
        "advice": advice.value,
        "position_suggestion": _suggest_position(score, UserProfile()),
        "core_logic": f"综合评分{score}分({dimension_text})",
        "risk_warnings": ["LLM决策不可用，建议仅供参考"],
        "operation": {
            "entry_condition": "建议进一步人工确认",
            "stop_loss": "建议设置-8%止损",
            "target": "需进一步分析",
            "holding_period": "中线",
        },
        "key_signals": [],
    }
