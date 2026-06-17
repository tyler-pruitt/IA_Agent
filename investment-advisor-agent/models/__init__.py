"""
数据模型 — Agent 分析结果、投资建议、用户偏好

所有 Agent 共享统一的数据模型，保证结构化输出可被下游决策 Agent 解析
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# 用户偏好模型
# ──────────────────────────────────────────────

class RiskTolerance(str, Enum):
    CONSERVATIVE = "保守"
    MODERATE = "稳健"
    AGGRESSIVE = "积极"
    RADICAL = "激进"


class InvestmentHorizon(str, Enum):
    SHORT = "短线"    # < 1月
    MEDIUM = "中线"   # 1-6月
    LONG = "长线"     # > 6月


class InvestmentStyle(str, Enum):
    VALUE = "价值"
    GROWTH = "成长"
    BALANCED = "均衡"
    THEME = "主题"


class CapitalSize(str, Enum):
    SMALL = "小"     # < 10万
    MEDIUM = "中"    # 10-50万
    LARGE = "大"     # > 50万


class UserProfile(BaseModel):
    """投资者偏好画像"""
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.MODERATE, description="风险承受能力")
    investment_horizon: InvestmentHorizon = Field(default=InvestmentHorizon.MEDIUM, description="投资周期")
    style: InvestmentStyle = Field(default=InvestmentStyle.BALANCED, description="投资风格")
    capital_size: CapitalSize = Field(default=CapitalSize.MEDIUM, description="资金规模")
    sectors_preference: list[str] = Field(default_factory=list, description="偏好的行业/概念")
    avoid_sectors: list[str] = Field(default_factory=list, description="回避的行业/概念")
    max_position_per_stock: float = Field(default=15.0, description="单股最大仓位百分比")
    max_portfolio_stocks: int = Field(default=10, description="组合最大持仓数")
    stop_loss_threshold: float = Field(default=-8.0, description="止损线百分比")


# ──────────────────────────────────────────────
# 信号与评分模型
# ──────────────────────────────────────────────

class SignalType(str, Enum):
    POSITIVE = "利好"
    NEGATIVE = "风险"
    NEUTRAL = "中性"


class Signal(BaseModel):
    """单个分析信号"""
    type: SignalType = Field(description="信号类型: 利好/风险/中性")
    source: str = Field(description="数据来源，如 估值/业绩/资金流")
    description: str = Field(description="信号描述，必须引用具体数值")
    significance: float = Field(default=5.0, ge=0, le=10, description="重要程度 0-10")


class DimensionScore(BaseModel):
    """单维度评分"""
    dimension: str = Field(description="维度名称，如 财务健康度/趋势判断")
    score: float = Field(ge=0, le=100, description="0-100 分")
    signals: list[Signal] = Field(default_factory=list, description="支撑评分的信号列表")
    summary: str = Field(default="", description="一两句话的总结")


# ──────────────────────────────────────────────
# 各子 Agent 输出模型
# ──────────────────────────────────────────────

class FundamentalResult(BaseModel):
    """基本面分析结果"""
    symbol: str = Field(description="股票代码")
    financial_health: DimensionScore = Field(description="财务健康度")
    valuation: DimensionScore = Field(description="估值合理性")
    growth: DimensionScore = Field(description="盈利成长性")
    risk_warnings: list[str] = Field(default_factory=list, description="风险提示")
    total_score: float = Field(ge=0, le=100, description="综合基本面得分")
    raw_data_summary: dict = Field(default_factory=dict, description="原始数据摘要(供决策Agent参考)")


class TechnicalResult(BaseModel):
    """量价技术分析结果"""
    symbol: str = Field(description="股票代码")
    trend: DimensionScore = Field(description="趋势判断")
    momentum: DimensionScore = Field(description="动量信号")
    capital_flow: DimensionScore = Field(description="资金流")
    anomalies: list[dict] = Field(default_factory=list, description="异动事件列表")
    total_score: float = Field(ge=0, le=100, description="量价综合得分")
    raw_data_summary: dict = Field(default_factory=dict, description="原始数据摘要")


class SentimentResult(BaseModel):
    """舆情景气分析结果"""
    symbol: str = Field(description="股票代码")
    market_emotion: DimensionScore = Field(description="市场情绪")
    news_events: list[dict] = Field(default_factory=list, description="新闻事件列表")
    social_heat: DimensionScore = Field(description="社交热度")
    overall_market_emotion: DimensionScore = Field(description="市场整体情绪")
    total_score: float = Field(ge=0, le=100, description="舆情综合得分")
    raw_data_summary: dict = Field(default_factory=dict, description="原始数据摘要")


class MacroResult(BaseModel):
    """宏观景气分析结果"""
    economic_cycle: DimensionScore = Field(description="经济周期")
    monetary_policy: DimensionScore = Field(description="货币环境")
    industry_cycle: DimensionScore = Field(description="产业景气")
    total_score: float = Field(ge=0, le=100, description="宏观综合得分")
    raw_data_summary: dict = Field(default_factory=dict, description="原始数据摘要")


# ──────────────────────────────────────────────
# 决策层输出模型
# ──────────────────────────────────────────────

class AdviceLevel(str, Enum):
    STRONG_BUY = "强烈推荐"     # 85-100
    BUY = "积极关注"            # 70-84
    HOLD = "适度配置"           # 55-69
    WATCH = "谨慎观望"          # 40-54
    AVOID = "回避"              # 0-39


class OperationStrategy(BaseModel):
    """操作策略"""
    entry_condition: str = Field(default="", description="入场条件")
    stop_loss: str = Field(default="", description="止损位")
    target: str = Field(default="", description="目标位")
    holding_period: str = Field(default="", description="建议持有周期")


class Recommendation(BaseModel):
    """投资建议"""
    symbol: str = Field(description="股票代码")
    name: str = Field(default="", description="股票名称")
    total_score: float = Field(ge=0, le=100, description="综合评分")
    advice: AdviceLevel = Field(description="投资建议等级")
    position_suggestion: str = Field(default="", description="仓位建议")
    core_logic: str = Field(description="核心逻辑，1-3句话")
    risk_warnings: list[str] = Field(default_factory=list, description="风险提示")
    operation: OperationStrategy = Field(default_factory=OperationStrategy, description="操作策略")
    dimension_scores: dict[str, float] = Field(default_factory=dict, description="各维度得分")
    key_signals: list[Signal] = Field(default_factory=list, description="关键信号")
    disclaimer: str = Field(
        default="⚠️ 以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。",
        description="免责声明",
    )


# ──────────────────────────────────────────────
# Agent 运行状态
# ──────────────────────────────────────────────

class AgentState(BaseModel):
    """Agent 编排状态，在 LangGraph 中流转"""
    symbol: str = Field(default="", description="当前分析的股票代码")
    user_query: str = Field(default="", description="用户原始问题")
    profile: UserProfile = Field(default_factory=UserProfile, description="用户偏好")
    # 子 Agent 结果
    fundamental_result: Optional[FundamentalResult] = Field(default=None)
    technical_result: Optional[TechnicalResult] = Field(default=None)
    sentiment_result: Optional[SentimentResult] = Field(default=None)
    macro_result: Optional[MacroResult] = Field(default=None)
    # 最终建议
    recommendation: Optional[Recommendation] = Field(default=None)
    # 错误记录
    errors: list[str] = Field(default_factory=list)
