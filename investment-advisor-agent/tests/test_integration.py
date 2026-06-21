"""
Phase 3+4 集成测试 — 风险过滤、偏好匹配、UI 组件、端到端流程
"""

import pytest

from models import (
    UserProfile,
    RiskTolerance,
    InvestmentHorizon,
    InvestmentStyle,
    CapitalSize,
    Recommendation,
    AdviceLevel,
    DimensionScore,
    FundamentalResult,
    TechnicalResult,
    SentimentResult,
    MacroResult,
    OperationStrategy,
    Signal,
    SignalType,
)
from orchestrator.risk_filter import (
    RiskFlag,
    check_risk_flags,
    apply_risk_filter,
    apply_preference_match,
    full_risk_and_preference_filter,
)
from orchestrator.decision_agent import (
    compute_weighted_score,
    generate_recommendation,
    _suggest_position,
    _format_dimension_detail,
)
from orchestrator.planner import _fallback_parse
from orchestrator.scheduler import analyze_stock


class StubRiskDataProvider:
    """离线风险测试用的最小数据提供者。"""

    def __init__(self, fundamental_data=None, price_data=None):
        self._fundamental_data = fundamental_data or {"valuation": []}
        self._price_data = price_data or {"data": []}

    def get_fundamental(self, symbol: str, report_type: str = "valuation") -> dict:
        return self._fundamental_data

    def get_price_volume(self, symbol: str, period: str = "daily") -> dict:
        return self._price_data


# ──────────────────────────────────────────────
# 风险过滤测试
# ──────────────────────────────────────────────

class TestRiskFlag:
    def test_block_penalty(self):
        flag = RiskFlag(level="block", message="ST股", score_penalty=1.0)
        assert flag.level == "block"
        assert flag.score_penalty == 1.0

    def test_warning_penalty(self):
        flag = RiskFlag(level="warning", message="质押率偏高", score_penalty=0.2)
        assert flag.level == "warning"
        assert flag.score_penalty == 0.2


class TestApplyRiskFilter:
    def test_no_flags(self):
        score, warnings = apply_risk_filter(80.0, [])
        assert score == 80.0
        assert warnings == []

    def test_block_sets_zero(self):
        flags = [RiskFlag(level="block", message="ST股", score_penalty=1.0)]
        score, warnings = apply_risk_filter(80.0, flags)
        assert score == 0
        assert any("⛔" in w for w in warnings)

    def test_warning_reduces_score(self):
        flags = [RiskFlag(level="warning", message="质押率50%", score_penalty=0.2)]
        score, warnings = apply_risk_filter(80.0, flags)
        assert score == 64.0  # 80 * 0.8
        assert any("⚠️" in w for w in warnings)

    def test_multiple_warnings_accumulate(self):
        flags = [
            RiskFlag(level="warning", message="风险1", score_penalty=0.2),
            RiskFlag(level="warning", message="风险2", score_penalty=0.1),
        ]
        score, warnings = apply_risk_filter(100.0, flags)
        # 100 * 0.8 * 0.9 = 72.0
        assert score == 72.0
        assert len(warnings) == 2

    def test_info_no_penalty(self):
        flags = [RiskFlag(level="info", message="流动性偏低", score_penalty=0.0)]
        score, warnings = apply_risk_filter(80.0, flags)
        assert score == 80.0
        assert any("ℹ️" in w for w in warnings)

    def test_score_clamped_to_zero(self):
        flags = [RiskFlag(level="warning", message="巨大风险", score_penalty=0.9)]
        score, warnings = apply_risk_filter(10.0, flags)
        assert score == 1.0  # 10 * 0.1 = 1.0

    def test_score_clamped_to_100(self):
        # 边界: 确保不超过100
        score, _ = apply_risk_filter(100.0, [])
        assert score == 100.0


class TestCheckRiskFlags:
    def test_returns_list(self):
        flags = check_risk_flags("999999", dp=StubRiskDataProvider())
        assert isinstance(flags, list)

    def test_pe_negative_detection(self):
        """PE为负的股票应该被标记为亏损风险"""
        # 模拟估值数据
        fundamental_data = {
            "valuation": [
                {"PE(TTM)": -5.3, "市净率": 1.2},
            ]
        }
        flags = check_risk_flags(
            "000001",
            fundamental_data=fundamental_data,
            price_data={"data": []},
        )
        has_loss_warning = any("亏损" in f.message for f in flags)
        assert has_loss_warning

    def test_pe_extremely_high_detection(self):
        """PE极高应该被标记为估值风险"""
        fundamental_data = {
            "valuation": [
                {"PE(TTM)": 500, "市净率": 15},
            ]
        }
        flags = check_risk_flags(
            "000001",
            fundamental_data=fundamental_data,
            price_data={"data": []},
        )
        has_high_pe = any("估值极高" in f.message for f in flags)
        assert has_high_pe

    def test_low_liquidity_detection(self):
        """低成交额应该被标记为流动性风险"""
        price_data = {
            "data": [
                {"成交额": 5000000},  # 500万
            ]
        }
        flags = check_risk_flags(
            "000001",
            fundamental_data={"valuation": []},
            price_data=price_data,
        )
        has_liquidity = any("流动性不足" in f.message for f in flags)
        assert has_liquidity


# ──────────────────────────────────────────────
# 偏好匹配测试
# ──────────────────────────────────────────────

class TestPreferenceMatch:
    def test_basic_match(self):
        profile = UserProfile()
        score, info = apply_preference_match("000001", 75.0, profile)
        assert 0 <= score <= 100
        assert isinstance(info, list)

    def test_avoid_st(self):
        profile = UserProfile(avoid_sectors=["ST"])
        score, info = apply_preference_match("000001", 75.0, profile)
        assert isinstance(score, float)


class TestFullRiskAndPreferenceFilter:
    def test_combined_filter(self):
        profile = UserProfile()
        score, info = full_risk_and_preference_filter(
            "000001",
            75.0,
            profile,
            dp=StubRiskDataProvider(),
        )
        assert 0 <= score <= 100
        assert isinstance(info, list)

    def test_with_risk_flags(self):
        """有风险标记时应降低分数"""
        profile = UserProfile()
        dp = StubRiskDataProvider(
            fundamental_data={"valuation": [{"PE(TTM)": -10}]},
            price_data={"data": [{"成交额": 5000000}]},
        )
        score, info = full_risk_and_preference_filter(
            "000001",
            80.0,
            profile,
            dp=dp,
        )
        assert score < 80.0
        assert info


# ──────────────────────────────────────────────
# 决策 Agent 集成测试
# ──────────────────────────────────────────────

class TestDecisionAgentIntegration:
    def _make_fundamental(self, total=75):
        return FundamentalResult(
            symbol="000001",
            financial_health=DimensionScore(dimension="财务健康度", score=80, summary="良好"),
            valuation=DimensionScore(dimension="估值合理性", score=70, summary="合理"),
            growth=DimensionScore(dimension="盈利成长性", score=75, summary="良好"),
            total_score=total,
        )

    def _make_technical(self, total=65):
        return TechnicalResult(
            symbol="000001",
            trend=DimensionScore(dimension="趋势判断", score=70, summary="上行"),
            momentum=DimensionScore(dimension="动量信号", score=60, summary="中性偏强"),
            capital_flow=DimensionScore(dimension="资金流", score=65, summary="流入"),
            total_score=total,
        )

    def _make_sentiment(self, total=70):
        return SentimentResult(
            symbol="000001",
            market_emotion=DimensionScore(dimension="市场情绪", score=75, summary="偏暖"),
            social_heat=DimensionScore(dimension="社交热度", score=65, summary="适中"),
            overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=70, summary="中性"),
            total_score=total,
        )

    def _make_macro(self, total=72):
        return MacroResult(
            economic_cycle=DimensionScore(dimension="经济周期", score=75, summary="复苏"),
            monetary_policy=DimensionScore(dimension="货币环境", score=70, summary="宽松"),
            industry_cycle=DimensionScore(dimension="产业景气", score=70, summary="分化"),
            total_score=total,
        )

    def test_generate_recommendation_without_llm(self):
        """测试决策Agent在无LLM时的规则回退"""
        profile = UserProfile(style=InvestmentStyle.BALANCED)
        # 不传dp，跳过风险过滤（需要网络）
        rec = generate_recommendation(
            symbol="000001",
            profile=profile,
            fundamental_result=self._make_fundamental(),
            technical_result=self._make_technical(),
            sentiment_result=self._make_sentiment(),
            macro_result=self._make_macro(),
        )
        assert isinstance(rec, Recommendation)
        assert rec.symbol == "000001"
        assert 0 <= rec.total_score <= 100
        assert len(rec.dimension_scores) == 4

    def test_recommendation_advice_matches_score(self):
        """建议等级应与评分一致"""
        # 高分
        profile = UserProfile()
        f = self._make_fundamental(90)
        t = self._make_technical(90)
        s = self._make_sentiment(90)
        m = self._make_macro(90)
        rec = generate_recommendation("000001", profile, f, t, s, m)
        assert rec.total_score >= 85
        assert rec.advice in [AdviceLevel.STRONG_BUY, AdviceLevel.BUY]

    def test_format_dimension_detail(self):
        """测试维度详情格式化"""
        f = self._make_fundamental()
        detail = _format_dimension_detail(f)
        assert "财务健康度" in detail
        assert "估值合理性" in detail
        assert "盈利成长性" in detail

    def test_format_dimension_detail_none(self):
        """None 应返回未分析"""
        detail = _format_dimension_detail(None)
        assert "未分析" in detail

    def test_weighted_score_with_different_profiles(self):
        """不同偏好应产生不同加权分"""
        profile_value = UserProfile(style=InvestmentStyle.VALUE)
        profile_theme = UserProfile(style=InvestmentStyle.THEME)

        # 基本面高, 舆情低
        score_value = compute_weighted_score(90, 50, 30, 60, profile_value)
        score_theme = compute_weighted_score(90, 50, 30, 60, profile_theme)

        # 价值型(重基本面) > 主题型(重舆情)
        assert score_value > score_theme

    def test_stale_agent_result_is_excluded_from_recommendation(self):
        """数据时效未通过的 Agent 不应参与最终评分。"""
        profile = UserProfile(style=InvestmentStyle.BALANCED)
        f = self._make_fundamental(90)
        f.raw_data_summary = {
            "data_freshness": {
                "usable": False,
                "reason": "基本面核心数据已滞后，Agent 暂不参与评分。",
            }
        }
        t = self._make_technical(60)
        s = self._make_sentiment(30)

        rec = generate_recommendation(
            symbol="000001",
            profile=profile,
            fundamental_result=f,
            technical_result=t,
            sentiment_result=s,
        )

        assert "基本面" not in rec.dimension_scores
        assert rec.dimension_scores == {"量价": 60, "舆情": 30}
        assert rec.total_score == 45.0


# ──────────────────────────────────────────────
# UI 组件测试
# ──────────────────────────────────────────────

class TestUIComponents:
    def test_score_to_color(self):
        """测试评分颜色映射"""
        from ui.app import _score_to_color
        assert _score_to_color(90) == "#4CAF50"  # 绿色
        assert _score_to_color(70) == "#FF9800"  # 橙色
        assert _score_to_color(50) == "#FFC107"  # 黄色
        assert _score_to_color(30) == "#F44336"  # 红色

    def test_advice_to_emoji(self):
        """测试建议等级 emoji 映射"""
        from ui.app import _advice_to_emoji
        assert _advice_to_emoji(AdviceLevel.STRONG_BUY) == "🔥"
        assert _advice_to_emoji(AdviceLevel.BUY) == "👍"
        assert _advice_to_emoji(AdviceLevel.HOLD) == "🤝"
        assert _advice_to_emoji(AdviceLevel.WATCH) == "👀"
        assert _advice_to_emoji(AdviceLevel.AVOID) == "🚫"


# ──────────────────────────────────────────────
# 完整端到端模拟测试
# ──────────────────────────────────────────────

class TestEndToEndSimulation:
    """模拟完整流程: 从子Agent结果到最终建议 (跳过数据获取和LLM调用)"""

    def test_full_pipeline(self):
        """模拟完整分析流程"""
        # 1. 模拟子Agent结果
        f = FundamentalResult(
            symbol="600519",
            financial_health=DimensionScore(
                dimension="财务健康度", score=92, summary="卓越",
                signals=[
                    Signal(type=SignalType.POSITIVE, source="ROE", description="ROE连续3年>30%", significance=9),
                ],
            ),
            valuation=DimensionScore(
                dimension="估值合理性", score=45, summary="偏高",
                signals=[
                    Signal(type=SignalType.NEGATIVE, source="PE", description="PE处于近5年85%分位", significance=7),
                ],
            ),
            growth=DimensionScore(
                dimension="盈利成长性", score=78, summary="稳健增长",
            ),
            risk_warnings=["估值处于历史高位"],
            total_score=74,
        )

        t = TechnicalResult(
            symbol="600519",
            trend=DimensionScore(dimension="趋势判断", score=72, summary="震荡上行"),
            momentum=DimensionScore(dimension="动量信号", score=60, summary="中性"),
            capital_flow=DimensionScore(dimension="资金流", score=80, summary="北向持续流入"),
            total_score=71,
        )

        s = SentimentResult(
            symbol="600519",
            market_emotion=DimensionScore(dimension="市场情绪", score=82, summary="千股千评85分"),
            social_heat=DimensionScore(dimension="社交热度", score=75, summary="热度高"),
            overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=65, summary="中性"),
            total_score=74,
        )

        m = MacroResult(
            economic_cycle=DimensionScore(dimension="经济周期", score=70, summary="温和复苏"),
            monetary_policy=DimensionScore(dimension="货币环境", score=80, summary="宽松"),
            industry_cycle=DimensionScore(dimension="产业景气", score=65, summary="消费分化"),
            total_score=72,
        )

        # 2. 投资者偏好
        profile = UserProfile(
            style=InvestmentStyle.VALUE,
            investment_horizon=InvestmentHorizon.LONG,
            risk_tolerance=RiskTolerance.MODERATE,
            sectors_preference=["白酒"],
        )

        # 3. 综合决策
        rec = generate_recommendation(
            symbol="600519",
            profile=profile,
            fundamental_result=f,
            technical_result=t,
            sentiment_result=s,
            macro_result=m,
        )

        # 4. 验证
        assert rec.symbol == "600519"
        assert 0 <= rec.total_score <= 100
        assert rec.advice in list(AdviceLevel)
        assert len(rec.dimension_scores) == 4
        assert rec.dimension_scores["基本面"] == 74
        assert rec.dimension_scores["量价"] == 71
        assert rec.dimension_scores["舆情"] == 74
        assert rec.dimension_scores["宏观"] == 72
        assert rec.disclaimer  # 有免责声明

        # 价值型+长线，基本面权重应最大，得分应偏高
        assert rec.total_score > 60

    def test_low_score_triggers_avoid(self):
        """低评分应触发回避建议"""
        f = FundamentalResult(
            symbol="000001",
            financial_health=DimensionScore(dimension="财务健康度", score=20, summary="恶化"),
            valuation=DimensionScore(dimension="估值合理性", score=25, summary="泡沫"),
            growth=DimensionScore(dimension="盈利成长性", score=15, summary="下滑"),
            risk_warnings=["连续亏损", "负债率高"],
            total_score=20,
        )
        t = TechnicalResult(
            symbol="000001",
            trend=DimensionScore(dimension="趋势判断", score=20, summary="空头"),
            momentum=DimensionScore(dimension="动量信号", score=25, summary="超卖"),
            capital_flow=DimensionScore(dimension="资金流", score=15, summary="大幅流出"),
            total_score=20,
        )
        s = SentimentResult(
            symbol="000001",
            market_emotion=DimensionScore(dimension="市场情绪", score=30, summary="低迷"),
            social_heat=DimensionScore(dimension="社交热度", score=20, summary="无人关注"),
            overall_market_emotion=DimensionScore(dimension="市场整体情绪", score=35, summary="恐慌"),
            total_score=28,
        )
        m = MacroResult(
            economic_cycle=DimensionScore(dimension="经济周期", score=40, summary="衰退"),
            monetary_policy=DimensionScore(dimension="货币环境", score=35, summary="紧缩"),
            industry_cycle=DimensionScore(dimension="产业景气", score=25, summary="全行业低迷"),
            total_score=33,
        )

        profile = UserProfile()
        rec = generate_recommendation("000001", profile, f, t, s, m)

        assert rec.total_score < 40
        assert rec.advice == AdviceLevel.AVOID
