"""
Agent 层单元测试 — 数据模型、评分逻辑、意图解析
(不依赖 LLM API 的纯逻辑测试)
"""

import json
from datetime import date, timedelta

import pytest

from models import (
    UserProfile,
    RiskTolerance,
    InvestmentHorizon,
    InvestmentStyle,
    CapitalSize,
    DimensionScore,
    Signal,
    SignalType,
    FundamentalResult,
    TechnicalResult,
    SentimentResult,
    MacroResult,
    Recommendation,
    AdviceLevel,
    OperationStrategy,
    AgentState,
)
from orchestrator.decision_agent import compute_weighted_score, _suggest_position
from orchestrator.planner import _fallback_parse, IntentType


# ──────────────────────────────────────────────
# 数据模型测试
# ──────────────────────────────────────────────

class TestUserProfile:
    def test_default_values(self):
        profile = UserProfile()
        assert profile.risk_tolerance == RiskTolerance.MODERATE
        assert profile.investment_horizon == InvestmentHorizon.MEDIUM
        assert profile.style == InvestmentStyle.BALANCED
        assert profile.capital_size == CapitalSize.MEDIUM
        assert profile.sectors_preference == []
        assert profile.avoid_sectors == []

    def test_custom_values(self):
        profile = UserProfile(
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            investment_horizon=InvestmentHorizon.SHORT,
            style=InvestmentStyle.THEME,
            capital_size=CapitalSize.SMALL,
            sectors_preference=["新能源", "半导体"],
            avoid_sectors=["ST"],
        )
        assert profile.risk_tolerance == RiskTolerance.AGGRESSIVE
        assert profile.style == InvestmentStyle.THEME
        assert len(profile.sectors_preference) == 2


class TestDimensionScore:
    def test_valid_score(self):
        ds = DimensionScore(dimension="测试", score=75, summary="还行")
        assert ds.score == 75
        assert ds.dimension == "测试"

    def test_boundary_scores(self):
        ds_0 = DimensionScore(dimension="测试", score=0, summary="")
        ds_100 = DimensionScore(dimension="测试", score=100, summary="")
        assert ds_0.score == 0
        assert ds_100.score == 100

    def test_signals(self):
        signal = Signal(type=SignalType.POSITIVE, source="估值", description="PE=8.5, 低于行业均值", significance=8)
        ds = DimensionScore(dimension="估值", score=85, signals=[signal], summary="低估")
        assert len(ds.signals) == 1
        assert ds.signals[0].type == SignalType.POSITIVE


class TestFundamentalResult:
    def test_construction(self):
        result = FundamentalResult(
            symbol="000001",
            financial_health=DimensionScore(dimension="财务健康度", score=80, summary="优秀"),
            valuation=DimensionScore(dimension="估值合理性", score=60, summary="合理"),
            growth=DimensionScore(dimension="盈利成长性", score=70, summary="良好"),
            risk_warnings=["质押率偏高"],
            total_score=72,
        )
        assert result.symbol == "000001"
        assert result.financial_health.score == 80
        assert len(result.risk_warnings) == 1

    def test_serialization(self):
        result = FundamentalResult(
            symbol="600519",
            financial_health=DimensionScore(dimension="财务健康度", score=90, summary="卓越"),
            valuation=DimensionScore(dimension="估值合理性", score=50, summary="中性"),
            growth=DimensionScore(dimension="盈利成长性", score=80, summary="强劲"),
            total_score=78,
        )
        data = result.model_dump()
        assert data["symbol"] == "600519"
        assert isinstance(data, dict)


class TestRecommendation:
    def test_construction(self):
        rec = Recommendation(
            symbol="000001",
            name="平安银行",
            total_score=72,
            advice=AdviceLevel.BUY,
            core_logic="基本面优秀+北向流入+业绩超预期",
            risk_warnings=["估值偏高"],
        )
        assert rec.total_score == 72
        assert rec.advice == AdviceLevel.BUY
        assert rec.disclaimer  # 自动附带免责声明

    def test_serialization(self):
        rec = Recommendation(
            symbol="000001",
            total_score=55,
            advice=AdviceLevel.HOLD,
            core_logic="综合评分55",
            dimension_scores={"基本面": 70, "量价": 50, "舆情": 55, "宏观": 45},
        )
        data = rec.model_dump()
        assert "dimension_scores" in data
        assert data["dimension_scores"]["基本面"] == 70


class TestAgentState:
    def test_default_state(self):
        state = AgentState()
        assert state.symbol == ""
        assert state.fundamental_result is None
        assert state.recommendation is None

    def test_with_results(self):
        state = AgentState(
            symbol="000001",
            user_query="分析平安银行",
            fundamental_result=FundamentalResult(
                symbol="000001",
                financial_health=DimensionScore(dimension="财务健康度", score=75, summary=""),
                valuation=DimensionScore(dimension="估值合理性", score=60, summary=""),
                growth=DimensionScore(dimension="盈利成长性", score=65, summary=""),
                total_score=68,
            ),
        )
        assert state.fundamental_result.total_score == 68


# ──────────────────────────────────────────────
# 评分权重测试
# ──────────────────────────────────────────────

class TestWeightedScore:
    def test_balanced_style(self):
        profile = UserProfile(style=InvestmentStyle.BALANCED)
        score = compute_weighted_score(80, 60, 70, 65, profile)
        # 均衡: 各25%, 预期 ≈ 80*0.25+60*0.25+70*0.25+65*0.25 = 68.75
        assert 65 <= score <= 72

    def test_macro_disabled_renormalizes_weights(self):
        profile = UserProfile(style=InvestmentStyle.BALANCED)
        score = compute_weighted_score(90, 60, 30, 0, profile, include_macro=False)
        # 宏观暂停时，均衡型三项有效权重重新归一为各 1/3。
        assert score == 60.0

    def test_stale_dimension_is_excluded_from_weights(self):
        profile = UserProfile(style=InvestmentStyle.BALANCED)
        score = compute_weighted_score(
            90,
            60,
            30,
            0,
            profile,
            include_macro=False,
            active_dimensions=["technical", "sentiment"],
        )
        assert score == 45.0

    def test_value_style_emphasizes_fundamental(self):
        profile_value = UserProfile(style=InvestmentStyle.VALUE)
        profile_balanced = UserProfile(style=InvestmentStyle.BALANCED)
        score_value = compute_weighted_score(90, 40, 50, 60, profile_value)
        score_balanced = compute_weighted_score(90, 40, 50, 60, profile_balanced)
        # 价值型应更重视基本面, 基本面90分时价值型得分应更高
        assert score_value > score_balanced

    def test_theme_style_emphasizes_sentiment(self):
        profile_theme = UserProfile(style=InvestmentStyle.THEME)
        profile_balanced = UserProfile(style=InvestmentStyle.BALANCED)
        score_theme = compute_weighted_score(40, 50, 90, 60, profile_theme)
        score_balanced = compute_weighted_score(40, 50, 90, 60, profile_balanced)
        # 主题型应更重视舆情, 舆情90分时主题型得分应更高
        assert score_theme > score_balanced

    def test_short_horizon_boosts_technical(self):
        profile_short = UserProfile(investment_horizon=InvestmentHorizon.SHORT, style=InvestmentStyle.BALANCED)
        profile_long = UserProfile(investment_horizon=InvestmentHorizon.LONG, style=InvestmentStyle.BALANCED)
        # 量价高, 基本面低时, 短线应得更高分
        score_short = compute_weighted_score(40, 90, 60, 60, profile_short)
        score_long = compute_weighted_score(40, 90, 60, 60, profile_long)
        assert score_short > score_long

    def test_score_range(self):
        profile = UserProfile()
        for _ in range(20):
            score = compute_weighted_score(50, 50, 50, 50, profile)
            assert 0 <= score <= 100


class TestDataFreshness:
    def test_technical_stale_price_is_not_usable(self):
        from agents.data_freshness import apply_freshness_policy

        stale_day = (date.today() - timedelta(days=30)).isoformat()
        raw_data = {
            "errors": [],
            "price": [{"日期": stale_day, "收盘": 10.5}],
            "capital_flow": {"data": []},
            "north_flow": {"data": []},
            "lhb": {"data": []},
            "margin": {"data": []},
        }

        checked = apply_freshness_policy("technical", raw_data)

        assert checked["freshness"]["usable"] is False
        assert checked["freshness"]["checks"][0]["status"] == "stale"
        assert checked["price"] == []
        assert checked["stale_raw"]["price"][0]["日期"] == stale_day

    def test_fundamental_keeps_agent_when_report_is_fresh_and_masks_stale_valuation(self):
        from agents.data_freshness import apply_freshness_policy

        recent_report = (date.today() - timedelta(days=90)).isoformat()
        raw_data = {
            "errors": [],
            "valuation": [{"数据日期": "2018-01-02", "PE(TTM)": 10}],
            "indicator": [],
            "balance": [{"REPORT_DATE": recent_report, "TOTAL_ASSETS": 100}],
            "profit": [],
            "cashflow": [],
            "forecast": {},
            "earnings": {"data": []},
        }

        checked = apply_freshness_policy("fundamental", raw_data)

        assert checked["freshness"]["usable"] is True
        assert checked["valuation"] == []
        assert checked["stale_raw"]["valuation"][0]["数据日期"] == "2018-01-02"


class TestPositionSuggestion:
    def test_high_score(self):
        pos = _suggest_position(90, UserProfile())
        assert "%" in pos
        assert "0" not in pos or int(pos.split("-")[-1].replace("%", "")) > 0

    def test_low_score(self):
        pos = _suggest_position(30, UserProfile())
        assert "观望" in pos or "0" in pos

    def test_medium_score(self):
        pos = _suggest_position(65, UserProfile())
        assert "%" in pos


# ──────────────────────────────────────────────
# 意图解析测试 (规则回退)
# ──────────────────────────────────────────────

class TestFallbackIntentParser:
    def test_stock_code_detection(self):
        result = _fallback_parse("帮我分析 000001")
        assert result["intent"] == "stock_analysis"
        assert "000001" in result["symbols"]

    def test_sector_keywords(self):
        result = _fallback_parse("新能源板块怎么样")
        assert result["intent"] == "sector_analysis"

    def test_macro_keywords(self):
        result = _fallback_parse("最近GDP和PMI怎么样")
        assert result["intent"] == "macro_analysis"

    def test_comparison_keywords(self):
        result = _fallback_parse("比亚迪和长城汽车对比")
        assert result["intent"] == "comparison"

    def test_screening_keywords(self):
        result = _fallback_parse("帮我筛选值得关注的股票")
        assert result["intent"] == "screening"

    def test_general_query(self):
        result = _fallback_parse("今天天气怎么样")
        assert result["intent"] == "general"


# ──────────────────────────────────────────────
# Agent 模块导入测试
# ──────────────────────────────────────────────

class TestAgentImports:
    def test_fundamental_agent_import(self):
        from agents.fundamental_agent import analyze_fundamental, fetch_fundamental_data
        assert callable(analyze_fundamental)
        assert callable(fetch_fundamental_data)

    def test_technical_agent_import(self):
        from agents.technical_agent import analyze_technical, fetch_technical_data
        assert callable(analyze_technical)
        assert callable(fetch_technical_data)

    def test_sentiment_agent_import(self):
        from agents.sentiment_agent import analyze_sentiment, fetch_sentiment_data
        assert callable(analyze_sentiment)
        assert callable(fetch_sentiment_data)

    def test_macro_agent_import(self):
        from agents.macro_agent import analyze_macro, fetch_macro_data
        assert callable(analyze_macro)
        assert callable(fetch_macro_data)

    def test_data_provider_import(self):
        from agents.data_provider import DataProvider
        dp = DataProvider()
        assert hasattr(dp, "get_fundamental")
        assert hasattr(dp, "get_price_volume")
        assert hasattr(dp, "get_sentiment")
        assert hasattr(dp, "get_macro_china")

    def test_scheduler_import(self):
        from orchestrator.scheduler import analyze_stock, run_query
        assert callable(analyze_stock)
        assert callable(run_query)

    def test_decision_agent_import(self):
        from orchestrator.decision_agent import generate_recommendation, compute_weighted_score
        assert callable(generate_recommendation)
        assert callable(compute_weighted_score)


# ──────────────────────────────────────────────
# 数据获取测试 (不依赖网络)
# ──────────────────────────────────────────────

class TestDataProviderMethods:
    """测试 DataProvider 的方法存在性和参数验证"""

    def test_method_signatures(self):
        from agents.data_provider import DataProvider
        dp = DataProvider()
        # 基本面
        assert callable(dp.get_fundamental)
        assert callable(dp.get_profit_forecast)
        assert callable(dp.get_earnings_preview)
        # 量价
        assert callable(dp.get_price_volume)
        assert callable(dp.get_capital_flow)
        assert callable(dp.get_lhb)
        assert callable(dp.get_margin)
        # 舆情
        assert callable(dp.get_sentiment)
        assert callable(dp.get_news)
        assert callable(dp.get_market_emotion)
        # 宏观
        assert callable(dp.get_macro_china)
        assert callable(dp.get_global_interest)
        # 行业
        assert callable(dp.get_sector)
        assert callable(dp.get_market_valuation)

    def test_cache_stats(self):
        from agents.data_provider import DataProvider
        dp = DataProvider()
        stats = dp.cache_stats
        assert "hits" in stats
        assert "misses" in stats

    def test_routes_calls_through_mcp_client(self):
        from agents.data_provider import DataProvider

        class DummyClient:
            def __init__(self):
                self.calls = []

            def call_tool(self, name, arguments=None):
                self.calls.append((name, arguments or {}))
                return {"tool": name, "arguments": arguments or {}}

            @property
            def stats(self):
                return {"hits": 1, "misses": 2, "hit_rate": "33.3%"}

        client = DummyClient()
        dp = DataProvider(client=client, local_store_enabled=False)

        assert dp.get_fundamental("000001", "valuation")["tool"] == "get_stock_fundamental"
        assert dp.get_price_volume("000001")["tool"] == "get_stock_price_volume"
        assert dp.get_macro_china()["tool"] == "get_macro_china_overview"
        assert dp.get_margin()["tool"] == "get_stock_margin_detail"
        assert dp.cache_stats["hit_rate"] == "33.3%"

        assert client.calls == [
            ("get_stock_fundamental", {"symbol": "000001", "report_type": "valuation"}),
            (
                "get_stock_price_volume",
                {
                    "symbol": "000001",
                    "period": "daily",
                    "start_date": "",
                    "end_date": "",
                    "adjust": "qfq",
                },
            ),
            ("get_macro_china_overview", {}),
            ("get_stock_margin_detail", {}),
        ]

    def test_local_store_read_through(self, tmp_path):
        from agents.data_provider import DataProvider

        class DummyClient:
            def __init__(self):
                self.calls = []

            def call_tool(self, name, arguments=None):
                self.calls.append((name, arguments or {}))
                return {
                    "symbol": arguments["symbol"],
                    "period": arguments["period"],
                    "data": [{"日期": "2026-06-05", "收盘": 12.3}],
                }

            @property
            def stats(self):
                return {"hits": 0, "misses": 0, "hit_rate": "0.0%"}

        client = DummyClient()
        dp = DataProvider(
            client=client,
            local_store_enabled=True,
            local_store_dir=str(tmp_path),
        )

        first = dp.get_price_volume("000001")
        second = dp.get_price_volume("000001")

        assert first == second
        assert len(client.calls) == 1
        assert dp.local_store_stats["hits"] == 1


class TestMCPToolClientParsing:
    def test_unwraps_fastmcp_string_result(self):
        from agents.mcp_client import MCPToolClient

        payload = {"result": json.dumps({"symbol": "000001", "valuation": [{"PE(TTM)": 6.5}]})}
        normalized = MCPToolClient._normalize_payload(payload)

        assert normalized["symbol"] == "000001"
        assert normalized["valuation"][0]["PE(TTM)"] == 6.5

    def test_keeps_regular_dict_payload(self):
        from agents.mcp_client import MCPToolClient

        normalized = MCPToolClient._normalize_payload({"symbol": "000001", "data": []})

        assert normalized == {"symbol": "000001", "data": []}
