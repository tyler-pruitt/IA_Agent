"""
MCP Server 核心组件单元测试
"""

import json
import time
import pytest
import pandas as pd

from akshare_mcp_server.cache import DataCache, SQLiteCache, TTL_CONFIG, _make_cache_key
from akshare_mcp_server.rate_limiter import RateLimiter, TokenBucket, FUNC_SOURCE_MAP


# ──────────────────────────────────────────────
# 缓存层测试
# ──────────────────────────────────────────────

class TestCacheKey:
    def test_same_args_same_key(self):
        key1 = _make_cache_key("func", ("a",), {"x": 1})
        key2 = _make_cache_key("func", ("a",), {"x": 1})
        assert key1 == key2

    def test_different_args_different_key(self):
        key1 = _make_cache_key("func", ("a",), {})
        key2 = _make_cache_key("func", ("b",), {})
        assert key1 != key2

    def test_different_func_different_key(self):
        key1 = _make_cache_key("func_a", (), {})
        key2 = _make_cache_key("func_b", (), {})
        assert key1 != key2


class TestSQLiteCache:
    def test_set_and_get(self, tmp_path):
        cache = SQLiteCache(cache_dir=str(tmp_path))
        cache.set("key1", "value1", ttl=60)
        assert cache.get("key1") == "value1"

    def test_expired_returns_none(self, tmp_path):
        cache = SQLiteCache(cache_dir=str(tmp_path))
        cache.set("key1", "value1", ttl=0)
        time.sleep(0.1)
        assert cache.get("key1") is None

    def test_delete(self, tmp_path):
        cache = SQLiteCache(cache_dir=str(tmp_path))
        cache.set("key1", "value1", ttl=60)
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear_expired(self, tmp_path):
        cache = SQLiteCache(cache_dir=str(tmp_path))
        cache.set("key1", "value1", ttl=0)
        cache.set("key2", "value2", ttl=60)
        time.sleep(0.1)
        cache.clear_expired()
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_overwrite(self, tmp_path):
        cache = SQLiteCache(cache_dir=str(tmp_path))
        cache.set("key1", "old", ttl=60)
        cache.set("key1", "new", ttl=60)
        assert cache.get("key1") == "new"


class TestDataCache:
    def test_miss_returns_none(self, tmp_path):
        cache = DataCache(cache_dir=str(tmp_path))
        assert cache.get("quote", "nonexistent", ()) is None

    def test_stats(self, tmp_path):
        cache = DataCache(cache_dir=str(tmp_path))
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        cache.get("quote", "nonexistent", ())
        assert cache.stats["misses"] == 1

    def test_ttl_config(self):
        assert TTL_CONFIG["quote"] == 300
        assert TTL_CONFIG["fundamental"] == 86400
        assert TTL_CONFIG["sentiment"] == 1800


class TestLocalDataStore:
    def test_save_and_read_stock_payload(self, tmp_path):
        from agents.local_data_store import LocalDataStore

        store = LocalDataStore(root_dir=str(tmp_path))
        payload = {
            "symbol": "000001",
            "period": "daily",
            "sort_order": "date_desc",
            "data": [{"日期": "2026-06-05", "收盘": 12.3}],
        }
        meta = store.save(
            "get_stock_price_volume",
            {"symbol": "000001", "period": "daily", "start_date": "", "end_date": "", "adjust": "qfq"},
            payload,
        )

        cached = store.get_fresh(
            "get_stock_price_volume",
            {"symbol": "000001", "period": "daily", "start_date": "", "end_date": "", "adjust": "qfq"},
        )

        assert cached == payload
        assert meta["scope"] == "stock"
        assert meta["entity_id"] == "000001"
        assert meta["latest_date"] == "2026-06-05"
        assert (tmp_path / "payloads" / "stock" / "000001" / "technical" / "price_volume" / "daily_qfq" / "latest.json").exists()

    def test_error_payload_is_not_read_through(self, tmp_path):
        from agents.local_data_store import LocalDataStore

        store = LocalDataStore(root_dir=str(tmp_path))
        store.save("get_stock_margin_detail", {}, {"error": "接口失败", "data": []})

        assert store.get_fresh("get_stock_margin_detail", {}) is None


class TestRQDataStore:
    def _write_fixture(self, root):
        (root / "metadata_universe_2020q1_2025q4.csv").write_text(
            "order_book_id,symbol,trading_code,exchange,status,listed_date,"
            "first_industry_name,second_industry_name,third_industry_name\n"
            "000001.XSHE,平安银行,000001,XSHE,Active,1991-04-03,"
            "银行,全国性股份制银行Ⅱ,全国性股份制银行Ⅲ\n",
            encoding="utf-8",
        )
        (root / "concept_tags_2020q1_2025q4.csv").write_text(
            "order_book_id,concept_name,inclusion_date\n"
            "000001.XSHE,深港通,2019-01-17\n",
            encoding="utf-8",
        )
        (root / "raw_market_labels_2020q1_2025q4.csv").write_text(
            "panel_id,order_book_id,quarter,info_date,valuation_date,"
            "close_price,volume,total_turnover,total_market_cap,"
            "pe_ratio_ttm,pb_ratio,ps_ratio_ttm,ev_to_ebitda\n"
            "000001.XSHE__2025q4,000001.XSHE,2025q4,2026-04-21,"
            "2026-04-22,13.23,103280274,1368222854,256740297760,"
            "8.76,0.91,1.79,10.5\n",
            encoding="utf-8",
        )
        (root / "pit_disclosures_2020q1_2025q4.csv").write_text(
            "order_book_id,quarter,info_date,operating_revenue,"
            "net_profit,total_assets,total_liabilities,equity_parent_company\n"
            "000001.XSHE,2025q4,2026-04-21,143408000000,"
            "29297000000,4132298000000,3779943000000,352355000000\n",
            encoding="utf-8",
        )
        (root / "raw_rqdatac_database_2020q1_2025q4.csv").write_text(
            "panel_id,order_book_id,symbol,first_industry_name,"
            "second_industry_name,third_industry_name,quarter,info_date,factor_date,"
            "operating_revenue,net_profit,total_assets,total_liabilities,"
            "total_equity,equity_parent_company,cash_equivalent,"
            "operating_revenue_ttm_0,net_profit_ttm_0,ebitda_ttm,"
            "return_on_equity_weighted_average,net_operate_cashflowTTM,"
            "fcff_ttm,fcfe_ttm,pe_ratio_ttm,pb_ratio,ps_ratio_ttm\n"
            "000001.XSHE__2025q4,000001.XSHE,平安银行,银行,"
            "全国性股份制银行Ⅱ,全国性股份制银行Ⅲ,2025q4,"
            "2026-04-21,2026-04-22,143408000000,29297000000,"
            "4132298000000,3779943000000,352355000000,352355000000,"
            "283706000000,143408000000,29297000000,37527000000,"
            "2.77,-75220000000,38513250000,56014000000,8.76,0.91,1.79\n",
            encoding="utf-8",
        )

    def _write_technical_fixture(self, root):
        cache_dir = root / "technical_fetch_cache_20260610_20260610"
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(
            [
                {
                    "order_book_id": "000001.XSHE",
                    "date": pd.Timestamp("2026-06-10"),
                    "MA5": 10.5,
                    "MA10": 10.2,
                    "MA20": 9.8,
                    "MA60": 9.1,
                    "MACD_DIFF": 0.12,
                    "MACD_DEA": 0.08,
                    "MACD_HIST": 0.04,
                    "KDJ_K": 61.2,
                    "KDJ_D": 55.4,
                    "RSI6": 58.3,
                    "BOLL": 10.0,
                    "BOLL_UP": 11.2,
                    "BOLL_DOWN": 8.8,
                    "VOL5": 123456.0,
                }
            ]
        )
        frame.to_parquet(cache_dir / "technical_20260610.parquet", index=False)

    def test_symbol_normalization(self, tmp_path):
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        store = RQDataStore(root_dir=str(tmp_path))

        assert store.normalize_symbol("000001") == "000001.XSHE"
        assert store.normalize_symbol("SH600519") == "600519.XSHG"
        assert store.normalize_symbol("000001.XSHE") == "000001.XSHE"

    def test_get_fundamental_payload(self, tmp_path):
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        store = RQDataStore(root_dir=str(tmp_path))
        payload = store.get_fundamental("000001")

        assert payload["source"] == "rqdatac_offline"
        assert payload["order_book_id"] == "000001.XSHE"
        assert payload["quarter"] == "2025q4"
        assert payload["valuation"][0]["pe_ratio_ttm"] == 8.76
        assert payload["metadata"]["symbol"] == "平安银行"
        assert payload["concept_tags"][0]["concept_name"] == "深港通"

    def test_get_industry_scorecard(self, tmp_path):
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        store = RQDataStore(root_dir=str(tmp_path))
        payload = store.get_industry_scorecard("000001")

        assert payload["dataset"] == "industry_scorecard"
        assert payload["industry_name"] == "全国性股份制银行Ⅲ"
        assert payload["industry_rank"] == 1
        assert payload["peer_count"] == 1
        assert payload["metrics"]
        assert payload["peer_table"][0]["order_book_id"] == "000001.XSHE"

    def test_get_technical_indicators_from_parquet(self, tmp_path):
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        technical_root = tmp_path / "technical"
        self._write_technical_fixture(technical_root)
        store = RQDataStore(root_dir=str(tmp_path), technical_root_dir=str(technical_root))

        payload = store.get_technical_indicators("000001")

        assert payload["dataset"] == "technical_indicators"
        assert payload["source"] == "rqdatac_technical_parquet"
        assert payload["date"] == "2026-06-10"
        assert payload["non_null_indicator_count"] > 0
        assert payload["key_indicators"]["MA5"] == 10.5
        assert payload["key_indicators"]["MACD_HIST"] == 0.04

    def test_data_provider_uses_injected_rqdata_store(self, tmp_path):
        from agents.data_provider import DataProvider
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        store = RQDataStore(root_dir=str(tmp_path))
        dp = DataProvider(rqdata_store=store, local_store_enabled=False)

        concepts = dp.get_rqdata_concepts("000001")

        assert concepts["concept_names"] == ["深港通"]
        assert dp.rqdata_store_stats["available"] is True

    def test_data_provider_exposes_technical_indicators(self, tmp_path):
        from agents.data_provider import DataProvider
        from agents.rqdata_store import RQDataStore

        self._write_fixture(tmp_path)
        technical_root = tmp_path / "technical"
        self._write_technical_fixture(technical_root)
        store = RQDataStore(root_dir=str(tmp_path), technical_root_dir=str(technical_root))
        dp = DataProvider(rqdata_store=store, local_store_enabled=False)

        payload = dp.get_rqdata_technical_indicators("000001")

        assert payload["order_book_id"] == "000001.XSHE"
        assert payload["key_indicators"]["RSI6"] == 58.3


# ──────────────────────────────────────────────
# 限流层测试
# ──────────────────────────────────────────────

class TestTokenBucket:
    def test_acquire_immediately(self):
        bucket = TokenBucket(rate=10.0, capacity=10)
        assert bucket.acquire(timeout=0.1) is True

    def test_depleted_bucket_waits(self):
        bucket = TokenBucket(rate=100.0, capacity=1)
        assert bucket.acquire(timeout=0.01) is True
        # 第二次应该需要等待
        assert bucket.acquire(timeout=0.05) is True

    def test_available_tokens(self):
        bucket = TokenBucket(rate=10.0, capacity=5)
        assert bucket.available > 0


class TestRateLimiter:
    def test_known_function_maps_to_source(self):
        limiter = RateLimiter()
        assert limiter.get_source("stock_zh_a_hist") == "eastmoney"
        assert limiter.get_source("stock_financial_abstract") == "sina"
        assert limiter.get_source("news_cctv") == "cctv"
        assert limiter.get_source("unknown_func") == "default"

    def test_func_source_map_coverage(self):
        """确保常用函数都有映射"""
        essential_funcs = [
            "stock_zh_a_hist", "stock_value_em", "stock_hot_rank_em",
            "stock_comment_em", "macro_china_gdp_yearly",
            "stock_board_industry_name_em",
        ]
        for func in essential_funcs:
            assert func in FUNC_SOURCE_MAP, f"{func} 未映射到数据源"

    def test_acquire_success(self):
        limiter = RateLimiter()
        assert limiter.acquire("stock_zh_a_hist", timeout=2.0) is True

    def test_status(self):
        limiter = RateLimiter()
        status = limiter.status
        assert "eastmoney" in status
        assert "default" in status


# ──────────────────────────────────────────────
# 工具层测试 (不依赖网络的逻辑测试)
# ──────────────────────────────────────────────

class TestToolImports:
    def test_fundamental_import(self):
        from akshare_mcp_server.tools.fundamental import (
            stock_fundamental, stock_profit_forecast, stock_earnings_preview,
        )
        assert callable(stock_fundamental)
        assert callable(stock_profit_forecast)
        assert callable(stock_earnings_preview)

    def test_price_volume_import(self):
        from akshare_mcp_server.tools.price_volume import (
            stock_price_volume, stock_capital_flow, stock_lhb, stock_margin_detail,
        )
        assert callable(stock_price_volume)
        assert callable(stock_capital_flow)
        assert callable(stock_lhb)
        assert callable(stock_margin_detail)

    def test_sentiment_import(self):
        from akshare_mcp_server.tools.sentiment import (
            stock_sentiment, stock_news, stock_market_emotion,
        )
        assert callable(stock_sentiment)
        assert callable(stock_news)
        assert callable(stock_market_emotion)

    def test_macro_import(self):
        from akshare_mcp_server.tools.macro import (
            macro_china_overview, macro_global_interest,
        )
        assert callable(macro_china_overview)
        assert callable(macro_global_interest)

    def test_sector_import(self):
        from akshare_mcp_server.tools.sector import (
            stock_sector_analysis, stock_market_valuation,
        )
        assert callable(stock_sector_analysis)
        assert callable(stock_market_valuation)


class TestToolParameterValidation:
    """测试工具参数校验逻辑"""

    def test_capital_flow_individual_without_symbol(self):
        from akshare_mcp_server.tools.price_volume import stock_capital_flow
        result = stock_capital_flow(symbol="", scope="individual")
        assert "error" in result

    def test_lhb_detail_without_date(self):
        from akshare_mcp_server.tools.price_volume import stock_lhb
        result = stock_lhb(date="", detail_type="detail")
        assert "error" in result

    def test_news_individual_without_symbol(self):
        from akshare_mcp_server.tools.sentiment import stock_news
        result = stock_news(symbol="", scope="individual")
        assert "error" in result

    def test_sector_without_name(self):
        from akshare_mcp_server.tools.sector import stock_sector_analysis
        result = stock_sector_analysis(sector_type="industry", name="")
        assert "error" in result

    def test_earnings_preview_bad_type(self):
        from akshare_mcp_server.tools.fundamental import stock_earnings_preview
        result = stock_earnings_preview(date="20240331", preview_type="bad_type")
        assert "error" in result


class TestPriceVolumeHelpers:
    def test_latest_window_returns_date_desc(self):
        import pandas as pd
        from akshare_mcp_server.tools.price_volume import _df_to_records

        df = pd.DataFrame(
            [
                {"日期": "2024-01-01", "收盘": 10},
                {"日期": "2024-01-03", "收盘": 12},
                {"日期": "2024-01-02", "收盘": 11},
            ]
        )

        records = _df_to_records(df, max_rows=2, latest_window=True)

        assert [row["日期"] for row in records] == ["2024-01-03", "2024-01-02"]


# ──────────────────────────────────────────────
# MCP Server 注册测试
# ──────────────────────────────────────────────

class TestMCPServer:
    def test_server_creation(self):
        from akshare_mcp_server.server import mcp
        assert mcp.name == "AKShare Finance Data"

    def test_tool_count(self):
        from akshare_mcp_server.server import mcp
        tools = mcp._tool_manager.list_tools()
        assert len(tools) == 14

    def test_all_tool_names(self):
        from akshare_mcp_server.server import mcp
        tools = mcp._tool_manager.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "get_stock_fundamental",
            "get_stock_profit_forecast",
            "get_stock_earnings_preview",
            "get_stock_price_volume",
            "get_stock_capital_flow",
            "get_stock_lhb",
            "get_stock_margin_detail",
            "get_stock_sentiment",
            "get_stock_news",
            "get_stock_market_emotion",
            "get_macro_china_overview",
            "get_macro_global_interest",
            "get_stock_sector_analysis",
            "get_stock_market_valuation",
        }
        assert tool_names == expected

    def test_tool_has_description(self):
        from akshare_mcp_server.server import mcp
        tools = mcp._tool_manager.list_tools()
        for tool in tools:
            assert tool.description, f"工具 {tool.name} 缺少描述"
