"""
数据提供者 — Agent 统一通过 MCP 工具访问数据

无论是本地 stdio 模式还是远端 SSE 模式，
Agent 层都只通过 MCP 协议调用数据工具，不再直接 import tools 代码。
"""

from __future__ import annotations

import logging

from agents.local_data_store import LocalDataStore
from agents.mcp_client import MCPClientConfig, MCPToolClient, get_shared_client
from agents.rqdata_store import RQDataStore
from config.settings import (
    LOCAL_DATA_STORE_ENABLED,
    LOCAL_DATA_STORE_READ_THROUGH,
    LOCAL_DATA_STORE_WRITE_THROUGH,
    RQDATA_STORE_ENABLED,
)

logger = logging.getLogger(__name__)


class DataProvider:
    """
    统一数据获取接口 — 供 Agent 调用 MCP 工具

    默认复用共享 MCP client，避免重复建立会话。
    如需测试或特殊配置，可注入自定义 client。
    """

    def __init__(
        self,
        cache_dir: str = None,
        client: MCPToolClient | None = None,
        use_shared_client: bool = True,
        transport: str | None = None,
        server_url: str | None = None,
        local_store: LocalDataStore | None = None,
        local_store_enabled: bool | None = None,
        local_store_dir: str | None = None,
        rqdata_store: RQDataStore | None = None,
        rqdata_enabled: bool | None = None,
        rqdata_dir: str | None = None,
        rqdata_technical_dir: str | None = None,
    ):
        self._cache_dir = cache_dir
        self._client = client
        self._injected_client = client is not None
        self._use_shared_client = use_shared_client and client is None
        self._transport = transport
        self._server_url = server_url
        self._owns_client = False
        self._local_store = local_store
        self._local_store_dir = local_store_dir
        self._local_store_enabled = (
            LOCAL_DATA_STORE_ENABLED
            if local_store_enabled is None
            else local_store_enabled
        )
        self._local_store_failed = False
        self._rqdata_store = rqdata_store
        self._rqdata_dir = rqdata_dir
        self._rqdata_technical_dir = rqdata_technical_dir
        self._rqdata_enabled = (
            RQDATA_STORE_ENABLED if rqdata_enabled is None else rqdata_enabled
        )
        self._rqdata_failed = False

    def _get_client(self) -> MCPToolClient:
        if self._client is not None:
            return self._client

        default_config = MCPClientConfig()
        config = MCPClientConfig(
            transport=self._transport or default_config.transport,
            server_url=self._server_url or default_config.server_url,
            cache_dir=self._cache_dir,
        )

        if self._use_shared_client and self._cache_dir is None:
            self._client = get_shared_client(config)
        else:
            self._client = MCPToolClient(config)
            self._owns_client = True
        return self._client

    def _get_local_store(self) -> LocalDataStore | None:
        if not self._local_store_enabled or self._local_store_failed:
            return None
        if self._local_store is not None:
            return self._local_store
        try:
            self._local_store = LocalDataStore(root_dir=self._local_store_dir)
        except Exception as exc:
            logger.warning("[LocalDataStore] 初始化失败，降级为直接 MCP 调用: %s", exc)
            self._local_store_failed = True
            return None
        return self._local_store

    def _get_rqdata_store(self) -> RQDataStore | None:
        if not self._rqdata_enabled or self._rqdata_failed:
            return None
        if self._rqdata_store is not None:
            return self._rqdata_store
        try:
            self._rqdata_store = RQDataStore(
                root_dir=self._rqdata_dir,
                technical_root_dir=self._rqdata_technical_dir,
            )
        except Exception as exc:
            logger.warning("[RQDataStore] 初始化失败，跳过离线数据源: %s", exc)
            self._rqdata_failed = True
            return None
        if not self._rqdata_store.available:
            logger.info("[RQDataStore] 离线数据目录不可用: %s", self._rqdata_store.root_dir)
        return self._rqdata_store

    def _call_tool(self, name: str, arguments: dict | None = None) -> dict:
        arguments = arguments or {}
        local_store = self._get_local_store()

        if local_store is not None and LOCAL_DATA_STORE_READ_THROUGH:
            try:
                cached = local_store.get_fresh(name, arguments)
                if cached is not None:
                    return cached
            except Exception as exc:
                logger.warning("[LocalDataStore] 读取失败，继续调用 MCP: %s", exc)

        payload = self._get_client().call_tool(name, arguments)

        if local_store is not None and LOCAL_DATA_STORE_WRITE_THROUGH:
            try:
                local_store.save(name, arguments, payload)
            except Exception as exc:
                logger.warning("[LocalDataStore] 写入失败，本次结果仍正常返回: %s", exc)

        return payload

    @property
    def local_store_stats(self) -> dict:
        local_store = self._get_local_store()
        return local_store.stats if local_store is not None else {}

    @property
    def rqdata_store_stats(self) -> dict:
        rqdata_store = self._get_rqdata_store()
        return rqdata_store.stats if rqdata_store is not None else {}


    # ─── 基本面 ───

    def get_fundamental(self, symbol: str, report_type: str = "all") -> dict:
        """获取个股基本面全景数据"""
        return self._call_tool(
            "get_stock_fundamental",
            {"symbol": symbol, "report_type": report_type},
        )

    def get_profit_forecast(self, symbol: str) -> dict:
        """获取分析师盈利预测"""
        return self._call_tool(
            "get_stock_profit_forecast",
            {"symbol": symbol},
        )

    def get_earnings_preview(self, date: str, preview_type: str = "yjyg", symbol: str = "") -> dict:
        """获取业绩预告/快报"""
        return self._call_tool(
            "get_stock_earnings_preview",
            {"date": date, "preview_type": preview_type, "symbol": symbol},
        )

    # ─── RQData 离线补充源 ───

    def get_rqdata_fundamental(
        self,
        symbol: str,
        report_type: str = "all",
        quarter: str | None = None,
        limit: int = 8,
    ) -> dict:
        """获取 RQData 离线基本面/估值/PIT 数据。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "fundamental",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
            }
        return rqdata_store.get_fundamental(symbol, report_type, quarter, limit)

    def get_rqdata_metadata(self, symbol: str) -> dict:
        """获取 RQData 股票元数据和行业分类。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "metadata",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
            }
        return rqdata_store.get_metadata(symbol)

    def get_rqdata_concepts(self, symbol: str, limit: int = 100) -> dict:
        """获取 RQData 概念标签。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "concept_tags",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
                "data": [],
            }
        return rqdata_store.get_concepts(symbol, limit=limit)

    def get_rqdata_market_labels(
        self,
        symbol: str,
        quarter: str | None = None,
        limit: int = 8,
    ) -> dict:
        """获取 RQData 市场估值标签。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "market_labels",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
                "data": [],
            }
        return rqdata_store.get_market_labels(symbol, quarter=quarter, limit=limit)

    def get_rqdata_industry_scorecard(
        self,
        symbol: str,
        quarter: str | None = None,
        min_peer_count: int = 8,
        peer_limit: int = 20,
    ) -> dict:
        """获取 RQData 行业内相对得分和同业排名。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "industry_scorecard",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
            }
        return rqdata_store.get_industry_scorecard(
            symbol,
            quarter=quarter,
            min_peer_count=min_peer_count,
            peer_limit=peer_limit,
        )

    def get_rqdata_technical_indicators(
        self,
        symbol: str,
        date: str | None = None,
    ) -> dict:
        """获取 RQData 离线技术指标缓存。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "technical_indicators",
                "source": "rqdatac_technical_parquet",
                "error": "rqdata_store_unavailable",
                "data": [],
                "key_indicators": {},
            }
        return rqdata_store.get_technical_indicators(symbol, date=date)

    def get_rqdata_technical_timeseries(
        self,
        symbol: str,
        limit: int = 30,
        indicators: list[str] | None = None,
    ) -> dict:
        """获取 RQData 离线技术指标时序缓存。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "technical_indicator_timeseries",
                "source": "rqdatac_technical_parquet",
                "error": "rqdata_store_unavailable",
                "data": [],
            }
        return rqdata_store.get_technical_indicator_timeseries(
            symbol,
            limit=limit,
            indicators=indicators,
        )

    def get_rqdata_consensus(
        self,
        symbol: str,
        limit: int = 30,
    ) -> dict:
        """获取 RQData 离线分析师一致预期目标价和评级。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "symbol": symbol,
                "dataset": "analyst_consensus",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
                "data": [],
                "summary": {},
            }
        return rqdata_store.get_consensus(symbol, limit=limit)

    def get_rqdata_sector_constituents(
        self,
        name: str = "",
        limit: int = 50,
    ) -> dict:
        """获取 RQData 离线行业成分股映射。"""
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is None:
            return {
                "dataset": "rqdata_sector_constituents",
                "source": "rqdatac_offline",
                "error": "rqdata_store_unavailable",
                "sector": name,
                "data": [],
            }
        return rqdata_store.get_sector_constituents(name=name, limit=limit)

    # ─── 量价 ───

    def get_price_volume(self, symbol: str, period: str = "daily",
                         start_date: str = "", end_date: str = "",
                         adjust: str = "qfq") -> dict:
        """获取K线量价数据，日线优先读取本地 RQData price.csv。"""
        if period == "daily" and not self._injected_client:
            rqdata_store = self._get_rqdata_store()
            if rqdata_store is not None:
                try:
                    local_payload = rqdata_store.get_price_volume(
                        symbol,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if local_payload.get("data"):
                        return local_payload
                except Exception as exc:
                    logger.warning("[RQDataStore] price.csv 读取失败，降级 MCP K线接口: %s", exc)
        return self._call_tool(
            "get_stock_price_volume",
            {
                "symbol": symbol,
                "period": period,
                "start_date": start_date,
                "end_date": end_date,
                "adjust": adjust,
            },
        )

    def get_capital_flow(self, symbol: str = "", scope: str = "individual",
                         indicator: str = "今日") -> dict:
        """获取资金流向，个股资金流优先读取本地 RQData inflow_outflow.csv。"""
        if scope == "individual" and symbol and not self._injected_client:
            rqdata_store = self._get_rqdata_store()
            if rqdata_store is not None:
                try:
                    local_payload = rqdata_store.get_capital_flow(symbol, scope=scope)
                    if local_payload.get("data"):
                        return local_payload
                except Exception as exc:
                    logger.warning("[RQDataStore] inflow_outflow.csv 读取失败，降级 MCP 资金流接口: %s", exc)
        return self._call_tool(
            "get_stock_capital_flow",
            {"symbol": symbol, "scope": scope, "indicator": indicator},
        )

    def get_lhb(self, date: str = "", symbol: str = "",
                detail_type: str = "detail") -> dict:
        """获取龙虎榜"""
        return self._call_tool(
            "get_stock_lhb",
            {"date": date, "symbol": symbol, "detail_type": detail_type},
        )

    def get_margin(self) -> dict:
        """获取融资融券"""
        return self._call_tool("get_stock_margin_detail")

    # ─── 舆情 ───

    def get_sentiment(self, symbol: str) -> dict:
        """获取个股综合舆情"""
        return self._call_tool(
            "get_stock_sentiment",
            {"symbol": symbol},
        )

    def get_news(self, symbol: str = "", scope: str = "individual") -> dict:
        """获取新闻"""
        return self._call_tool(
            "get_stock_news",
            {"symbol": symbol, "scope": scope},
        )

    def get_market_emotion(self, date: str = "") -> dict:
        """获取市场整体情绪"""
        return self._call_tool(
            "get_stock_market_emotion",
            {"date": date},
        )

    # ─── 宏观 ───

    def get_macro_china(self) -> dict:
        """获取中国宏观全景"""
        return self._call_tool("get_macro_china_overview")

    def get_global_interest(self) -> dict:
        """获取全球利率"""
        return self._call_tool("get_macro_global_interest")

    # ─── 行业 ───

    def get_sector(self, sector_type: str = "industry", name: str = "") -> dict:
        """获取板块数据"""
        return self._call_tool(
            "get_stock_sector_analysis",
            {"sector_type": sector_type, "name": name},
        )

    def get_market_valuation(self, index: str = "沪深300") -> dict:
        """获取市场估值"""
        return self._call_tool(
            "get_stock_market_valuation",
            {"index": index},
        )

    # ─── 缓存统计 ───

    @property
    def cache_stats(self) -> dict:
        stats = {"hits": 0, "misses": 0, "hit_rate": "0.0%"} if self._client is None else self._client.stats
        local_store = self._get_local_store()
        if local_store is not None:
            stats = stats.copy()
            stats["local_store"] = local_store.stats
        rqdata_store = self._get_rqdata_store()
        if rqdata_store is not None:
            stats = stats.copy()
            stats["rqdata_store"] = rqdata_store.stats
        return stats

    def close(self):
        """关闭私有 MCP client；共享 client 由进程级复用管理。"""
        if self._client is not None and self._owns_client:
            self._client.close()
