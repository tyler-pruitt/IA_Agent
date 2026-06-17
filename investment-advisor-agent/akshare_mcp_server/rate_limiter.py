"""
限流控制 — 令牌桶算法，按数据源分组限流

默认: 1 req/s (全局)，各数据源可独立配置
防止 AKShare 接口被反爬封禁
"""

import time
import threading
from typing import Dict, Optional


class TokenBucket:
    """令牌桶限流器"""

    def __init__(self, rate: float = 1.0, capacity: int = 5):
        """
        :param rate: 每秒填充令牌数
        :param capacity: 桶容量 (允许短时突发)
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 10.0) -> bool:
        """
        获取一个令牌，阻塞等待最多 timeout 秒
        :return: True=获取成功, False=超时
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True
            if time.monotonic() >= deadline:
                return False
            # 等待令牌补充
            time.sleep(1.0 / self._rate)

    def _refill(self):
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill = elapsed * self._rate
        self._tokens = min(self._capacity, self._tokens + refill)
        self._last_refill = now

    @property
    def available(self) -> float:
        """当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens


# 数据源分组限流配置
# AKShare 不同数据源来源不同，需分别限流
SOURCE_LIMITS = {
    "eastmoney": {"rate": 2.0, "capacity": 8},    # 东方财富: 主要数据源，2 req/s
    "sina": {"rate": 1.0, "capacity": 5},          # 新浪财经: 1 req/s
    "qq": {"rate": 1.0, "capacity": 5},            # 腾讯财经: 1 req/s
    "ths": {"rate": 0.8, "capacity": 4},           # 同花顺: 0.8 req/s
    "cninfo": {"rate": 0.5, "capacity": 3},        # 巨潮资讯: 0.5 req/s
    "baidu": {"rate": 1.0, "capacity": 5},         # 百度股市: 1 req/s
    "jin10": {"rate": 1.0, "capacity": 5},          # 金十数据: 1 req/s
    "xueqiu": {"rate": 0.8, "capacity": 4},        # 雪球: 0.8 req/s
    "legu": {"rate": 1.0, "capacity": 5},          # 乐估: 1 req/s
    "cctv": {"rate": 0.5, "capacity": 2},           # 央视: 0.5 req/s
    "nbs": {"rate": 0.5, "capacity": 3},            # 国家统计局: 0.5 req/s
    "default": {"rate": 1.0, "capacity": 5},        # 默认: 1 req/s
}

# 函数名 → 数据源映射
FUNC_SOURCE_MAP = {
    # 东方财富 (eastmoney) — em 后缀
    "stock_zh_a_hist": "eastmoney",
    "stock_zh_a_hist_tx": "qq",
    "stock_zh_a_spot_em": "eastmoney",
    "stock_zh_a_hist_min_em": "eastmoney",
    "stock_individual_fund_flow": "eastmoney",
    "stock_individual_fund_flow_rank": "eastmoney",
    "stock_market_fund_flow": "eastmoney",
    "stock_sector_fund_flow_rank": "eastmoney",
    "stock_lhb_detail_em": "eastmoney",
    "stock_lhb_stock_statistic_em": "eastmoney",
    "stock_lhb_jgmmtj_em": "eastmoney",
    "stock_hot_rank_em": "eastmoney",
    "stock_hot_rank_detail_em": "eastmoney",
    "stock_hot_rank_detail_realtime_em": "eastmoney",
    "stock_hot_keyword_em": "eastmoney",
    "stock_hot_rank_latest_em": "eastmoney",
    "stock_hot_rank_relate_em": "eastmoney",
    "stock_hot_up_em": "eastmoney",
    "stock_value_em": "eastmoney",
    "stock_comment_em": "eastmoney",
    "stock_comment_detail_zlkp_jgcyd_em": "eastmoney",
    "stock_comment_detail_zhpj_lspf_em": "eastmoney",
    "stock_comment_detail_scrd_focus_em": "eastmoney",
    "stock_comment_detail_scrd_desire_em": "eastmoney",
    "stock_yjbb_em": "eastmoney",
    "stock_yjkb_em": "eastmoney",
    "stock_yjyg_em": "eastmoney",
    "stock_yysj_em": "eastmoney",
    "stock_zcfz_em": "eastmoney",
    "stock_lrb_em": "eastmoney",
    "stock_xjll_em": "eastmoney",
    "stock_a_ttm_lyr": "eastmoney",
    "stock_balance_sheet_by_report_em": "eastmoney",
    "stock_profit_sheet_by_report_em": "eastmoney",
    "stock_cash_flow_sheet_by_report_em": "eastmoney",
    "stock_financial_analysis_indicator_em": "eastmoney",
    "stock_profit_forecast_em": "eastmoney",
    "stock_board_industry_name_em": "eastmoney",
    "stock_board_industry_hist_em": "eastmoney",
    "stock_board_concept_name_em": "eastmoney",
    "stock_board_concept_hist_em": "eastmoney",
    "stock_hsgt_hist_em": "eastmoney",
    "stock_hsgt_hold_stock_em": "eastmoney",
    "stock_hsgt_board_rank_em": "eastmoney",
    "stock_zt_pool_em": "eastmoney",
    "stock_zt_pool_previous_em": "eastmoney",
    "stock_zt_pool_strong_em": "eastmoney",
    "stock_news_em": "eastmoney",
    "stock_dxsyl_em": "eastmoney",
    "stock_xgsglb_em": "eastmoney",
    "stock_changes_em": "eastmoney",
    "stock_board_change_em": "eastmoney",
    "stock_intraday_em": "eastmoney",
    "stock_bid_ask_em": "eastmoney",
    # 新浪 (sina) — sina 后缀
    "stock_financial_abstract": "sina",
    "stock_financial_analysis_indicator": "sina",
    "stock_financial_report_sina": "sina",
    # 同花顺 (ths) — ths 后缀
    "stock_financial_abstract_ths": "ths",
    "stock_financial_debt_ths": "ths",
    "stock_financial_benefit_ths": "ths",
    "stock_financial_cash_ths": "ths",
    "stock_financial_abstract_new_ths": "ths",
    "stock_financial_debt_new_ths": "ths",
    "stock_financial_benefit_new_ths": "ths",
    "stock_financial_cash_new_ths": "ths",
    "stock_board_industry_summary_ths": "ths",
    "stock_board_concept_summary_ths": "ths",
    "stock_fund_flow_individual": "ths",
    "stock_fund_flow_concept": "ths",
    "stock_fund_flow_industry": "ths",
    "stock_fund_flow_big_deal": "ths",
    # 巨潮资讯 (cninfo)
    "stock_irm_cninfo": "cninfo",
    "stock_allotment_cninfo": "cninfo",
    "stock_dividend_cninfo": "cninfo",
    # 百度 (baidu)
    "stock_hot_search_baidu": "baidu",
    "news_economic_baidu": "baidu",
    "news_report_time_baidu": "baidu",
    # 金十 (jin10)
    "stock_js_weibo_report": "jin10",
    # 雪球 (xueqiu)
    "stock_hot_follow_xq": "xueqiu",
    "stock_hot_tweet_xq": "xueqiu",
    "stock_individual_basic_info_xq": "xueqiu",
    # 乐估 (legu)
    "stock_a_congestion_lg": "legu",
    "stock_buffett_index_lg": "legu",
    "stock_market_pe_lg": "legu",
    "stock_index_pe_lg": "legu",
    "stock_ebs_lg": "legu",
    # 央视 (cctv)
    "news_cctv": "cctv",
    # 国家统计局 (nbs)
    "macro_china_nbs_nation": "nbs",
    "macro_china_nbs_region": "nbs",
    # 宏观综合 (macro_china.py 来自金十/东财等多源，统一归为 macro_china 组)
    "macro_china_gdp_yearly": "eastmoney",
    "macro_china_cpi_yearly": "eastmoney",
    "macro_china_cpi_monthly": "eastmoney",
    "macro_china_ppi_yearly": "eastmoney",
    "macro_china_pmi_yearly": "eastmoney",
    "macro_china_m2_yearly": "eastmoney",
    "macro_china_exports_yoy": "eastmoney",
    "macro_china_imports_yoy": "eastmoney",
    "macro_china_trade_balance": "eastmoney",
    "macro_china_industrial_production_yoy": "eastmoney",
    "macro_china_urban_unemployment": "eastmoney",
    "macro_china_fx_reserves_yearly": "eastmoney",
    "macro_china_shibor_all": "eastmoney",
    "macro_china_lpr": "eastmoney",
    # 财新 (cx)
    "index_pmi_com_cx": "eastmoney",
    "index_pmi_man_cx": "eastmoney",
    "index_dei_cx": "eastmoney",
}


class RateLimiter:
    """
    统一限流接口 — 按数据源分组管理令牌桶

    用法:
        limiter = RateLimiter()
        limiter.acquire("stock_zh_a_hist")  # 自动映射到 eastmoney 限流组
        data = ak.stock_zh_a_hist(symbol="000001", period="daily")
    """

    def __init__(self, custom_limits: Optional[Dict[str, dict]] = None):
        self._buckets: Dict[str, TokenBucket] = {}
        self._init_buckets(custom_limits or {})

    def _init_buckets(self, custom_limits: dict):
        """初始化所有数据源的令牌桶"""
        all_sources = set(SOURCE_LIMITS.keys())
        for source in all_sources:
            config = custom_limits.get(source, SOURCE_LIMITS.get(source, SOURCE_LIMITS["default"]))
            self._buckets[source] = TokenBucket(
                rate=config.get("rate", 1.0),
                capacity=config.get("capacity", 5),
            )

    def acquire(self, func_name: str, timeout: float = 10.0) -> bool:
        """
        按函数名自动映射数据源，获取令牌
        :param func_name: akshare 函数名
        :param timeout: 最长等待秒数
        :return: True=获取成功, False=超时
        """
        source = FUNC_SOURCE_MAP.get(func_name, "default")
        bucket = self._buckets.get(source, self._buckets["default"])
        return bucket.acquire(timeout=timeout)

    def get_source(self, func_name: str) -> str:
        """获取函数对应的数据源"""
        return FUNC_SOURCE_MAP.get(func_name, "default")

    @property
    def status(self) -> dict:
        """各数据源令牌桶状态"""
        return {source: bucket.available for source, bucket in self._buckets.items()}
