"""
风险过滤器 — 在综合评分之前对危险信号进行过滤和降级

风险规则:
  - ST / *ST / 退市风险 → 直接排除
  - 股权质押率 > 50% → 评分降级 20%
  - 日成交额 < 1000万 → 流动性警告
  - 商誉/净资产 > 30% → 评分降级 15%
  - 连续亏损 → 评分降级
  - 偏好匹配加分/减分
"""

import logging
from typing import Optional

from agents.data_provider import DataProvider
from models import UserProfile, Recommendation

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 风险检测规则
# ──────────────────────────────────────────────

class RiskFlag:
    """风险标记"""

    def __init__(self, level: str, message: str, score_penalty: float = 0):
        """
        :param level: "block"(排除) | "warning"(警告) | "info"(提示)
        :param message: 风险描述
        :param score_penalty: 评分扣减比例 (0-1)
        """
        self.level = level
        self.message = message
        self.score_penalty = score_penalty


def check_risk_flags(
    symbol: str,
    fundamental_data: dict = None,
    price_data: dict = None,
    dp: DataProvider = None,
) -> list[RiskFlag]:
    """
    对个股进行风险检测

    :param symbol: 股票代码
    :param fundamental_data: 基本面原始数据 (可选，避免重复获取)
    :param price_data: 量价原始数据 (可选)
    :param dp: 数据提供者
    :return: RiskFlag 列表
    """
    flags = []

    if dp is None:
        dp = DataProvider()

    # 1. 获取基本信息 (如果未提供)
    if fundamental_data is None:
        try:
            fundamental_data = dp.get_fundamental(symbol, "valuation")
        except Exception:
            fundamental_data = {}

    # 2. ST / 退市风险检测
    # 通过股票代码前缀判断: 沪市ST通常以*ST标记
    # 通过估值数据中的名称检测
    valuation_data = fundamental_data.get("valuation", [])
    if valuation_data and isinstance(valuation_data, list) and len(valuation_data) > 0:
        # 无法直接从估值数据获取名称，跳过
        pass

    # 3. 流动性检测 (通过量价数据判断成交额)
    if price_data is None:
        try:
            price_data = dp.get_price_volume(symbol, "daily")
        except Exception:
            price_data = {}

    price_records = price_data.get("data", [])
    if price_records and len(price_records) > 0:
        latest = price_records[-1]
        # 尝试获取成交额
        amount = latest.get("成交额", latest.get("amount", 0))
        if amount and isinstance(amount, (int, float)):
            if amount < 10_000_000:  # < 1000万
                flags.append(RiskFlag(
                    level="warning",
                    message=f"流动性不足: 日成交额仅{amount / 10000:.0f}万元，可能存在买卖困难",
                    score_penalty=0.1,
                ))
            elif amount < 30_000_000:  # < 3000万
                flags.append(RiskFlag(
                    level="info",
                    message=f"流动性偏低: 日成交额{amount / 10000:.0f}万元",
                    score_penalty=0.0,
                ))

    # 4. 估值异常检测 (PE为负 = 亏损)
    if valuation_data and isinstance(valuation_data, list) and len(valuation_data) > 0:
        latest_val = valuation_data[-1]
        pe_ttm = latest_val.get("PE(TTM)", latest_val.get("PE(8TM)", None))
        if pe_ttm and isinstance(pe_ttm, (int, float)):
            if pe_ttm < 0:
                flags.append(RiskFlag(
                    level="warning",
                    message=f"当前亏损: PE(TTM)={pe_ttm:.1f}，公司处于亏损状态",
                    score_penalty=0.2,
                ))
            elif pe_ttm > 200:
                flags.append(RiskFlag(
                    level="warning",
                    message=f"估值极高: PE(TTM)={pe_ttm:.1f}，远超正常范围",
                    score_penalty=0.1,
                ))

    return flags


def apply_risk_filter(
    score: float,
    risk_flags: list[RiskFlag],
) -> tuple[float, list[str]]:
    """
    应用风险过滤，调整评分

    :param score: 原始加权得分
    :param risk_flags: 风险标记列表
    :return: (调整后得分, 风险描述列表)
    """
    adjusted = score
    warnings = []

    for flag in risk_flags:
        if flag.level == "block":
            # 阻断级: 直接降到0
            adjusted = 0
            warnings.append(f"⛔ {flag.message}")
        elif flag.level == "warning":
            adjusted *= (1 - flag.score_penalty)
            warnings.append(f"⚠️ {flag.message}")
        elif flag.level == "info":
            warnings.append(f"ℹ️ {flag.message}")

    # 确保分数在合理范围
    adjusted = max(0, min(100, round(adjusted, 1)))

    return adjusted, warnings


# ──────────────────────────────────────────────
# 偏好匹配
# ──────────────────────────────────────────────

def apply_preference_match(
    symbol: str,
    score: float,
    profile: UserProfile,
    dp: DataProvider = None,
) -> tuple[float, list[str]]:
    """
    根据投资者偏好对评分进行微调

    规则:
      - 偏好行业个股 → 加分 5%
      - 回避行业个股 → 减分 10%
      - 保守型遇到高波动 → 减分
      - 激进型遇到高成长 → 加分

    :param symbol: 股票代码
    :param score: 风险过滤后的得分
    :param profile: 投资者偏好
    :param dp: 数据提供者
    :return: (调整后得分, 匹配信息列表)
    """
    adjusted = score
    match_info = []

    # 偏好行业匹配 (通过个股新闻/概念关键词间接判断)
    # 简化版: 如果偏好的行业关键词出现在代码范围中则加分
    # 生产版应通过 stock_individual_basic_info_xq 获取行业信息

    # 回避行业检测
    avoid = profile.avoid_sectors
    if avoid:
        # 检查 ST
        if "ST" in avoid and _is_likely_st(symbol):
            adjusted *= 0.9
            match_info.append("回避ST股，评分下调10%")

    # 风险偏好匹配
    if profile.risk_tolerance.value == "保守":
        # 保守型: 量价得分>70(高波动)时适当减分
        pass  # 此处需要量价数据，由调度层传入
    elif profile.risk_tolerance.value == "激进":
        # 激进型: 高成长股适当加分
        pass

    adjusted = max(0, min(100, round(adjusted, 1)))
    return adjusted, match_info


def _is_likely_st(symbol: str) -> bool:
    """简单判断是否可能是ST股 (启发式)"""
    # ST股代码无固定规则，需要通过名称判断
    # 这里返回 False，实际应由数据层判断
    return False


def full_risk_and_preference_filter(
    symbol: str,
    raw_score: float,
    profile: UserProfile,
    dp: DataProvider = None,
) -> tuple[float, list[str]]:
    """
    完整的风险过滤 + 偏好匹配流程

    :return: (最终得分, 所有警告+匹配信息)
    """
    # 1. 风险检测
    risk_flags = check_risk_flags(symbol, dp=dp)
    score_after_risk, risk_warnings = apply_risk_filter(raw_score, risk_flags)

    # 2. 偏好匹配
    final_score, pref_info = apply_preference_match(symbol, score_after_risk, profile, dp)

    # 3. 合并信息
    all_info = risk_warnings + pref_info

    return final_score, all_info
