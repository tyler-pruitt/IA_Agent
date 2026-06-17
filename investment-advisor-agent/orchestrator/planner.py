"""
意图解析器 — 解析用户自然语言，提取分析意图和参数

支持意图:
  - 个股分析: "分析一下比亚迪"
  - 板块分析: "新能源板块怎么样"
  - 宏观分析: "最近经济形势"
  - 对比分析: "比亚迪和长城汽车比"
  - 筛选推荐: "帮我找找值得关注的股票"
  - 通用问答: 其他
"""

import json
import logging
import re
from enum import Enum
from typing import Optional

from agents.llm_utils import call_llm_json

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    STOCK_ANALYSIS = "stock_analysis"       # 个股分析
    SECTOR_ANALYSIS = "sector_analysis"      # 板块分析
    MACRO_ANALYSIS = "macro_analysis"        # 宏观分析
    COMPARISON = "comparison"                # 对比分析
    SCREENING = "screening"                  # 筛选推荐
    GENERAL = "general"                      # 通用问答


INTENT_SYSTEM_PROMPT = """你是一个意图解析器。根据用户的自然语言输入，提取分析意图和参数。

你必须返回以下JSON格式:
```json
{
  "intent": "stock_analysis/sector_analysis/macro_analysis/comparison/screening/general",
  "symbols": ["股票代码1", "股票代码2"],
  "sector": "板块名称(如有)",
  "detail": "用户具体想了解的内容摘要"
}
```

意图判断规则:
- 提到具体股票名称/代码 → stock_analysis
- 提到"板块/行业/概念" → sector_analysis
- 提到"经济/宏观/政策/利率" → macro_analysis
- 提到"对比/比较/VS" → comparison
- 提到"筛选/推荐/值得关注的/找找" → screening
- 其他 → general

股票代码映射(常见):
- 比亚迪: 002594
- 贵州茅台: 600519
- 宁德时代: 300750
- 平安银行: 000001
- 招商银行: 600036
- 中芯国际: 688981
- 隆基绿能: 601012
- 长城汽车: 601633
- 中国平安: 601318
- 美的集团: 000333
- 格力电器: 000651
- 腾讯控股: 00700

如果用户提到股票名称但不在映射表中，symbols 返回空列表，detail 中保留股票名称。
只返回JSON，不要其他文字。"""


def parse_intent(user_query: str) -> dict:
    """
    解析用户意图

    :param user_query: 用户自然语言输入
    :return: {intent, symbols, sector, detail}
    """
    logger.info(f"[意图解析] 输入: {user_query}")

    try:
        result = call_llm_json(
            system_prompt=INTENT_SYSTEM_PROMPT,
            user_prompt=user_query,
            max_tokens=512,
        )
    except Exception as e:
        logger.warning(f"[意图解析] LLM解析失败: {e}, 使用规则回退")
        result = _fallback_parse(user_query)

    # 验证和清理
    result["intent"] = result.get("intent", "general")
    result["symbols"] = result.get("symbols", [])
    result["sector"] = result.get("sector", "")
    result["detail"] = result.get("detail", user_query)

    logger.info(f"[意图解析] 结果: intent={result['intent']}, symbols={result['symbols']}")
    return result


def _fallback_parse(query: str) -> dict:
    """规则回退: 当 LLM 不可用时的简易意图解析"""
    result = {"intent": "general", "symbols": [], "sector": "", "detail": query}

    # 检测股票代码格式 (6位数字)
    code_pattern = re.compile(r"\b(\d{6})\b")
    codes = code_pattern.findall(query)
    if codes:
        result["intent"] = "stock_analysis"
        result["symbols"] = codes
        return result

    # 检测板块关键词
    sector_keywords = ["板块", "行业", "概念", "赛道"]
    for kw in sector_keywords:
        if kw in query:
            result["intent"] = "sector_analysis"
            return result

    # 检测宏观关键词
    macro_keywords = ["经济", "宏观", "GDP", "CPI", "PMI", "利率", "政策"]
    for kw in macro_keywords:
        if kw in query:
            result["intent"] = "macro_analysis"
            return result

    # 检测对比关键词
    compare_keywords = ["对比", "比较", "vs", "VS", "和.*比"]
    for kw in compare_keywords:
        if re.search(kw, query, re.IGNORECASE):
            result["intent"] = "comparison"
            return result

    # 检测筛选关键词
    screen_keywords = ["筛选", "推荐", "值得关注", "找找", "选股"]
    for kw in screen_keywords:
        if kw in query:
            result["intent"] = "screening"
            return result

    return result
