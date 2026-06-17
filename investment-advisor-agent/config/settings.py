"""
全局配置 — LLM 连接、模型选择、Agent 参数

支持 OpenAI 兼容 API (Claude/GPT/国产模型均可)
通过环境变量覆盖默认值
"""

import os
import sys

# ──────────────────────────────────────────────
# LLM 配置
# ──────────────────────────────────────────────

# 基础模型 (分析推理)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# 快速模型 (数据提取/格式化)
LLM_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gpt-4o-mini")

# 温度
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ──────────────────────────────────────────────
# Agent 配置
# ──────────────────────────────────────────────

# 最大重试次数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# 单次 LLM 调用超时 (秒)
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))

# MCP Server 地址 (SSE 模式)
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")
MCP_STDIO_COMMAND = os.getenv("MCP_STDIO_COMMAND", sys.executable)
MCP_STDIO_ARGS = os.getenv("MCP_STDIO_ARGS", "-m akshare_mcp_server")
MCP_CONNECT_TIMEOUT = int(os.getenv("MCP_CONNECT_TIMEOUT", "10"))
MCP_SSE_READ_TIMEOUT = int(os.getenv("MCP_SSE_READ_TIMEOUT", "300"))
MCP_TOOL_TIMEOUT = int(os.getenv("MCP_TOOL_TIMEOUT", "120"))


def _env_bool(name: str, default: str = "false") -> bool:
    """Parse a boolean-like environment variable."""
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


# 宏观 Agent 当前默认暂停: 数据源滞后时不参与个股评分。
# 如需重新启用: export MACRO_AGENT_ENABLED=true
MACRO_AGENT_ENABLED = _env_bool("MACRO_AGENT_ENABLED", "false")
MACRO_AGENT_DISABLED_REASON = os.getenv(
    "MACRO_AGENT_DISABLED_REASON",
    "宏观 Agent 已暂停：当前宏观数据源滞后性较大，暂不参与个股评分。",
)

# 数据时效闸门: 拉取数据后先判断最新日期，核心数据滞后则跳过对应 Agent。
DATA_FRESHNESS_CHECK_ENABLED = _env_bool("DATA_FRESHNESS_CHECK_ENABLED", "true")

# 本地数据仓储: SQLite 索引 + JSON payload，用于跨股票复用公共数据和沉淀个股数据。
LOCAL_DATA_STORE_ENABLED = _env_bool("LOCAL_DATA_STORE_ENABLED", "true")
LOCAL_DATA_STORE_READ_THROUGH = _env_bool("LOCAL_DATA_STORE_READ_THROUGH", "true")
LOCAL_DATA_STORE_WRITE_THROUGH = _env_bool("LOCAL_DATA_STORE_WRITE_THROUGH", "true")
LOCAL_DATA_STORE_HISTORY_ENABLED = _env_bool("LOCAL_DATA_STORE_HISTORY_ENABLED", "true")
LOCAL_DATA_STORE_DIR = os.getenv(
    "LOCAL_DATA_STORE_DIR",
    os.path.expanduser("~/.cache/investment_advisor_agent"),
)

# RQData 离线数据源: 用于补强结构化财务、估值、行业、概念标签。
RQDATA_STORE_ENABLED = _env_bool("RQDATA_STORE_ENABLED", "true")
RQDATA_STORE_DIR = os.getenv(
    "RQDATA_STORE_DIR",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "raw_rqdatac_database",
        )
    ),
)
RQDATA_TECHNICAL_STORE_DIR = os.getenv(
    "RQDATA_TECHNICAL_STORE_DIR",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "outputs",
            "raw_rqdatac_technical_indicators",
        )
    ),
)

DATA_FRESHNESS_POLICIES = {
    "fundamental": {
        "required_any": ["indicator", "balance", "profit", "cashflow"],
        "sections": {
            "valuation": 30,
            "indicator": 240,
            "balance": 240,
            "profit": 240,
            "cashflow": 240,
            "forecast": 180,
            "earnings": 400,
        },
    },
    "technical": {
        "required_all": ["price"],
        "sections": {
            "price": 10,
            "capital_flow": 10,
            "north_flow": 10,
            "lhb": 45,
            "margin": 10,
        },
    },
    "sentiment": {
        "required_any": ["sentiment", "news", "market_emotion"],
        "sections": {
            "sentiment": 30,
            "news": 14,
            "market_emotion": 14,
        },
    },
    "macro": {
        "required_any": ["macro"],
        "sections": {
            "macro": 120,
            "interest": 180,
            "valuation": 30,
        },
    },
}

# ──────────────────────────────────────────────
# 评分权重配置 (决策 Agent 使用)
# ──────────────────────────────────────────────

WEIGHTS_BY_STYLE = {
    "价值": {"fundamental": 0.40, "technical": 0.10, "sentiment": 0.15, "macro": 0.35},
    "成长": {"fundamental": 0.30, "technical": 0.25, "sentiment": 0.20, "macro": 0.25},
    "均衡": {"fundamental": 0.25, "technical": 0.25, "sentiment": 0.25, "macro": 0.25},
    "主题": {"fundamental": 0.15, "technical": 0.30, "sentiment": 0.35, "macro": 0.20},
}

# 周期对量价权重的调整
HORIZON_ADJUST = {
    "短线": {"technical": 0.20, "fundamental": -0.15},
    "中线": {"technical": 0.00, "fundamental": 0.00},
    "长线": {"technical": -0.15, "fundamental": 0.20},
}


def get_effective_weights(
    style: str = "均衡",
    horizon: str = "中线",
    include_macro: bool | None = None,
    active_dimensions: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, float]:
    """
    返回决策层实际使用的归一化权重。

    当部分 Agent 未启用或时效校验失败时，对应权重置 0，其余维度按原有
    风格/周期权重重新归一。
    """
    if include_macro is None:
        include_macro = MACRO_AGENT_ENABLED

    weights = WEIGHTS_BY_STYLE.get(style, WEIGHTS_BY_STYLE["均衡"]).copy()

    adjustments = HORIZON_ADJUST.get(horizon, {})
    for key, delta in adjustments.items():
        if key not in weights:
            continue
        weights[key] = max(0.05, weights[key] + delta)
        others = [k for k in weights if k != key]
        per_deduct = -delta / len(others) if others else 0
        for other_key in others:
            weights[other_key] = max(0.05, weights[other_key] + per_deduct)

    if not include_macro:
        weights["macro"] = 0.0

    if active_dimensions is not None:
        active_set = set(active_dimensions)
        for key in list(weights):
            if key not in active_set:
                weights[key] = 0.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        return weights
    return {key: value / total_weight for key, value in weights.items()}
