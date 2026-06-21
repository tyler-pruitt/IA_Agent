"""
LLM 调用工具 — 统一 LLM 调用接口，支持结构化 JSON 输出

使用 OpenAI 兼容 API，可对接 Claude/GPT/国产模型
"""

import json
import logging
import re

from openai import OpenAI

from config.settings import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_FAST_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """获取 OpenAI 兼容客户端"""
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY 未配置，无法调用 LLM")
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        timeout=LLM_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = 4096,
) -> str:
    """
    调用 LLM 并返回原始文本

    :param system_prompt: 系统提示词
    :param user_prompt: 用户提示词
    :param model: 模型名称 (默认使用配置中的主模型)
    :param temperature: 温度
    :param max_tokens: 最大输出 token
    :return: LLM 输出文本
    """
    client = get_client()
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content
    return content


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str = None,
    temperature: float = None,
    max_tokens: int = 4096,
) -> dict:
    """
    调用 LLM 并解析 JSON 输出

    自动从 LLM 输出中提取 JSON 内容，支持:
    - 纯 JSON 字符串
    - ```json ... ``` 代码块包裹
    - 混合文字+JSON

    :return: 解析后的 dict
    :raises: ValueError 如果无法解析 JSON
    """
    raw = call_llm(system_prompt, user_prompt, model, temperature, max_tokens)

    # 尝试直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 代码块
    json_block_pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    match = json_block_pattern.search(raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取最外层 { ... }
    brace_pattern = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)
    match = brace_pattern.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 最后尝试: 找到第一个 { 和最后一个 }
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(raw[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"无法从 LLM 输出中解析 JSON。原始输出前200字: {raw[:200]}")


def call_llm_fast(system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
    """使用快速模型调用 (适合数据提取等简单任务)"""
    return call_llm(system_prompt, user_prompt, model=LLM_FAST_MODEL, max_tokens=max_tokens)
