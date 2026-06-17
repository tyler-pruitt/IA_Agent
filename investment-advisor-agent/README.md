# Investment Advisor Agent

`investment-advisor-agent` 是本项目的多智能体投顾子系统，负责 Streamlit 前端、Agent 编排、数据适配、风险过滤和最终投资建议生成。完整项目说明见根目录 [README.md](../README.md)。

## 1. 当前定位

该子系统当前聚焦单只股票分析。用户输入股票代码、RQData `order_book_id` 或常见股票名称后，系统会:

1. 通过 `orchestrator.planner` 识别个股分析意图。
2. 通过 `DataProvider` 获取本地 RQData、技术指标、资金流、换手率、分析师一致预期和外部 MCP 数据。
3. 并行运行基本面 Agent、量价 Agent、舆情 Agent。
4. 使用 Decision Agent 按投资者偏好加权融合评分。
5. 在 Streamlit 页面展示综合建议、AI 报告、技术指标时序、DCF 估值和相对估值结果。

宏观 Agent 当前默认暂停，不参与主评分；板块和对比分析不是当前主面板重点。

## 2. 架构概览

```text
ui/app.py
  -> orchestrator.scheduler.run_query
  -> planner.parse_intent
  -> fundamental_agent / technical_agent / sentiment_agent
  -> decision_agent.generate_recommendation
  -> Streamlit 展示 Recommendation + 数据快照 + 估值模型
```

数据入口统一在:

```text
agents/data_provider.py
```

本地 RQData 离线数据读取在:

```text
agents/rqdata_store.py
```

## 3. 目录说明

```text
investment-advisor-agent/
├── ui/
│   └── app.py                         # Streamlit 前端
├── agents/
│   ├── data_provider.py               # 统一数据入口
│   ├── rqdata_store.py                # RQData 本地 CSV/Parquet 适配
│   ├── local_data_store.py            # MCP payload 本地缓存
│   ├── fundamental_agent.py           # 基本面 Agent
│   ├── technical_agent.py             # 量价 Agent
│   ├── sentiment_agent.py             # 舆情 Agent
│   ├── macro_agent.py                 # 宏观 Agent，默认暂停
│   ├── ai_analysis_report.py          # AI 分析页结构化报告
│   ├── dcf_analysis_adapter.py        # program.py DCF 接入
│   └── relative_valuation_adapter.py  # Relative_prediction 接入
├── orchestrator/
│   ├── planner.py                     # 意图识别
│   ├── scheduler.py                   # 并行调度
│   ├── decision_agent.py              # 综合决策
│   └── risk_filter.py                 # 风险过滤
├── akshare_mcp_server/                # MCP/AKShare 工具服务
├── config/
│   ├── settings.py                    # 权重、时效、路径和环境变量
│   └── agent_prompts/                 # Agent prompt 模板
├── models/
│   └── __init__.py                    # Pydantic 数据模型
├── tests/
└── DATA_STORAGE.md                    # 本地数据存储设计
```

## 4. 安装

从项目根目录安装冻结依赖:

```bash
cd /Users/julia/AI_and_Financial_Analysis_Final_Project
pip install -r requirements.txt
pip install -e investment-advisor-agent
```

仅安装子项目开发依赖:

```bash
cd investment-advisor-agent
pip install -e ".[dev]"
```

## 5. 配置

### LLM

```bash
cp ../.env.example ../.env
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
export LLM_FAST_MODEL="gpt-4o-mini"
```

### 数据源

默认启用本地 RQData 仓储:

```text
outputs/raw_rqdatac_database/
outputs/raw_rqdatac_technical_indicators/
```

关键环境变量:

| 变量 | 说明 |
|------|------|
| `RQDATA_STORE_ENABLED` | 是否启用本地 RQData 读取 |
| `RQDATA_STORE_DIR` | RQData CSV 目录 |
| `RQDATA_TECHNICAL_STORE_DIR` | 技术指标 parquet 目录 |
| `LOCAL_DATA_STORE_ENABLED` | 是否启用 MCP payload 本地缓存 |
| `DATA_FRESHNESS_CHECK_ENABLED` | 是否启用数据时效校验 |
| `MACRO_AGENT_ENABLED` | 是否启用宏观 Agent |
| `MCP_TRANSPORT` | MCP 连接方式，`stdio` 或 `sse` |

## 6. 启动前端

推荐从项目根目录运行，保证 `program.py`、`Relative_prediction/` 和 `model/` 都能被正确导入:

```bash
PYTHONPATH=investment-advisor-agent streamlit run investment-advisor-agent/ui/app.py
```

页面入口:

- `AI 投顾分析`: 主分析面板。
- `DCF 估值`: 现金流预测和 DCF 每股价值。
- `相对估值可视化`: Fair Multiple 结果查询和模型产物展示。

## 7. 命令行分析

```bash
PYTHONPATH=investment-advisor-agent python - <<'PY'
from orchestrator.scheduler import analyze_stock
from models import UserProfile, InvestmentStyle

profile = UserProfile(style=InvestmentStyle.BALANCED)
rec = analyze_stock("000001", profile)
print(rec.model_dump_json(indent=2, ensure_ascii=False))
PY
```

## 8. MCP 工具服务

默认情况下，`DataProvider` 会通过 stdio 自动拉起 MCP Server。也可以手动启动:

```bash
PYTHONPATH=investment-advisor-agent python -m akshare_mcp_server
```

SSE 模式:

```bash
PYTHONPATH=investment-advisor-agent python -m akshare_mcp_server --transport sse --port 8080
```

## 9. 测试

```bash
PYTHONPATH=investment-advisor-agent pytest investment-advisor-agent/tests -q
```

## 10. 当前已接入的关键能力

- RQData 财务、市场估值、行业标签、概念标签。
- `price.csv` 日线 OHLCV。
- `turnover.csv` 日换手率。
- `inflow_outflow.csv` 个股买入/卖出量额与净流入。
- `concensus.csv` 分析师一致预期和评级分布。
- 技术指标 parquet 时序，支持 MA、MACD、RSI、KDJ、BOLL 展示。
- DCF 每股价值自动运行。
- Relative Prediction 相对估值查询。
- AI 分析页中文报告和结构化表格。

## 11. 后续待完善

- 增加 DCF 和相对估值 adapter 的测试覆盖。
- 增加 Agent 运行耗时和数据源耗时监控。
- 对新闻/舆情建立本地缓存，减少外部接口不稳定影响。
- 为宏观和板块分析重新设计数据源时效校验后再接入主评分。
- 将 `.env.example`、Dockerfile、部署说明补齐。
- 明确大体量数据和模型文件的 Git LFS 或外部存储方案。
