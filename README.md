# IA Agent: 多源数据驱动的智能投顾分析系统

本项目是“人工智能与财务分析”期末项目，目标是构建一个面向 A 股个股分析的多智能体投顾系统。系统以 `investment-advisor-agent` 为前端与 Agent 编排核心，融合本地 RQData 离线数据库、DCF 现金流估值、Relative Prediction 相对估值、技术指标时序、资金流、分析师一致预期和新闻舆情数据，输出可解释、可追溯的投资分析结果。

> 免责声明: 本项目仅用于课程研究和技术展示，不构成任何投资建议。

## 1. 项目主线

用户在 Investment Advisor 面板输入股票代码或名称后，系统会自动完成:

1. 意图识别: 当前主流程聚焦单只股票分析。
2. 数据读取: 优先读取本地 RQData 离线数据，必要时通过 MCP/AKShare 接口补充。
3. 多 Agent 并行分析: 基本面 Agent、量价 Agent、舆情 Agent 同时运行。
4. 估值交叉验证: 接入 DCF 每股价值、相对估值 Fair Multiple、分析师一致预期目标价。
5. 决策融合: 根据用户风险偏好、投资周期和投资风格加权生成综合评分与建议。
6. 前端展示: 用 Streamlit 展示推荐结论、关键证据、技术指标时序、估值模型和 AI 分析报告。

简化流程如下:

```text
股票输入
  -> DataProvider 统一取数
  -> RQData 离线库 / 技术指标 parquet / DCF / 相对估值 / MCP 接口
  -> Fundamental Agent + Technical Agent + Sentiment Agent
  -> Decision Agent 加权融合
  -> Investment Advisor 前端展示
```

## 2. 核心模块

| 模块 | 路径 | 作用 |
|------|------|------|
| 投顾前端 | `investment-advisor-agent/ui/app.py` | Streamlit 主界面，包含 AI 投顾分析、DCF 估值、相对估值页面 |
| 数据统一入口 | `investment-advisor-agent/agents/data_provider.py` | 封装 RQData 离线数据、本地缓存和 MCP 工具调用 |
| RQData 离线适配 | `investment-advisor-agent/agents/rqdata_store.py` | 读取财务、行情、技术指标、资金流、换手率、行业标签、一致预期等本地 CSV/Parquet |
| Agent 层 | `investment-advisor-agent/agents/` | 基本面、量价、舆情、宏观等分析 Agent |
| 决策编排 | `investment-advisor-agent/orchestrator/` | 意图识别、并行调度、风险过滤、最终建议生成 |
| DCF 估值 | `program.py` + `model/` | ML 预测未来现金流/EPS，并计算 DCF 每股价值 |
| DCF 适配器 | `investment-advisor-agent/agents/dcf_analysis_adapter.py` | 将 `program.py` 的结果接入 Streamlit |
| 相对估值 | `Relative_prediction/` | 可比公司、倍数选择、公允倍数预测和估值查询 |
| 相对估值适配器 | `investment-advisor-agent/agents/relative_valuation_adapter.py` | 不修改原相对估值文件，复用其查询结果 |
| 数据产物 | `outputs/` | 存放 RQData 原始数据、技术指标缓存、估值结果等 |

## 3. 数据与模型来源

当前系统主要使用以下数据源:

| 数据类型 | 本地路径/来源 | 用途 |
|----------|---------------|------|
| 财务面板与市场标签 | `outputs/raw_rqdatac_database/` | 基本面、估值、行业比较 |
| 日线 OHLCV | `outputs/raw_rqdatac_database/price.csv` | K 线与趋势判断 |
| 换手率 | `outputs/raw_rqdatac_database/turnover.csv` | 量价活跃度分析 |
| 个股资金流 | `outputs/raw_rqdatac_database/inflow_outflow.csv` | 买入/卖出额、净流入分析 |
| 分析师一致预期 | `outputs/raw_rqdatac_database/concensus.csv` | 目标价、评级、覆盖机构 |
| 技术指标时序 | `outputs/raw_rqdatac_technical_indicators/technical_fetch_cache_*/` | MA、MACD、RSI、KDJ、BOLL 等时序可视化 |
| 相对估值产物 | `outputs/calculated_feature_database/` | Fair Multiple、公允价格、隐含空间 |
| DCF 模型文件 | `model/` | 现金流预测、盈利预测、DCF 估值 |
| 新闻/舆情 | MCP/AKShare 工具 | 舆情与事件补充 |

本地数据优先级高于外部接口。若本地数据存在，系统会优先使用本地 CSV/Parquet，减少网络代理或第三方接口不稳定带来的影响。

## 4. 环境配置

建议使用 Python 3.11。当前依赖已冻结在根目录 `requirements.txt`。

```bash
cd /Users/julia/AI_and_Financial_Analysis_Final_Project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e investment-advisor-agent
```

如使用 conda:

```bash
conda create -n ia-agent python=3.11
conda activate ia-agent
pip install -r requirements.txt
pip install -e investment-advisor-agent
```

### LLM 配置

如果需要调用 AI 报告和 Agent 推理，需要配置兼容 OpenAI SDK 的 API:

```bash
cp .env.example .env
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4o"
export LLM_FAST_MODEL="gpt-4o-mini"
```

如只查看 DCF 结构化结果、相对估值结果和本地数据面板，可先不配置 LLM Key，但完整 Agent 分析会受影响。

### RQData 配置

如果只读取已下载的本地 CSV/Parquet，不需要实时登录 RQData。如果要重新拉取数据，需要在本地环境中配置 `rqdatac` 账号权限，并确认:

```bash
python -c "import rqdatac as rq; rq.init(); print('rqdatac ok')"
```

### 常用环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_API_KEY` | 空 | LLM API Key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM 接口地址 |
| `LLM_MODEL` | `gpt-4o` | 主要分析模型 |
| `LLM_FAST_MODEL` | `gpt-4o-mini` | 快速抽取模型 |
| `MACRO_AGENT_ENABLED` | `false` | 宏观 Agent 当前默认暂停 |
| `DATA_FRESHNESS_CHECK_ENABLED` | `true` | 是否启用数据时效校验 |
| `RQDATA_STORE_ENABLED` | `true` | 是否启用本地 RQData 仓储 |
| `MCP_TRANSPORT` | `stdio` | MCP 工具调用方式 |
| `MCP_SERVER_URL` | `http://localhost:8080` | SSE 模式 MCP 地址 |

## 5. 运行方式

### 5.1 启动 Investment Advisor 前端

推荐从项目根目录运行:

```bash
PYTHONPATH=investment-advisor-agent streamlit run investment-advisor-agent/ui/app.py
```

页面包含:

- `AI 投顾分析`: 输入股票，生成综合建议、技术指标时序、估值模型联动、AI 报告。
- `DCF 估值`: 调用 `program.py` 的现金流预测与 DCF 模型。
- `相对估值可视化`: 嵌入 Fair Multiple 查询与模型产物展示。

### 5.2 命令行运行单股分析

```bash
PYTHONPATH=investment-advisor-agent python - <<'PY'
from orchestrator.scheduler import analyze_stock
from models import UserProfile, InvestmentStyle

profile = UserProfile(style=InvestmentStyle.BALANCED)
rec = analyze_stock("000001", profile)
print(rec.model_dump_json(indent=2, ensure_ascii=False))
PY
```

### 5.3 运行测试

```bash
PYTHONPATH=investment-advisor-agent pytest investment-advisor-agent/tests -q
```

当前核心测试应通过:

```text
110 passed
```

## 6. 主要脚本使用方法

### 6.1 RQData 财务与市场面板

脚本:

```text
Relative_prediction/fetch_rqdatac_panel_data.py
```

用途: 拉取季度财务面板、市场估值标签和 Step 1 输入文件。

```bash
python Relative_prediction/fetch_rqdatac_panel_data.py \
  --start-quarter 2020q1 \
  --end-quarter 2025q4
```

主要输出:

```text
outputs/raw_rqdatac_database/raw_rqdatac_database_2020q1_2025q4.csv
outputs/raw_rqdatac_database/raw_market_labels_2020q1_2025q4.csv
outputs/calculated_feature_database/calculated_feature_input_2020q1_2025q4.csv
```

### 6.2 RQData 技术指标缓存

脚本:

```text
Relative_prediction/fetch_rqdatac_technical_indicators.py
```

用途: 分交易日拉取全市场技术指标，默认保存为 parquet，支持断点续跑。

```bash
python Relative_prediction/fetch_rqdatac_technical_indicators.py \
  --start-date 2026-05-08 \
  --end-date 2026-06-09 \
  --output-format parquet
```

主要输出:

```text
outputs/raw_rqdatac_technical_indicators/technical_fetch_cache_YYYYMMDD_YYYYMMDD/
```

这些文件会被 Investment Advisor 读取，用于展示 MA、MACD、RSI、KDJ、BOLL 等近月技术指标时序。

### 6.3 相对估值流水线

脚本目录:

```text
Relative_prediction/
```

完整流水线:

```bash
python Relative_prediction/step1_comparable_companies.py
python Relative_prediction/step2_multiple_selection.py
python Relative_prediction/step3_market_implied_multiple_prediction.py
```

查询单家公司:

```bash
python Relative_prediction/company_valuation_cli.py 平安银行 --quarter latest
python Relative_prediction/company_valuation_cli.py 000001.XSHE --quarter 2025q4
```

启动原相对估值 API 页面:

```bash
python Relative_prediction/valuation_api_server.py
```

Investment Advisor 不修改原 `valuation_api_server.py`，而是通过 `relative_valuation_adapter.py` 读取同一套产物。

### 6.4 DCF 估值

核心文件:

```text
program.py
model/
```

在前端中，`DCF 估值` 页面会自动调用 `program.py` 的 `FinancialAnalysisSystem`。如需在 Python 中直接使用:

```bash
python - <<'PY'
from program import FinancialAnalysisSystem

system = FinancialAnalysisSystem(model_dir="./model")
results, report, chat_context = system.analyze_company("000001", generate_llm_report=False)
print(results["dcf_value"])
PY
```

### 6.5 MCP/AKShare 工具服务

默认情况下，`DataProvider` 会通过 stdio 自动拉起 MCP Server。也可以手动启动:

```bash
PYTHONPATH=investment-advisor-agent python -m akshare_mcp_server
```

SSE 模式:

```bash
PYTHONPATH=investment-advisor-agent python -m akshare_mcp_server --transport sse --port 8080
```

## 7. 前端展示内容

Investment Advisor 主面板当前包括:

- 综合投资建议: 评分、建议等级、仓位、核心逻辑、风险提示。
- 分析师一致预期: 覆盖机构、平均目标价、评级分布、隐含空间。
- 近月技术指标时序: 均线趋势、MACD 动能、RSI/KDJ、BOLL 通道。
- 估值模型联动: DCF 每股价值与相对估值公允价格。
- AI 分析报告: 基本面、技术面、资金面、资讯面、估值面。
- 数据快照: 展示本次分析实际使用的数据结构和错误信息。

## 8. 推送到远程新分支的建议流程

当前工作区存在较多未跟踪文件和生成数据，推送前建议先确认 `.gitignore` 是否排除大体量数据和缓存目录。

推荐流程:

```bash
git status --short
git switch -c feature/investment-advisor-docs
git add README.md requirements.txt investment-advisor-agent/README.md investment-advisor-agent/pyproject.toml
git add investment-advisor-agent Relative_prediction/fetch_rqdatac_technical_indicators.py
git commit -m "docs: update investment advisor project documentation"
git push -u origin feature/investment-advisor-docs
```

如果不希望上传本地大数据文件，应避免直接 `git add outputs model __pycache__`。建议将这些目录放入 `.gitignore`，或用 Git LFS/网盘/对象存储管理。

## 9. 仍需完善的地方

1. 数据体量管理: `outputs/`、`model/` 和 parquet 缓存较大，远程仓库应明确是否上传，必要时使用 Git LFS。
2. 配置文件模板: 可以补充 `.env.example`，统一说明 LLM、RQData、MCP、缓存目录配置。
3. 运行耗时监控: 当前 Agent 并行运行，但缺少每个 Agent、每个数据源的耗时统计面板。
4. 舆情数据稳定性: 新闻和舆情仍依赖外部接口，后续可增加本地新闻缓存和失败降级摘要。
5. 技术面策略规则: 已有 MA/MACD/RSI/KDJ/BOLL 时序展示，后续可进一步规则化突破、回踩、超买超卖和资金确认信号。
6. 估值结果一致性检查: DCF、相对估值、分析师目标价可进一步形成统一的估值区间和置信度。
7. 宏观与板块功能: 当前主流程聚焦个股分析，宏观 Agent 默认暂停，板块分析尚未作为主评分维度。
8. 测试覆盖: 需要为 DCF adapter、relative valuation adapter、技术指标时序读取增加更多单元测试。
9. 部署文档: 后续可补充 Dockerfile、云端部署和远程 MCP Server 配置说明。

## 10. 项目目录速览

```text
.
├── README.md
├── requirements.txt
├── program.py
├── model/
├── outputs/
├── investment-advisor-agent/
│   ├── ui/app.py
│   ├── agents/
│   ├── orchestrator/
│   ├── akshare_mcp_server/
│   ├── config/
│   ├── models/
│   ├── tests/
│   ├── README.md
│   └── DATA_STORAGE.md
├── Relative_prediction/
│   ├── fetch_rqdatac_panel_data.py
│   ├── fetch_rqdatac_technical_indicators.py
│   ├── step1_comparable_companies.py
│   ├── step2_multiple_selection.py
│   ├── step3_market_implied_multiple_prediction.py
│   ├── company_valuation_cli.py
│   └── valuation_api_server.py
└── ricequant_factor_names.txt
```
