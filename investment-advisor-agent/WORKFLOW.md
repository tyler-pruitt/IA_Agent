# 投资分析工作流说明

本文档说明 `investment-advisor-agent` 从用户输入到最终投资建议的完整路径，包括调用的 MCP 工具、底层 AKShare API、目标数据，以及当前评分逻辑的确定性边界。

## 1. 总体流程

```text
用户输入股票/问题
  -> 意图解析 planner.parse_intent
  -> 调度器 scheduler.run_query / analyze_stock
  -> DataProvider 统一调用 MCP 工具
  -> 数据时效闸门 agents.data_freshness
     - 提取最新日期
     - 判断是否超过该数据类型阈值
     - 滞后核心数据对应 Agent 不参与评分
  -> 子 Agent 并行分析
     - 基本面 Agent
     - 量价 Agent
     - 舆情 Agent
     - 宏观 Agent（当前默认暂停）
  -> 决策 Agent 汇总有效维度得分
  -> 风险过滤和偏好匹配
  -> 输出 Recommendation
```

当前所有数据访问都经过 MCP。默认模式是 `stdio`，由 `DataProvider` 自动拉起本地 `python -m akshare_mcp_server`；也可以设置 `MCP_TRANSPORT=sse` 连接独立 MCP Server。宏观 Agent 因数据源滞后性较大，当前默认 `MACRO_AGENT_ENABLED=false`，不参与个股评分；如需恢复，可设置 `MACRO_AGENT_ENABLED=true`。数据时效校验默认开启，可通过 `DATA_FRESHNESS_CHECK_ENABLED=false` 临时关闭。

本地数据存储分两层理解: 当前已有 MCP 函数级 TTL 缓存，用于减少短时间重复调用；更适合长期沉淀和审计的规范化存储设计见 [DATA_STORAGE.md](DATA_STORAGE.md)。

相关入口:

| 阶段 | 文件 | 说明 |
|------|------|------|
| 用户入口 | `ui/app.py` | Streamlit 页面入口 |
| 调度入口 | `orchestrator/scheduler.py` | 解析意图并路由到个股、对比、板块、宏观等流程 |
| MCP 数据入口 | `agents/data_provider.py` | 把所有数据请求映射到 MCP 工具 |
| MCP 客户端 | `agents/mcp_client.py` | 管理 stdio/SSE MCP 会话 |
| MCP 服务端 | `akshare_mcp_server/server.py` | 注册 14 个数据工具 |
| 数据时效闸门 | `agents/data_freshness.py` | 抽取最新日期、标记 stale/empty/unknown_date，并决定 Agent 是否参与评分 |
| 综合决策 | `orchestrator/decision_agent.py` | 加权、风险过滤、生成最终建议 |

## 2. 数据获取流程

### 2.1 基本面 Agent

目标: 判断财务健康度、估值合理性、盈利成长性。

| MCP 工具 | 底层 AKShare API | 目标数据 | 用途 |
|----------|------------------|----------|------|
| `get_stock_fundamental(report_type="valuation")` | `ak.stock_value_em` | PE(TTM)、PE(静)、PB、市值、PEG、市销率、市现率 | 估值水平与历史样本比较 |
| `get_stock_fundamental(report_type="indicator")` | `ak.stock_financial_analysis_indicator_em` | ROE、毛利率、净利率、资产负债率等财务指标 | 财务质量、盈利能力、偿债能力 |
| `get_stock_fundamental(report_type="balance")` | `ak.stock_balance_sheet_by_report_em` | 总资产、总负债、所有者权益、商誉、货币资金等 | 资产负债结构 |
| `get_stock_fundamental(report_type="profit")` | `ak.stock_profit_sheet_by_report_em` | 营业收入、营业成本、营业利润、净利润、EPS 及同比 | 盈利与成长趋势 |
| `get_stock_fundamental(report_type="cashflow")` | `ak.stock_cash_flow_sheet_by_report_em` | 经营现金流、投资现金流、筹资现金流 | 现金流质量与经营现金流/净利润 |
| `get_stock_profit_forecast` | `ak.stock_profit_forecast_em` | 分析师一致预期、盈利预测 | 预期变化与估值辅助；底层接口不支持直接按股票代码查询，MCP 会先取全市场预测再按代码过滤 |
| `get_stock_earnings_preview(symbol=...)` | `ak.stock_yjyg_em` / `ak.stock_yjkb_em` / `ak.stock_yjbb_em` | 业绩预告、业绩快报、业绩报表 | 业绩超预期或低于预期信号；传入股票代码时 MCP 先过滤完整样本再截取展示 |

注意:
估值接口 `ak.stock_value_em` 在 AKShare 内部会按日期升序返回，MCP 会重新按日期倒序截取，确保 UI 中默认展示最新样本而不是 2018 年初的最早样本。财务指标接口需要东方财富 `SECUCODE` 格式，如 `000001.SZ`，MCP 会从用户输入的 `000001` 自动转换。

业绩预告接口返回的是全市场样本，因此 MCP 会按股票代码过滤，只保留目标股票记录。找不到目标股票时，会明确写入“不能使用其他股票样本替代”。

### 2.2 量价 Agent

目标: 判断趋势、动量和资金流。

| MCP 工具 | 底层 AKShare API | 目标数据 | 用途 |
|----------|------------------|----------|------|
| `get_stock_price_volume` | 主源 `ak.stock_zh_a_hist`，日线备用源 `ak.stock_zh_a_hist_tx` | 日/周/月 K 线，开高低收、成交量、成交额、涨跌幅 | 均线、趋势、动量、成交活跃度；东方财富 push2his 不可用时日线切腾讯源 |
| `get_stock_capital_flow(scope="individual")` | 主源 `ak.stock_individual_fund_flow`，兜底 `ak.stock_individual_fund_flow_rank` | 个股主力资金净流入、超大单/大单/中单/小单流向 | 买盘强度与资金行为；历史资金流失败时用今日资金流排名过滤目标股票 |
| `get_stock_capital_flow(scope="north")` | `ak.stock_hsgt_hist_em` | 北向资金流入流出 | 外资方向辅助判断 |
| `get_stock_lhb(detail_type="stock_statistic")` | `ak.stock_lhb_stock_statistic_em` | 近一月龙虎榜上榜统计 | 游资/机构交易活跃度 |
| `get_stock_margin_detail` | `ak.stock_margin_account_info` | 融资融券余额、市场杠杆情绪 | 杠杆资金风险与热度 |

注意:
北向资金和融资融券等长历史接口在 AKShare 内部按日期升序返回，MCP 会先截取最新窗口，避免 UI 显示 2014/2012 等最早样本。当前量价指标中的 MACD、RSI、均线状态主要由 LLM 从 K 线数据推理，没有在代码中用确定性公式预计算；如果主源失败但备用源成功，页面会展示 `source_warning`，如果主源和备用源都失败，量价通常回到中性 50 分。

### 2.3 舆情 Agent

目标: 判断个股关注度、市场情绪和新闻影响。

| MCP 工具 | 底层 AKShare API | 目标数据 | 用途 |
|----------|------------------|----------|------|
| `get_stock_sentiment` | `ak.stock_hot_rank_em` | 东方财富人气排名 | 个股关注度 |
| `get_stock_sentiment` | `ak.stock_hot_keyword_em` | 热门关键词 | 讨论主题 |
| `get_stock_sentiment` | `ak.stock_comment_detail_zhpj_lspf_em` | 千股千评综合评价 | 个股情绪评分 |
| `get_stock_sentiment` | `ak.stock_comment_detail_scrd_focus_em` | 用户关注指数 | 关注度趋势 |
| `get_stock_sentiment` | `ak.stock_comment_detail_scrd_desire_em` | 市场参与意愿 | 交易意愿 |
| `get_stock_sentiment` | `ak.stock_comment_detail_zlkp_jgcyd_em` | 机构参与度 | 机构侧信号 |
| `get_stock_news` | `ak.stock_news_em` | 个股新闻 | 事件影响 |
| `get_stock_market_emotion` | `ak.stock_comment_em` | 全市场千股千评分布 | 市场整体情绪 |
| `get_stock_market_emotion` | `ak.stock_a_congestion_lg` | 拥挤度 | 市场过热/冷清 |
| `get_stock_market_emotion` | `ak.stock_buffett_index_lg` | 巴菲特指标 | 市场估值状态 |
| `get_stock_market_emotion` | `ak.stock_zt_pool_em` | 涨停池 | 短线风险偏好 |

注意:
千股千评明细、用户关注指数、机构参与度等子接口取自东方财富特色数据，接口本身可能阶段性停更或只返回源站最新可用日期。MCP 会展示 `*_latest_date` 和 `*_note`，如果最新日期明显滞后，应把它视为历史舆情参考，而不是实时情绪。拥挤度、巴菲特指标等长历史接口同样会先截取最新窗口；如果乐咕乐股字段或页面结构变化，MCP 会尽量保留可用字段，并通过 `*_note` 或 `*_error` 说明数据质量。

### 2.4 宏观 Agent

当前状态: 默认暂停。宏观数据仍保留 MCP 工具和 Agent 实现，但普通个股分析、宏观分析入口不会主动调用；UI 会展示“宏观 Agent 已暂停”。暂停期间，综合评分只使用基本面、量价、舆情，并按原风格权重重新归一。

目标: 判断经济周期、货币环境和产业景气。

| MCP 工具 | 底层 AKShare API | 目标数据 | 用途 |
|----------|------------------|----------|------|
| `get_macro_china_overview` | `ak.macro_china_gdp_yearly` | GDP 增速 | 经济周期 |
| `get_macro_china_overview` | `ak.macro_china_cpi_yearly` / `ak.macro_china_cpi_monthly` | CPI 同比/环比 | 通胀环境 |
| `get_macro_china_overview` | `ak.macro_china_ppi_yearly` | PPI 同比 | 工业价格周期 |
| `get_macro_china_overview` | `ak.macro_china_pmi_yearly` | 官方 PMI | 扩张/收缩判断 |
| `get_macro_china_overview` | `ak.index_pmi_com_cx` | 财新 PMI | 民企/制造业景气辅助 |
| `get_macro_china_overview` | `ak.macro_china_m2_yearly` | M2 增速 | 流动性环境 |
| `get_macro_china_overview` | `ak.macro_china_lpr` | 1年期/5年期 LPR | 中国利率与信用环境 |
| `get_macro_china_overview` | `ak.macro_china_exports_yoy` / `ak.macro_china_imports_yoy` | 进出口同比 | 外需与内需 |
| `get_macro_china_overview` | `ak.macro_china_trade_balance` | 贸易差额 | 外贸压力 |
| `get_macro_china_overview` | `ak.macro_china_industrial_production_yoy` | 工业增加值 | 生产景气 |
| `get_macro_china_overview` | `ak.macro_china_urban_unemployment` | 城镇失业率 | 就业压力 |
| `get_macro_global_interest` | 多国央行利率 API | 美国、欧元区、中国、日本等利率 | 全球货币环境 |

注意:
多数中国宏观和全球利率数据来自金十经济日历，接口会包含“公布日期已有、但今值尚未填充”的记录。MCP 会按日期查找最新的非空 `今值`，并把 `latest.status`、`value_col` 和 `note` 一起展示；如果源站本身只返回到较旧日期，则页面会标注为数据源滞后。财新 PMI 使用 `综合PMI` 作为实际值，`变化值` 只作为环比变化参考。中国利率会同时展示金十央行决议报告和东方财富 LPR，其中 LPR 更适合判断当前信用利率环境。市场估值 PE/PB/全A PB 等长历史接口会截取最新窗口，而不是最早样本。

## 3. 评分流程

### 3.0 数据时效闸门

每个 Agent 在调用 LLM 前都会先执行 `agents.data_freshness.apply_freshness_policy`。它会递归识别常见日期字段，例如 `日期`、`数据日期`、`REPORT_DATE`、`发布时间`、`交易日`、`latest.date` 等，并为每个数据块生成时效检查结果。

默认阈值:

| Agent | 核心启用条件 | 主要阈值 |
|-------|--------------|----------|
| 基本面 | 财务指标/资产负债表/利润表/现金流中至少一个核心财报块新鲜 | 财报 240 天，估值 30 天，盈利预测 180 天 |
| 量价 | K 线量价必须新鲜 | K 线 10 天，资金流 10 天，龙虎榜 45 天 |
| 舆情 | 个股舆情/新闻/市场情绪至少一个块新鲜 | 新闻 14 天，市场情绪 14 天，个股舆情 30 天 |
| 宏观 | 中国宏观数据必须新鲜 | 宏观 120 天，全球利率 180 天，市场估值 30 天 |

如果某个 Agent 的核心数据不达标，该 Agent 会返回“未启用”结构化结果，保留原始数据快照，但不调用 LLM，也不参与最终综合得分。如果只是某个辅助数据块滞后，系统会把该块从 prompt 中移除，并在 `stale_raw` 中保留审计样本。

### 3.1 子 Agent 内部分数

各子 Agent 的内部评分目前主要由 LLM 按 prompt 规则生成。也就是说，代码负责取数、整理数据、约束输出结构；LLM 负责判断各维度分数。

| Agent | 子维度 | 子维度权重 |
|-------|--------|------------|
| 基本面 | 财务健康度、估值合理性、盈利成长性 | 40% / 35% / 25% |
| 量价 | 趋势判断、动量信号、资金流 | 30% / 25% / 45% |
| 舆情 | 市场情绪、社交热度、市场整体情绪 | 35% / 30% / 35% |
| 宏观 | 经济周期、货币环境、产业景气 | 40% / 35% / 25%（默认暂停） |

当前边界:
这些子维度分数不是完全由代码公式计算出来的，而是由 LLM 根据数据和评分规则返回。因此同一组数据在不同模型或不同 prompt 下可能有轻微波动。

### 3.2 综合得分

最终综合得分由代码确定性计算，位置在 `orchestrator/decision_agent.py::compute_weighted_score`。只有通过数据时效闸门的维度会参与最终评分；未通过的维度不会以 50 分中性值占位。

当前默认宏观暂停时，均衡型:

```text
综合得分 = 基本面 * 33.3% + 量价 * 33.3% + 舆情 * 33.3%
```

如果重新启用宏观 Agent，则恢复四维权重:

| 风格 | 基本面 | 量价 | 舆情 | 宏观 |
|------|--------|------|------|------|
| 价值 | 40% | 10% | 15% | 35% |
| 成长 | 30% | 25% | 20% | 25% |
| 均衡 | 25% | 25% | 25% | 25% |
| 主题 | 15% | 30% | 35% | 20% |

宏观暂停或任意维度时效不达标时，对应权重置为 0，其余有效维度按上述权重重新归一。例如价值型在宏观暂停时会从 `40/10/15/35` 变为约 `61.5/15.4/23.1/0`；如果量价也因 K 线滞后被剔除，则只在基本面和舆情之间重新分配。

投资周期会进一步调整:

| 周期 | 调整逻辑 |
|------|----------|
| 短线 | 提高量价权重，降低基本面权重 |
| 中线 | 不调整 |
| 长线 | 提高基本面权重，降低量价权重 |

### 3.3 风险过滤

综合得分计算后，会进入风险过滤器 `orchestrator/risk_filter.py`。

当前已实现规则:

| 风险信号 | 处理 |
|----------|------|
| PE(TTM) < 0 | 综合评分乘以 0.8 |
| PE(TTM) > 200 | 综合评分乘以 0.9 |
| 日成交额 < 1000 万 | 综合评分乘以 0.9 |
| 日成交额 1000 万到 3000 万 | 只提示，不扣分 |

当前仍偏弱的部分:
ST、股权质押、商誉占比、行业回避、保守/激进偏好匹配还没有完整数据闭环，部分是占位逻辑。

### 3.4 建议等级

风险过滤后的总分映射为建议等级:

| 分数 | 建议 |
|------|------|
| 85-100 | 强烈推荐 |
| 70-84 | 积极关注 |
| 55-69 | 适度配置 |
| 40-54 | 谨慎观望 |
| 0-39 | 回避 |

## 4. 当前透明度问题

当前工作流已经能跑通，但确实有几处不够透明或容易显得随意:

1. 子 Agent 的分数由 LLM 判断，不是完全可复现的代码公式。
2. 输出里没有展示“本次实际调用了哪些工具、每个工具是否成功、返回多少条数据”。
3. 量价指标没有预计算 MACD、RSI、均线等确定性指标，导致数据缺失时只能中性处理。
4. 基本面存在银行、券商、制造业等行业差异，但评分规则目前比较通用。
5. 风险过滤规则还不完整，和 README 中的目标设计有差距。

## 5. 建议的透明化改造

当前 UI 已经展示 `analysis_data` 数据快照，包含每个子 Agent 的得分摘要、原始数据行数、字段样本和接口错误。后续可以进一步升级为更完整的 `analysis_trace`，随每次 `Recommendation` 一起返回:

```json
{
  "symbol": "000001",
  "tools_called": [
    {
      "tool": "get_stock_fundamental",
      "arguments": {"symbol": "000001", "report_type": "valuation"},
      "status": "ok",
      "rows": 30,
      "fields": ["PE(TTM)", "市净率", "PEG值"]
    }
  ],
  "dimension_formula": {
    "fundamental": "LLM评分: 财务健康度40% + 估值35% + 成长25%",
    "freshness_gate": "代码校验: 核心数据超过阈值时，该 Agent 不调用 LLM 且不参与评分",
    "final": "代码加权: 只对通过时效校验的有效维度重新分配权重"
  },
  "data_quality": [
    "K线接口失败: push2his.eastmoney.com DNS解析失败",
    "业绩预告未找到目标股票样本"
  ]
}
```

这个对象可以直接展示在 UI 里，用户就能看到每一次建议背后的数据来源、成功/失败情况和得分路径。
