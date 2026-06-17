"""
Investment Advisor Agent — Streamlit 交互界面

启动方式:
  streamlit run ui/app.py

功能:
  - 侧边栏: 投资者偏好设置
  - 主页: 个股分析面板
  - 对比分析: 两只股票对比
  - 宏观分析: 宏观景气度总览
"""

import sys
import os
import json
import logging
import html
import re
from datetime import datetime

# 确保项目根目录在 path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

from agents.ai_analysis_report import build_ai_analysis_report
from agents.dcf_analysis_adapter import chat_with_dcf_analysis, create_dcf_system, run_dcf_analysis
from agents.relative_valuation_adapter import (
    get_model_metrics,
    get_prediction_artifact_stats,
    get_relative_valuation,
)
from models import (
    UserProfile,
    RiskTolerance,
    InvestmentHorizon,
    InvestmentStyle,
    CapitalSize,
    Recommendation,
    AdviceLevel,
)
from orchestrator.planner import parse_intent
from orchestrator.scheduler import analyze_stock, run_query
from agents.data_provider import DataProvider
from config.settings import MACRO_AGENT_DISABLED_REASON, MACRO_AGENT_ENABLED, get_effective_weights

# ──────────────────────────────────────────────
# 页面配置
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="AI 投资顾问",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO)


def inject_sneat_theme():
    """Apply a Sneat-like visual layer inspired by valuation_agent_dashboard.html."""
    st.markdown(
        """
        <style>
        :root {
            --rq-bg: #f5f5f9;
            --rq-card: #ffffff;
            --rq-text: #566a7f;
            --rq-heading: #384551;
            --rq-primary: #696cff;
            --rq-primary-soft: #e7e7ff;
            --rq-success: #71dd37;
            --rq-warning: #ffab00;
            --rq-danger: #ff3e1d;
            --rq-border: #d9dee3;
            --rq-shadow: 0 0.125rem 0.55rem rgba(67, 89, 113, 0.13);
        }

        .stApp {
            background: var(--rq-bg);
            color: var(--rq-text);
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stSidebar"] {
            background: var(--rq-card);
            border-right: 1px solid rgba(67, 89, 113, 0.08);
            box-shadow: 0.125rem 0 0.5rem rgba(67, 89, 113, 0.06);
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.55rem;
        }

        .rq-brand {
            align-items: center;
            color: var(--rq-heading);
            display: flex;
            font-weight: 760;
            gap: 0.75rem;
            margin: 0.3rem 0 1.35rem;
            padding: 0 0.45rem;
        }

        .rq-brand-mark {
            align-items: center;
            background: var(--rq-primary);
            border-radius: 0.75rem;
            color: white;
            display: grid;
            font-weight: 800;
            height: 2.5rem;
            place-items: center;
            width: 2.5rem;
        }

        .rq-brand-subtitle {
            color: #a1acb8;
            display: block;
            font-size: 0.78rem;
            font-weight: 600;
            margin-top: 0.1rem;
        }

        .rq-menu-label {
            color: #a1acb8;
            font-size: 0.75rem;
            font-weight: 760;
            letter-spacing: 0.08em;
            margin: 1.2rem 0 0.35rem;
            padding: 0 0.45rem;
            text-transform: uppercase;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            align-items: center;
            background: #fff;
            border: 0 !important;
            border-radius: 0.9rem;
            color: #2f3542;
            display: flex;
            font-size: 1.02rem;
            font-weight: 650;
            margin: 0.22rem 0;
            min-height: 2.85rem;
            padding: 0.62rem 0.78rem !important;
            transition: background-color 160ms ease, color 160ms ease, box-shadow 160ms ease;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stRadio"] {
            display: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: var(--rq-primary-soft);
            color: var(--rq-primary);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--rq-primary-soft);
            box-shadow: none;
            color: var(--rq-primary);
            font-weight: 760;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
            display: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover > div:first-child,
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) > div:first-child {
            display: none !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label svg,
        [data-testid="stSidebar"] div[role="radiogroup"] label [style*="rgb(255, 75, 75)"],
        [data-testid="stSidebar"] div[role="radiogroup"] label [style*="#ff4b4b"] {
            color: var(--rq-primary) !important;
            fill: var(--rq-primary) !important;
            stroke: var(--rq-primary) !important;
        }

        .rq-sidebar-link {
            align-items: center;
            background: #fff;
            border-radius: 0.9rem;
            color: #566a7f !important;
            display: flex;
            font-size: 1.02rem;
            font-weight: 650;
            gap: 0.85rem;
            margin: 0.28rem 0;
            padding: 0.78rem 0.85rem;
            text-decoration: none !important;
            transition: background-color 160ms ease, color 160ms ease;
        }

        .rq-sidebar-link:hover,
        .rq-sidebar-link.active {
            background: var(--rq-primary-soft);
            color: var(--rq-primary) !important;
        }

        .rq-sidebar-icon {
            align-items: center;
            display: grid;
            font-size: 1.05rem;
            height: 1.45rem;
            place-items: center;
            width: 1.45rem;
        }

        .rq-sidebar-divider {
            border-top: 1px solid rgba(67, 89, 113, 0.16);
            margin: 1.6rem 0 1.25rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        h1, h2, h3, h4, h5, h6 {
            color: var(--rq-heading);
            font-weight: 750;
            letter-spacing: -0.01em;
        }

        .block-container {
            padding-top: 1.15rem;
            max-width: 1380px;
        }

        [data-testid="stHeader"] {
            background: rgba(245, 245, 249, 0.72);
            backdrop-filter: blur(8px);
        }

        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"],
        div[data-testid="stExpander"],
        div[data-testid="stAlert"] {
            border-radius: 1rem;
        }

        div[data-testid="stMetric"] {
            background: var(--rq-card);
            border: 0;
            box-shadow: var(--rq-shadow);
            padding: 1.15rem 1.25rem;
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--rq-text);
            font-size: 0.88rem;
        }

        div[data-testid="stMetricValue"] {
            color: var(--rq-heading);
            font-weight: 780;
        }

        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]),
        div[data-testid="stPlotlyChart"],
        div[data-testid="stJson"],
        div[data-testid="stCodeBlock"] {
            background: var(--rq-card);
            border-radius: 1rem;
            box-shadow: var(--rq-shadow);
            padding: 0.35rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.72);
            border-radius: 0.875rem;
            box-shadow: var(--rq-shadow);
            gap: 0.25rem;
            padding: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0.7rem;
            color: var(--rq-text);
            font-weight: 650;
            padding: 0.65rem 1rem;
        }

        .stTabs [aria-selected="true"] {
            background: var(--rq-primary-soft);
            color: var(--rq-primary);
        }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--rq-primary);
            border: 1px solid var(--rq-primary);
            border-radius: 0.75rem;
            box-shadow: 0 0.125rem 0.35rem rgba(105, 108, 255, 0.35);
            color: #fff;
            font-weight: 700;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: #5f61e6;
            border-color: #5f61e6;
            color: #fff;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div,
        textarea {
            border-color: var(--rq-border);
            border-radius: 0.75rem;
        }

        div[data-testid="stCaptionContainer"],
        .stMarkdown p {
            color: var(--rq-text);
        }

        .rq-card {
            background: var(--rq-card);
            border-radius: 1rem;
            box-shadow: var(--rq-shadow);
            padding: 1.35rem;
            margin-bottom: 1rem;
        }

        .rq-topbar {
            align-items: center;
            background: rgba(255, 255, 255, 0.88);
            border-radius: 0.875rem;
            box-shadow: var(--rq-shadow);
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.25rem;
            padding: 1rem 1.25rem;
        }

        .rq-topbar-title {
            color: var(--rq-heading);
            font-size: 1.35rem;
            font-weight: 800;
            line-height: 1.25;
            margin: 0;
        }

        .rq-topbar-subtitle {
            color: var(--rq-text);
            font-size: 0.92rem;
            margin-top: 0.2rem;
        }

        .rq-card-title {
            color: var(--rq-heading);
            font-size: 1.08rem;
            font-weight: 760;
            margin-bottom: 0.45rem;
        }

        .rq-soft-badge {
            background: var(--rq-primary-soft);
            border-radius: 999px;
            color: var(--rq-primary);
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 760;
            padding: 0.35rem 0.7rem;
        }

        .rq-search-card {
            align-items: center;
            background: var(--rq-card);
            border-radius: 1rem;
            box-shadow: var(--rq-shadow);
            display: block;
            margin: 1rem 0 1.25rem;
            padding: 1.1rem 1.25rem 0.65rem;
        }

        div[data-testid="stForm"] {
            background: var(--rq-card);
            border: 0;
            border-radius: 1rem;
            box-shadow: var(--rq-shadow);
            margin: 1rem 0 1.25rem;
            padding: 1.1rem 1.25rem 0.65rem;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] label {
            display: none;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] input {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            color: var(--rq-text);
            font-size: 1.12rem;
            font-weight: 650;
            padding-left: 0;
            outline: none !important;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
            color: #767b86;
            opacity: 0.9;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus,
        div[data-testid="stForm"] div[data-testid="stTextInput"] input:active,
        div[data-testid="stForm"] div[data-testid="stTextInput"] input:invalid,
        div[data-testid="stForm"] div[data-testid="stTextInput"] input[aria-invalid="true"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        div[data-testid="stForm"] div[data-testid="stTextInput"] [data-baseweb="input"],
        div[data-testid="stForm"] div[data-testid="stTextInput"] [data-baseweb="base-input"],
        div[data-testid="stForm"] div[data-testid="stTextInput"] [data-baseweb="input"] > div,
        div[data-testid="stForm"] div[data-testid="stTextInput"] [data-baseweb="base-input"] > div {
            background: transparent !important;
            border: 0 !important;
            border-color: transparent !important;
            box-shadow: none !important;
            outline: none !important;
        }

        div[data-testid="stForm"] div[data-testid="InputInstructions"] {
            display: none !important;
        }

        div[data-testid="stForm"] div[aria-invalid="true"],
        div[data-testid="stForm"] [data-invalid="true"] {
            border-color: transparent !important;
            box-shadow: none !important;
        }

        .rq-lookup-row div[data-testid="stTextInput"] input {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            color: var(--rq-text);
            font-size: 1.12rem;
            font-weight: 650;
            padding-left: 0;
        }

        .rq-lookup-row div[data-testid="stTextInput"] input:focus,
        .rq-lookup-row div[data-testid="stTextInput"] input:active,
        .rq-lookup-row div[data-testid="stTextInput"] input:invalid {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            outline: none !important;
        }

        .rq-lookup-row div[data-testid="stTextInput"] [data-baseweb="input"],
        .rq-lookup-row div[data-testid="stTextInput"] [data-baseweb="base-input"] {
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }

        .rq-lookup-row div[data-testid="InputInstructions"] {
            display: none !important;
        }

        div[data-testid="stForm"] .stButton,
        div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
            display: none;
        }

        .rq-search-icon-only {
            color: #566a7f;
            font-size: 2rem;
            line-height: 2.45rem;
            padding-left: 0.25rem;
            text-align: center;
        }

        .rq-search-badge {
            background: var(--rq-primary-soft);
            border-radius: 999px;
            color: var(--rq-primary);
            font-size: 1rem;
            font-weight: 800;
            padding: 0.4rem 0.8rem;
            text-align: center;
            white-space: nowrap;
        }

        .rq-pipeline-step {
            border-left: 3px solid var(--rq-primary);
            padding: 0.15rem 0 0.15rem 1rem;
            margin-bottom: 1rem;
        }

        .rq-dcf-input-anchor {
            border: 0;
            height: 0;
            margin: 0;
            padding: 0;
        }

        div[data-testid="stVerticalBlock"]:has(.rq-dcf-input-anchor) {
            background:
                radial-gradient(circle at top left, rgba(105, 108, 255, 0.16), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f7f7ff 100%);
            border: 1px solid rgba(105, 108, 255, 0.16);
            border-left: 5px solid var(--rq-primary);
            border-radius: 1.25rem;
            box-shadow: 0 0.35rem 1.2rem rgba(67, 89, 113, 0.15);
            margin: 1rem 0 1.35rem;
            padding: 1.25rem 1.35rem 1.1rem;
        }

        .rq-dcf-input-title {
            align-items: center;
            display: flex;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
        }

        .rq-dcf-input-icon {
            align-items: center;
            background: var(--rq-primary);
            border-radius: 0.85rem;
            color: white;
            display: grid;
            font-weight: 850;
            height: 2.35rem;
            place-items: center;
            width: 2.35rem;
        }

        .rq-dcf-input-title strong {
            color: var(--rq-heading);
            display: block;
            font-size: 1.08rem;
            font-weight: 820;
        }

        .rq-dcf-input-title span {
            color: #8b95a4;
            display: block;
            font-size: 0.86rem;
            margin-top: 0.1rem;
        }

        div[data-testid="stVerticalBlock"]:has(.rq-dcf-input-anchor) div[data-testid="stTextInput"] input,
        div[data-testid="stVerticalBlock"]:has(.rq-dcf-input-anchor) div[data-testid="stNumberInput"] input {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(105, 108, 255, 0.10);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }

        div[data-testid="stVerticalBlock"]:has(.rq-dcf-input-anchor) div[data-testid="stTextInput"] input:focus,
        div[data-testid="stVerticalBlock"]:has(.rq-dcf-input-anchor) div[data-testid="stNumberInput"] input:focus {
            border-color: var(--rq-primary);
            box-shadow: 0 0 0 0.18rem rgba(105, 108, 255, 0.14);
        }

        .ia-hero {
            background:
                radial-gradient(circle at top left, rgba(105, 108, 255, 0.18), transparent 30%),
                linear-gradient(135deg, #ffffff 0%, #f7f8ff 100%);
            border: 1px solid rgba(105, 108, 255, 0.14);
            border-radius: 1.35rem;
            box-shadow: 0 0.5rem 1.5rem rgba(67, 89, 113, 0.13);
            margin: 1rem 0 1.35rem;
            padding: 1.3rem 1.45rem;
        }

        .ia-hero-title {
            align-items: center;
            color: var(--rq-heading);
            display: flex;
            font-size: 1.2rem;
            font-weight: 830;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
        }

        .ia-hero-mark {
            align-items: center;
            background: var(--rq-primary);
            border-radius: 0.9rem;
            color: #fff;
            display: grid;
            font-weight: 850;
            height: 2.45rem;
            place-items: center;
            width: 2.45rem;
        }

        .ia-hero-subtitle {
            color: #8b95a4;
            font-size: 0.94rem;
            line-height: 1.7;
            margin: 0 0 1rem;
        }

        .ia-feature-grid {
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin-top: 1rem;
        }

        .ia-feature-card {
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid rgba(67, 89, 113, 0.08);
            border-radius: 1rem;
            padding: 1rem;
        }

        .ia-feature-card strong {
            color: var(--rq-heading);
            display: block;
            font-size: 1rem;
            margin-bottom: 0.3rem;
        }

        .ia-feature-card span {
            color: #8b95a4;
            font-size: 0.88rem;
            line-height: 1.55;
        }

        .ia-example-chip {
            background: var(--rq-primary-soft);
            border-radius: 999px;
            color: var(--rq-primary);
            display: inline-block;
            font-weight: 760;
            margin: 0.25rem 0.35rem 0 0;
            padding: 0.35rem 0.7rem;
        }

        @media (max-width: 992px) {
            .rq-topbar {
                align-items: stretch;
                flex-direction: column;
            }

            .ia-feature-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str = "", badge: str = ""):
    badge_html = f"<span class='rq-soft-badge'>{badge}</span>" if badge else ""
    st.markdown(
        f"""
        <div class="rq-topbar">
          <div>
            <h1 class="rq-topbar-title">{title}</h1>
            <div class="rq-topbar-subtitle">{subtitle}</div>
          </div>
          <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand():
    st.sidebar.markdown(
        """
        <div class="rq-brand">
          <div class="rq-brand-mark">IA</div>
          <div>
            <div>Investment Advisor</div>
            <span class="rq-brand-subtitle">AI Valuation Agent</span>
          </div>
        </div>
        <div class="rq-menu-label">Dashboard</div>
        """,
        unsafe_allow_html=True,
    )


def render_fair_multiple_sidebar():
    st.sidebar.markdown(
        """
        <div class="rq-menu-label">Fair Multiple</div>
        <a class="rq-sidebar-link active" href="#fair-overview">
          <span class="rq-sidebar-icon">▦</span><span>Overview</span>
        </a>
        <a class="rq-sidebar-link" href="#fair-search">
          <span class="rq-sidebar-icon">⌕</span><span>Valuation Search</span>
        </a>
        <a class="rq-sidebar-link" href="#fair-checks">
          <span class="rq-sidebar-icon">▤</span><span>Report Checks</span>
        </a>
        <a class="rq-sidebar-link" href="#fair-pipeline">
          <span class="rq-sidebar-icon">⌘</span><span>Pipeline</span>
        </a>
        <a class="rq-sidebar-link" href="#fair-outputs">
          <span class="rq-sidebar-icon">▦</span><span>Outputs</span>
        </a>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# 侧边栏 — 投资者偏好
# ──────────────────────────────────────────────

def render_sidebar() -> UserProfile:
    """渲染侧边栏，返回用户偏好"""
    st.sidebar.markdown("<div class='rq-menu-label'>Preferences</div>", unsafe_allow_html=True)

    risk = st.sidebar.selectbox(
        "风险承受",
        options=[e.value for e in RiskTolerance],
        index=1,  # 默认"稳健"
        help="影响建议的激进程度和仓位推荐",
    )

    horizon = st.sidebar.selectbox(
        "投资周期",
        options=[e.value for e in InvestmentHorizon],
        index=1,  # 默认"中线"
        help="影响量价维度的权重: 短线看重量价，长线看重基本面",
    )

    style = st.sidebar.selectbox(
        "投资风格",
        options=[e.value for e in InvestmentStyle],
        index=2,  # 默认"均衡"
        help="影响各维度权重: 价值型重基本面，主题型重舆情",
    )

    capital = st.sidebar.selectbox(
        "资金规模",
        options=[e.value for e in CapitalSize],
        index=1,  # 默认"中"
        help="影响仓位建议；宏观 Agent 暂停期间不影响宏观权重",
    )

    st.sidebar.divider()

    preferred_sectors = st.sidebar.multiselect(
        "偏好行业/概念",
        options=["新能源", "半导体", "消费电子", "白酒", "医药", "银行",
                 "地产", "军工", "人工智能", "机器人", "汽车", "光伏"],
        default=[],
        help="偏好行业的个股会获得小幅加分",
    )

    avoid_sectors = st.sidebar.multiselect(
        "回避行业/概念",
        options=["ST", "次新股", "房地产", "影视", "教育"],
        default=[],
        help="回避行业的个股会被排除或大幅降级",
    )

    st.sidebar.divider()

    max_position = st.sidebar.slider(
        "单股最大仓位 (%)",
        min_value=5, max_value=30, value=15, step=5,
    )

    stop_loss = st.sidebar.slider(
        "止损线 (%)",
        min_value=-15, max_value=-3, value=-8, step=1,
    )

    # 构建 UserProfile
    risk_enum = next(e for e in RiskTolerance if e.value == risk)
    horizon_enum = next(e for e in InvestmentHorizon if e.value == horizon)
    style_enum = next(e for e in InvestmentStyle if e.value == style)
    capital_enum = next(e for e in CapitalSize if e.value == capital)

    profile = UserProfile(
        risk_tolerance=risk_enum,
        investment_horizon=horizon_enum,
        style=style_enum,
        capital_size=capital_enum,
        sectors_preference=preferred_sectors,
        avoid_sectors=avoid_sectors,
        max_position_per_stock=float(max_position),
        stop_loss_threshold=float(stop_loss),
    )

    # 显示当前实际权重配置
    weights = get_effective_weights(style, horizon, include_macro=MACRO_AGENT_ENABLED)
    caption = (
        f"当前权重: 基本面{weights['fundamental']:.0%} | "
        f"量价{weights['technical']:.0%} | "
        f"舆情{weights['sentiment']:.0%}"
    )
    if MACRO_AGENT_ENABLED:
        caption += f" | 宏观{weights['macro']:.0%}"
    else:
        caption += " | 宏观已暂停"
    st.sidebar.caption(caption)

    return profile


# ──────────────────────────────────────────────
# 主页 — 个股分析
# ──────────────────────────────────────────────

def render_main_page(profile: UserProfile):
    """渲染主页: 个股分析面板"""
    render_page_header(
        "AI 投资顾问",
        "聚焦个股分析：融合基本面、量价、舆情、行业相对表现与分析师一致预期。",
        "Stock Agent",
    )

    with st.container():
        st.markdown(
            """
            <div class="ia-hero">
              <div class="ia-hero-title">
                <div class="ia-hero-mark">IA</div>
                <div>输入个股，生成多维投顾分析</div>
              </div>
              <p class="ia-hero-subtitle">
                当前主面板仅支持单只股票分析。请输入 6 位股票代码、RQData order_book_id 或常见股票名称。
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "输入股票代码或名称",
                placeholder="例如: 000001、000001.XSHE、比亚迪",
                key="stock_query",
            )
        with col2:
            analyze_btn = st.button("开始个股分析", width="stretch", type="primary")

    if not query and not analyze_btn:
        _render_guide()
        return

    if not query:
        st.warning("请输入股票代码或名称")
        return

    if not _looks_like_stock_query(query):
        st.info("当前投顾分析主面板仅支持单只股票。请直接输入股票代码或常见股票名称，例如 `000001`、`000001.XSHE`、`比亚迪`。")
        return

    # 执行分析
    with st.spinner("正在获取数据并进行 AI 分析，请稍候..."):
        try:
            dp = DataProvider()
            result = run_query(query, profile, dp)

            intent_type = result.get("type", "general")

            if intent_type == "stock_analysis":
                rec_data = result.get("recommendation", {})
                rec = Recommendation(**rec_data) if isinstance(rec_data, dict) and "symbol" in rec_data else None
                if rec:
                    _render_recommendation(rec)
                    _render_consensus_card(rec.symbol, dp)
                    _render_technical_timeseries_card(rec.symbol, dp)
                    _render_integrated_valuation_models(rec.symbol, rec.name)
                    ai_tab, industry_tab, data_tab = st.tabs(["AI分析", "行业比较", "数据快照"])
                    with ai_tab:
                        _render_ai_analysis_page(rec.symbol, dp, result.get("analysis_data", {}))
                    with industry_tab:
                        _render_industry_scorecard(rec.symbol, dp)
                    with data_tab:
                        _render_initial_data(result.get("analysis_data", {}))
                else:
                    st.error("分析结果解析失败，请重试")

            elif intent_type == "comparison":
                st.info("当前主面板暂只支持单只股票分析。请输入一个股票代码或名称，例如 `000001`。")

            elif intent_type == "sector_analysis":
                st.info("当前主面板暂不展示板块分析。请输入具体个股代码或名称进行分析。")

            elif intent_type == "macro_analysis":
                st.info("当前主面板暂不展示宏观分析。请输入具体个股代码或名称进行分析。")

            elif intent_type == "screening":
                st.info("当前主面板暂不提供选股筛选。请输入具体个股代码或名称进行分析。")

            else:
                st.info(result.get("message", "请输入具体股票代码或名称进行分析"))

        except Exception as e:
            st.error(f"分析过程中出现错误: {e}")
            with st.expander("查看详细错误信息"):
                st.code(str(e))


# ──────────────────────────────────────────────
# 本地 DCF 估值页面
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_cached_dcf_system():
    return create_dcf_system()


def render_dcf_page():
    """渲染基于 program.py 的本地 DCF 估值页面。"""
    render_page_header(
        "DCF 估值",
        "接入顶层 program.py 的 ML 现金流预测、DCF、护城河和财务健康度分析。",
        "DCF Engine",
    )

    with st.expander("运行说明", expanded=False):
        st.markdown(
            """
            - 输入可以是公司名、6 位股票代码或 RQData `order_book_id`，例如 `神开股份`、`002278`、`002278.XSHE`。
            - 首次运行会初始化 rqdatac 并加载本地 `/model` 目录下的模型文件，耗时会更长。
            - LLM 报告沿用 `program.py` 的配置；未配置 API Key 时，页面仍会展示 DCF、预测、健康度和护城河结构化结果。
            """
        )

    with st.container():
        st.markdown(
            """
            <div class="rq-dcf-input-anchor"></div>
            <div class="rq-dcf-input-title">
              <div class="rq-dcf-input-icon">DCF</div>
              <div>
                <strong>估值参数输入</strong>
                <span>输入标的并调整核心假设，系统将调用 program.py 完成现金流估值。</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_cols = st.columns([2.2, 1, 1])
        with input_cols[0]:
            company_name = st.text_input(
                "公司名/股票代码",
                placeholder="例如: 神开股份、002278、000001.XSHE",
                key="dcf_company_name",
            )
        with input_cols[1]:
            discount_rate = st.number_input(
                "折现率",
                min_value=0.01,
                max_value=0.50,
                value=0.12,
                step=0.01,
                format="%.2f",
                help="DCF 使用的 required return / discount rate。",
            )
        with input_cols[2]:
            terminal_growth = st.number_input(
                "永续增长率",
                min_value=-0.05,
                max_value=0.10,
                value=0.03,
                step=0.005,
                format="%.3f",
                help="Gordon Growth 终值模型中的长期增长率。",
            )

        action_cols = st.columns([1.2, 2.8])
        with action_cols[0]:
            generate_llm_report = st.checkbox(
                "生成 AI 文本报告",
                value=True,
                help="勾选后会调用 program.py 的 LLMAnalyzer 生成中文报告；取消勾选时仍会生成完整结构化 DCF 结果。",
            )
        with action_cols[1]:
            analyze_btn = st.button("开始 DCF 估值", type="primary", width="stretch")

    if terminal_growth >= discount_rate:
        st.warning("永续增长率必须低于折现率，否则终值公式不可用。")
        return

    if not analyze_btn and "dcf_payload" not in st.session_state:
        _render_dcf_guide()
        return

    if analyze_btn and not company_name.strip():
        st.warning("请输入公司名或股票代码。")
        return

    if analyze_btn:
        try:
            with st.spinner("正在运行 DCF 估值流水线，请稍候..."):
                system = _get_cached_dcf_system()
                payload = run_dcf_analysis(
                    system,
                    company_name.strip(),
                    discount_rate=discount_rate,
                    terminal_growth=terminal_growth,
                    generate_llm_report=generate_llm_report,
                )
                st.session_state["dcf_payload"] = payload
                st.session_state["dcf_chat_messages"] = []
                st.session_state["dcf_chat_context"] = payload.get("chat_context", [])
        except Exception as exc:
            st.error(f"DCF 估值运行失败: {exc}")
            with st.expander("排查建议", expanded=True):
                st.markdown(
                    """
                    - 确认 rqdatac 已登录且本机可以访问米筐数据。
                    - 确认项目根目录 `/model` 下的随机森林、GBR 和 selector 模型文件存在。
                    - 若输入公司名无法识别，可尝试 6 位股票代码或 `000001.XSHE` 格式。
                    """
                )
                st.code(str(exc))
            return

    payload = st.session_state.get("dcf_payload")
    if payload:
        _render_dcf_result(payload)
        _render_dcf_chat(payload)


def _render_dcf_downloads(payload: dict):
    results = payload.get("results", {})
    company = results.get("company_name") or "dcf"
    order_book_id = results.get("order_book_id") or "unknown"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"dcf_{order_book_id}_{timestamp}"

    report_md = _build_dcf_markdown(payload)
    json_text = json.dumps(_json_safe(payload), ensure_ascii=False, indent=2)

    cols = st.columns([1, 1, 3])
    with cols[0]:
        st.download_button(
            "下载报告 Markdown",
            data=report_md.encode("utf-8"),
            file_name=f"{base_name}.md",
            mime="text/markdown",
            width="stretch",
        )
    with cols[1]:
        st.download_button(
            "下载结果 JSON",
            data=json_text.encode("utf-8"),
            file_name=f"{base_name}.json",
            mime="application/json",
            width="stretch",
        )
    with cols[2]:
        st.caption(f"下载文件将保存 {company}（{order_book_id}）当前页面展示的报告和结构化结果。")


def _build_dcf_markdown(payload: dict) -> str:
    results = payload.get("results", {})
    tables = payload.get("tables", {})
    assumptions = payload.get("assumptions", {})
    health = results.get("health_assessment", {})
    moat = results.get("moat_analysis", {})

    lines = [
        f"# DCF 估值报告: {results.get('company_name', '')} ({results.get('order_book_id', '')})",
        "",
        "## 核心结论",
        f"- DCF 每股价值: {_format_dcf_value(results.get('dcf_value'))}",
        f"- 健康度评分: {_format_dcf_value(health.get('overall_score'), suffix='/100')}",
        f"- 健康状态: {health.get('overall_status', '—')}",
        f"- 系统建议: {health.get('recommendation', '—')}",
        f"- 护城河评分: {_format_dcf_value(moat.get('score'), suffix='/10')}",
        f"- 折现率: {_format_percent_value(assumptions.get('discount_rate'))}",
        f"- 永续增长率: {_format_percent_value(assumptions.get('terminal_growth'))}",
        "",
        "## 未来 5 年预测",
        _markdown_table(tables.get("prediction_table", [])),
        "",
        "## 财务健康度",
        _markdown_table(tables.get("health_table", [])),
        "",
        "## 护城河",
        f"{moat.get('moat_size', '—')} · {moat.get('durability', '—')}",
        "",
        moat.get("description", ""),
        "",
        _markdown_table(tables.get("moat_table", [])),
        "",
        "## AI 报告",
        payload.get("report", ""),
        "",
        "> 本报告由本地模型与 RQData 数据生成，仅供分析参考，不构成投资建议。",
    ]
    return "\n".join(lines)


def _markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "暂无数据"
    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.reset_index().to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if isinstance(value, float):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    return value


def _render_dcf_chat(payload: dict):
    st.divider()
    st.subheader("继续问")
    st.caption("沿用 program.py 的交互问答能力，基于本次 DCF、预测、护城河和健康度结果继续追问。")

    if "dcf_chat_messages" not in st.session_state:
        st.session_state["dcf_chat_messages"] = []
    if "dcf_chat_context" not in st.session_state:
        st.session_state["dcf_chat_context"] = payload.get("chat_context", [])

    for message in st.session_state["dcf_chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("例如: 这个 DCF 对折现率敏感吗？未来最需要跟踪什么指标？")
    if not question:
        return

    st.session_state["dcf_chat_messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在基于本次 DCF 结果回答..."):
            system = _get_cached_dcf_system()
            answer = chat_with_dcf_analysis(
                system,
                st.session_state["dcf_chat_context"],
                question,
            )
            st.markdown(answer)

    st.session_state["dcf_chat_messages"].append({"role": "assistant", "content": answer})
    st.session_state["dcf_chat_context"].extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )


def _render_dcf_guide():
    st.markdown(
        """
        ### 页面会输出什么

        | 模块 | 内容 |
        | --- | --- |
        | DCF 估值 | 基于 ML 预测的未来 5 年 FCF/Share 和终值折现 |
        | 预测表 | FCF/Share 与 EPS 的 5 年预测路径 |
        | 护城河 | 盈利能力、效率、成长和财务结构打分 |
        | 财务健康度 | 盈利、流动性、杠杆、成长、估值五维评分 |
        | AI 报告 | 使用 `program.py` 的 LLMAnalyzer 生成的文本报告 |
        """
    )


def _render_dcf_result(payload: dict):
    results = payload.get("results", {})
    tables = payload.get("tables", {})
    assumptions = payload.get("assumptions", {})
    health = results.get("health_assessment", {})
    moat = results.get("moat_analysis", {})

    st.subheader(f"{results.get('company_name', '')} · {results.get('order_book_id', '')}")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("DCF 每股价值", _format_dcf_value(results.get("dcf_value"), suffix=""))
    with metric_cols[1]:
        st.metric("健康度评分", _format_dcf_value(health.get("overall_score"), suffix="/100"))
    with metric_cols[2]:
        st.metric("健康状态", health.get("overall_status", "—"))
    with metric_cols[3]:
        st.metric("系统建议", health.get("recommendation", "—"))
    with metric_cols[4]:
        st.metric("护城河评分", _format_dcf_value(moat.get("score"), suffix="/10"))

    st.caption(
        f"估值假设: 折现率 {_format_percent_value(assumptions.get('discount_rate'))}，"
        f"永续增长率 {_format_percent_value(assumptions.get('terminal_growth'))}。"
    )
    _render_dcf_downloads(payload)

    warning_signs = health.get("warning_signs", [])
    if warning_signs:
        with st.expander("风险提示", expanded=True):
            for warning in warning_signs:
                st.warning(warning)

    tab_forecast, tab_health, tab_moat, tab_report, tab_raw = st.tabs(
        ["预测与估值", "财务健康度", "护城河", "AI报告", "运行细节"]
    )

    with tab_forecast:
        prediction_table = tables.get("prediction_table", [])
        if prediction_table:
            st.markdown("**未来 5 年预测路径**")
            _render_dataframe(prediction_table)
            _render_prediction_chart(prediction_table)

    with tab_health:
        st.markdown(f"**总体状态:** {health.get('overall_status', '—')}；**建议:** {health.get('recommendation', '—')}")
        health_table = tables.get("health_table", [])
        if health_table:
            _render_dataframe(health_table)

    with tab_moat:
        st.markdown(
            f"**{moat.get('moat_size', '—')}** · {moat.get('durability', '—')}  \n\n"
            f"{moat.get('description', '')}"
        )
        moat_table = tables.get("moat_table", [])
        if moat_table:
            _render_dataframe(moat_table)

    with tab_report:
        report = payload.get("report", "")
        if report:
            st.markdown(report)
        else:
            st.info("暂无 LLM 报告。")

    with tab_raw:
        logs = payload.get("logs", "")
        if logs:
            with st.expander("program.py 运行日志", expanded=False):
                st.code(logs)
        with st.expander("结构化结果 JSON", expanded=False):
            st.json(results, expanded=False)


def _render_prediction_chart(prediction_table: list[dict]):
    try:
        import plotly.express as px

        frame = pd.DataFrame(prediction_table)
        melted = frame.melt(id_vars="年份", var_name="指标", value_name="预测值")
        fig = px.line(melted, x="年份", y="预测值", color="指标", markers=True)
        fig.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    except Exception:
        pass


def _format_dcf_value(value, suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value or "—")


def _format_percent_value(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _format_price_value(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.2f} 元"
    except (TypeError, ValueError):
        return "—"


def _format_plain_number(value, digits: int = 2) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


# ──────────────────────────────────────────────
# 相对估值可视化页面
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_relative_stats():
    return get_prediction_artifact_stats()


@st.cache_data(show_spinner=False)
def _cached_relative_metrics():
    return get_model_metrics()


def render_relative_valuation_page():
    """渲染 Relative_prediction/valuation_api_server.py 的投顾内置版可视化。"""
    render_page_header(
        "Fair Multiple 相对估值",
        "Overview、估值查询、检查数据、Pipeline 与输出文件按顺序展示；复用 Relative_prediction 原计算函数。",
        "2020q1 - 2025q4",
    )

    try:
        stats = _cached_relative_stats()
    except Exception as exc:
        st.error(f"相对估值结果文件不可用: {exc}")
        st.info("请先运行 Relative_prediction 的 Step 1-3 工作流，生成 step3_market_implied_multiple_predictions.csv。")
        return

    st.markdown("<span id='fair-overview'></span>", unsafe_allow_html=True)
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("预测行数", f"{stats.get('rows', 0):,}")
    with stat_cols[1]:
        st.metric("覆盖公司", f"{stats.get('companies', 0):,}")
    with stat_cols[2]:
        st.metric("报告期数", f"{stats.get('quarters', 0):,}")
    with stat_cols[3]:
        st.metric("有效信号", f"{stats.get('signals', 0):,}")
    st.caption(f"数据文件: {stats.get('path', '')}")

    st.markdown("<span id='fair-search'></span>", unsafe_allow_html=True)
    with st.form("relative_lookup_form", border=False):
        search_cols = st.columns([0.16, 4.4, 0.95])
        with search_cols[0]:
            st.markdown("<div class='rq-search-icon-only'>⌕</div>", unsafe_allow_html=True)
        with search_cols[1]:
            company = st.text_input(
                "公司名 / order_book_id",
                value=st.session_state.get("relative_query", ""),
                placeholder="请输入企业名、企业号或代码",
                key="relative_company_query",
                label_visibility="collapsed",
            )
        with search_cols[2]:
            st.markdown("<div class='rq-search-badge'>2020q1 - 2025q4</div>", unsafe_allow_html=True)
        run_lookup = st.form_submit_button("查询相对估值", type="primary", width="stretch")
    should_lookup = run_lookup or (
        company.strip()
        and company.strip() != st.session_state.get("relative_query_loaded")
    )
    if should_lookup:
        if not company.strip():
            st.warning("请输入公司名或 order_book_id。")
            return
        try:
            with st.spinner("正在读取相对估值预测结果..."):
                payload = get_relative_valuation(company.strip(), quarter="latest", top_peers=5)
                st.session_state["relative_payload"] = payload
                st.session_state["relative_query"] = company.strip()
                st.session_state["relative_query_loaded"] = company.strip()
        except Exception as exc:
            st.error(f"查询失败: {exc}")
            return

    projection = "SGD"

    payload = st.session_state.get("relative_payload")
    if payload:
        metrics = _cached_relative_metrics()
        _render_relative_original_dashboard(payload, projection, stats, metrics, "Valuation Search")
        st.markdown("<span id='fair-checks'></span>", unsafe_allow_html=True)
        _render_relative_original_dashboard(payload, projection, stats, metrics, "Report Checks")
        st.markdown("<span id='fair-pipeline'></span>", unsafe_allow_html=True)
        _render_relative_original_dashboard(payload, projection, stats, metrics, "Pipeline")
        st.markdown("<span id='fair-outputs'></span>", unsafe_allow_html=True)
        _render_relative_original_dashboard(payload, projection, stats, metrics, "Outputs")
    else:
        st.info("请在上方搜索框输入公司名或 order_book_id，以加载 Step 3 fair multiple 估值结果。")


def _render_relative_original_dashboard(payload: dict, projection: str, stats: dict, metrics: list[dict], section: str):
    selected = projection.lower()
    fair_price = payload.get("fair_price_hgb_formatted") if selected == "hgb" else payload.get("fair_price_formatted")
    upside = payload.get("upside_downside_hgb_formatted") if selected == "hgb" else payload.get("upside_downside_formatted")
    upside_raw = payload.get("upside_downside_hgb") if selected == "hgb" else payload.get("upside_downside")
    sgd_fair_price = payload.get("fair_price_formatted")
    sgd_upside = payload.get("upside_downside_formatted")
    sgd_upside_raw = _safe_float_for_ui(payload.get("upside_downside"))
    hgb_fair_price = payload.get("fair_price_hgb_formatted") or payload.get("fair_price_formatted")
    hgb_upside = payload.get("upside_downside_hgb_formatted") or payload.get("upside_downside_formatted")
    hgb_upside_raw = _safe_float_for_ui(payload.get("upside_downside_hgb"))
    if hgb_upside_raw is None:
        hgb_upside_raw = sgd_upside_raw
    is_undervalued = _safe_float_for_ui(upside_raw) is not None and _safe_float_for_ui(upside_raw) >= 0
    signal_text = "UNDERVALUED" if is_undervalued else "OVERVALUED"
    signal_class = "text-bg-success" if is_undervalued else "text-bg-danger"
    icon_class = "soft-success" if is_undervalued else "soft-danger"
    icon_glyph = "↗" if is_undervalued else "↘"
    hgb_note_class = "hgb-note visible" if selected == "hgb" else "hgb-note"
    body_class = {
        "All": "fm-all",
        "Overview": "fm-overview",
        "Valuation Search": "fm-search",
        "Report Checks": "fm-checks",
        "Pipeline": "fm-pipeline",
        "Outputs": "fm-outputs",
    }.get(section, "fm-overview")

    metrics_rows = "".join(
        f"""
        <tr>
          <td>{_html_escape(row.get('selected_multiple'))}</td>
          <td>{_html_number(row.get('trainable_rows'), digits=0)}</td>
          <td>{_html_number(row.get('test_r2_log'))}</td>
          <td><span class="badge rounded-pill text-bg-primary">{_html_escape(row.get('model_method') or 'Ridge')}</span></td>
        </tr>
        """
        for row in metrics
    ) or "<tr><td colspan='4'>No model metrics available</td></tr>"

    secondary_rows = "".join(
        f"<tr><td>{_html_escape(key)}</td><td>{_html_escape(value)}</td></tr>"
        for key, value in (payload.get("secondary_multiples") or {}).items()
    )

    quality_rows = "".join(
        f"<tr><td>{_html_escape(label)}</td><td>{_html_escape(value)}</td></tr>"
        for label, value in [
            ("Peer similarity score", payload.get("peer_similarity_mean_formatted")),
            ("Peer blend weight", payload.get("peer_blend_weight_formatted")),
            ("Usable peer multiples", payload.get("peer_multiple_count_used")),
            ("Range clipping applied", payload.get("multiple_clip_applied")),
            ("Fair value / market cap", payload.get("fair_to_market_cap_formatted")),
            ("Confidence", str(payload.get("valuation_confidence", "")).upper()),
            ("Sanity flag", payload.get("valuation_sanity_flag")),
        ]
    )

    current_model_rows = "".join(
        f"<tr><td>{_html_escape(label)}</td><td>{_html_escape(value)}</td></tr>"
        for label, value in [
            ("Model method", payload.get("model_method")),
            ("Model training rows", payload.get("model_training_rows")),
            ("Metric trainable rows", payload.get("metric_trainable_rows")),
            ("Split method", payload.get("metric_split_method")),
            ("Test R2(log)", _html_number(payload.get("metric_test_r2_log"))),
            ("Test MAE(log)", _html_number(payload.get("metric_test_mae_log"))),
        ]
    )

    report_text = _html_escape(payload.get("text_report", ""))
    html_doc = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #f5f5f9;
      --card: #ffffff;
      --text: #566a7f;
      --heading: #384551;
      --primary: #696cff;
      --primary-soft: #e7e7ff;
      --success: #71dd37;
      --warning: #ffab00;
      --danger: #ff3e1d;
      --border: #d9dee3;
      --shadow: 0 0.125rem 0.375rem rgba(67, 89, 113, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 0.95rem;
    }}
    .app-shell {{
      display: block;
      min-height: auto;
    }}
    .sidebar {{
      display: none;
      background: var(--card);
      border-right: 1px solid rgba(67, 89, 113, 0.08);
      padding: 1.5rem 1rem;
      position: sticky;
      top: 0;
      height: 100vh;
    }}
    .brand {{
      align-items: center;
      color: var(--heading);
      display: flex;
      font-weight: 700;
      gap: 0.75rem;
      margin-bottom: 2rem;
      padding: 0 0.75rem;
    }}
    .brand-mark {{
      align-items: center;
      background: var(--primary);
      border-radius: 0.75rem;
      color: white;
      display: grid;
      height: 2.5rem;
      place-items: center;
      width: 2.5rem;
    }}
    .menu-label {{
      color: #a1acb8;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      margin: 1.5rem 0 0.5rem;
      padding: 0 0.75rem;
      text-transform: uppercase;
    }}
    .menu-item {{
      align-items: center;
      border-radius: 0.625rem;
      color: var(--text);
      display: flex;
      gap: 0.75rem;
      margin-bottom: 0.25rem;
      padding: 0.75rem;
      text-decoration: none;
    }}
    .menu-item.active, .menu-item:hover {{
      background: var(--primary-soft);
      color: var(--primary);
    }}
    .content {{ padding: 0; }}
    .topbar {{
      align-items: center;
      background: rgba(255, 255, 255, 0.86);
      border-radius: 0.875rem;
      box-shadow: var(--shadow);
      display: flex;
      gap: 1rem;
      justify-content: space-between;
      margin-bottom: 1.5rem;
      padding: 1rem 1.25rem;
    }}
    .topbar {{
      display: none;
    }}
    .card-lite {{
      background: var(--card);
      border: 0;
      border-radius: 1rem;
      box-shadow: var(--shadow);
      height: 100%;
    }}
    .card-body {{ padding: 1.5rem; }}
    h1, h2, h3, h4, h5, h6 {{
      color: var(--heading);
      font-weight: 700;
      margin: 0;
    }}
    p {{ margin-top: 0; }}
    .text-muted {{ color: #a1acb8; }}
    .small {{ font-size: 0.85rem; }}
    .row {{
      display: grid;
      gap: 1.5rem;
      margin-bottom: 1.5rem;
    }}
    .overview-grid {{ grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr); }}
    .stats-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .two-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .three-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .stat-icon {{
      align-items: center;
      border-radius: 0.75rem;
      display: grid;
      height: 2.75rem;
      place-items: center;
      width: 2.75rem;
      font-weight: 800;
    }}
    .soft-primary {{ background: var(--primary-soft); color: var(--primary); }}
    .soft-success {{ background: #e8fadf; color: #2f9f0a; }}
    .soft-warning {{ background: #fff2d6; color: #b76e00; }}
    .soft-danger {{ background: #ffe0db; color: var(--danger); }}
    .pipeline-step {{
      border-left: 3px solid var(--primary);
      padding-left: 1rem;
      margin-bottom: 1.35rem;
    }}
    .badge-soft {{
      background: var(--primary-soft);
      color: var(--primary);
      font-weight: 600;
    }}
    .badge {{
      border-radius: 999px;
      display: inline-block;
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.35rem 0.65rem;
    }}
    .text-bg-success {{ background: #e8fadf; color: #2f9f0a; }}
    .text-bg-danger {{ background: #ffe0db; color: var(--danger); }}
    .text-bg-primary {{ background: var(--primary-soft); color: var(--primary); }}
    .alert {{
      border-radius: 0.75rem;
      margin: 1.15rem 0;
      padding: 0.7rem 0.9rem;
    }}
    .alert-primary {{ background: var(--primary-soft); color: var(--primary); }}
    .alert-success {{ background: #e8fadf; color: #2f9f0a; }}
    .report-card-grid {{
      display: grid;
      gap: 1.5rem;
      grid-template-columns: 1fr auto;
    }}
    .metric-grid {{
      display: grid;
      gap: 1rem;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .metric-block {{ margin-bottom: 1.25rem; }}
    .metric-block p {{ margin-bottom: 0.25rem; }}
    .metric-block h3 {{ font-size: 1.08rem; }}
    .projection-selector {{
      align-self: stretch;
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      padding-left: 1rem;
    }}
    .projection-selector button {{
      background: var(--primary-soft);
      border: 0;
      border-radius: 999px;
      color: var(--primary);
      font-size: 0.75rem;
      font-weight: 700;
      padding: 0.5rem 0.65rem;
      writing-mode: vertical-rl;
    }}
    .projection-selector button.active {{
      background: var(--primary);
      color: white;
    }}
    .hgb-note {{ display: none; }}
    .hgb-note.visible {{ display: block; }}
    table {{
      border-collapse: collapse;
      width: 100%;
    }}
    th, td {{
      border-bottom: 1px solid rgba(67, 89, 113, 0.10);
      color: var(--text);
      padding: 0.875rem 1rem;
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--heading); font-weight: 700; }}
    pre {{
      background: #f8f8fb;
      border: 1px solid rgba(67, 89, 113, 0.08);
      border-radius: 0.75rem;
      color: var(--text);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      line-height: 1.55;
      overflow: auto;
      padding: 1rem;
      white-space: pre-wrap;
    }}
    .mb-0 {{ margin-bottom: 0; }}
    .mb-1 {{ margin-bottom: 0.25rem; }}
    .mb-3 {{ margin-bottom: 1rem; }}
    .mb-4 {{ margin-bottom: 1.5rem; }}
    .h4 {{ font-size: 1.45rem; }}
    .h5 {{ font-size: 1.1rem; }}
    .close-price-info {{ font-size: 0.85rem; }}
    body.fm-all .overview-grid,
    body.fm-all .two-grid,
    body.fm-all .three-grid {{
      grid-template-columns: 1fr;
    }}
    body.fm-all .stats-grid {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}
    body.fm-all .card-lite {{
      height: auto;
    }}
    body.fm-all section {{
      scroll-margin-top: 1rem;
    }}
    body.fm-search .overview-grid,
    body.fm-pipeline .two-grid,
    body.fm-checks .three-grid {{
      grid-template-columns: 1fr;
    }}
    body.fm-search .card-lite,
    body.fm-checks .card-lite,
    body.fm-pipeline .card-lite,
    body.fm-outputs .card-lite {{
      height: auto;
    }}
    body.fm-overview #checks,
    body.fm-overview #pipeline,
    body.fm-overview #outputs,
    body.fm-overview #valuation-mechanics {{
      display: none;
    }}
    body.fm-search .stats-grid,
    body.fm-search #checks,
    body.fm-search #pipeline,
    body.fm-search #outputs {{
      display: none;
    }}
    body.fm-checks #overview,
    body.fm-checks .stats-grid,
    body.fm-checks #pipeline,
    body.fm-checks #outputs,
    body.fm-checks #valuation-mechanics {{
      display: none;
    }}
    body.fm-pipeline #overview,
    body.fm-pipeline .stats-grid,
    body.fm-pipeline #checks,
    body.fm-pipeline #outputs,
    body.fm-pipeline #valuation-mechanics {{
      display: none;
    }}
    body.fm-outputs #overview,
    body.fm-outputs .stats-grid,
    body.fm-outputs #checks,
    body.fm-outputs #pipeline,
    body.fm-outputs #valuation-mechanics {{
      display: none;
    }}
    @media (max-width: 992px) {{
      .app-shell {{ grid-template-columns: 1fr; }}
      .sidebar {{ height: auto; position: static; }}
      .content {{ padding: 1rem; }}
      .overview-grid, .stats-grid, .two-grid, .three-grid, .metric-grid {{ grid-template-columns: 1fr; }}
      .report-card-grid {{ grid-template-columns: 1fr; }}
      .projection-selector {{
        border-left: 0;
        border-top: 1px solid var(--border);
        flex-direction: row;
        padding-left: 0;
        padding-top: 1rem;
      }}
      .projection-selector button {{ writing-mode: horizontal-tb; }}
    }}
  </style>
</head>
<body class="{body_class}">
  <div class="app-shell">
    <main class="content">
      <header class="topbar">
        <div>
          <h1 class="h4 mb-1">Fair Multiple Engine Dashboard</h1>
          <p class="text-muted mb-0">{_html_escape(payload.get('symbol'))} ({_html_escape(payload.get('order_book_id'))}) | {_html_escape(payload.get('industry'))}</p>
        </div>
        <div><span class="badge badge-soft">2020q1 - 2025q4</span></div>
      </header>

      <span id="fair-overview"></span>
      <section id="overview" class="row overview-grid">
        <div id="lookup" class="card-lite">
          <div class="card-body">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
              <div>
                <h2 class="h5 mb-1">{_html_escape(payload.get('symbol'))} valuation report</h2>
                <p class="text-muted mb-0">{_html_escape(payload.get('order_book_id'))} | {_html_escape(payload.get('quarter'))} | valuation date {_html_escape(payload.get('valuation_date'))}</p>
              </div>
              <span id="reportSignal" class="badge {signal_class}">{signal_text}</span>
            </div>
            <div class="alert alert-success">{_html_escape(payload.get('signal_sentence'))}</div>
            <div class="report-card-grid">
              <div>
                <div class="metric-grid">
                  <div>
                    <div class="metric-block"><p class="text-muted mb-1">Selected multiple</p><h3>{_html_escape(payload.get('selected_multiple'))}</h3></div>
                    <div class="metric-block"><p class="text-muted mb-1">Final fair multiple</p><h3>{_html_escape(payload.get('final_fair_multiple_formatted'))}</h3></div>
                  </div>
                  <div>
                    <div class="metric-block"><p class="text-muted mb-1">Fair stock price</p><h3 id="reportFairValue">{_html_escape(fair_price)}</h3></div>
                    <div class="metric-block"><p class="text-muted mb-1">Upside/downside</p><h3 id="reportUpside">{_html_escape(upside)}</h3></div>
                  </div>
                </div>
                <p id="hgbModelNote" class="{hgb_note_class} text-muted small mb-0">这是基于市场趋势学习后推导出的 HGB 估值结果。</p>
              </div>
              <div class="projection-selector">
                <button class="{'' if selected == 'hgb' else 'active'}" type="button" data-projection="sgd">SGD</button>
                <button class="{'active' if selected == 'hgb' else ''}" type="button" data-projection="hgb">HGB</button>
              </div>
            </div>
          </div>
        </div>
        <div class="card-lite">
          <div class="card-body">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
              <div>
                <p class="text-muted mb-1">Valuation signal</p>
                <h2 id="heroSignal" class="h4 mb-0">{'Undervalued' if is_undervalued else 'Overvalued'}</h2>
              </div>
              <div id="heroSignalIcon" class="stat-icon {icon_class}">{icon_glyph}</div>
            </div>
            <p id="heroSignalText" class="mb-3">{_html_escape(payload.get('symbol'))} ({_html_escape(payload.get('order_book_id'))}) is {'undervalued' if is_undervalued else 'overvalued'} for {_html_escape(payload.get('quarter'))}: {_html_escape(upside)} upside/downside versus actual market cap.</p>
            <p class="close-price-info text-muted mb-0">当前股价: {_html_escape(payload.get('close_price_formatted'))} 元 (基于{_html_escape(payload.get('quarter'))}数据)</p>
          </div>
        </div>
      </section>

      <section class="row stats-grid">
        <div class="card-lite"><div class="card-body"><div class="stat-icon soft-primary mb-3">DB</div><p class="text-muted mb-1">Prediction rows</p><h3 class="h4">{int(stats.get('rows', 0)):,}</h3></div></div>
        <div class="card-lite"><div class="card-body"><div class="stat-icon soft-success mb-3">CO</div><p class="text-muted mb-1">Companies</p><h3 class="h4">{int(stats.get('companies', 0)):,}</h3></div></div>
        <div class="card-lite"><div class="card-body"><div class="stat-icon soft-warning mb-3">Q</div><p class="text-muted mb-1">Quarters</p><h3 class="h4">{int(stats.get('quarters', 0)):,}</h3></div></div>
        <div class="card-lite"><div class="card-body"><div class="stat-icon soft-danger mb-3">S</div><p class="text-muted mb-1">Signals ready</p><h3 class="h4">{int(stats.get('signals', 0)):,}</h3></div></div>
      </section>

      <span id="fair-checks"></span>
      <section id="checks" class="row three-grid">
        <div class="card-lite"><div class="card-body"><h2 class="h5 mb-4">Valuation quality checks</h2><table><tbody>{quality_rows}</tbody></table></div></div>
        <div class="card-lite"><div class="card-body"><h2 class="h5 mb-4">Secondary market multiple checks</h2><table><tbody>{secondary_rows}</tbody></table></div></div>
        <div class="card-lite"><div class="card-body"><h2 class="h5 mb-4">Current model diagnostics</h2><table><tbody>{current_model_rows}</tbody></table></div></div>
      </section>

      <span id="fair-pipeline"></span>
      <section id="pipeline" class="row two-grid">
        <div class="card-lite">
          <div class="card-body">
            <h2 class="h5 mb-4">Connected pipeline</h2>
            <div class="pipeline-step"><h3 class="h5 mb-1">1. Fetch RQData</h3><p class="mb-0">Build raw database and market labels from quarterly panel data.</p></div>
            <div class="pipeline-step"><h3 class="h5 mb-1">2. Comparable companies</h3><p class="mb-0">Create financial features, K-means clusters, DBSCAN diagnostics, and peer sets.</p></div>
            <div class="pipeline-step"><h3 class="h5 mb-1">3. Multiple selection</h3><p class="mb-0">Choose P/E, P/B, P/S, or EV/EBITDA using deterministic finance rules.</p></div>
            <div class="pipeline-step"><h3 class="h5 mb-1">4. Fair multiple prediction</h3><p class="mb-0">Train Ridge log-multiple models and blend with peer median multiples.</p></div>
          </div>
        </div>
        <div class="card-lite">
          <div class="card-body">
            <h2 class="h5 mb-4">Model metrics</h2>
            <table>
              <thead><tr><th>Multiple</th><th>Rows</th><th>Test R2</th><th>Status</th></tr></thead>
              <tbody>{metrics_rows}</tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="valuation-mechanics" class="row two-grid">
        <div class="card-lite">
          <div class="card-body">
            <h2 class="h5 mb-4">Valuation mechanics</h2>
            <p class="text-muted mb-1">Selection reason</p>
            <p>{_html_escape(payload.get('selection_reason'))}</p>
            <p class="text-muted mb-1">Comparable peers used by Step 1</p>
            <p>{_html_escape(payload.get('peer_symbols'))}</p>
            <p class="text-muted mb-1">Formula applied</p>
            <pre>{_html_escape(payload.get('fair_value_formula'))}</pre>
          </div>
        </div>
        <div class="card-lite">
          <div class="card-body">
            <h2 class="h5 mb-4">Company valuation result</h2>
            <pre>{report_text}</pre>
          </div>
        </div>
      </section>

      <span id="fair-outputs"></span>
      <section id="outputs" class="card-lite">
        <div class="card-body">
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;">
            <div>
              <h2 class="h5 mb-1">Generated artifacts</h2>
              <p class="text-muted mb-0">Key files produced by the existing Python workflow.</p>
            </div>
            <span class="badge badge-soft">outputs/</span>
          </div>
          <table style="margin-top:1rem;">
            <thead><tr><th>File</th><th>Purpose</th><th>Layer</th></tr></thead>
            <tbody>
              <tr><td>raw_rqdatac_database_2020q1_2025q4.csv</td><td>Vendor financial and market source values</td><td>Raw</td></tr>
              <tr><td>step1_to_step2_input.csv</td><td>Clean universe with peer context</td><td>Features</td></tr>
              <tr><td>step2_selected_multiples.csv</td><td>Selected valuation multiple and reason</td><td>Rules</td></tr>
              <tr><td>step3_market_implied_multiple_predictions.csv</td><td>Fair multiple, fair stock price, and valuation signal</td><td>Prediction</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </div>
  <script>
    const projectionButtons = document.querySelectorAll("[data-projection]");
    const reportSignal = document.getElementById("reportSignal");
    const reportFairValue = document.getElementById("reportFairValue");
    const reportUpside = document.getElementById("reportUpside");
    const hgbModelNote = document.getElementById("hgbModelNote");
    const heroSignal = document.getElementById("heroSignal");
    const heroSignalIcon = document.getElementById("heroSignalIcon");
    const heroSignalText = document.getElementById("heroSignalText");
    const companySymbol = {_json_for_script(payload.get('symbol'))};
    const orderBookId = {_json_for_script(payload.get('order_book_id'))};
    const quarter = {_json_for_script(payload.get('quarter'))};
    const projections = {{
      sgd: {{
        fairPrice: {_json_for_script(sgd_fair_price)},
        upside: {_json_for_script(sgd_upside)},
        upsideRaw: {_json_for_script(sgd_upside_raw)}
      }},
      hgb: {{
        fairPrice: {_json_for_script(hgb_fair_price)},
        upside: {_json_for_script(hgb_upside)},
        upsideRaw: {_json_for_script(hgb_upside_raw)}
      }}
    }};

    function updateProjection(projection) {{
      const data = projections[projection] || projections.sgd;
      const isUndervalued = Number(data.upsideRaw) >= 0;
      reportSignal.textContent = isUndervalued ? "UNDERVALUED" : "OVERVALUED";
      reportSignal.className = `badge ${{isUndervalued ? "text-bg-success" : "text-bg-danger"}}`;
      reportFairValue.textContent = data.fairPrice || "N/A";
      reportUpside.textContent = data.upside || "N/A";
      hgbModelNote.classList.toggle("visible", projection === "hgb");
      heroSignal.textContent = isUndervalued ? "Undervalued" : "Overvalued";
      heroSignalIcon.className = `stat-icon ${{isUndervalued ? "soft-success" : "soft-danger"}}`;
      heroSignalIcon.textContent = isUndervalued ? "↗" : "↘";
      heroSignalText.textContent = `${{companySymbol}} (${{orderBookId}}) is ${{isUndervalued ? "undervalued" : "overvalued"}} for ${{quarter}}: ${{data.upside || "N/A"}} upside/downside versus actual market cap.`;
      projectionButtons.forEach((button) => {{
        button.classList.toggle("active", button.dataset.projection === projection);
      }});
    }}

    projectionButtons.forEach((button) => {{
      button.addEventListener("click", () => updateProjection(button.dataset.projection));
    }});
  </script>
</body>
</html>
"""
    component_height = {
        "All": 3400,
        "Valuation Search": 1050,
        "Report Checks": 980,
        "Pipeline": 900,
        "Outputs": 360,
    }.get(section, 900)
    components.html(html_doc, height=component_height, scrolling=False)


def _render_relative_valuation_result(payload: dict, projection: str):
    selected = projection.lower()
    fair_price = payload.get("fair_price_hgb_formatted") if selected == "hgb" else payload.get("fair_price_formatted")
    upside = payload.get("upside_downside_hgb_formatted") if selected == "hgb" else payload.get("upside_downside_formatted")
    upside_raw = payload.get("upside_downside_hgb") if selected == "hgb" else payload.get("upside_downside")
    is_undervalued = _safe_float_for_ui(upside_raw) is not None and _safe_float_for_ui(upside_raw) >= 0
    signal_label = "低估" if is_undervalued else "高估"
    signal_delta = upside if upside else "—"

    st.subheader(f"{payload.get('symbol', '')} · {payload.get('order_book_id', '')}")
    st.caption(
        f"{payload.get('industry', '')} | {payload.get('quarter', '')} | "
        f"valuation date {payload.get('valuation_date', '')}"
    )

    top_cols = st.columns(5)
    with top_cols[0]:
        st.metric("估值信号", signal_label, delta=signal_delta)
    with top_cols[1]:
        st.metric("当前价格", f"{payload.get('close_price_formatted', 'N/A')} 元")
    with top_cols[2]:
        st.metric(f"{projection} 公允价格", fair_price or "N/A")
    with top_cols[3]:
        st.metric("最终公允倍数", payload.get("final_fair_multiple_formatted", "N/A"))
    with top_cols[4]:
        st.metric("选定倍数", payload.get("selected_multiple", "N/A"))

    st.info(payload.get("signal_sentence", ""))

    tab_overview, tab_model, tab_report, tab_pipeline = st.tabs(["估值概览", "模型指标", "文本报告", "Pipeline"])
    with tab_overview:
        col_chart, col_detail = st.columns([1.2, 1])
        with col_chart:
            _render_relative_multiple_chart(payload)
            _render_relative_value_chart(payload, fair_price, projection)
        with col_detail:
            st.markdown("**倍数选择逻辑**")
            st.write(payload.get("selection_reason", ""))
            st.markdown("**估值公式**")
            st.code(payload.get("fair_value_formula", ""), language="text")
            st.markdown("**可比公司预览**")
            st.write(payload.get("peer_symbols", "N/A"))
            st.markdown("**二级市场倍数检查**")
            _render_dataframe(
                [
                    {"指标": key, "数值": value}
                    for key, value in (payload.get("secondary_multiples") or {}).items()
                ]
            )

        quality = pd.DataFrame(
            [
                {"指标": "同业相似度", "数值": payload.get("peer_similarity_mean_formatted")},
                {"指标": "同业混合权重", "数值": payload.get("peer_blend_weight_formatted")},
                {"指标": "同业数量", "数值": payload.get("peer_multiple_count_used")},
                {"指标": "置信度", "数值": str(payload.get("valuation_confidence", "")).upper()},
                {"指标": "Sanity Flag", "数值": payload.get("valuation_sanity_flag")},
                {"指标": "Range clipping", "数值": payload.get("multiple_clip_applied")},
            ]
        )
        st.markdown("**估值质量检查**")
        _render_dataframe(quality.to_dict(orient="records"))

    with tab_model:
        metrics = _cached_relative_metrics()
        if metrics:
            _render_dataframe(metrics)
        else:
            st.info("未找到 step3_model_metrics.csv。")
        this_metric = pd.DataFrame(
            [
                {"指标": "模型方法", "数值": payload.get("model_method")},
                {"指标": "训练行数", "数值": payload.get("model_training_rows")},
                {"指标": "Metric trainable rows", "数值": payload.get("metric_trainable_rows")},
                {"指标": "Split method", "数值": payload.get("metric_split_method")},
                {"指标": "Test R2(log)", "数值": payload.get("metric_test_r2_log")},
                {"指标": "Test MAE(log)", "数值": payload.get("metric_test_mae_log")},
            ]
        )
        st.markdown("**当前选定倍数模型表现**")
        _render_dataframe(this_metric.to_dict(orient="records"))

    with tab_report:
        st.download_button(
            "下载相对估值报告",
            data=str(payload.get("text_report", "")).encode("utf-8"),
            file_name=f"relative_valuation_{payload.get('order_book_id', 'report')}.txt",
            mime="text/plain",
        )
        st.code(payload.get("text_report", ""), language="text")
        with st.expander("查看原始 JSON", expanded=False):
            st.json(payload, expanded=False)

    with tab_pipeline:
        _render_relative_pipeline()


def _render_relative_multiple_chart(payload: dict):
    try:
        import plotly.express as px

        rows = [
            {"指标": "实际市场倍数", "数值": payload.get("actual_selected_multiple")},
            {"指标": "模型公允倍数", "数值": payload.get("model_predicted_fair_multiple")},
            {"指标": "同业中位倍数", "数值": payload.get("peer_median_multiple")},
            {"指标": "最终融合倍数", "数值": payload.get("final_fair_multiple")},
        ]
        frame = pd.DataFrame(rows)
        fig = px.bar(frame, x="指标", y="数值", color="指标", text="数值")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20), showlegend=False, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    except Exception:
        pass


def _render_relative_value_chart(payload: dict, fair_price: str | None, projection: str):
    try:
        import plotly.express as px

        fair_numeric = payload.get("fair_price_hgb") if projection.lower() == "hgb" else payload.get("fair_price")
        frame = pd.DataFrame(
            [
                {"指标": "当前价格", "数值": payload.get("close_price")},
                {"指标": f"{projection} 公允价格", "数值": fair_numeric},
            ]
        )
        fig = px.bar(frame, x="指标", y="数值", color="指标", text="数值")
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=20), showlegend=False, yaxis_title="价格")
        st.plotly_chart(fig, width="stretch")
    except Exception:
        st.caption(f"{projection} 公允价格: {fair_price or 'N/A'}")


def _render_relative_pipeline():
    st.markdown(
        """
        <div class="rq-card">
          <div class="rq-card-title">Connected pipeline</div>
          <div class="rq-pipeline-step"><b>1. Fetch RQData</b><br><span>生成季度原始数据库与市场标签。</span></div>
          <div class="rq-pipeline-step"><b>2. Comparable companies</b><br><span>构建财务特征、K-means 分组、DBSCAN 诊断和可比公司集合。</span></div>
          <div class="rq-pipeline-step"><b>3. Multiple selection</b><br><span>用确定性财务规则选择 P/E、P/B、P/S 或 EV/EBITDA。</span></div>
          <div class="rq-pipeline-step"><b>4. Fair multiple prediction</b><br><span>训练 log-multiple 模型，并与同业中位数融合。</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Generated artifacts**

        | File | Purpose | Layer |
        | --- | --- | --- |
        | raw_rqdatac_database_2020q1_2025q4.csv | Vendor financial and market source values | Raw |
        | step1_to_step2_input.csv | Clean universe with peer context | Features |
        | step2_selected_multiples.csv | Selected valuation multiple and reason | Rules |
        | step3_market_implied_multiple_predictions.csv | Fair multiple, fair price, and valuation signal | Prediction |
        """
    )


def _safe_float_for_ui(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _html_escape(value) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return html.escape(str(value))


def _html_number(value, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        number = float(value)
    except (TypeError, ValueError):
        return _html_escape(value)
    if digits <= 0:
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _json_for_script(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return json.dumps(value, ensure_ascii=False)


# ──────────────────────────────────────────────
# 投资建议展示
# ──────────────────────────────────────────────

def _render_recommendation(rec: Recommendation):
    """渲染投资建议卡片"""
    # 顶部: 综合评分 + 建议等级
    st.divider()

    # 评分颜色
    score_color = _score_to_color(rec.total_score)
    advice_emoji = _advice_to_emoji(rec.advice)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("综合评分", f"{rec.total_score}")
    with col2:
        st.metric("投资建议", f"{advice_emoji} {rec.advice.value}")
    with col3:
        st.metric("仓位建议", rec.position_suggestion or "—")
    with col4:
        st.metric("股票", f"{rec.name or rec.symbol}")

    # 核心逻辑
    st.subheader("💡 核心逻辑")
    st.info(rec.core_logic)

    # 各维度评分
    st.subheader("📐 维度评分")
    dim_names = list(rec.dimension_scores.keys()) or ["基本面", "量价", "舆情"]
    dim_cols = st.columns(len(dim_names))
    dim_icons = {
        "基本面": "📋",
        "量价": "📈",
        "舆情": "🔥",
        "宏观": "🌐",
    }

    for i, dim_name in enumerate(dim_names):
        with dim_cols[i]:
            score = rec.dimension_scores.get(dim_name, 0)
            st.markdown(
                f"<div style='text-align:center; padding:10px;'>"
                f"<div style='font-size:2em;'>{dim_icons.get(dim_name, '📊')}</div>"
                f"<div style='font-size:1.5em; font-weight:bold; color:{_score_to_color(score)}'>{score}</div>"
                f"<div style='color:#666;'>{dim_name}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # 雷达图
    try:
        _render_radar_chart(rec)
    except Exception:
        pass

    # 关键信号
    if rec.key_signals:
        st.subheader("📡 关键信号")
        for signal in rec.key_signals[:8]:
            icon = "🟢" if signal.type.value == "利好" else ("🔴" if signal.type.value == "风险" else "⚪")
            st.markdown(f"{icon} **[{signal.source}]** {signal.description}")

    # 操作策略
    st.subheader("🎯 操作策略")
    op_cols = st.columns(4)
    with op_cols[0]:
        st.markdown(f"**入场条件**\n\n{rec.operation.entry_condition or '—'}")
    with op_cols[1]:
        st.markdown(f"**止损位**\n\n{rec.operation.stop_loss or '—'}")
    with op_cols[2]:
        st.markdown(f"**目标位**\n\n{rec.operation.target or '—'}")
    with op_cols[3]:
        st.markdown(f"**持有周期**\n\n{rec.operation.holding_period or '—'}")

    # 风险提示
    if rec.risk_warnings:
        st.subheader("⚠️ 风险提示")
        for warning in rec.risk_warnings:
            st.warning(warning)

    # 免责声明
    st.caption(rec.disclaimer)


def _render_consensus_card(symbol: str, dp: DataProvider):
    """渲染 RQData 分析师一致预期补充卡片。"""
    try:
        consensus = dp.get_rqdata_consensus(symbol)
    except Exception as exc:
        st.caption(f"分析师一致预期暂不可用: {exc}")
        return

    if not isinstance(consensus, dict) or consensus.get("error"):
        warning = consensus.get("source_warning") or consensus.get("error") if isinstance(consensus, dict) else ""
        st.caption(f"分析师一致预期暂不可用: {warning}")
        return

    summary = consensus.get("summary") or {}
    data = consensus.get("latest_by_institute") or consensus.get("data") or []
    if not summary and not data:
        st.caption("分析师一致预期暂无可展示数据。")
        return

    st.markdown(
        """
        <div class="rq-card" style="border-left:5px solid var(--rq-primary);">
          <div class="rq-card-title">分析师一致预期</div>
          <div style="color:#8b95a4;font-size:0.92rem;margin-bottom:0.4rem;">
            来自 RQData concensus.csv，按机构最新观点聚合，作为目标价与评级的补充参考。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("覆盖机构", f"{summary.get('institute_count', 0):,} 家")
    with metric_cols[1]:
        st.metric("平均目标价", _format_price_value(summary.get("target_price_mean")))
    with metric_cols[2]:
        st.metric("隐含空间", _format_percent_value(summary.get("target_price_upside")))
    with metric_cols[3]:
        st.metric("平均评级", summary.get("rating_label") or "—")
    with metric_cols[4]:
        st.metric("最新发布日期", summary.get("latest_date") or "—")

    detail_cols = st.columns([1, 2])
    with detail_cols[0]:
        distribution = summary.get("rating_distribution") or {}
        if distribution:
            st.markdown("**评级分布**")
            dist_frame = pd.DataFrame(
                [{"评级": key, "机构数": value} for key, value in distribution.items()]
            )
            st.bar_chart(dist_frame.set_index("评级"))
        else:
            st.caption("暂无评级分布。")
    with detail_cols[1]:
        st.markdown("**目标价口径**")
        target_table = [
            {"口径": "原始目标价均值", "数值": _format_price_value(summary.get("raw_target_price_mean"))},
            {"口径": "6个月目标价均值", "数值": _format_price_value(summary.get("half_year_target_price_mean"))},
            {"口径": "1年目标价均值", "数值": _format_price_value(summary.get("one_year_target_price_mean"))},
            {"口径": "当前价格", "数值": _format_price_value(summary.get("close_price"))},
            {"口径": "平均评级系数", "数值": _format_plain_number(summary.get("rating_coef_mean"), digits=2)},
        ]
        _render_dataframe(target_table)

    with st.expander("查看机构一致预期明细", expanded=False):
        display_rows = []
        for row in data[:30]:
            display_rows.append(
                {
                    "发布日期": row.get("date"),
                    "机构": row.get("institute"),
                    "目标价原始值": row.get("price_raw"),
                    "6个月目标价": row.get("half_year_target_price"),
                    "1年目标价": row.get("one_year_target_price"),
                    "评级系数": row.get("grd_coef") or row.get("quarter_recommendation"),
                    "评级时段": row.get("grd_prd"),
                }
            )
        _render_dataframe(display_rows)


def _render_technical_timeseries_card(symbol: str, dp: DataProvider):
    """渲染近月技术指标时序变化。"""
    indicators = [
        "MA5",
        "MA10",
        "MA20",
        "MA60",
        "MACD_DIFF",
        "MACD_DEA",
        "MACD_HIST",
        "RSI6",
        "RSI10",
        "KDJ_K",
        "KDJ_D",
        "KDJ_J",
        "BOLL",
        "BOLL_UP",
        "BOLL_DOWN",
        "WR",
        "OBV",
        "VOL5",
        "VOL10",
        "VOL20",
    ]
    try:
        payload = dp.get_rqdata_technical_timeseries(symbol, limit=30, indicators=indicators)
    except Exception as exc:
        st.caption(f"技术指标时序暂不可用: {exc}")
        return

    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        warning = payload.get("source_warning") or payload.get("error") if isinstance(payload, dict) else ""
        st.caption(f"技术指标时序暂不可用: {warning}")
        return

    summary = payload.get("trend_summary") or {}
    st.markdown(
        """
        <div class="rq-card" style="border-left:5px solid #696cff;">
          <div class="rq-card-title">近月技术指标时序</div>
          <div style="color:#8b95a4;font-size:0.92rem;margin-bottom:0.4rem;">
            来自 RQData 技术指标 parquet 缓存，展示近一个月均线、动能和摆动指标的变化，用于辅助趋势判断与操作点建议。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("窗口", f"{summary.get('window_days') or len(rows)} 日")
    with metric_cols[1]:
        st.metric("趋势状态", summary.get("trend_bias") or "—")
    with metric_cols[2]:
        st.metric("最新日期", summary.get("latest_date") or payload.get("end_date") or "—")
    with metric_cols[3]:
        latest = rows[-1]
        st.metric("RSI6", _format_plain_number(latest.get("RSI6"), digits=2))

    st.info(
        "；".join(
            item
            for item in [
                summary.get("trend_signal"),
                summary.get("momentum_signal"),
                summary.get("risk_signal"),
                summary.get("operation_hint"),
            ]
            if item
        )
    )

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        return
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values("date")
    for column in indicators:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    try:
        import plotly.graph_objects as go

        chart_tabs = st.tabs(["均线趋势", "MACD 动能", "RSI/KDJ", "BOLL 通道"])
        with chart_tabs[0]:
            fig = go.Figure()
            for column in ["MA5", "MA10", "MA20", "MA60"]:
                if column in frame.columns and frame[column].notna().any():
                    fig.add_trace(go.Scatter(x=frame["date"], y=frame[column], mode="lines+markers", name=column))
            _style_technical_figure(fig, "均线时序")
            st.plotly_chart(fig, width="stretch")
        with chart_tabs[1]:
            fig = go.Figure()
            if "MACD_HIST" in frame.columns:
                fig.add_trace(go.Bar(x=frame["date"], y=frame["MACD_HIST"], name="MACD_HIST", marker_color="#696cff"))
            for column in ["MACD_DIFF", "MACD_DEA"]:
                if column in frame.columns and frame[column].notna().any():
                    fig.add_trace(go.Scatter(x=frame["date"], y=frame[column], mode="lines+markers", name=column))
            _style_technical_figure(fig, "MACD 动能变化")
            st.plotly_chart(fig, width="stretch")
        with chart_tabs[2]:
            fig = go.Figure()
            for column in ["RSI6", "RSI10", "KDJ_K", "KDJ_D", "KDJ_J", "WR"]:
                if column in frame.columns and frame[column].notna().any():
                    fig.add_trace(go.Scatter(x=frame["date"], y=frame[column], mode="lines+markers", name=column))
            fig.add_hline(y=80, line_dash="dot", line_color="#ff3e1d", opacity=0.45)
            fig.add_hline(y=20, line_dash="dot", line_color="#71dd37", opacity=0.45)
            _style_technical_figure(fig, "摆动指标变化")
            st.plotly_chart(fig, width="stretch")
        with chart_tabs[3]:
            fig = go.Figure()
            for column in ["BOLL_UP", "BOLL", "BOLL_DOWN"]:
                if column in frame.columns and frame[column].notna().any():
                    fig.add_trace(go.Scatter(x=frame["date"], y=frame[column], mode="lines+markers", name=column))
            _style_technical_figure(fig, "BOLL 通道变化")
            st.plotly_chart(fig, width="stretch")
    except ImportError:
        _render_dataframe(_technical_timeseries_preview(frame))

    with st.expander("查看最新技术指标快照", expanded=False):
        latest = rows[-1]
        display_rows = [
            {"指标": column, "数值": _format_plain_number(latest.get(column), digits=4)}
            for column in indicators
            if latest.get(column) is not None
        ]
        _render_dataframe(display_rows)


def _style_technical_figure(fig, title: str):
    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=48, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(86,106,127,0.12)"),
    )


def _technical_timeseries_preview(frame: pd.DataFrame) -> list[dict]:
    preview_columns = ["date", "MA5", "MA20", "MA60", "MACD_HIST", "RSI6", "KDJ_J"]
    columns = [column for column in preview_columns if column in frame.columns]
    return frame.loc[:, columns].tail(10).to_dict(orient="records")


def _render_integrated_valuation_models(symbol: str, name: str = ""):
    """在 Investment Advisor 主面板展示相对估值和本地 DCF 摘要。"""
    st.markdown(
        """
        <div class="rq-card" style="border-left:5px solid #71dd37;">
          <div class="rq-card-title">估值模型联动</div>
          <div style="color:#8b95a4;font-size:0.92rem;margin-bottom:0.4rem;">
            汇总 Relative Prediction 公允倍数模型与 DCF 现金流估值结果，作为主投顾建议的外部校验。
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rel_col, dcf_col = st.columns(2)
    with rel_col:
        _render_relative_valuation_summary(symbol)
    with dcf_col:
        _render_dcf_summary_for_main(symbol, name)


def _render_relative_valuation_summary(symbol: str):
    st.markdown("**相对估值 · Fair Multiple**")
    try:
        payload = get_relative_valuation(symbol, quarter="latest", top_peers=5)
    except Exception as exc:
        st.caption(f"相对估值结果暂不可用: {exc}")
        return

    if not payload or payload.get("error"):
        st.caption(f"相对估值结果暂不可用: {payload.get('error', '无可用结果') if isinstance(payload, dict) else '无可用结果'}")
        return

    signal = _relative_signal_label(payload)
    metric_rows = [
        [
            ("选定倍数", payload.get("selected_multiple", "—")),
            ("公允价格", payload.get("fair_price_formatted") or _format_price_value(payload.get("fair_price"))),
        ],
        [
            ("隐含空间", payload.get("upside_downside_formatted") or _format_percent_value(payload.get("upside_downside"))),
            ("估值信号", signal),
        ],
    ]
    for row in metric_rows:
        cols = st.columns(2)
        for col, (label, value) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="rq-card" style="padding:0.9rem 1rem;margin-bottom:0.65rem;box-shadow:0 0.125rem 0.35rem rgba(67,89,113,0.10);">
                      <div style="color:#8b95a4;font-size:0.82rem;font-weight:700;margin-bottom:0.25rem;">{html.escape(str(label))}</div>
                      <div style="color:#384551;font-size:1.22rem;font-weight:820;white-space:normal;word-break:break-word;">{html.escape(str(value or '—'))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.caption(payload.get("signal_sentence", ""))
    with st.expander("查看相对估值详情", expanded=False):
        rows = [
            {"指标": "报告期", "数值": payload.get("quarter")},
            {"指标": "估值日期", "数值": payload.get("valuation_date")},
            {"指标": "最终公允倍数", "数值": payload.get("final_fair_multiple_formatted")},
            {"指标": "HGB 公允价格", "数值": payload.get("fair_price_hgb_formatted")},
            {"指标": "HGB 隐含空间", "数值": payload.get("upside_downside_hgb_formatted")},
            {"指标": "可比公司", "数值": payload.get("peer_symbols")},
        ]
        _render_dataframe(rows)


def _relative_signal_label(payload: dict) -> str:
    raw = str(payload.get("valuation_signal") or "").strip().upper()
    if raw in {"UNDERVALUED", "UNDERVALUE", "LOW", "低估"}:
        return "低估"
    if raw in {"OVERVALUED", "OVERVALUE", "HIGH", "高估"}:
        return "高估"
    upside = _safe_float_for_ui(payload.get("upside_downside"))
    if upside is None:
        return raw or "—"
    return "低估" if upside >= 0 else "高估"


def _render_dcf_summary_for_main(symbol: str, name: str = ""):
    st.markdown("**DCF 估值 · Cash Flow**")
    state_key = f"main_dcf_payload_{symbol}"
    company_query = symbol or name

    if state_key not in st.session_state:
        try:
            with st.spinner("正在自动运行 DCF 估值..."):
                system = _get_cached_dcf_system()
                st.session_state[state_key] = run_dcf_analysis(
                    system,
                    company_query,
                    generate_llm_report=False,
                )
        except Exception as exc:
            st.warning(f"DCF 每股价值暂不可用: {exc}")
            return

    payload = st.session_state.get(state_key)
    if not payload:
        return

    results = payload.get("results", {})
    st.metric("DCF 每股价值", _format_dcf_value(results.get("dcf_value")))
    st.caption("自动调用本地 program.py DCF 流水线，仅取每股价值结果；不生成 AI 报告。")


def _render_radar_chart(rec: Recommendation):
    """渲染雷达图"""
    try:
        import plotly.graph_objects as go

        dims = list(rec.dimension_scores.keys()) or ["基本面", "量价", "舆情"]
        scores = [rec.dimension_scores.get(d, 0) for d in dims]

        fig = go.Figure(data=go.Scatterpolar(
            r=scores,
            theta=dims,
            fill="toself",
            fillcolor="rgba(99, 110, 255, 0.2)",
            line=dict(color="rgb(99, 110, 255)", width=2),
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=350,
            margin=dict(l=30, r=30, t=30, b=30),
        )

        st.plotly_chart(fig, width="stretch")
    except ImportError:
        return


def _render_ai_analysis_page(symbol: str, dp: DataProvider, analysis_data: dict):
    """渲染接近券商 App 样式的 AI 分析详情页。"""
    with st.spinner("正在生成 AI 分析详情..."):
        report = build_ai_analysis_report(symbol, dp, analysis_data)

    st.markdown(
        """
        <style>
        .ai-report-title {
            font-size: 1.45rem;
            font-weight: 800;
            color: #171717;
            margin-bottom: 14px;
        }
        .ai-report-line {
            font-size: 1rem;
            line-height: 1.85;
            color: #262626;
            margin: 8px 0;
        }
        .source-chip {
            display: inline-block;
            background: #eaf3ff;
            color: #1d4ed8;
            padding: 4px 9px;
            border-radius: 6px;
            margin: 4px 6px 4px 0;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.caption(f"{report.get('name', symbol)} · 更新时间 {report.get('updated_at', '')}")
    _render_fundamental_analysis_card(report.get("fundamental", {}))
    _render_technical_analysis_card(report.get("technical", {}))
    _render_capital_analysis_card(report.get("capital", {}))
    _render_news_analysis_card(report.get("news", {}))
    _render_valuation_analysis_card(report.get("valuation", {}))
    st.caption(report.get("source_note", ""))


def _render_fundamental_analysis_card(section: dict):
    with st.container(border=True):
        _render_ai_card_title("一、基本面分析")
        _render_summary_lines(section.get("summary", []))

        table = section.get("financial_table", [])
        if table:
            st.markdown("**营业收入与净利润数据表**")
            _render_dataframe(table)

        health_table = section.get("health_table", [])
        if health_table:
            st.markdown("**财务健康度指标表**")
            _render_dataframe(health_table)


def _render_technical_analysis_card(section: dict):
    with st.container(border=True):
        _render_ai_card_title("二、技术面分析")
        _render_summary_lines(section.get("summary", []))

        chart = section.get("chart", [])
        if chart:
            try:
                import plotly.express as px

                frame = pd.DataFrame(chart)
                y_column = "close" if "close" in frame.columns else None
                if y_column is None:
                    numeric_columns = [
                        column
                        for column in frame.columns
                        if column != "date" and pd.to_numeric(frame[column], errors="coerce").notna().any()
                    ]
                    y_column = numeric_columns[:3]
                if y_column:
                    fig = px.line(frame, x="date", y=y_column)
                    fig.update_traces(line=dict(width=2))
                    fig.update_layout(
                        height=260,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="",
                        yaxis_title="技术指标",
                    )
                    st.plotly_chart(fig, width="stretch")
            except Exception:
                pass

        table = section.get("table", [])
        if table:
            st.markdown("**技术指标数据表**")
            _render_dataframe(table)


def _render_capital_analysis_card(section: dict):
    with st.container(border=True):
        _render_ai_card_title("三、资金面分析")
        _render_summary_lines(section.get("summary", []))

        table = section.get("table", [])
        if table:
            st.markdown("**资金面数据表**")
            _render_dataframe(table)


def _render_news_analysis_card(section: dict):
    with st.container(border=True):
        _render_ai_card_title("四、资讯面分析")
        _render_summary_lines(section.get("summary", []))

        sources = section.get("sources", [])
        if sources:
            st.markdown("**资讯溯源**")
            chips = "".join(f"<span class='source-chip'>{source}</span>" for source in sources)
            st.markdown(chips, unsafe_allow_html=True)


def _render_valuation_analysis_card(section: dict):
    with st.container(border=True):
        _render_ai_card_title("五、估值面分析")
        _render_summary_lines(section.get("summary", []))

        table = section.get("table", [])
        if table:
            st.markdown("**估值指标对比表**")
            _render_dataframe(table)


def _render_summary_lines(lines: list[str]):
    if not lines:
        st.info("暂无可展示分析。")
        return
    for line in lines:
        if not line:
            continue
        if ":" in line:
            head, tail = line.split(":", 1)
            st.markdown(f"<p class='ai-report-line'><b>{head}：</b>{tail.strip()}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='ai-report-line'>{line}</p>", unsafe_allow_html=True)


def _render_ai_card_title(title: str):
    st.markdown(f"<div class='ai-report-title'>{title}</div>", unsafe_allow_html=True)


def _render_industry_scorecard(symbol: str, dp: DataProvider):
    """渲染 RQData 行业内相对得分组件。"""
    try:
        scorecard = dp.get_rqdata_industry_scorecard(symbol)
    except Exception as exc:
        st.caption(f"行业内相对表现暂不可用: {exc}")
        return

    if not isinstance(scorecard, dict) or scorecard.get("error"):
        st.caption("行业内相对表现暂不可用")
        return

    st.subheader("行业内相对表现")
    st.caption("基于 RQData 离线 PIT 面板计算，展示该股票在所属行业内的估值、盈利、规模、流动性和现金流质量分位。")

    top_cols = st.columns(4)
    with top_cols[0]:
        st.metric("所属行业", scorecard.get("industry_name", ""))
    with top_cols[1]:
        st.metric("行业层级", scorecard.get("industry_level", ""))
    with top_cols[2]:
        rank = scorecard.get("industry_rank")
        peer_count = scorecard.get("peer_count")
        rank_text = f"{rank}/{peer_count}" if rank and peer_count else "—"
        st.metric("行业排名", rank_text)
    with top_cols[3]:
        overall = scorecard.get("overall_score")
        st.metric("行业综合分位", f"{overall:.1f}" if isinstance(overall, (int, float)) else "—")

    metrics = scorecard.get("metrics", [])
    if metrics:
        metric_df = pd.DataFrame(metrics)
        display_df = metric_df.rename(
            columns={
                "label": "指标",
                "value": "当前值",
                "score": "行业分位",
                "rank": "指标排名",
                "peer_count": "有效样本",
                "direction": "方向",
            }
        )

        chart_cols = st.columns([3, 2])
        with chart_cols[0]:
            try:
                import plotly.express as px

                fig = px.bar(
                    metric_df.sort_values("score"),
                    x="score",
                    y="label",
                    orientation="h",
                    text="score",
                    color="score",
                    color_continuous_scale=["#b91c1c", "#f59e0b", "#15803d"],
                    range_color=[0, 100],
                )
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig.update_layout(
                    height=320,
                    margin=dict(l=20, r=30, t=10, b=20),
                    xaxis_title="行业分位",
                    yaxis_title="",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, width="stretch")
            except Exception:
                for item in metrics:
                    st.progress(int(item.get("score", 0)), text=f"{item.get('label')}: {item.get('score')}")

        with chart_cols[1]:
            st.dataframe(
                _normalize_display_dataframe(display_df[["指标", "当前值", "行业分位", "指标排名", "有效样本", "方向"]]),
                width="stretch",
                hide_index=True,
            )

    quality_notes = scorecard.get("data_quality", [])
    if quality_notes:
        with st.expander("数据质量提示", expanded=False):
            for note in quality_notes:
                st.warning(note)

    peer_table = scorecard.get("peer_table", [])
    if peer_table:
        with st.expander("查看行业同业样本", expanded=False):
            peer_df = pd.DataFrame(peer_table)
            rename_map = {
                "industry_rank": "排名",
                "order_book_id": "证券ID",
                "symbol": "名称",
                "overall_score": "综合分",
                "overall_percentile": "综合分位",
                "available_metric_count": "可用指标数",
                "pe_ratio_ttm": "PE(TTM)",
                "pb_ratio": "PB",
                "return_on_equity_weighted_average": "ROE",
                "net_profit_ttm_0": "净利润TTM",
                "market_cap": "市值",
                "total_turnover": "成交额",
            }
            peer_df = peer_df.rename(columns=rename_map)
            st.dataframe(_normalize_display_dataframe(peer_df), width="stretch", hide_index=True)

    with st.expander("口径说明", expanded=False):
        methodology = scorecard.get("methodology", {})
        for item in methodology.values():
            st.markdown(f"- {item}")


# ──────────────────────────────────────────────
# 初步数据展示
# ──────────────────────────────────────────────

def _render_initial_data(analysis_data: dict):
    """渲染 Agent 已获取的初步数据快照。"""
    if not analysis_data:
        return

    st.divider()
    st.subheader("🧾 数据快照")

    agent_labels = [
        ("fundamental", "基本面"),
        ("technical", "量价"),
        ("sentiment", "舆情"),
        ("macro", "宏观"),
    ]
    tabs = st.tabs([label for _, label in agent_labels])

    for tab, (agent_key, label) in zip(tabs, agent_labels):
        with tab:
            agent_result = analysis_data.get(agent_key)
            if not agent_result:
                st.info(f"{label}数据暂未返回")
                continue
            if isinstance(agent_result, dict) and agent_result.get("disabled"):
                st.info(agent_result.get("message", f"{label} Agent 已暂停"))
                continue

            raw_summary = agent_result.get("raw_data_summary", {})
            freshness = raw_summary.get("data_freshness", {}) if isinstance(raw_summary, dict) else {}
            usable = freshness.get("usable", True) if isinstance(freshness, dict) else True
            score = agent_result.get("total_score")
            cols = st.columns([1, 3])
            with cols[0]:
                st.metric(f"{label}得分", score if usable and score is not None else "未启用")
            with cols[1]:
                if usable:
                    _render_agent_score_summary(agent_key, agent_result)
                elif freshness:
                    st.warning(freshness.get("reason", f"{label}数据时效未通过"))

            if raw_summary:
                _render_freshness_summary(freshness)
                _render_snapshot_sections(raw_summary)
            else:
                st.info("暂无原始数据摘要")


def _render_agent_score_summary(agent_key: str, agent_result: dict):
    """展示子维度评分摘要。"""
    dimension_map = {
        "fundamental": ["financial_health", "valuation", "growth"],
        "technical": ["trend", "momentum", "capital_flow"],
        "sentiment": ["market_emotion", "social_heat", "overall_market_emotion"],
        "macro": ["economic_cycle", "monetary_policy", "industry_cycle"],
    }
    rows = []
    for key in dimension_map.get(agent_key, []):
        item = agent_result.get(key, {})
        rows.append({
            "维度": item.get("dimension", key),
            "分数": item.get("score", ""),
            "摘要": item.get("summary", ""),
        })
    if rows:
        _render_dataframe(rows)


def _render_snapshot_sections(snapshot: dict):
    """展示原始数据摘要分组。"""
    quality = snapshot.get("data_quality", {})
    errors = quality.get("errors", []) if isinstance(quality, dict) else []
    if errors:
        with st.expander(f"数据质量 / 接口错误 ({len(errors)})", expanded=False):
            for error in errors:
                st.warning(error)

    for section, payload in snapshot.items():
        if section in {"data_quality", "data_freshness"}:
            continue
        with st.expander(_section_label(section), expanded=False):
            _render_snapshot_payload(payload)


def _render_freshness_summary(freshness: dict):
    """展示数据时效校验结果。"""
    if not isinstance(freshness, dict) or not freshness:
        return

    usable = freshness.get("usable", True)
    reason = freshness.get("reason", "")
    if usable:
        st.success(reason or "数据时效校验通过")
    else:
        st.warning(reason or "数据时效校验未通过")

    checks = freshness.get("checks", [])
    if checks:
        with st.expander("数据时效明细", expanded=not usable):
            _render_dataframe(checks)


def _render_snapshot_payload(payload):
    """递归展示数据快照。"""
    if isinstance(payload, dict) and payload.get("type") == "records":
        rows = payload.get("rows", 0)
        fields = payload.get("fields", [])
        sample = payload.get("sample", [])
        metric_cols = st.columns(2)
        with metric_cols[0]:
            st.metric("返回行数", rows)
        with metric_cols[1]:
            st.metric("字段数", len(fields))
        if fields:
            st.caption("字段: " + "、".join(fields[:20]))
        if sample:
            _render_dataframe(sample)
        return

    if isinstance(payload, dict):
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"type", "keys"} and not isinstance(value, (dict, list))
        }
        if metadata:
            st.json(metadata, expanded=False)

        for key, value in payload.items():
            if key in {"type", "keys"}:
                continue
            if isinstance(value, dict):
                st.markdown(f"**{_section_label(key)}**")
                _render_snapshot_payload(value)
            elif isinstance(value, list):
                st.markdown(f"**{_section_label(key)}**")
                _render_snapshot_payload({
                    "type": "records",
                    "rows": len(value),
                    "fields": list(value[0].keys()) if value and isinstance(value[0], dict) else [],
                    "sample": value[:5],
                })
        return

    if isinstance(payload, list):
        _render_snapshot_payload({
            "type": "records",
            "rows": len(payload),
            "fields": list(payload[0].keys()) if payload and isinstance(payload[0], dict) else [],
            "sample": payload[:5],
        })
    else:
        st.write(payload)


def _render_dataframe(rows):
    """安全展示表格，避免混合类型列触发 PyArrow 序列化警告。"""
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("暂无表格数据")
        return
    st.dataframe(_normalize_display_dataframe(df), width="stretch", hide_index=True)


def _normalize_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """把复杂/混合类型单元格转为稳定展示值。"""
    display_df = df.copy()
    for col in display_df.columns:
        display_df[col] = display_df[col].map(_normalize_display_cell)
    return display_df


def _normalize_display_cell(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _section_label(section: str) -> str:
    """把内部字段名转换为页面标签。"""
    labels = {
        "valuation": "估值数据",
        "indicator": "财务指标",
        "balance": "资产负债表",
        "profit": "利润表",
        "cashflow": "现金流量表",
        "forecast": "盈利预测",
        "earnings": "业绩预告",
        "price": "K线量价",
        "capital_flow": "资金流",
        "north_flow": "北向资金",
        "lhb": "龙虎榜",
        "margin": "融资融券",
        "sentiment": "个股舆情",
        "news": "个股新闻",
        "market_emotion": "市场情绪",
        "macro": "中国宏观",
        "interest": "全球利率",
        "valuation_data": "市场估值",
        "price_meta": "K线来源",
        "stale_raw": "滞后原始数据(未参与分析)",
    }
    return labels.get(section, section)


# ──────────────────────────────────────────────
# 对比分析展示
# ──────────────────────────────────────────────

def _render_comparison(result: dict):
    """渲染对比分析"""
    st.subheader("📊 对比分析")

    stock_a = result.get("stock_a", {})
    stock_b = result.get("stock_b", {})
    comparison = result.get("comparison", "")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### A股")
        if stock_a:
            rec_a = Recommendation(**stock_a) if isinstance(stock_a, dict) and "symbol" in stock_a else None
            if rec_a:
                st.metric("综合评分", f"{rec_a.total_score}", delta=rec_a.advice.value)
                for dim, score in rec_a.dimension_scores.items():
                    st.progress(int(score), text=f"{dim}: {score}")

    with col2:
        st.markdown("### B股")
        if stock_b:
            rec_b = Recommendation(**stock_b) if isinstance(stock_b, dict) and "symbol" in stock_b else None
            if rec_b:
                st.metric("综合评分", f"{rec_b.total_score}", delta=rec_b.advice.value)
                for dim, score in rec_b.dimension_scores.items():
                    st.progress(int(score), text=f"{dim}: {score}")

    if comparison:
        st.divider()
        st.markdown(comparison)


# ──────────────────────────────────────────────
# 板块分析展示
# ──────────────────────────────────────────────

def _render_sector(result: dict):
    """渲染板块分析"""
    sector = result.get("sector", "")
    st.subheader(f"🏭 {sector} 板块分析")

    macro_score = result.get("macro_score", 0)
    macro_summary = result.get("macro_summary", "")
    sector_data = result.get("sector_data", {})
    offline_sector = result.get("offline_sector", {})
    macro_disabled = result.get("macro_disabled", False)

    col1, col2 = st.columns(2)
    with col1:
        if macro_disabled:
            st.metric("宏观景气评分", "已暂停")
            st.info(macro_summary or MACRO_AGENT_DISABLED_REASON)
            st.caption("这是系统配置行为：当前 `MACRO_AGENT_ENABLED=false`，宏观维度不会参与个股或板块评分。")
        else:
            st.metric("宏观景气评分", f"{macro_score}")
            st.info(macro_summary)

    with col2:
        errors = _sector_errors(sector_data)
        spot = sector_data.get("spot") if isinstance(sector_data, dict) else None
        hist = sector_data.get("hist") if isinstance(sector_data, dict) else None
        constituents = sector_data.get("constituents") if isinstance(sector_data, dict) else None

        if spot or hist or constituents:
            st.success("实时板块接口返回成功")
            if spot:
                st.markdown("**实时行情**")
                _render_dataframe(spot[:5])
            if constituents:
                st.markdown("**实时成分股**")
                _render_dataframe(constituents[:10])
        elif errors:
            st.warning("实时板块接口暂不可用，已尝试使用本地 RQData 行业映射兜底。")
            st.caption(_friendly_sector_error(errors))
        else:
            st.info("暂无实时板块数据。")

        offline_rows = offline_sector.get("data", []) if isinstance(offline_sector, dict) else []
        if offline_rows:
            names = "、".join(offline_sector.get("sector_names", [])[:3])
            st.markdown("**本地 RQData 行业映射兜底**")
            st.caption(
                f"匹配行业: {names or sector}；覆盖成分股 "
                f"{offline_sector.get('constituent_count', len(offline_rows))} 只。该表不包含实时涨跌幅。"
            )
            _render_dataframe(offline_rows[:20])
        elif isinstance(offline_sector, dict) and offline_sector.get("source_warning"):
            st.info(offline_sector.get("source_warning"))

        if errors:
            with st.expander("查看实时接口错误详情", expanded=False):
                st.json(errors, expanded=False)


def _sector_errors(sector_data: dict) -> dict:
    if not isinstance(sector_data, dict):
        return {}
    return {
        key: value
        for key, value in sector_data.items()
        if key.endswith("_error") or key == "error"
    }


def _friendly_sector_error(errors: dict) -> str:
    text = " ".join(str(value) for value in errors.values())
    if "ProxyError" in text or "Unable to connect to proxy" in text:
        return "AKShare 访问东方财富实时接口时代理连接失败，通常是本地网络/代理配置导致，不是投顾分析逻辑错误。"
    if "Max retries exceeded" in text or "HTTPSConnectionPool" in text:
        return "AKShare 实时行情接口请求多次失败，可能是网络、代理或数据源临时不可用。"
    return "实时行情接口返回错误，详情可在下方折叠项中查看。"


# ──────────────────────────────────────────────
# 宏观分析展示
# ──────────────────────────────────────────────

def _render_macro(result: dict):
    """渲染宏观分析"""
    st.subheader("🌐 宏观景气分析")
    if result.get("disabled"):
        st.info(result.get("message", MACRO_AGENT_DISABLED_REASON))
        macro_result = result.get("result", {})
        if macro_result:
            with st.expander("查看宏观数据时效快照", expanded=False):
                st.json(macro_result.get("raw_data_summary", macro_result), expanded=False)
        return

    macro_result = result.get("result", {})
    if macro_result:
        st.json(macro_result, expanded=True)
    else:
        st.info("暂无宏观数据")


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _render_guide():
    """渲染使用指引"""
    st.markdown(
        """
        <div class="ia-feature-grid">
          <div class="ia-feature-card">
            <strong>一站式个股画像</strong>
            <span>综合基本面、量价技术、舆情热度、行业相对得分和一致预期。</span>
          </div>
          <div class="ia-feature-card">
            <strong>本地 RQData 补充</strong>
            <span>优先融合离线财务面板、技术指标、行业标签与分析师目标价。</span>
          </div>
          <div class="ia-feature-card">
            <strong>结果可追溯</strong>
            <span>分析页保留数据快照、关键指标表和可解释的信号摘要。</span>
          </div>
        </div>
        <div class="rq-card" style="margin-top:1rem;">
          <div class="rq-card-title">可以这样输入</div>
          <span class="ia-example-chip">000001</span>
          <span class="ia-example-chip">000001.XSHE</span>
          <span class="ia-example-chip">比亚迪</span>
          <span class="ia-example-chip">平安银行</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _looks_like_stock_query(query: str) -> bool:
    text = str(query or "").strip()
    if not text:
        return False
    blocked_keywords = [
        "板块",
        "行业",
        "概念",
        "赛道",
        "宏观",
        "经济",
        "政策",
        "利率",
        "对比",
        "比较",
        "筛选",
        "推荐",
        "选股",
        "找找",
    ]
    if any(keyword in text for keyword in blocked_keywords):
        return False
    if re.search(r"和.+比|vs|VS", text):
        return False
    if re.search(r"\d{6}(\.(XSHE|XSHG|XBSE))?", text.upper()):
        return True
    known_names = [
        "比亚迪",
        "贵州茅台",
        "宁德时代",
        "平安银行",
        "招商银行",
        "中芯国际",
        "隆基绿能",
        "长城汽车",
        "中国平安",
        "美的集团",
        "格力电器",
    ]
    return any(name in text for name in known_names)


def _score_to_color(score: float) -> str:
    """评分转颜色"""
    if score >= 80:
        return "#4CAF50"
    elif score >= 60:
        return "#FF9800"
    elif score >= 40:
        return "#FFC107"
    else:
        return "#F44336"


def _advice_to_emoji(advice: AdviceLevel) -> str:
    """建议等级转 emoji"""
    emoji_map = {
        AdviceLevel.STRONG_BUY: "🔥",
        AdviceLevel.BUY: "👍",
        AdviceLevel.HOLD: "🤝",
        AdviceLevel.WATCH: "👀",
        AdviceLevel.AVOID: "🚫",
    }
    return emoji_map.get(advice, "📊")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    inject_sneat_theme()
    render_sidebar_brand()
    page = st.sidebar.radio(
        "选择功能",
        ["AI 投顾分析", "DCF 估值", "相对估值可视化"],
        label_visibility="collapsed",
        key="main_page",
    )
    st.sidebar.divider()

    if page == "相对估值可视化":
        render_fair_multiple_sidebar()
        render_relative_valuation_page()
        return

    if page == "DCF 估值":
        render_dcf_page()
        return

    profile = render_sidebar()
    render_main_page(profile)


if __name__ == "__main__":
    main()
