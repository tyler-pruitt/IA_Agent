
# %%
"""
Complete Financial Analysis System
Combines ML predictions, fundamental data, and LLM analysis for comprehensive company evaluation
"""

import numpy as np
import pandas as pd
import joblib
import pickle
import rqdatac as rq
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


def initialize_rqdatac() -> None:
    """Initialize rqdatac only when the analysis system actually needs it."""
    try:
        rq.init()
    except Exception as exc:
        raise RuntimeError(
            "rqdatac initialization failed. Please confirm your local network access and rqdatac credentials."
        ) from exc



# %% [markdown]
# ## MODEL LOADING AND PREDICTION FUNCTIONS

# %%

# ============================================
# 1. MODEL LOADING AND PREDICTION FUNCTIONS
# ============================================

class FinancialPredictor:
    """Handles the non-TensorFlow ML predictions for cash flows and earnings."""
    
    def __init__(self, model_dir='./model'):
        self.model_dir = model_dir
        self.models = self._load_models()
        
    def _load_models(self):
        """Load the tree-based models used by the analysis system."""
        print("Loading ML models (Random Forest + Gradient Boosting)...")
        models = {
            # Cash Flow Models
            'ffcf_rf_model': self._load_joblib_model('ffcf_random_forest_model.pkl', required=True),
            'ffcf_rf_selector': self._load_joblib_model('ffcf_rf_selector.pkl', required=True),
            'ffcf_gbr_model': self._load_joblib_model('ffcf_gradient_boosting_model.pkl', required=False),

            # Earnings Models
            'earnings_rf_model': self._load_joblib_model('earnings_random_forest_model.pkl', required=True),
            'earnings_rf_selector': self._load_joblib_model('earnings_rf_selector.pkl', required=True),
            'earnings_gbr_model': self._load_joblib_model('earnings_gradient_boosting_model.pkl', required=False),
        }

        models['ffcf_config'] = self._load_pickle_config('ffcf_model_config.pkl')
        models['earnings_config'] = self._load_pickle_config('earnings_model_config.pkl')

        print("✓ ML models loaded successfully")
        return models

    def _load_joblib_model(self, filename: str, required: bool) -> object | None:
        path = f'{self.model_dir}/{filename}'
        try:
            return joblib.load(path)
        except Exception as exc:
            if required:
                print(f"Error loading required model artifact {filename}: {exc}")
                raise
            print(f"Warning: Skipping optional model artifact {filename}: {exc}")
            return None

    def _load_pickle_config(self, filename: str) -> object | None:
        path = f'{self.model_dir}/{filename}'
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as exc:
            print(f"Warning: Skipping optional config {filename}: {exc}")
            return None

    @staticmethod
    def _ensemble_average(predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Average the available model outputs into one ensemble forecast."""
        return np.mean(list(predictions.values()), axis=0)

    @staticmethod
    def _align_feature_columns(
        financial_df: pd.DataFrame,
        expected_factors: list[str] | None,
    ) -> pd.DataFrame:
        """Align a feature frame to the factor order used during model training.

        The saved earnings model config contains a legacy typo,
        `return_on_assetrevenue`, produced by a missing comma in the original
        training notebook. We recreate that placeholder column as zeros so the
        serialized selector/model receive the exact feature count they expect.
        """
        if not expected_factors:
            return financial_df

        aligned = financial_df.copy()
        if "return_on_assetrevenue" in expected_factors and "return_on_assetrevenue" not in aligned.columns:
            aligned["return_on_assetrevenue"] = 0.0

        for factor in expected_factors:
            if factor not in aligned.columns:
                aligned[factor] = 0.0

        return aligned[expected_factors]
    
    def _prepare_features(self, financial_df: pd.DataFrame, expected_factors: list[str] | None = None) -> np.ndarray:
        """Prepare features for model prediction."""
        financial_df = self._align_feature_columns(financial_df, expected_factors)

        # Group by year and take last value
        X = financial_df.groupby([
            pd.Grouper(level='order_book_id'),
            pd.Grouper(level='date', freq='Y')
        ]).last().fillna(0)
        
        X = X.reorder_levels(['order_book_id', 'date'])
        X = X.tail(5)
        
        # Flatten for ML models
        prediction_input = X.values.flatten().reshape(1, -1)
        return prediction_input
    
    def predict_cash_flows(self, financial_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict 5-year cash flows using the available tree-based models."""
        expected_factors = None
        if isinstance(self.models.get('ffcf_config'), dict):
            expected_factors = self.models['ffcf_config'].get('factors')
        features = self._prepare_features(financial_df, expected_factors)
        
        predictions = {}
        
        # Random Forest
        features_selected = self.models['ffcf_rf_selector'].transform(features)
        predictions['random_forest'] = self.models['ffcf_rf_model'].predict(features_selected)[0]
        
        # Gradient Boosting
        if self.models['ffcf_gbr_model'] is not None:
            predictions['gradient_boosting'] = self.models['ffcf_gbr_model'].predict(features)[0]

        predictions['ensemble'] = self._ensemble_average(predictions)
        
        return predictions
    
    def predict_earnings(self, financial_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict 5-year earnings per share using the available tree-based models."""
        expected_factors = None
        if isinstance(self.models.get('earnings_config'), dict):
            expected_factors = self.models['earnings_config'].get('factors')
        features = self._prepare_features(financial_df, expected_factors)
        
        predictions = {}
        
        # Random Forest
        features_selected = self.models['earnings_rf_selector'].transform(features)
        predictions['random_forest'] = self.models['earnings_rf_model'].predict(features_selected)[0]
        
        # Gradient Boosting
        if self.models['earnings_gbr_model'] is not None:
            predictions['gradient_boosting'] = self.models['earnings_gbr_model'].predict(features)[0]

        predictions['ensemble'] = self._ensemble_average(predictions)
        
        return predictions

# %% [markdown]
# ## Fundamental Data Collection

# %%

# ============================================
# 2. FUNDAMENTAL DATA COLLECTION
# ============================================

class FundamentalDataCollector:
    """Collects and processes fundamental financial data"""
    
    def __init__(self):
        self.factors = [
            # Revenue & Profitability
            'revenue', 'operating_revenue', 'net_profit', 'net_profit_parent_company',
            'ebit_ttm', 'ebitda_ttm', 'profit_before_tax',
            
            # Ratios
            'return_on_equity_lyr', 'return_on_asset_lyr', 'debt_to_equity_ratio_ttm', 'current_ratio_ttm',
            'operating_revenue_growth_ratio_ttm', 'net_profit_growth_ratio_ttm',
            
            # Per share metrics
            'basic_earnings_per_share', 'book_value_per_share_ttm', 'free_cash_flow_company_per_share_ttm',
            
            # Balance sheet
            'total_assets', 'total_equity', 'current_assets', 'current_liabilities',
            'long_term_loans', 'short_term_loans', 'cash_equivalent',
            'net_accts_receivable', 'inventory', 'net_fixed_assets_lyr_0',
            'intangible_assets', 'total_liabilities',
        ]
        
        # Valuation factors for DCF
        self.valuation_factors = [
            'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio'
        ]

        # Earnings prediction factors        
        self.earnings_factors = [
            'basic_earnings_per_share',
            'net_profitTTM',
            'net_profit_parent_company',
            'ebit',
            'ebitda',
            'return_on_equity',
            'return_on_asset',

            # revenue
            'revenue',
            'operating_revenue_growth_ratio_ttm',
            'net_profit_growth_ratio_ttm',
            
            # valuation
            'book_value_per_share_ttm',

            # working capital
            'working_capital_ttm',                 # TTM working capital
            'inventory',                           # Inventory levels
            'net_accts_receivable',                # Receivables collection
            'current_assets',                      # Total current assets
            'current_liabilities',                 # Total current liabilities
            'cash_equivalent',                     # Cash position

            # investment and fixed assets
            'net_fixed_assets',                    # Asset base for earnings
            'depreciation_and_amortization',       # Non-cash expense affecting net income

            # balance sheet core metrics
            'total_assets',                        # Size basis
            'total_equity',                        # Book value base
            'equity_parent_company',               # Parent equity
            'intangible_assets',                   # Goodwill/intangibles (earnings quality)

            # liabilites structure
            'short_term_loans',                    # Short-term debt
            'long_term_loans',                     # Long-term debt
            'bond_payable',                        # Bond obligations
            'long_term_liabilities_due_one_year',  # Current portion of long-term debt

            # leverage/liquidity
            'debt_to_equity_ratio_ttm',
            'current_ratio_ttm',
        ]

        # Cash flow prediction factors
        self.cash_flow_factors = [
            # revenue
            'revenue',
            'operating_revenue',
            'net_operating_revenue',
            
            # valuation
            'book_value_per_share_ttm',

            # profitability
            'return_on_equity',
            'return_on_asset',
            "profit_before_tax",
            "net_profit",
            "net_profit_parent_company",
            "ebit",

            # leverage/liquidity
            'debt_to_equity_ratio_ttm',
            'current_ratio',
            
            # growth
            'operating_revenue_growth_ratio_ttm',
            'net_profit_growth_ratio_ttm',

            # free cash flow
            'free_cash_flow_company_per_share_ttm',

            # working capital
            "working_capital_ttm",

            # investment
            "net_current_investment",

            # assets and liabilities
            "current_assets",
            "current_liabilities",

            "depreciation_and_amortization",

            "ebitda",

            # Balance
            "cash_equivalent", 
            "bill_receivable", 
            "net_accts_receivable", 
            "inventory", 
            "long_term_equity_investment", 
            "net_long_term_equity_investment", 
            "net_fixed_assets", 
            "intangible_assets", 
            "short_term_loans", 
            "long_term_liabilities_due_one_year", 
            "long_term_loans", 
            "bond_payable", 
            "long_term_payable", 
            "total_assets", 
            "equity_parent_company", 
            "total_equity",
        ]
    
    def get_order_book_id(self, company_name: str) -> Optional[str]:
        """Get order_book_id from company name"""
        query = str(company_name or "").strip().upper()
        if not query:
            return None

        all_stocks = rq.all_instruments(type='CS')

        # Support direct stock-code inputs from the Streamlit panel.
        if "." in query:
            stock = all_stocks[all_stocks['order_book_id'].str.upper() == query]
            if not stock.empty:
                return stock['order_book_id'].values[0]

        digits = ''.join(ch for ch in query if ch.isdigit())
        if len(digits) == 6:
            stock = all_stocks[all_stocks['order_book_id'].str.startswith(digits)]
            if not stock.empty:
                return stock['order_book_id'].values[0]
            if 'trading_code' in all_stocks.columns:
                stock = all_stocks[all_stocks['trading_code'].astype(str) == digits]
                if not stock.empty:
                    return stock['order_book_id'].values[0]
        
        # Try exact match first
        stock = all_stocks[all_stocks['symbol'] == company_name]
        if not stock.empty:
            return stock['order_book_id'].values[0]
        
        # Try partial match
        stock = all_stocks[all_stocks['abbrev_symbol'].str.contains(company_name, na=False)]
        if not stock.empty:
            return stock['order_book_id'].values[0]
        
        return None

    @staticmethod
    def _date_window() -> tuple[str, str]:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = '2022-01-01'
        return start_date, end_date

    def _query_factors(self, order_book_id: str, logical_factors: list[str]) -> pd.DataFrame:
        data = rq.get_factor(order_book_id, logical_factors, start_date=self._date_window()[0], end_date=self._date_window()[1])
        return data

    @staticmethod
    def _resample_yearly_last_valid(data: pd.DataFrame) -> pd.DataFrame:
        grouped = data.groupby([
            pd.Grouper(level='order_book_id'),
            pd.Grouper(level='date', freq='Y')
        ]).last()
        return grouped.dropna(how='all')
    
    def collect_fundamental_data(self, order_book_id: str) -> pd.DataFrame:
        """Collect latest 5 years of fundamental data"""
        data = self._query_factors(order_book_id, self.factors)
        return self._resample_yearly_last_valid(data)
    
    def collect_earnings_prediction_data(self, order_book_id: str) -> pd.DataFrame:
        """Collect latest 5 years of fundamental data"""
        data = self._query_factors(order_book_id, self.earnings_factors)
        return self._resample_yearly_last_valid(data)
    
    def collect_cash_flow_prediction_data(self, order_book_id: str) -> pd.DataFrame:
        """Collect latest 5 years of fundamental data"""
        data = self._query_factors(order_book_id, self.cash_flow_factors)
        return self._resample_yearly_last_valid(data)
    
    def collect_valuation_data(self, order_book_id: str) -> pd.DataFrame:
        """Collect latest valuation ratios"""
        start_date, end_date = self._date_window()

        data = rq.get_factor(order_book_id, self.valuation_factors, start_date=start_date, end_date=end_date)

        latest = data.groupby([
            pd.Grouper(level='order_book_id'),
            pd.Grouper(level='date', freq='M')
        ]).last().dropna(how='all').tail(1)
        
        return latest


def coalesce_value(data: pd.Series, *keys: str, default: object = np.nan) -> object:
    """Return the first non-null value found in the provided series keys."""
    for key in keys:
        if key in data.index:
            value = data[key]
            if pd.notna(value):
                return value
    return default


def as_percentage_points(value: object) -> float:
    """Normalize ratio-like values into percentage points for display/scoring."""
    if pd.isna(value):
        return float("nan")
    number = float(value)
    return number * 100 if abs(number) <= 1 else number


def format_prompt_number(value: object, decimals: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.{decimals}f}"


def format_prompt_percent(value: object) -> str:
    number = as_percentage_points(value)
    if pd.isna(number):
        return "N/A"
    return f"{number:.1f}%"

# %% [markdown]
# ## DCF Valuation

# %%

# ============================================
# 3. DCF VALUATION
# ============================================

class DCFValuator:
    """Performs Discounted Cash Flow valuation"""
    
    def __init__(self, discount_rate: float = 0.12, terminal_growth: float = 0.03):
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth
    
    def calculate_terminal_value(self, final_cf: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        return final_cf * (1 + self.terminal_growth) / (self.discount_rate - self.terminal_growth)
    
    def calculate_dcf(self, cash_flows: np.ndarray) -> float:
        """Calculate DCF value from 5-year cash flow projections"""
        pv = 0
        
        # Discount each year's cash flow
        for i, cf in enumerate(cash_flows[:5], start=1):
            pv += cf / (1 + self.discount_rate) ** i
        
        # Add terminal value
        terminal_value = self.calculate_terminal_value(cash_flows[4])
        pv += terminal_value / (1 + self.discount_rate) ** 5
        
        return pv

# %% [markdown]
# ## Moat Analysis

# %%

# ============================================
# 4. MOAT ANALYSIS
# ============================================

class MoatAnalyzer:
    """Analyzes company's competitive moat based on fundamentals"""
    
    def analyze_moat(self, fundamental_data: pd.DataFrame) -> Dict:
        """Analyze the durability and size of the company's moat"""
        latest = fundamental_data.tail(1).iloc[0]
        
        moat_indicators = {
            'profitability': {
                'roe': latest.get('return_on_equity_lyr', 0)*100,
                'roa': latest.get('return_on_asset_lyr', 0)*100,
                'net_margin': latest.get('net_profit', 0) / latest.get('revenue', 1) if latest.get('revenue', 0) > 0 else 0,
            },
            'efficiency': {
                'receivables_turnover': latest.get('revenue', 0) / latest.get('net_accts_receivable', 1) if latest.get('net_accts_receivable', 0) > 0 else 0,
                'inventory_turnover': latest.get('revenue', 0) / latest.get('inventory', 1) if latest.get('inventory', 0) > 0 else 0,
                'asset_turnover': latest.get('revenue', 0) / latest.get('total_assets', 1) if latest.get('total_assets', 0) > 0 else 0,
            },
            'financial_health': {
                'debt_to_equity': latest.get('debt_to_equity_ratio_ttm', 0),
                'current_ratio': latest.get('current_ratio_ttm', 0),
                'revenue_growth': latest.get('operating_revenue_growth_ratio_ttm', 0)*100,
                'profit_growth': latest.get('net_profit_growth_ratio_ttm', 0)*100,
            }
        }
        
        # Calculate moat score (0-10)
        score = 0
        
        # Profitability (max 4 points)
        if moat_indicators['profitability']['roe'] > 15:
            score += 2
        elif moat_indicators['profitability']['roe'] > 10:
            score += 1
            
        if moat_indicators['profitability']['net_margin'] > 0.15:
            score += 2
        elif moat_indicators['profitability']['net_margin'] > 0.10:
            score += 1
        
        # Growth (max 3 points)
        if moat_indicators['financial_health']['revenue_growth'] > 0.15:
            score += 1.5
        elif moat_indicators['financial_health']['revenue_growth'] > 0.05:
            score += 0.75
            
        if moat_indicators['financial_health']['profit_growth'] > 0.15:
            score += 1.5
        elif moat_indicators['financial_health']['profit_growth'] > 0.05:
            score += 0.75
        
        # Financial health (max 3 points)
        if moat_indicators['financial_health']['debt_to_equity'] < 0.5:
            score += 1.5
        elif moat_indicators['financial_health']['debt_to_equity'] < 1.0:
            score += 0.75
            
        if moat_indicators['financial_health']['current_ratio'] > 2:
            score += 1.5
        elif moat_indicators['financial_health']['current_ratio'] > 1:
            score += 0.75
        
        # Determine moat size and durability
        if score >= 7:
            moat_size = "Wide Moat"
            durability = "Very Durable"
            description = "Exceptional competitive advantages with high barriers to entry"
        elif score >= 5:
            moat_size = "Narrow Moat"
            durability = "Moderately Durable"
            description = "Good competitive position but with some vulnerabilities"
        elif score >= 3:
            moat_size = "No Moat"
            durability = "Limited Durability"
            description = "Weak competitive advantages, facing significant competition"
        else:
            moat_size = "Negative Moat"
            durability = "Unstable"
            description = "Disadvantages relative to competitors"
        
        return {
            'score': score,
            'moat_size': moat_size,
            'durability': durability,
            'description': description,
            'indicators': moat_indicators
        }

# %% [markdown]
# ## Health Assessment

# %%

# ============================================
# 5. HEALTH ASSESSMENT
# ============================================

class HealthAssessor:
    """Assesses overall financial health of the company"""
    
    def assess_health(self, fundamental_data: pd.DataFrame, 
                      predictions: Dict) -> Dict:
        """Comprehensive health assessment"""
        latest = fundamental_data.tail(1).iloc[0]
        
        health_metrics = {
            'profitability_health': self._assess_profitability(latest),
            'liquidity_health': self._assess_liquidity(latest),
            'leverage_health': self._assess_leverage(latest),
            'growth_health': self._assess_growth(latest, predictions),
            'valuation_health': self._assess_valuation(latest),
        }
        
        # Calculate overall health score (0-100)
        scores = [v['score'] for v in health_metrics.values()]
        overall_score = np.mean(scores)
        
        if overall_score >= 80:
            overall_status = "Excellent"
            recommendation = "Strong Buy"
        elif overall_score >= 65:
            overall_status = "Good"
            recommendation = "Buy"
        elif overall_score >= 50:
            overall_status = "Fair"
            recommendation = "Hold"
        elif overall_score >= 35:
            overall_status = "Poor"
            recommendation = "Sell"
        else:
            overall_status = "Critical"
            recommendation = "Strong Sell"
        
        return {
            'overall_score': overall_score,
            'overall_status': overall_status,
            'recommendation': recommendation,
            'metrics': health_metrics,
            'warning_signs': self._identify_warning_signs(latest, health_metrics)
        }
    
    def _assess_profitability(self, data: pd.Series) -> Dict:
        score = 0
        roe = as_percentage_points(coalesce_value(data, 'return_on_equity_lyr', 'return_on_equity', default=0.0))
        roa = as_percentage_points(coalesce_value(data, 'return_on_asset_lyr', 'return_on_asset', default=0.0))
        
        if roe > 15:
            score += 40
        elif roe > 10:
            score += 30
        elif roe > 5:
            score += 20
        elif roe > 0:
            score += 10
        
        if roa > 5:
            score += 30
        elif roa > 3:
            score += 20
        elif roa > 0:
            score += 10
        
        return {'score': score, 'roe': roe, 'roa': roa}
    
    def _assess_liquidity(self, data: pd.Series) -> Dict:
        score = 0
        current_ratio = data.get('current_ratio_ttm', 0)
        cash = data.get('cash_equivalent', 0)
        
        if current_ratio > 2:
            score += 50
        elif current_ratio > 1.5:
            score += 35
        elif current_ratio > 1:
            score += 20
        elif current_ratio > 0.5:
            score += 10
        
        if cash > 0:
            score += 20
        
        return {'score': score, 'current_ratio': current_ratio, 'cash': cash}
    
    def _assess_leverage(self, data: pd.Series) -> Dict:
        score = 0
        debt_to_equity = data.get('debt_to_equity_ratio_ttm', 0)
        
        if debt_to_equity < 0.3:
            score += 50
        elif debt_to_equity < 0.5:
            score += 40
        elif debt_to_equity < 1:
            score += 25
        elif debt_to_equity < 2:
            score += 10
        
        return {'score': score, 'debt_to_equity': debt_to_equity}
    
    def _assess_growth(self, data: pd.Series, predictions: Dict) -> Dict:
        score = 0
        revenue_growth = data.get('operating_revenue_growth_ratio_ttm', 0)
        profit_growth = data.get('net_profit_growth_ratio_ttm', 0)
        
        # Historical growth
        if revenue_growth > 0.15:
            score += 20
        elif revenue_growth > 0.05:
            score += 15
        elif revenue_growth > 0:
            score += 10
        
        if profit_growth > 0.15:
            score += 20
        elif profit_growth > 0.05:
            score += 15
        elif profit_growth > 0:
            score += 10
        
        # Future growth from ML predictions
        forecast_key = 'random_forest' if 'random_forest' in predictions else 'ensemble'
        if forecast_key in predictions:
            avg_future_earnings = np.mean(predictions[forecast_key][:3])
            current_eps = data.get('basic_earnings_per_share', 1)
            if current_eps > 0:
                future_growth_rate = (avg_future_earnings / current_eps - 1)
                if future_growth_rate > 0.1:
                    score += 20
                elif future_growth_rate > 0.05:
                    score += 15
                elif future_growth_rate > 0:
                    score += 10
        
        return {'score': score, 'revenue_growth': revenue_growth, 'profit_growth': profit_growth}
    
    def _assess_valuation(self, data: pd.Series) -> Dict:
        score = 70  # Default neutral score
        pe = data.get('pe_ratio', 0)
        
        if 10 < pe < 20:
            score = 80
        elif pe <= 10:
            score = 90
        elif pe >= 30:
            score = 50
        elif pe >= 25:
            score = 60
        
        return {'score': score, 'pe_ratio': pe}
    
    def _identify_warning_signs(self, data: pd.Series, metrics: Dict) -> List[str]:
        warnings = []
        
        if data.get('debt_to_equity_ratio_ttm', 0) > 1.5:
            warnings.append("High leverage: Debt-to-equity ratio exceeds 1.5x")
        
        if data.get('current_ratio_ttm', 0) < 1:
            warnings.append("Liquidity concern: Current ratio below 1.0")
        
        if data.get('return_on_equity_lyr', 0) * 100 < 5:
            warnings.append("Weak profitability: ROE below 5%")
        
        if data.get('operating_revenue_growth_ratio_ttm', 0) < 0:
            warnings.append("Declining revenue: Negative growth rate")
        
        if data.get('net_profit_growth_ratio_ttm', 0) < 0:
            warnings.append("Declining profits: Negative earnings growth")
        
        return warnings

# %% [markdown]
# ## LLM Integration

# %%

# ============================================
# 6. LLM INTEGRATION (DeepSeek/OpenAI)
# ============================================

from openai import OpenAI
import os
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"


def resolve_llm_settings(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Optional[str]]:
    """Resolve one normalized LLM config from explicit args + environment.

    Supported providers:
    - openai: uses OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL
    - deepseek: uses DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL

    Shared overrides:
    - LLM_PROVIDER
    - LLM_API_KEY
    - LLM_BASE_URL
    - LLM_MODEL
    """
    resolved_provider = (provider or os.environ.get("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)).strip().lower()

    if resolved_provider == "deepseek":
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LLM_API_KEY")
        resolved_base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("LLM_BASE_URL") or DEFAULT_DEEPSEEK_BASE_URL
        resolved_model = model or os.environ.get("DEEPSEEK_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_DEEPSEEK_MODEL
    else:
        resolved_provider = "openai"
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        resolved_model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or DEFAULT_OPENAI_MODEL

    return {
        "provider": resolved_provider,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
        "model": resolved_model,
    }

class LLMAnalyzer:
    """Handles LLM-powered analysis and report generation"""
    
    def __init__(
            self,
            provider: Optional[str] = None,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None
    ):
        settings = resolve_llm_settings(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
        self.provider = settings["provider"] or DEFAULT_LLM_PROVIDER
        self.api_key = settings["api_key"]
        self.base_url = settings["base_url"] or DEFAULT_OPENAI_BASE_URL
        self.model = settings["model"] or DEFAULT_OPENAI_MODEL
        
        if not self.api_key:
            print(
                "Warning: 未找到 LLM API Key。"
                "请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY。"
            )
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def generate_report(self, company_name: str, order_book_id: str,
                       fundamental_data: pd.DataFrame,
                       cash_flow_predictions: Dict,
                       earnings_predictions: Dict,
                       dcf_results: Dict,
                       moat_analysis: Dict,
                       health_assessment: Dict,
                       price_data: pd.DataFrame) -> str:
        """Generate comprehensive financial analysis report"""
        
        if not self.client:
            return (
                "LLM 未配置。请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，"
                "也可选配 LLM_PROVIDER / OPENAI_MODEL / DEEPSEEK_MODEL。"
            )
        
        # Prepare data for LLM
        latest_data = fundamental_data.tail(1).iloc[0]
        latest_close = np.nan
        if not price_data.empty and 'close' in price_data.columns:
            latest_close = price_data['close'].iloc[0]
        cashflow_key = 'random_forest' if 'random_forest' in cash_flow_predictions else 'ensemble'
        earnings_key = 'random_forest' if 'random_forest' in earnings_predictions else 'ensemble'
        roe = coalesce_value(latest_data, 'return_on_equity_lyr', 'return_on_equity', default=np.nan)
        roa = coalesce_value(latest_data, 'return_on_asset_lyr', 'return_on_asset', default=np.nan)
        debt_to_equity = coalesce_value(latest_data, 'debt_to_equity_ratio_ttm', 'debt_to_equity', default=np.nan)
        current_ratio = coalesce_value(latest_data, 'current_ratio_ttm', 'current_ratio', default=np.nan)
        revenue_growth = coalesce_value(latest_data, 'operating_revenue_growth_ratio_ttm', 'revenue_growth_rate', default=np.nan)
        profit_growth = coalesce_value(latest_data, 'net_profit_growth_ratio_ttm', 'net_profit_growth_rate', default=np.nan)
        
        prompt = f"""你是一名专业的中文金融分析师。请基于以下数据，为 {company_name}（{order_book_id}）生成一份结构清晰、数据驱动的中文投资分析报告。

## 当前股价（人民币）
- 最新收盘价: {format_prompt_number(latest_close, 2)}

## 当前财务状况（最新年度，货币单位为人民币）

### 收入与盈利
- Revenue: {format_prompt_number(latest_data.get('revenue'), 0)}
- Operating Revenue: {format_prompt_number(latest_data.get('operating_revenue'), 0)}
- Net Profit: {format_prompt_number(latest_data.get('net_profit'), 0)}
- Net Profit (Parent Company): {format_prompt_number(latest_data.get('net_profit_parent_company'), 0)}
- EBIT (TTM): {format_prompt_number(coalesce_value(latest_data, 'ebit_ttm', 'ebit'), 0)}
- EBITDA (TTM): {format_prompt_number(coalesce_value(latest_data, 'ebitda_ttm', 'ebitda'), 0)}
- Profit Before Tax: {format_prompt_number(latest_data.get('profit_before_tax'), 0)}

### 比率
- ROE（上年）: {format_prompt_percent(roe)}
- ROA（上年）: {format_prompt_percent(roa)}
- Debt/Equity（TTM）: {format_prompt_number(debt_to_equity, 2)}
- Current Ratio（TTM）: {format_prompt_number(current_ratio, 2)}
- Operating Revenue Growth（TTM）: {format_prompt_percent(revenue_growth)}
- Net Profit Growth（TTM）: {format_prompt_percent(profit_growth)}

### 每股指标
- Basic EPS: {format_prompt_number(latest_data.get('basic_earnings_per_share'), 2)}
- Book Value Per Share (TTM): {format_prompt_number(coalesce_value(latest_data, 'book_value_per_share_ttm', 'book_value_per_share'), 2)}
- Free Cash Flow Per Share (TTM): {format_prompt_number(coalesce_value(latest_data, 'free_cash_flow_company_per_share_ttm', 'free_cash_flow_company_per_share'), 2)}

### 资产负债表
- Total Assets: {format_prompt_number(latest_data.get('total_assets'), 0)}
- Total Equity: {format_prompt_number(latest_data.get('total_equity'), 0)}
- Current Assets: {format_prompt_number(latest_data.get('current_assets'), 0)}
- Current Liabilities: {format_prompt_number(latest_data.get('current_liabilities'), 0)}
- Long-term Loans: {format_prompt_number(latest_data.get('long_term_loans'), 0)}
- Short-term Loans: {format_prompt_number(latest_data.get('short_term_loans'), 0)}
- Cash & Equivalents: {format_prompt_number(latest_data.get('cash_equivalent'), 0)}
- Net Accounts Receivable: {format_prompt_number(latest_data.get('net_accts_receivable'), 0)}
- Inventory: {format_prompt_number(latest_data.get('inventory'), 0)}
- Net Fixed Assets: {format_prompt_number(coalesce_value(latest_data, 'net_fixed_assets_lyr_0', 'net_fixed_assets'), 0)}
- Intangible Assets: {format_prompt_number(latest_data.get('intangible_assets'), 0)}
- Total Liabilities: {format_prompt_number(latest_data.get('total_liabilities'), 0)}

### 估值倍数
- P/E Ratio: {format_prompt_number(latest_data.get('pe_ratio'), 2)}
- P/B Ratio: {format_prompt_number(latest_data.get('pb_ratio'), 2)}
- P/S Ratio: {format_prompt_number(latest_data.get('ps_ratio'), 2)}
- P/CF Ratio: {format_prompt_number(latest_data.get('pcf_ratio'), 2)}

## 机器学习模型预测（未来 5 年）

### 每股自由现金流预测（Random Forest）
{', '.join([f'第 {i+1} 年: {cf:.2f}' for i, cf in enumerate(cash_flow_predictions[cashflow_key])])}

### 每股收益预测（Random Forest）
{', '.join([f'第 {i+1} 年: {eps:.2f}' for i, eps in enumerate(earnings_predictions[earnings_key])])}

## DCF 估值
- DCF 每股价值: {dcf_results:.2f}

## 护城河分析
- 护城河评分: {moat_analysis['score']:.1f}/10
- 护城河类型: {moat_analysis['moat_size']}
- 持久性: {moat_analysis['durability']}
- 描述: {moat_analysis['description']}

## 财务健康度
- 总体健康度评分: {health_assessment['overall_score']:.1f}/100
- 健康状态: {health_assessment['overall_status']}
- 系统建议: {health_assessment['recommendation']}

### 风险信号
{chr(10).join(['- ' + w for w in health_assessment['warning_signs']]) if health_assessment['warning_signs'] else '- 暂未识别到主要风险信号'}

## 写作要求
请用中文撰写，并包含以下部分：
1. 执行摘要
2. 财务健康度分析
3. 护城河评估
4. DCF 估值分析
5. 未来展望
6. 风险因素
7. 投资建议
8. 后续跟踪指标

请保持客观，不要夸大模型结论；明确说明这是基于本地模型和 RQData 数据的辅助分析，不构成投资建议。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一名严谨的中文金融分析师，擅长 DCF 估值、竞争优势分析和投资研究。请始终用中文回答。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                # max_tokens=4000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"生成 LLM 报告失败: {str(e)}"
    
    def chat(self, messages: List[Dict], user_message: str) -> str:
        """Interactive chat with LLM about the analysis"""
        if not self.client:
            return (
                "LLM 未配置。请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，"
                "也可选配 LLM_PROVIDER / OPENAI_MODEL / DEEPSEEK_MODEL。"
            )
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
                # max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"问答失败: {str(e)}"

# %% [markdown]
# ## Main Integration System

# %%

# ============================================
# 7. MAIN INTEGRATION SYSTEM
# ============================================

DISCOUNT_RATE=0.12
TERMINAL_GROWTH=0.03

class FinancialAnalysisSystem:
    """Main system orchestrating all components"""
    
    def __init__(self, model_dir='./model'):
        print("Initializing Financial Analysis System...")
        initialize_rqdatac()
        self.predictor = FinancialPredictor(model_dir)
        self.data_collector = FundamentalDataCollector()
        self.dcf_valuator = DCFValuator(DISCOUNT_RATE, TERMINAL_GROWTH)
        self.moat_analyzer = MoatAnalyzer()
        self.health_assessor = HealthAssessor()
        self.llm_analyzer = LLMAnalyzer()
        print("System ready!\n")
    
    def analyze_company(self, company_name: str, generate_llm_report: bool = True) -> Tuple[Dict, str, List]:
        """Complete analysis pipeline for a company"""
        
        print(f"\n{'='*60}")
        print(f"Analyzing {company_name}...")
        print(f"{'='*60}\n")
        
        # Step 1: Get order_book_id
        print("1. Looking up company...")
        order_book_id = self.data_collector.get_order_book_id(company_name)
        if not order_book_id:
            raise ValueError(f"Company '{company_name}' not found")
        print(f"   Found: {order_book_id}\n")
        
        # Step 2: Collect fundamental data
        print("2. Collecting fundamental data...")
        fundamental_data = self.data_collector.collect_fundamental_data(order_book_id)
        if fundamental_data.empty:
            raise ValueError("No fundamental data available")
        print(f"   Collected {len(fundamental_data)} years of data\n")
        print(f"   Fundamental Data: {fundamental_data.head(5)}")
        
        # Step 3: Get latest valuation and prediction data
        print("3. Collecting valuation and prediction data...")
        valuation_data = self.data_collector.collect_valuation_data(order_book_id)
        if not valuation_data.empty:
            # Broadcast the latest market multiples onto the analysis frame so
            # downstream health/valuation checks can read them from the latest row.
            for col in valuation_data.columns:
                fundamental_data[col] = valuation_data[col].iloc[0]
        print("   Valuation data collected\n")
        print(f"   Valuation Data: {valuation_data}")

        earnings_prediction_data = self.data_collector.collect_earnings_prediction_data(order_book_id)
        print("   Earnings prediction data collected\n")

        cash_flow_prediction_data = self.data_collector.collect_cash_flow_prediction_data(order_book_id)
        print("   Cash flow prediction data collected\n")
        
        # Step 4: Make ML predictions
        print("4. Generating ML predictions...")
        cash_flow_preds = self.predictor.predict_cash_flows(cash_flow_prediction_data)
        earnings_preds = self.predictor.predict_earnings(earnings_prediction_data)
        print("   Predictions complete\n")
        
        # Step 5: Calculate DCF
        print("5. Performing DCF valuation...")
        dcf_key = 'random_forest' if 'random_forest' in cash_flow_preds else 'ensemble'
        dcf_value = self.dcf_valuator.calculate_dcf(cash_flow_preds[dcf_key])
        print(f"   DCF Value: ${dcf_value:.2f}")
        
        # Step 6: Moat analysis
        print("6. Analyzing competitive moat...")
        moat_analysis = self.moat_analyzer.analyze_moat(fundamental_data)
        print(f"   Moat Score: {moat_analysis['score']:.1f}/10")
        print(f"   Moat Size: {moat_analysis['moat_size']}\n")
        
        # Step 7: Health assessment
        print("7. Assessing financial health...")
        health_assessment = self.health_assessor.assess_health(fundamental_data, earnings_preds)
        print(f"   Health Score: {health_assessment['overall_score']:.1f}/100")
        print(f"   Recommendation: {health_assessment['recommendation']}\n")
        
        # Step 8: Get latest price data
        print("8. Getting latest price data...")
        price_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        price_data = rq.get_price(
            order_book_id,
            start_date=price_start,
            end_date=datetime.now().strftime('%Y-%m-%d'),
            fields=['close'],
        )
        if isinstance(price_data, pd.DataFrame):
            price_data = price_data.dropna(subset=['close']).tail(1)
        else:
            price_data = pd.DataFrame(columns=['close'])
        print("   Latest price data collected\n")
        print(f"   Price Data: {price_data}")

        # Step 9: Generate LLM report
        if generate_llm_report:
            print("9. Generating AI-powered report...")
            report = self.llm_analyzer.generate_report(
                company_name, order_book_id, fundamental_data,
                cash_flow_preds, earnings_preds, dcf_value,
                moat_analysis, health_assessment, price_data
            )
            print("   Report generated!\n")
        else:
            print("9. Skipping AI-powered report by user choice.\n")
            report = "已跳过 AI 报告生成；结构化 DCF、预测、护城河和健康度结果已完成。"
        
        # Package results for chat
        cashflow_context_key = 'random_forest' if 'random_forest' in cash_flow_preds else 'ensemble'
        earnings_context_key = 'random_forest' if 'random_forest' in earnings_preds else 'ensemble'
        chat_context = [
            {"role": "system", "content": f"你正在分析 {company_name}（{order_book_id}）。请始终用中文回答用户关于该公司 DCF、财务健康度、护城河、预测和风险的问题。"},
            {"role": "system", "content": f"财务健康度: {health_assessment}"},
            {"role": "system", "content": f"护城河分析: {moat_analysis}"},
            {"role": "system", "content": f"DCF 每股价值: {dcf_value}"},
            {"role": "system", "content": f"最新收盘价: {price_data['close'].iloc[0] if not price_data.empty else 'N/A'}"},
            {"role": "system", "content": f"现金流预测: {cash_flow_preds[cashflow_context_key]}"},
            {"role": "system", "content": f"EPS 预测: {earnings_preds[earnings_context_key]}"},
        ]
        
        results = {
            'company_name': company_name,
            'order_book_id': order_book_id,
            'fundamental_data': fundamental_data,
            'cash_flow_predictions': cash_flow_preds,
            'earnings_predictions': earnings_preds,
            'dcf_value': dcf_value,
            'moat_analysis': moat_analysis,
            'health_assessment': health_assessment,
            'price_data': price_data,
        }
        
        return results, report, chat_context
    
    def interactive_chat(self, chat_context: List[Dict]):
        """Interactive Q&A session with the LLM"""
        print("\n" + "="*60)
        print("INTERACTIVE Q&A SESSION")
        print("="*60)
        print("Ask questions about the company's financial health, valuation, or investment prospects.")
        print("Type 'quit' to exit.\n")
        
        messages = chat_context.copy()
        
        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the Financial Analysis System!")
                break
            
            if not question:
                continue
            
            print("\n🤔 Analyzing...\n")
            response = self.llm_analyzer.chat(messages, question)
            print(f"💡 Answer:\n{response}\n")
            
            # Update context with the Q&A
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": response})

# %% [markdown]
# ## Main Execution

# %%

# ============================================
# 8. MAIN EXECUTION
# ============================================

def main():
    """
    Run the Financial Analysis System
    """
    # Initialize the system
    system = FinancialAnalysisSystem(model_dir='./model')
    
    # Get company name from user
    print("\n" + "="*60)
    company_name = input("Enter company name (e.g., '中国平安' or 'Ping An'): ").strip()
    
    # try:
    # Run complete analysis
    results, report, chat_context = system.analyze_company(company_name)
    
    # Display the report
    print("\n" + "="*60)
    print("INVESTMENT ANALYSIS REPORT")
    print("="*60)
    print(report)
    
    # Ask if user wants interactive Q&A
    print("\n" + "="*60)
    interactive = input("\nDo you want to ask follow-up questions? (yes/no): ").strip().lower()
    
    if interactive in ['yes', 'y']:
        system.interactive_chat(chat_context)
    else:
        print("\nThank you for using the Financial Analysis System!")
            
    # except Exception as e:
    #     print(f"\n❌ Error: {e}")
    #     print("\nPossible issues:")
    #     print("1. Check if company name is correct")
    #     print("2. Verify rqdatac is properly initialized")
    #     print("3. Ensure all model files exist in the directory")
    #     print("4. Check your internet connection")

if __name__ == "__main__":
    main()

# %%
