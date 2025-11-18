"""
Claude AI-Powered Recommendation Engine

Provides intelligent, context-aware recommendations for fraud detection using Claude API.
"""

import os
import streamlit as st
from typing import Dict, Any, List, Optional
from functools import lru_cache
import hashlib
import json

# Note: Using anthropic API - requires ANTHROPIC_API_KEY environment variable
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AIRecommendationEngine:
    """
    AI-powered recommendation engine using Claude API.
    Provides context-aware insights for fraud detection and risk analysis.
    """

    def __init__(self):
        """Initialize the AI recommendation engine."""
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None

        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception:
                self.client = None

    def _get_cache_key(self, context: str, data: Dict[str, Any]) -> str:
        """Generate cache key for recommendations."""
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(f"{context}:{data_str}".encode()).hexdigest()

    def _call_claude(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Call Claude API with caching.

        Args:
            prompt: The prompt to send to Claude
            max_tokens: Maximum tokens in response

        Returns:
            Claude's response text
        """
        if not self.client:
            return self._get_fallback_recommendation(prompt)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception:
            return self._get_fallback_recommendation(prompt)

    def _get_fallback_recommendation(self, prompt: str) -> str:
        """Generate fallback recommendation when API is unavailable."""
        # Simple rule-based fallbacks based on keywords
        if "high risk" in prompt.lower():
            return "This pattern indicates elevated fraud risk. Recommend enhanced monitoring and manual review."
        elif "threshold" in prompt.lower():
            return "Current thresholds appear stable. Monitor for seasonal variations and adjust if false positive rate changes."
        elif "trend" in prompt.lower():
            return "Trend analysis suggests normal variation. Continue monitoring for anomalous patterns."
        elif "correlation" in prompt.lower():
            return "Strong correlation detected. Consider creating composite rules for improved detection."
        else:
            return "Continue monitoring. System operating within expected parameters."

    def get_risk_recommendation(self, risk_score: float, amount: float,
                                context: Dict[str, Any]) -> str:
        """
        Get AI recommendation for a specific risk score.

        Args:
            risk_score: Transaction risk score (0-1)
            amount: Transaction amount
            context: Additional context (country, merchant, etc.)

        Returns:
            AI-generated recommendation
        """
        prompt = f"""Analyze this transaction and provide a brief fraud risk assessment:

Risk Score: {risk_score:.3f}
Amount: ${amount:,.2f}
Context: {context}

Provide a 1-2 sentence risk assessment with actionable recommendation. Be direct and specific.
Format as: "AI Analysis: [your assessment]"
"""

        cache_key = self._get_cache_key("risk", {"score": risk_score, "amount": amount, "context": context})

        # Check session cache
        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=150)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation

    def get_threshold_recommendation(self, current_threshold: float,
                                    recent_stats: Dict[str, Any]) -> str:
        """
        Get AI recommendation for threshold adjustment.

        Args:
            current_threshold: Current decision threshold
            recent_stats: Recent performance statistics

        Returns:
            AI-generated threshold recommendation
        """
        prompt = f"""Analyze these fraud detection threshold metrics and suggest adjustments:

Current Threshold: {current_threshold}
Recent Statistics:
- False Positive Rate: {recent_stats.get('false_positive_rate', 0):.2%}
- Detection Rate: {recent_stats.get('detection_rate', 0):.2%}
- Review Queue Size: {recent_stats.get('queue_size', 0)}
- Avg Processing Time: {recent_stats.get('avg_time', 0)} minutes

Provide a brief recommendation on whether to adjust thresholds and why. 2-3 sentences max.
Format as: "AI Analysis: [your recommendation]"
"""

        cache_key = self._get_cache_key("threshold", {"threshold": current_threshold, **recent_stats})

        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=200)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation

    def get_trend_analysis(self, metric_name: str, trend_data: List[float]) -> str:
        """
        Get AI analysis of a trend.

        Args:
            metric_name: Name of the metric being analyzed
            trend_data: Recent trend values

        Returns:
            AI-generated trend analysis
        """
        prompt = f"""Analyze this fraud detection metric trend:

Metric: {metric_name}
Recent Values (last 7 days): {trend_data}
Current Value: {trend_data[-1] if trend_data else 0}
Average: {sum(trend_data)/len(trend_data) if trend_data else 0:.2f}

Provide a brief 1-2 sentence analysis of the trend and what it suggests for fraud operations.
Format as: "AI Analysis: [your analysis]"
"""

        cache_key = self._get_cache_key("trend", {"metric": metric_name, "data": str(trend_data)})

        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=150)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation

    def get_rule_optimization(self, rule_name: str, performance: Dict[str, Any]) -> str:
        """
        Get AI recommendation for rule optimization.

        Args:
            rule_name: Name of the rule
            performance: Rule performance metrics

        Returns:
            AI-generated optimization recommendation
        """
        prompt = f"""Analyze this fraud detection rule's performance and suggest optimizations:

Rule: {rule_name}
Performance Metrics:
- Precision: {performance.get('precision', 0):.2%}
- Trigger Frequency: {performance.get('frequency', 0)}
- False Positive Rate: {performance.get('fp_rate', 0):.2%}
- Confirmed Fraud Catches: {performance.get('catches', 0)}

Provide a brief 1-2 sentence recommendation on how to optimize this rule.
Format as: "AI Analysis: [your recommendation]"
"""

        cache_key = self._get_cache_key("rule", {"name": rule_name, **performance})

        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=150)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation

    def get_pattern_insight(self, pattern_type: str, pattern_data: Dict[str, Any]) -> str:
        """
        Get AI insight about a detected pattern.

        Args:
            pattern_type: Type of pattern (e.g., "geographic", "temporal", "behavioral")
            pattern_data: Pattern details

        Returns:
            AI-generated pattern insight
        """
        prompt = f"""Analyze this detected fraud pattern:

Pattern Type: {pattern_type}
Pattern Details: {pattern_data}

Provide a brief 1-2 sentence insight about what this pattern suggests and recommended action.
Format as: "AI Analysis: [your insight]"
"""

        cache_key = self._get_cache_key("pattern", {"type": pattern_type, **pattern_data})

        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=150)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation

    def get_ml_performance_insight(self, accuracy: float, precision: float,
                                   recall: float, auc_roc: float, trend: str = 'stable') -> str:
        """
        Get AI insight about ML model performance.

        Args:
            accuracy: Model accuracy score
            precision: Model precision score
            recall: Model recall score
            auc_roc: AUC-ROC score
            trend: Performance trend ('improving', 'stable', 'declining')

        Returns:
            AI-generated ML performance insight
        """
        prompt = f"""Analyze these machine learning model performance metrics:

Model Performance:
- Accuracy: {accuracy:.1%}
- Precision: {precision:.1%}
- Recall: {recall:.1%}
- AUC-ROC: {auc_roc:.3f}
- Recent Trend: {trend}

Provide a brief 2-3 sentence analysis of the model's performance and any recommendations.
Format as: "AI Analysis: [your analysis]"
"""

        cache_key = self._get_cache_key("ml_performance", {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc_roc": auc_roc,
            "trend": trend
        })

        if 'ai_cache' not in st.session_state:
            st.session_state.ai_cache = {}

        if cache_key in st.session_state.ai_cache:
            return st.session_state.ai_cache[cache_key]

        recommendation = self._call_claude(prompt, max_tokens=200)
        st.session_state.ai_cache[cache_key] = recommendation
        return recommendation


# Global instance
_ai_engine = None

def get_ai_engine() -> AIRecommendationEngine:
    """Get or create the global AI recommendation engine instance."""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AIRecommendationEngine()
    return _ai_engine


def render_ai_insight(title: str, recommendation: str, icon: str = "🤖"):
    """
    Render an AI insight box.

    Args:
        title: Title of the insight
        recommendation: The AI recommendation text
        icon: Icon to display
    """
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #ffd700;
                margin: 10px 0;">
        <div style="color: white; font-weight: bold; margin-bottom: 8px;">
            {icon} {title}
        </div>
        <div style="color: #f0f0f0; font-size: 14px; line-height: 1.5;">
            {recommendation}
        </div>
    </div>
    """, unsafe_allow_html=True)



