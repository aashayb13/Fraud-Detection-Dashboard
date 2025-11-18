"""
Explainability Module - AI-Powered Hover Insights for Visualizations

Provides context-aware explanations for data points across all dashboard visualizations.
"""

from typing import Dict, Any, Optional


class ExplainabilityEngine:
    """Generates AI-powered insights for visualization data points"""

    def __init__(self):
        """Initialize the explainability engine"""
        pass

    # ============================================================================
    # RULE PERFORMANCE EXPLAINABILITY
    # ============================================================================

    def explain_rule_performance(self, rule_name: str, metrics: Dict[str, Any]) -> str:
        """
        Generate explanation for a rule's performance metrics

        Args:
            rule_name: Name of the fraud detection rule
            metrics: Dict containing precision, frequency, fp_rate, etc.

        Returns:
            HTML formatted explanation with insights and recommendations
        """
        precision = metrics.get('precision', 0)
        frequency = metrics.get('frequency', 0)
        fp_rate = metrics.get('fp_rate', 0)
        fraud_caught = metrics.get('fraud_caught', 0)

        # Performance assessment
        if precision >= 0.90:
            performance = "⭐ Excellent"
            assessment = "This rule has outstanding accuracy with minimal false positives."
        elif precision >= 0.75:
            performance = "✅ Good"
            assessment = "This rule performs well and reliably catches fraud."
        elif precision >= 0.60:
            performance = "⚠️ Fair"
            assessment = "This rule is moderately effective but may need tuning."
        else:
            performance = "🔴 Needs Improvement"
            assessment = "This rule generates many false positives and should be reviewed."

        # Frequency assessment
        if frequency > 300:
            freq_note = "High frequency - This is a commonly triggered rule."
        elif frequency > 100:
            freq_note = "Moderate frequency - Triggers regularly."
        else:
            freq_note = "Low frequency - Rare but potentially important signals."

        # False positive assessment
        if fp_rate < 0.10:
            fp_note = "Low false positive rate - Excellent specificity."
        elif fp_rate < 0.25:
            fp_note = "Moderate false positives - Acceptable for most use cases."
        else:
            fp_note = "High false positives - May overwhelm analysts."

        # Recommendations
        recommendations = []
        if precision < 0.70:
            recommendations.append("• Consider adjusting thresholds or rule logic")
        if fp_rate > 0.30:
            recommendations.append("• Review recent false positives for patterns")
        if fraud_caught < 20:
            recommendations.append("• Monitor - may not catch enough fraud to justify cost")
        if frequency > 400 and precision < 0.80:
            recommendations.append("• High volume + low precision = analyst burden")

        rec_text = "<br>".join(recommendations) if recommendations else "• No immediate actions needed - continue monitoring"

        return f"""
        <div style="background: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 400px;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">🔍 {rule_name}</h4>

            <div style="margin: 10px 0;">
                <strong>Performance:</strong> {performance}<br>
                <span style="color: #4b5563; font-size: 14px;">{assessment}</span>
            </div>

            <div style="background: #f3f4f6; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <div style="margin: 5px 0;"><strong>📊 Metrics</strong></div>
                <div style="font-size: 13px; color: #374151;">
                    • Precision: {precision*100:.1f}%<br>
                    • Frequency: {frequency} triggers<br>
                    • False Positive Rate: {fp_rate*100:.1f}%<br>
                    • Fraud Caught: {fraud_caught} cases
                </div>
            </div>

            <div style="margin: 10px 0;">
                <div style="font-size: 13px; color: #6b7280;">
                    {freq_note}<br>
                    {fp_note}
                </div>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <strong style="color: #059669;">💡 Recommendations:</strong><br>
                <div style="font-size: 13px; color: #065f46; margin-top: 5px;">
                    {rec_text}
                </div>
            </div>
        </div>
        """

    def explain_transaction_decision(self, transaction: Dict[str, Any],
                                     risk_score: float,
                                     triggered_rules: list) -> str:
        """
        Explain why a transaction was flagged/cleared

        Args:
            transaction: Transaction details
            risk_score: Calculated risk score
            triggered_rules: List of triggered rule names

        Returns:
            HTML formatted explanation
        """
        amount = transaction.get('amount', 0)
        tx_type = transaction.get('transaction_type', 'Unknown')

        # Decision logic
        if risk_score >= 0.80:
            decision = "🔴 High Risk - Manual Review Required"
            action = "ESCALATE to senior analyst for immediate review"
            color = "#ef4444"
        elif risk_score >= 0.60:
            decision = "🟠 Medium Risk - Review Needed"
            action = "REVIEW within standard queue processing time"
            color = "#f59e0b"
        elif risk_score >= 0.30:
            decision = "🟡 Low Risk - Quick Check"
            action = "MONITOR and perform quick validation"
            color = "#eab308"
        else:
            decision = "🟢 Auto-Cleared"
            action = "APPROVE automatically - low risk"
            color = "#10b981"

        # Top contributing rules
        rule_list = "<br>".join([f"• {rule}" for rule in triggered_rules[:5]])

        return f"""
        <div style="background: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 450px;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">Transaction Analysis</h4>

            <div style="background: {color}15; padding: 10px; border-radius: 5px;
                        border-left: 4px solid {color}; margin: 10px 0;">
                <strong style="color: {color};">{decision}</strong><br>
                <span style="font-size: 13px; color: #374151;">Risk Score: {risk_score*100:.1f}%</span>
            </div>

            <div style="margin: 10px 0;">
                <strong>📋 Transaction Details:</strong><br>
                <div style="font-size: 13px; color: #4b5563; margin: 5px 0;">
                    • Amount: ${amount:,.2f}<br>
                    • Type: {tx_type}<br>
                    • Rules Triggered: {len(triggered_rules)}
                </div>
            </div>

            <div style="margin: 10px 0;">
                <strong>🎯 Top Contributing Rules:</strong><br>
                <div style="font-size: 12px; color: #6b7280; margin: 5px 0;">
                    {rule_list or "• No major red flags"}
                </div>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <strong style="color: #1e40af;">⚡ Recommended Action:</strong><br>
                <div style="font-size: 13px; color: #1e3a8a; margin-top: 5px;">
                    {action}
                </div>
            </div>
        </div>
        """

    def explain_fraud_scenario(self, scenario_name: str, stats: Dict[str, Any]) -> str:
        """
        Explain a fraud scenario pattern

        Args:
            scenario_name: Name of the fraud scenario
            stats: Statistics about the scenario

        Returns:
            HTML formatted explanation
        """
        occurrence_rate = stats.get('occurrence_rate', 0)
        avg_loss = stats.get('avg_loss', 0)
        detection_rate = stats.get('detection_rate', 0)

        # Risk assessment
        if occurrence_rate > 0.15:
            severity = "🔴 High Frequency"
        elif occurrence_rate > 0.05:
            severity = "🟠 Moderate"
        else:
            severity = "🟡 Low Frequency"

        return f"""
        <div style="background: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 400px;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">🎭 {scenario_name}</h4>

            <div style="margin: 10px 0;">
                <strong>Severity:</strong> {severity}<br>
                <span style="font-size: 13px; color: #4b5563;">
                    Occurs in {occurrence_rate*100:.2f}% of transactions
                </span>
            </div>

            <div style="background: #fef3c7; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>💰 Financial Impact:</strong><br>
                <span style="font-size: 14px; color: #92400e;">
                    Average Loss: ${avg_loss:,.2f} per incident
                </span>
            </div>

            <div style="margin: 10px 0;">
                <strong>✅ Detection Rate:</strong> {detection_rate*100:.1f}%<br>
                <span style="font-size: 13px; color: #6b7280;">
                    {'Excellent detection capability' if detection_rate > 0.85 else
                     'Good detection' if detection_rate > 0.70 else
                     'Needs improvement - some cases may slip through'}
                </span>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <strong style="color: #7c3aed;">💡 What This Means:</strong><br>
                <div style="font-size: 13px; color: #5b21b6; margin-top: 5px;">
                    {'High priority - allocate resources to combat this pattern' if occurrence_rate > 0.10 else
                     'Monitor regularly and maintain current defenses' if occurrence_rate > 0.05 else
                     'Low occurrence but maintain vigilance'}
                </div>
            </div>
        </div>
        """

    def explain_geographic_pattern(self, location: str, stats: Dict[str, Any]) -> str:
        """
        Explain geographic fraud patterns

        Args:
            location: Geographic location (country, region, etc.)
            stats: Transaction and fraud statistics for the location

        Returns:
            HTML formatted explanation
        """
        fraud_rate = stats.get('fraud_rate', 0)
        volume = stats.get('volume', 0)
        trend = stats.get('trend', 'stable')

        # Risk level
        if fraud_rate > 0.15:
            risk = "🔴 High Risk Location"
            color = "#ef4444"
            recommendation = "Enhanced screening recommended for all transactions"
        elif fraud_rate > 0.08:
            risk = "🟠 Elevated Risk"
            color = "#f59e0b"
            recommendation = "Apply additional verification steps"
        else:
            risk = "🟢 Standard Risk"
            color = "#10b981"
            recommendation = "Normal processing procedures apply"

        trend_icon = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "➡️"

        return f"""
        <div style="background: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 400px;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">🌍 {location}</h4>

            <div style="background: {color}15; padding: 10px; border-radius: 5px;
                        border-left: 4px solid {color}; margin: 10px 0;">
                <strong style="color: {color};">{risk}</strong><br>
                <span style="font-size: 13px; color: #374151;">Fraud Rate: {fraud_rate*100:.2f}%</span>
            </div>

            <div style="margin: 10px 0;">
                <strong>📊 Activity Metrics:</strong><br>
                <div style="font-size: 13px; color: #4b5563; margin: 5px 0;">
                    • Transaction Volume: {volume:,}<br>
                    • Trend: {trend_icon} {trend.capitalize()}<br>
                    • Risk Level: {fraud_rate*100:.2f}% fraud rate
                </div>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <strong style="color: #059669;">🎯 Recommendation:</strong><br>
                <div style="font-size: 13px; color: #065f46; margin-top: 5px;">
                    {recommendation}
                </div>
            </div>
        </div>
        """

    def explain_ml_model_prediction(self, prediction: Dict[str, Any]) -> str:
        """
        Explain ML model prediction and feature importance

        Args:
            prediction: Dict with score, confidence, top_features

        Returns:
            HTML formatted explanation
        """
        score = prediction.get('score', 0)
        confidence = prediction.get('confidence', 0)
        features = prediction.get('top_features', [])

        # Confidence assessment
        if confidence >= 0.90:
            conf_level = "🎯 Very High Confidence"
            conf_note = "Model is highly certain about this prediction"
        elif confidence >= 0.75:
            conf_level = "✅ High Confidence"
            conf_note = "Model has strong conviction in this assessment"
        elif confidence >= 0.60:
            conf_level = "⚠️ Moderate Confidence"
            conf_note = "Consider additional manual review"
        else:
            conf_level = "🔍 Low Confidence"
            conf_note = "Manual review strongly recommended"

        feature_list = "<br>".join([f"• {f['name']}: {f['importance']*100:.1f}% impact"
                                   for f in features[:5]])

        return f"""
        <div style="background: white; padding: 15px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); max-width: 400px;">
            <h4 style="margin: 0 0 10px 0; color: #1e3a8a;">🤖 ML Model Analysis</h4>

            <div style="margin: 10px 0;">
                <strong>Risk Score:</strong> {score*100:.1f}%<br>
                <div style="font-size: 13px; color: #4b5563; margin: 5px 0;">
                    {conf_level}<br>
                    <span style="color: #6b7280;">{conf_note}</span>
                </div>
            </div>

            <div style="background: #eff6ff; padding: 10px; border-radius: 5px; margin: 10px 0;">
                <strong>🔑 Key Factors Driving This Score:</strong><br>
                <div style="font-size: 12px; color: #1e40af; margin: 5px 0;">
                    {feature_list or "• No dominant features identified"}
                </div>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #e5e7eb;">
                <strong style="color: #7c3aed;">💭 What This Means:</strong><br>
                <div style="font-size: 13px; color: #5b21b6; margin-top: 5px;">
                    The ML model analyzed {len(features)} features and identified
                    patterns {'consistent with fraudulent behavior' if score > 0.6 else
                            'suggesting legitimate transaction behavior'}.
                </div>
            </div>
        </div>
        """


# Global singleton instance
_explainability_engine = None

def get_explainability_engine() -> ExplainabilityEngine:
    """Get or create the global explainability engine instance"""
    global _explainability_engine
    if _explainability_engine is None:
        _explainability_engine = ExplainabilityEngine()
    return _explainability_engine
