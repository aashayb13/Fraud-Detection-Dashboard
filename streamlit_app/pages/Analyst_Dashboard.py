
"""
Homepage - Arriba Advisors Transaction Screening System

Executive dashboard with key performance indicators and system overview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from streamlit_app.api_client import get_api_client
from streamlit_app.theme import apply_master_theme, render_page_header, get_chart_colors
from streamlit_app.ai_recommendations import get_ai_engine, render_ai_insight
from streamlit_app.explainability import get_explainability_engine

# Generate synthetic dataset for visualization
np.random.seed(42)

# Rule performance data (20 rules)
rule_names = [
    "Transaction Amount Anomalies", "Transaction Frequency", "Recipient Verification Status",
    "Recipient Blacklist Status", "Device Fingerprinting", "VPN or Proxy Usage",
    "Geo-Location Flags", "Behavioral Biometrics", "Time Since Last Transaction",
    "Social Trust Score", "Account Age", "High-Risk Transaction Times",
    "Past Fraudulent Behavior", "Location-Inconsistent Transactions", "Normalized Transaction Amount",
    "Transaction Context Anomalies", "Fraud Complaints Count", "Merchant Category Mismatch",
    "User Daily Limit Exceeded", "Recent High-Value Transaction"
]

rule_performance_df = pd.DataFrame({
    'rule_name': rule_names,
    'trigger_frequency': np.random.randint(50, 500, 20),
    'precision': np.random.uniform(0.65, 0.98, 20),
    'false_positive_rate': np.random.uniform(0.02, 0.35, 20),
    'avg_contribution': np.random.uniform(5, 35, 20),
    'confirmed_fraud_count': np.random.randint(10, 200, 20),
    'rule_weight': [32, 35, 26, 22, 30, 22, 32, 15, 24, 18, 8, 28, 35, 20, 18, 24, 12, 4, 10, 6]
})

# Analyst decision data (30 days)
dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
analyst_decisions_df = pd.DataFrame({
    'date': dates,
    'cleared': np.random.randint(150, 250, 30),
    'rejected': np.random.randint(20, 80, 30),
    'escalated': np.random.randint(10, 40, 30)
})
analyst_decisions_df['total'] = analyst_decisions_df[['cleared', 'rejected', 'escalated']].sum(axis=1)
analyst_decisions_df['confidence'] = np.minimum(50 + np.arange(30) * 1.2 + np.random.uniform(-5, 5, 30), 95)

def render():
    """Render the Homepage"""

    # Apply theme
    apply_master_theme()

    # Header
    render_page_header(
        title="Arriba Advisors Transaction Screening System",
        subtitle="Real-Time Fraud Detection & Prevention Analytics",
        show_logo=False  # Logo is in sidebar
    )

    # Get standardized chart colors
    colors = get_chart_colors()

    # # Key Performance Indicators
    # st.markdown("## 📊 Key Performance Indicators")

    # col1, col2, col3, col4, col5 = st.columns(5)

    # with col1:
    #     st.metric("Total Transactions Today", "12,547", delta="↑ 8.2%")
    # with col2:
    #     st.metric("Auto-Approved", "11,915 (95%)", delta="↑ 2.1%")
    # with col3:
    #     st.metric("Flagged for Review", "632 (5%)", delta="↓ 1.3%")
    # with col4:
    #     st.metric("Fraud Detected", "47", delta="↓ 12%")
    # with col5:
    #     st.metric("Detection Accuracy", "94.2%", delta="↑ 1.5%")

    # st.markdown("---")

 # Recent Alerts Summary
    st.markdown("## ⚡ Threat Detection Command Center")

    recent_alerts = pd.DataFrame({
        'Time': ['10 min ago', '25 min ago', '1 hr ago', '2 hr ago', '3 hr ago'],
        'Transaction ID': ['TXN-78945', 'TXN-78932', 'TXN-78901', 'TXN-78876', 'TXN-78834'],
        'Amount': ['$15,000', '$12,500', '$45,000', '$3,200', '$8,900'],
        'Risk Score': [0.89, 0.96, 0.91, 0.88, 0.84],
        'Status': ['Under Review', 'Blocked', 'Escalated', 'Under Review', 'Cleared'],
        'Scenario': ['Large Transfer', 'Account Takeover', 'Vendor Impersonation', 'Duplicate Check', 'Odd Hours']
    })
      
    st.dataframe(
        recent_alerts,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Risk Score': st.column_config.ProgressColumn(
                'Risk Score',
                min_value=0,
                max_value=1,
                format='%.2f'
            )
        }
    )

    st.markdown("---")

       # Transaction Flow Funnel
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 Transaction Lifecycle Monitor")

        funnel_data = pd.DataFrame({
            'Stage': ['Total Transactions', 'Auto-Cleared', 'Manual Review', 'Rejected', 'Fraud Confirmed'],
            'Count': [12547, 11915, 632, 85, 47],
            'Percentage': [100, 95, 5, 0.68, 0.37]
        })

        # Enhanced hover with explainability
        funnel_hover_texts = []
        stage_descriptions = {
            'Total Transactions': {
                'desc': 'All incoming transactions processed by the system',
                'insight': 'Represents the complete transaction volume for the period',
                'action': 'Baseline metric for system capacity planning'
            },
            'Auto-Cleared': {
                'desc': 'Low-risk transactions automatically approved by AI',
                'insight': f'{11915/12547*100:.1f}% automation rate - Excellent efficiency',
                'action': 'These bypass manual review, saving significant analyst time'
            },
            'Manual Review': {
                'desc': 'Medium-risk transactions requiring analyst review',
                'insight': f'{632} cases in queue - approximately {632*45/60:.1f} hours of work',
                'action': 'Monitor queue size and consider threshold adjustments if backlog grows'
            },
            'Rejected': {
                'desc': 'Transactions declined after review',
                'insight': f'{85/12547*100:.2f}% rejection rate - Within normal bounds',
                'action': 'These prevented potential losses - review patterns regularly'
            },
            'Fraud Confirmed': {
                'desc': 'Verified fraudulent transactions caught by the system',
                'insight': f'{47} confirmed cases - Est. ${47*12400:,} in losses prevented',
                'action': 'Study these patterns to improve future detection'
            }
        }

        for _, row in funnel_data.iterrows():
            stage = row['Stage']
            count = row['Count']
            pct = row['Percentage']
            info = stage_descriptions[stage]

            hover_text = (
                f"<b style='font-size:14px'>{stage}</b><br><br>"
                f"<b>📊 Volume:</b> <b>{count:,}</b> transactions ({pct:.2f}%)<br><br>"
                f"<b>📝 Description:</b><br>{info['desc']}<br><br>"
                f"<b>💡 Insight:</b><br>{info['insight']}<br><br>"
                f"<b>🎯 Action:</b><br>{info['action']}"
            )
            funnel_hover_texts.append(hover_text)

        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent initial",
            marker=dict(color=colors['funnel']),  # Standardized Arriba palette
            hovertemplate='%{customdata}<extra></extra>',
            customdata=funnel_hover_texts
        ))

        fig_funnel.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_funnel, use_container_width=True, key="analyst_funnel_chart")

        # Cost savings
        st.info(f"💰 **Cost Savings**: Manual reviews prevented: 11,915 × $5 = **$59,575/day**")

    with col2:
        st.subheader("🧠 Decision Pattern Analytics")

        fig_decisions = go.Figure()

        # Enhanced hover for Cleared transactions
        cleared_hover = [
            f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>"
            f"<b>Cleared Transactions:</b> {cleared}<br><br>"
            f"<b>💡 Meaning:</b> Analyst reviewed and approved<br>"
            f"<b>🎯 Impact:</b> {cleared} legitimate transactions processed<br>"
            f"<b>📈 Trend:</b> {'Above average' if cleared > 200 else 'Normal volume' if cleared > 150 else 'Below average'}"
            for date, cleared in zip(analyst_decisions_df['date'], analyst_decisions_df['cleared'])
        ]

        fig_decisions.add_trace(go.Bar(
            x=analyst_decisions_df['date'],
            y=analyst_decisions_df['cleared'],
            name='Cleared',
            marker_color=colors['success'],
            hovertemplate='%{customdata}<extra></extra>',
            customdata=cleared_hover
        ))

        # Enhanced hover for Rejected transactions
        rejected_hover = [
            f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>"
            f"<b>Rejected Transactions:</b> {rejected}<br><br>"
            f"<b>💡 Meaning:</b> Blocked after analyst review<br>"
            f"<b>🎯 Impact:</b> Est. ${rejected * 3500:,} in fraud prevented<br>"
            f"<b>⚠️ Note:</b> {'High rejection day - investigate pattern' if rejected > 60 else 'Normal rejection rate'}"
            for date, rejected in zip(analyst_decisions_df['date'], analyst_decisions_df['rejected'])
        ]

        fig_decisions.add_trace(go.Bar(
            x=analyst_decisions_df['date'],
            y=analyst_decisions_df['rejected'],
            name='Rejected',
            marker_color=colors['danger'],
            hovertemplate='%{customdata}<extra></extra>',
            customdata=rejected_hover
        ))

        # Enhanced hover for Escalated transactions
        escalated_hover = [
            f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>"
            f"<b>Escalated Cases:</b> {escalated}<br><br>"
            f"<b>💡 Meaning:</b> Complex cases sent to senior analysts<br>"
            f"<b>🎯 Impact:</b> Requires additional expert review<br>"
            f"<b>📊 Volume:</b> {escalated/(cleared+rejected+escalated)*100:.1f}% of decisions"
            for date, escalated, cleared, rejected in zip(
                analyst_decisions_df['date'],
                analyst_decisions_df['escalated'],
                analyst_decisions_df['cleared'],
                analyst_decisions_df['rejected']
            )
        ]

        fig_decisions.add_trace(go.Bar(
            x=analyst_decisions_df['date'],
            y=analyst_decisions_df['escalated'],
            name='Escalated',
            marker_color=colors['warning'],
            hovertemplate='%{customdata}<extra></extra>',
            customdata=escalated_hover
        ))

        # Enhanced hover for Confidence line
        confidence_hover = [
            f"<b>Date:</b> {date.strftime('%Y-%m-%d')}<br>"
            f"<b>ML Confidence:</b> {conf:.1f}%<br><br>"
            f"<b>💡 Assessment:</b><br>"
            f"{'⭐ Very High - Models are very certain' if conf >= 90 else '✅ High - Strong model performance' if conf >= 75 else '⚠️ Moderate - Models less certain' if conf >= 60 else '🔴 Low - Manual review critical'}<br><br>"
            f"<b>🎯 Meaning:</b> Average confidence across all ML predictions this day"
            for date, conf in zip(analyst_decisions_df['date'], analyst_decisions_df['confidence'])
        ]

        fig_decisions.add_trace(go.Scatter(
            x=analyst_decisions_df['date'],
            y=analyst_decisions_df['confidence'],
            name='Confidence %',
            yaxis='y2',
            line=dict(color=colors['primary'], width=3),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=confidence_hover
        ))

        fig_decisions.update_layout(
            barmode='stack',
            height=400,
            yaxis=dict(title='Transaction Count'),
            yaxis2=dict(title='Confidence %', overlaying='y', side='right', range=[0, 100]),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig_decisions, use_container_width=True, key="analyst_decisions_chart")

    st.markdown("---")

    # System Activity Timeline
    st.markdown("## 📊 Live Transaction Pulse")

    # Generate sample time series data
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
    transactions = np.random.poisson(lam=500, size=24) + np.random.randint(-50, 100, 24)
    fraud_detected = np.random.poisson(lam=2, size=24)

    fig = go.Figure()

    # Enhanced hover for transaction volume
    transaction_hover = []
    for hour, txn_count in zip(hours, transactions):
        hour_of_day = hour.hour
        # Determine if it's peak hours
        if 9 <= hour_of_day <= 17:
            period = "Business Hours"
            note = "Peak transaction period"
        elif 18 <= hour_of_day <= 23:
            period = "Evening"
            note = "Moderate activity"
        elif 0 <= hour_of_day <= 6:
            period = "Overnight"
            note = "Low activity - elevated fraud risk"
        else:
            period = "Morning"
            note = "Activity ramping up"

        hover_text = (
            f"<b>Time:</b> {hour.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Period:</b> {period}<br>"
            f"<b>Transactions:</b> {txn_count}<br><br>"
            f"<b>💡 Context:</b> {note}<br>"
            f"<b>📊 Volume Status:</b> {'High volume' if txn_count > 550 else 'Normal' if txn_count > 450 else 'Below average'}<br>"
            f"<b>🎯 Impact:</b> ~{txn_count * 0.05:.0f} require manual review"
        )
        transaction_hover.append(hover_text)

    fig.add_trace(go.Scatter(
        x=hours,
        y=transactions,
        name='Total Transactions',
        line=dict(color=colors['primary'], width=2),
        fill='tozeroy',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=transaction_hover
    ))

    # Enhanced hover for fraud detected
    fraud_hover = []
    for hour, fraud_count, txn_count in zip(hours, fraud_detected, transactions):
        fraud_rate = (fraud_count / txn_count * 100) if txn_count > 0 else 0

        if fraud_count > 3:
            severity = "🔴 High"
            action = "Investigate immediately"
        elif fraud_count > 1:
            severity = "🟡 Moderate"
            action = "Monitor closely"
        else:
            severity = "🟢 Low"
            action = "Normal levels"

        hover_text = (
            f"<b>Time:</b> {hour.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Fraud Cases:</b> {fraud_count}<br>"
            f"<b>Total Transactions:</b> {txn_count}<br><br>"
            f"<b>📊 Fraud Rate:</b> {fraud_rate:.2f}%<br>"
            f"<b>⚠️ Severity:</b> {severity}<br>"
            f"<b>💰 Est. Loss Prevented:</b> ${fraud_count * 12400:,}<br><br>"
            f"<b>🎯 Action:</b> {action}"
        )
        fraud_hover.append(hover_text)

    fig.add_trace(go.Scatter(
        x=hours,
        y=fraud_detected,
        name='Fraud Detected',
        line=dict(color=colors['danger'], width=2),
        mode='lines+markers',
        yaxis='y2',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=fraud_hover
    ))

    fig.update_layout(
        yaxis=dict(title='Transaction Volume'),
        yaxis2=dict(title='Fraud Cases', overlaying='y', side='right'),
        hovermode='x unified',
        height=400,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    st.plotly_chart(fig, use_container_width=True, key="analyst_pulse_chart")

    st.markdown("---")

    # Quick Access Panels
    st.markdown("## 🚀 Quick Access")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 AI Transaction Intelligence")
        st.markdown("Real-time AI-powered transaction monitoring with automated decision intelligence.")
        if st.button("Open Transaction Monitor", use_container_width=True):
            st.info("Navigate using sidebar to Transaction Monitoring System")

    with col2:
        st.markdown("### 🧠 AI Scenario Intelligence")
        st.markdown("ML-driven analysis of 13 fraud scenarios with AI-powered rule intelligence.")
        if st.button("View Fraud Scenarios", use_container_width=True):
            st.info("Navigate using sidebar to Fraud Scenario Analysis")

    with col3:
        st.markdown("### 📊 AI Rule Performance Intelligence")
        st.markdown("ML-powered fraud detection rule analysis with predictive performance metrics.")
        if st.button("Analyze Rules", use_container_width=True):
            st.info("Navigate using sidebar to Rule Performance Analytics")

    st.markdown("---")



    st.markdown("---")

    # AI-Powered Dynamic Threshold Optimization
    st.markdown("## 🎯 AI-Powered Dynamic Threshold Optimization")
    st.markdown("**Next-Generation Adaptive Fraud Prevention**")

    st.markdown("""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea; margin: 15px 0;">
        <h4 style="color: #667eea; margin-top: 0;">🔮 Intelligent Threshold Management</h4>
        <p style="margin-bottom: 10px;">
            Our system continuously monitors fraud patterns and automatically adjusts detection thresholds
            based on real-time data analysis. This ensures optimal balance between fraud detection and
            operational efficiency.
        </p>
        <h5 style="color: #555; margin-top: 15px;">Dynamic Factors Monitored:</h5>
        <ul style="color: #666; line-height: 1.8;">
            <li><strong>Transaction Volume Trends:</strong> Adapts to seasonal peaks and valleys</li>
            <li><strong>Geographic Risk Shifts:</strong> Responds to emerging high-risk regions</li>
            <li><strong>Amount Pattern Changes:</strong> Adjusts for inflation and market conditions</li>
            <li><strong>False Positive Rates:</strong> Optimizes analyst workload</li>
            <li><strong>Emerging Fraud Patterns:</strong> Learns from new attack vectors</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # AI Recommendations for Current Thresholds
    threshold_col1, threshold_col2 = st.columns([2, 1])

    with threshold_col1:
        st.markdown("### 📊 Current Threshold Performance")

        # Mock current threshold metrics
        current_metrics = pd.DataFrame({
            'Threshold Type': ['Auto-Clear', 'Manual Review', 'High Priority', 'Critical'],
            'Current Value': [0.30, 0.60, 0.80, 0.90],
            'Utilization': [95.2, 4.5, 0.25, 0.05],
            'Efficiency': [98.5, 87.3, 92.1, 99.2]
        })

        st.dataframe(
            current_metrics,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Current Value': st.column_config.ProgressColumn(
                    'Current Value',
                    min_value=0,
                    max_value=1,
                    format='%.2f'
                ),
                'Utilization': st.column_config.ProgressColumn(
                    'Utilization %',
                    min_value=0,
                    max_value=100,
                    format='%.1f%%'
                ),
                'Efficiency': st.column_config.ProgressColumn(
                    'Efficiency %',
                    min_value=0,
                    max_value=100,
                    format='%.1f%%'
                )
            }
        )

    with threshold_col2:
        st.markdown("### 🤖 AI Recommendations")

        # Get AI recommendation for thresholds
        ai_engine = get_ai_engine()
        threshold_rec = ai_engine.get_threshold_recommendation(
            current_threshold=0.60,
            recent_stats={
                'false_positive_rate': 0.062,
                'detection_rate': 0.942,
                'queue_size': 632,
                'avg_time': 45
            }
        )

        st.info(threshold_rec)

        # Trend insight
        fraud_trend = [45, 47, 44, 52, 51, 48, 47]
        trend_analysis = ai_engine.get_trend_analysis(
            metric_name="Daily Fraud Detection",
            trend_data=fraud_trend
        )

        st.success(trend_analysis)

    st.markdown("---")

    # ML Intelligence Section
    st.markdown("## 🤖 Machine Learning Intelligence")
    st.markdown("*Real-time ML model performance and fraud prediction insights*")

    ml_col1, ml_col2, ml_col3, ml_col4 = st.columns(4)

    with ml_col1:
        st.metric("Model Accuracy", "94.3%", "+1.2%")
    with ml_col2:
        st.metric("AUC-ROC Score", "0.963", "+0.008")
    with ml_col3:
        st.metric("Predictions/Min", "1,247", "+156")
    with ml_col4:
        st.metric("Avg Confidence", "87.2%", "+2.3%")

    ml_viz_col1, ml_viz_col2 = st.columns(2)

    with ml_viz_col1:
        st.markdown("### 🎯 Model Performance Trends")

        # Model performance over last 7 days
        ml_days = pd.date_range(end=datetime.now(), periods=7, freq='D')
        ml_accuracy = [0.932 + i * 0.0015 + np.random.uniform(-0.005, 0.005) for i in range(7)]
        ml_precision = [0.918 + i * 0.002 + np.random.uniform(-0.005, 0.005) for i in range(7)]
        ml_recall = [0.895 + i * 0.0025 + np.random.uniform(-0.005, 0.005) for i in range(7)]

        fig_ml_perf = go.Figure()

        fig_ml_perf.add_trace(go.Scatter(
            x=ml_days,
            y=ml_accuracy,
            name='Accuracy',
            line=dict(color=colors['primary'], width=3),
            mode='lines+markers'
        ))

        fig_ml_perf.add_trace(go.Scatter(
            x=ml_days,
            y=ml_precision,
            name='Precision',
            line=dict(color=colors['success'], width=3),
            mode='lines+markers'
        ))

        fig_ml_perf.add_trace(go.Scatter(
            x=ml_days,
            y=ml_recall,
            name='Recall',
            line=dict(color=colors['info'], width=3),
            mode='lines+markers'
        ))

        fig_ml_perf.update_layout(
            height=350,
            yaxis=dict(title='Score', range=[0.85, 0.98]),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        st.plotly_chart(fig_ml_perf, use_container_width=True, key="ml_performance_trends")

    with ml_viz_col2:
        st.markdown("### 📊 Prediction Confidence Distribution")

        # Generate prediction confidence distribution
        np.random.seed(42)
        confidence_scores = np.concatenate([
            np.random.beta(8, 2, 400),  # High confidence predictions
            np.random.beta(2, 2, 100)   # Low confidence predictions
        ]) * 100

        fig_conf = go.Figure()

        fig_conf.add_trace(go.Histogram(
            x=confidence_scores,
            nbinsx=20,
            marker=dict(
                color=confidence_scores,
                colorscale='RdYlGn',
                showscale=False
            ),
            opacity=0.75
        ))

        fig_conf.update_layout(
            height=350,
            xaxis=dict(title='Confidence Score (%)'),
            yaxis=dict(title='Number of Predictions'),
            showlegend=False
        )

        st.plotly_chart(fig_conf, use_container_width=True, key="ml_confidence_dist")

    ml_viz_col3, ml_viz_col4 = st.columns(2)

    with ml_viz_col3:
        st.markdown("### 🔍 Top Feature Importance")

        feature_names = [
            'Transaction Amount',
            'Customer Risk Level',
            'Transaction Hour',
            'International Flag',
            'Account Balance',
            'Account Age',
            'Is PEP',
            'Weekend Flag'
        ]
        feature_importance = [0.28, 0.22, 0.14, 0.12, 0.10, 0.06, 0.05, 0.03]

        fig_features = go.Figure(go.Bar(
            y=feature_names,
            x=feature_importance,
            orientation='h',
            marker=dict(
                color=feature_importance,
                colorscale='Blues',
                showscale=False
            ),
            text=[f"{v:.1%}" for v in feature_importance],
            textposition='outside'
        ))

        fig_features.update_layout(
            height=350,
            xaxis=dict(title='Importance Score'),
            yaxis=dict(title=''),
            showlegend=False
        )

        st.plotly_chart(fig_features, use_container_width=True, key="ml_feature_importance")

    with ml_viz_col4:
        st.markdown("### ⚡ Real-time Model Health")

        # Model health metrics
        health_metrics = pd.DataFrame({
            'Metric': ['Data Quality', 'Model Drift', 'Latency', 'Throughput', 'Error Rate'],
            'Status': ['Excellent', 'Normal', 'Good', 'Excellent', 'Good'],
            'Score': [98, 92, 88, 97, 91]
        })

        # Color coding
        def get_status_color(score):
            if score >= 95:
                return '🟢'
            elif score >= 85:
                return '🟡'
            else:
                return '🔴'

        health_metrics['Indicator'] = health_metrics['Score'].apply(get_status_color)

        fig_health = go.Figure()

        colors_health = ['#2E865F' if s >= 95 else '#F3B65B' if s >= 85 else '#E54848'
                        for s in health_metrics['Score']]

        fig_health.add_trace(go.Bar(
            y=health_metrics['Metric'],
            x=health_metrics['Score'],
            orientation='h',
            marker=dict(color=colors_health),
            text=[f"{s}%" for s in health_metrics['Score']],
            textposition='outside'
        ))

        fig_health.update_layout(
            height=350,
            xaxis=dict(title='Health Score (%)', range=[0, 110]),
            yaxis=dict(title=''),
            showlegend=False
        )

        st.plotly_chart(fig_health, use_container_width=True, key="ml_health_metrics")

    # AI Insight for ML Performance
    st.markdown("### 💡 ML Performance Insights")

    ai_engine = get_ai_engine()
    ml_insight = ai_engine.get_ml_performance_insight(
        accuracy=0.943,
        precision=0.932,
        recall=0.906,
        auc_roc=0.963,
        trend='improving'
    )

    render_ai_insight("ML Performance Analysis", ml_insight, icon="🤖")

    st.markdown("---")

    # Footer
    st.markdown("## ℹ️ System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**System Version:** 2.5.3")
        st.markdown("**Database:** SQLite (Production-Ready)")
    with col2:
        st.markdown("**API Status:** ✅ Healthy")
        st.markdown("**Last Sync:** Just now")
    with col3:
        st.markdown("**Support:** support@arribaadvisors.com")
        st.markdown("**Documentation:** Available in sidebar")

    st.caption("© 2024 Arriba Advisors. All rights reserved. | Real-Time Fraud Detection Platform")

if __name__ == "__main__":
    render()