"""
Rule Performance Analytics Page

Advanced rule correlation, waterfall analysis, and performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

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

# Rule correlation pairs
rule_correlation_pairs = [
    ('VPN or Proxy Usage', 'Geo-Location Flags', 85),
    ('Device Fingerprinting', 'Behavioral Biometrics', 72),
    ('Transaction Amount Anomalies', 'Transaction Context Anomalies', 68),
    ('Recipient Verification Status', 'Social Trust Score', 78),
    ('High-Risk Transaction Times', 'Transaction Frequency', 65),
    ('Account Age', 'Transaction Amount Anomalies', 55),
    ('Past Fraudulent Behavior', 'Fraud Complaints Count', 80),
    ('Location-Inconsistent Transactions', 'VPN or Proxy Usage', 75),
    ('Recipient Blacklist Status', 'Past Fraudulent Behavior', 70),
    ('User Daily Limit Exceeded', 'Transaction Frequency', 60)
]


def render():
    """Render the Rule Performance Analytics page"""

    # Apply theme
    apply_master_theme()

    # Header
    render_page_header(
        title="Detection Rule Observatory",
        subtitle="Fraud Detection Rule Effectiveness & Optimization Metrics",
        show_logo=False
    )

    # Get standardized chart colors
    colors = get_chart_colors()

    # Rule Contribution Treemap and Bubble Chart
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 Detection Impact Scorecard")
        st.caption("Treemap showing confirmed fraud catches by rule")

        # Enhanced hover texts for treemap
        treemap_hover_texts = []
        total_fraud = rule_performance_df['confirmed_fraud_count'].sum()

        for _, row in rule_performance_df.iterrows():
            fraud_count = row['confirmed_fraud_count']
            percentage = (fraud_count / total_fraud * 100)
            precision = row['precision']
            frequency = row['trigger_frequency']

            # Estimate financial impact
            avg_fraud_value = 12400
            prevented_loss = fraud_count * avg_fraud_value

            # Performance badge
            if fraud_count > 150:
                badge = "🏆 TOP DETECTOR"
                badge_color = "#10b981"
                impact = "Elite fraud prevention performer"
            elif fraud_count > 100:
                badge = "⭐ HIGH IMPACT"
                badge_color = "#22c55e"
                impact = "Major fraud detection contributor"
            elif fraud_count > 50:
                badge = "✅ SOLID"
                badge_color = "#3b82f6"
                impact = "Reliable fraud catcher"
            else:
                badge = "📊 SUPPORTING"
                badge_color = "#6b7280"
                impact = "Supplementary detection role"

            hover_text = (
                f"<b style='font-size:14px'>{row['rule_name']}</b><br><br>"
                f"<b style='color:{badge_color}'>{badge}</b><br><br>"
                f"<b>📊 Detection Stats:</b><br>"
                f"• Fraud Caught: <b>{fraud_count}</b> cases<br>"
                f"• Share of Total: <b>{percentage:.1f}%</b><br>"
                f"• Precision: <b>{precision*100:.1f}%</b><br>"
                f"• Trigger Frequency: <b>{frequency}</b><br><br>"
                f"<b>💰 Business Impact:</b><br>"
                f"• Losses Prevented: <b>${prevented_loss:,}</b><br>"
                f"• Avg per Case: <b>${avg_fraud_value:,}</b><br><br>"
                f"<b>💡 Assessment:</b><br>"
                f"{impact}"
            )
            treemap_hover_texts.append(hover_text)

        fig_treemap = go.Figure(go.Treemap(
            labels=rule_performance_df['rule_name'],
            parents=[''] * len(rule_performance_df),
            values=rule_performance_df['confirmed_fraud_count'],
            textinfo='label+value+percent parent',
            marker=dict(
                colorscale='Reds',
                cmid=rule_performance_df['confirmed_fraud_count'].mean()
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=treemap_hover_texts
        ))

        fig_treemap.update_layout(height=500)
        st.plotly_chart(fig_treemap, use_container_width=True)

    with col2:
        st.subheader("🧠 Rule Precision Dashboard")
        st.caption("Bubble chart: Trigger Frequency vs Precision (size = weight)")

        fig_bubble = go.Figure()

        # Enhanced hover information with AI insights
        explainer = get_explainability_engine()
        bubble_hover_texts = []

        for _, row in rule_performance_df.iterrows():
            # Performance assessment
            precision = row['precision']
            frequency = row['trigger_frequency']
            fp_rate = row['false_positive_rate']
            fraud_caught = row['confirmed_fraud_count']
            weight = row['rule_weight']

            # Determine performance level
            if precision >= 0.90:
                perf_badge = "⭐ EXCELLENT"
                perf_color = "#10b981"
            elif precision >= 0.75:
                perf_badge = "✅ GOOD"
                perf_color = "#3b82f6"
            elif precision >= 0.60:
                perf_badge = "⚠️ FAIR"
                perf_color = "#f59e0b"
            else:
                perf_badge = "🔴 NEEDS WORK"
                perf_color = "#ef4444"

            # Generate recommendations
            recommendations = []
            if precision < 0.70:
                recommendations.append("• Adjust thresholds")
            if fp_rate > 0.30:
                recommendations.append("• Review false positives")
            if frequency > 400 and precision < 0.80:
                recommendations.append("• High volume burden")
            if fraud_caught < 20:
                recommendations.append("• Low fraud detection")

            rec_text = "<br>".join(recommendations) if recommendations else "• Continue monitoring"

            hover_text = (
                f"<b style='font-size:14px'>{row['rule_name']}</b><br><br>"
                f"<b style='color:{perf_color}'>Performance: {perf_badge}</b><br><br>"
                f"<b>📊 Key Metrics:</b><br>"
                f"• Precision: <b>{precision*100:.1f}%</b><br>"
                f"• Trigger Frequency: <b>{frequency}</b><br>"
                f"• False Positive Rate: <b>{fp_rate*100:.1f}%</b><br>"
                f"• Fraud Caught: <b>{fraud_caught} cases</b><br>"
                f"• Rule Weight: <b>{weight}</b><br><br>"
                f"<b>💡 Analysis:</b><br>"
                f"{'High-precision rule with excellent accuracy' if precision > 0.90 else 'Solid performer' if precision > 0.75 else 'Needs optimization'}<br>"
                f"{'High frequency - major workload driver' if frequency > 300 else 'Moderate activity' if frequency > 100 else 'Low frequency'}<br><br>"
                f"<b style='color:#059669'>🎯 Recommendations:</b><br>"
                f"{rec_text}"
            )
            bubble_hover_texts.append(hover_text)

        fig_bubble.add_trace(go.Scatter(
            x=rule_performance_df['trigger_frequency'],
            y=rule_performance_df['precision'] * 100,
            mode='markers',
            marker=dict(
                size=rule_performance_df['rule_weight'],
                sizemode='diameter',
                sizeref=2,
                color=rule_performance_df['precision'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Precision"),
                line=dict(width=1, color='white')
            ),
            text=rule_performance_df['rule_name'],
            hovertemplate='%{customdata}<extra></extra>',
            customdata=bubble_hover_texts
        ))

        fig_bubble.update_layout(
            xaxis_title="Trigger Frequency",
            yaxis_title="Precision (%)",
            height=500,
            hovermode='closest'
        )

        st.plotly_chart(fig_bubble, use_container_width=True)

    st.markdown("---")

    # Rule Correlation Network
    st.subheader("🔗 Detection Rule Nexus")
    st.caption("Shows which rules commonly fire together (typical 5-8 rule patterns)")

    # Create network visualization using scatter plot with connecting lines
    fig_network = go.Figure()

    # Get unique rules from pairs
    unique_rules = list(set([r[0] for r in rule_correlation_pairs] + [r[1] for r in rule_correlation_pairs]))
    rule_positions = {rule: (np.cos(i*2*np.pi/len(unique_rules)), np.sin(i*2*np.pi/len(unique_rules)))
                     for i, rule in enumerate(unique_rules)}

    # Calculate correlation stats for each rule
    rule_correlation_stats = {}
    for rule in unique_rules:
        connections = [pair for pair in rule_correlation_pairs if rule in [pair[0], pair[1]]]
        avg_strength = np.mean([pair[2] for pair in connections])
        max_strength = max([pair[2] for pair in connections])
        connection_count = len(connections)
        rule_correlation_stats[rule] = {
            'avg_strength': avg_strength,
            'max_strength': max_strength,
            'connection_count': connection_count,
            'connections': connections
        }

    # Draw edges (correlations)
    for rule1, rule2, strength in rule_correlation_pairs:
        x0, y0 = rule_positions[rule1]
        x1, y1 = rule_positions[rule2]

        # Line thickness based on correlation strength
        width = strength / 20

        fig_network.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(
                width=width,
                color=f'rgba(239, 68, 68, {strength/100})'
            ),
            hoverinfo='skip',
            showlegend=False
        ))

    # Enhanced hover texts for nodes
    node_hover_texts = []
    for rule, (x, y) in rule_positions.items():
        stats = rule_correlation_stats[rule]

        # Assess connectivity
        if stats['avg_strength'] > 75:
            status = "🔴 HIGHLY CORRELATED"
            status_color = "#ef4444"
            insight = "Frequently triggers with other rules - strong pattern"
        elif stats['avg_strength'] > 65:
            status = "🟡 MODERATELY CORRELATED"
            status_color = "#f59e0b"
            insight = "Regular co-occurrence with other rules"
        else:
            status = "🟢 INDEPENDENTLY USEFUL"
            status_color = "#10b981"
            insight = "More independent - less correlated triggers"

        # List top correlations
        top_correlations = sorted(stats['connections'], key=lambda x: x[2], reverse=True)[:3]
        corr_list = []
        for pair in top_correlations:
            other_rule = pair[1] if pair[0] == rule else pair[0]
            corr_list.append(f"• {other_rule}: {pair[2]}%")
        corr_text = "<br>".join(corr_list)

        hover_text = (
            f"<b style='font-size:14px'>{rule}</b><br><br>"
            f"<b style='color:{status_color}'>{status}</b><br><br>"
            f"<b>🔗 Correlation Stats:</b><br>"
            f"• Connections: <b>{stats['connection_count']}</b><br>"
            f"• Avg Strength: <b>{stats['avg_strength']:.1f}%</b><br>"
            f"• Max Strength: <b>{stats['max_strength']:.1f}%</b><br><br>"
            f"<b>🎯 Top Correlations:</b><br>"
            f"{corr_text}<br><br>"
            f"<b>💡 Insight:</b><br>"
            f"{insight}"
        )
        node_hover_texts.append(hover_text)

    # Draw nodes
    for idx, (rule, (x, y)) in enumerate(rule_positions.items()):
        fig_network.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text',
            marker=dict(size=15, color='#3b82f6', line=dict(width=2, color='white')),
            text=rule.split()[0],  # First word only
            textposition='top center',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=[node_hover_texts[idx]],
            showlegend=False
        ))

    fig_network.update_layout(
        height=600,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig_network, use_container_width=True)

    # Display correlation pairs
    st.markdown("**High-Correlation Rule Pairs:**")
    correlation_df = pd.DataFrame(rule_correlation_pairs, columns=['Rule 1', 'Rule 2', 'Correlation %'])
    correlation_df = correlation_df.sort_values('Correlation %', ascending=False)
    st.dataframe(correlation_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Rule Contribution Waterfall
    st.subheader("📈 Fraud Prevention Cascade")
    st.caption("Shows how each triggered rule contributes to final risk score")

    # Sample transaction with rules
    sample_rules = [
        {'name': 'Base Score', 'contribution': 0, 'cumulative': 0},
        {'name': 'Amount Anomaly', 'contribution': 25, 'cumulative': 25},
        {'name': 'VPN Usage', 'contribution': 15, 'cumulative': 40},
        {'name': 'Device Mismatch', 'contribution': 22, 'cumulative': 62},
        {'name': 'Geo-Location Flag', 'contribution': 18, 'cumulative': 80},
        {'name': 'High-Risk Time', 'contribution': 9, 'cumulative': 89},
        {'name': 'Final Risk Score', 'contribution': 0, 'cumulative': 89}
    ]

    # Enhanced hover texts for waterfall
    waterfall_hover_texts = []

    for idx, rule in enumerate(sample_rules):
        rule_name = rule['name']
        contribution = rule['contribution']
        cumulative = rule['cumulative']

        if rule_name == 'Base Score':
            status = "🔵 STARTING POINT"
            status_color = "#3b82f6"
            insight = "Transaction begins with neutral risk score"
            detail = "All transactions start at 0 risk points"
        elif rule_name == 'Final Risk Score':
            if cumulative >= 80:
                status = "🔴 HIGH RISK - BLOCK"
                status_color = "#ef4444"
                insight = f"Score {cumulative} exceeds threshold 75 - flagged for review"
                detail = "This transaction would be automatically blocked"
            elif cumulative >= 50:
                status = "🟡 MEDIUM RISK - REVIEW"
                status_color = "#f59e0b"
                insight = "Requires manual analyst review"
                detail = "Transaction queued for human verification"
            else:
                status = "🟢 LOW RISK - CLEAR"
                status_color = "#10b981"
                insight = "Below threshold - auto-approved"
                detail = "Transaction proceeds normally"
        else:
            # Regular rule
            if contribution >= 20:
                status = "🔴 MAJOR CONTRIBUTOR"
                status_color = "#ef4444"
                impact_level = "High-impact detection"
            elif contribution >= 15:
                status = "🟡 MODERATE CONTRIBUTOR"
                status_color = "#f59e0b"
                impact_level = "Significant risk signal"
            else:
                status = "🟢 MINOR CONTRIBUTOR"
                status_color = "#10b981"
                impact_level = "Supporting risk indicator"

            insight = f"Adds {contribution} risk points to cumulative score"
            detail = impact_level

        hover_text = (
            f"<b style='font-size:14px'>{rule_name}</b><br><br>"
            f"<b style='color:{status_color}'>{status}</b><br><br>"
            f"<b>📊 Risk Score Details:</b><br>"
            f"• Contribution: <b>+{contribution}</b> points<br>"
            f"• Cumulative Score: <b>{cumulative}</b><br>"
            f"• Progress: <b>{(cumulative/100)*100:.0f}%</b><br><br>"
            f"<b>💡 Assessment:</b><br>"
            f"{insight}<br>"
            f"{detail}"
        )
        waterfall_hover_texts.append(hover_text)

    fig_waterfall = go.Figure(go.Waterfall(
        name="Risk Score",
        orientation="v",
        measure=["absolute"] + ["relative"] * 5 + ["total"],
        x=[r['name'] for r in sample_rules],
        textposition="outside",
        text=[f"+{r['contribution']}" if r['contribution'] > 0 else "" for r in sample_rules],
        y=[r['contribution'] for r in sample_rules],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#ef4444"}},
        decreasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6366f1"}},
        hovertemplate='%{customdata}<extra></extra>',
        customdata=waterfall_hover_texts
    ))

    fig_waterfall.update_layout(
        title="Risk Score Build-up for Transaction #TXN-789456",
        yaxis_title="Risk Points",
        height=500
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("---")

    # Detailed Rule Performance Table
    st.subheader("📊 Rule Efficacy Intelligence")

    # Format the dataframe for display
    display_df = rule_performance_df.copy()
    display_df['precision'] = (display_df['precision'] * 100).round(1).astype(str) + '%'
    display_df['false_positive_rate'] = (display_df['false_positive_rate'] * 100).round(1).astype(str) + '%'
    display_df = display_df.rename(columns={
        'rule_name': 'Rule Name',
        'trigger_frequency': 'Trigger Frequency',
        'precision': 'Precision',
        'false_positive_rate': 'False Positive Rate',
        'avg_contribution': 'Avg Contribution',
        'confirmed_fraud_count': 'Fraud Caught',
        'rule_weight': 'Rule Weight'
    })

    st.dataframe(
        display_df[['Rule Name', 'Trigger Frequency', 'Precision', 'False Positive Rate',
                    'Avg Contribution', 'Fraud Caught', 'Rule Weight']],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # AI-Powered Rule Optimization Recommendations
    st.markdown("## 🤖 AI-Powered Rule Optimization")

    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        st.markdown("### 🎯 Top Performing Rules")

        # Get top 3 performing rules
        top_performers = rule_performance_df.nlargest(3, 'confirmed_fraud_count')

        ai_engine = get_ai_engine()

        for idx, row in top_performers.iterrows():
            recommendation = ai_engine.get_rule_optimization(
                rule_name=row['rule_name'],
                performance={
                    'precision': row['precision'],
                    'frequency': row['trigger_frequency'],
                    'fp_rate': row['false_positive_rate'],
                    'catches': row['confirmed_fraud_count']
                }
            )

            render_ai_insight(
                title=f"{row['rule_name']}",
                recommendation=recommendation,
                icon="✅"
            )

    with opt_col2:
        st.markdown("### ⚠️ Rules Needing Attention")

        # Get bottom 3 rules by precision
        needs_attention = rule_performance_df.nsmallest(3, 'precision')

        for idx, row in needs_attention.iterrows():
            recommendation = ai_engine.get_rule_optimization(
                rule_name=row['rule_name'],
                performance={
                    'precision': row['precision'],
                    'frequency': row['trigger_frequency'],
                    'fp_rate': row['false_positive_rate'],
                    'catches': row['confirmed_fraud_count']
                }
            )

            render_ai_insight(
                title=f"{row['rule_name']}",
                recommendation=recommendation,
                icon="🔧"
            )

    # ML-Enhanced Rule Intelligence
    st.markdown("---")
    st.markdown("## 🤖 ML-Enhanced Rule Intelligence")
    st.markdown("*Machine learning insights for rule optimization*")

    ml_rule_col1, ml_rule_col2, ml_rule_col3, ml_rule_col4 = st.columns(4)

    with ml_rule_col1:
        st.metric("ML-Optimized Rules", "12/20", "+3")
    with ml_rule_col2:
        st.metric("Avg Precision Gain", "+8.3%", "+1.2%")
    with ml_rule_col3:
        st.metric("False Positive Reduction", "22%", "-5%")
    with ml_rule_col4:
        st.metric("Ensemble Boost", "+15%", "+3%")

    ml_rule_viz_col1, ml_rule_viz_col2 = st.columns(2)

    with ml_rule_viz_col1:
        st.markdown("### 📊 ML vs Rule-Based Performance")

        comparison_metrics = ['Precision', 'Recall', 'F1 Score', 'FP Rate']
        ml_scores = [94.3, 91.8, 93.0, 6.2]
        rule_scores = [86.1, 84.5, 85.3, 24.5]

        # Enhanced hover texts for ML comparison
        ml_hover_texts = []
        rule_hover_texts = []

        for idx, metric in enumerate(comparison_metrics):
            ml_val = ml_scores[idx]
            rule_val = rule_scores[idx]

            # Calculate improvement
            if metric == 'FP Rate':
                improvement = ((rule_val - ml_val) / rule_val) * 100
                better = "lower is better"
            else:
                improvement = ((ml_val - rule_val) / rule_val) * 100
                better = "higher is better"

            # ML hover
            if improvement > 50:
                status_ml = "🏆 MAJOR IMPROVEMENT"
                status_color_ml = "#10b981"
            elif improvement > 20:
                status_ml = "⭐ STRONG IMPROVEMENT"
                status_color_ml = "#22c55e"
            else:
                status_ml = "✅ IMPROVEMENT"
                status_color_ml = "#3b82f6"

            ml_hover = (
                f"<b style='font-size:14px'>ML-Enhanced System</b><br><br>"
                f"<b style='color:{status_color_ml}'>{status_ml}</b><br><br>"
                f"<b>📊 {metric}:</b><br>"
                f"• ML Score: <b>{ml_val:.1f}%</b><br>"
                f"• Rules Only: <b>{rule_val:.1f}%</b><br>"
                f"• Improvement: <b>{improvement:.1f}%</b><br><br>"
                f"<b>💡 Advantage:</b><br>"
                f"ML outperforms by {improvement:.1f}% ({better})<br>"
                f"Machine learning optimization delivers measurable gains"
            )
            ml_hover_texts.append(ml_hover)

            # Rules-only hover
            rule_hover = (
                f"<b style='font-size:14px'>Rules-Only System</b><br><br>"
                f"<b style='color:#6b7280'>📊 BASELINE</b><br><br>"
                f"<b>📊 {metric}:</b><br>"
                f"• Rules Score: <b>{rule_val:.1f}%</b><br>"
                f"• ML Score: <b>{ml_val:.1f}%</b><br>"
                f"• Gap: <b>{improvement:.1f}%</b><br><br>"
                f"<b>💡 Context:</b><br>"
                f"Traditional rule-based approach serves as baseline<br>"
                f"ML enhancement provides {improvement:.1f}% boost"
            )
            rule_hover_texts.append(rule_hover)

        fig_ml_comparison = go.Figure()

        fig_ml_comparison.add_trace(go.Bar(
            x=comparison_metrics,
            y=ml_scores,
            name='ML-Enhanced',
            marker=dict(color=colors['primary']),
            text=[f"{v:.1f}%" for v in ml_scores],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=ml_hover_texts
        ))

        fig_ml_comparison.add_trace(go.Bar(
            x=comparison_metrics,
            y=rule_scores,
            name='Rules Only',
            marker=dict(color=colors['secondary']),
            text=[f"{v:.1f}%" for v in rule_scores],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=rule_hover_texts
        ))

        fig_ml_comparison.update_layout(
            title="Performance: ML-Enhanced vs Traditional Rules",
            yaxis_title="Score (%)",
            height=350,
            barmode='group'
        )

        st.plotly_chart(fig_ml_comparison, use_container_width=True, key="ml_rule_comparison")

    with ml_rule_viz_col2:
        st.markdown("### 🎯 Rule Optimization Impact")

        # Show before/after for optimized rules
        optimized_rules = rule_performance_df.nlargest(5, 'precision')['rule_name'].tolist()
        before_precision = [0.78, 0.82, 0.75, 0.80, 0.79]
        after_precision = optimized_rules_precision = [r['precision'] for _, r in rule_performance_df.nlargest(5, 'precision').iterrows()]

        # Enhanced hover texts for optimization chart
        before_hover_texts = []
        after_hover_texts = []

        for idx in range(5):
            rule_name = optimized_rules[idx][:30]  # Truncate long names
            before_val = before_precision[idx]
            after_val = after_precision[idx]
            improvement = ((after_val - before_val) / before_val) * 100

            # Before hover
            before_hover = (
                f"<b style='font-size:14px'>{rule_name}</b><br><br>"
                f"<b style='color:#ef4444'>📊 BEFORE ML</b><br><br>"
                f"<b>Performance Metrics:</b><br>"
                f"• Precision: <b>{before_val:.1%}</b><br>"
                f"• Rank: <b>#{idx+1}</b> of 5<br>"
                f"• Status: <b>Pre-optimization</b><br><br>"
                f"<b>💡 Context:</b><br>"
                f"Baseline rule performance before ML tuning<br>"
                f"Traditional threshold-based detection"
            )
            before_hover_texts.append(before_hover)

            # After hover
            if improvement > 10:
                status = "🏆 MAJOR IMPROVEMENT"
                status_color = "#10b981"
            elif improvement > 5:
                status = "⭐ GOOD IMPROVEMENT"
                status_color = "#22c55e"
            else:
                status = "✅ SLIGHT IMPROVEMENT"
                status_color = "#3b82f6"

            after_hover = (
                f"<b style='font-size:14px'>{rule_name}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 Performance Metrics:</b><br>"
                f"• Precision: <b>{after_val:.1%}</b><br>"
                f"• Previous: <b>{before_val:.1%}</b><br>"
                f"• Improvement: <b>+{improvement:.1f}%</b><br>"
                f"• Rank: <b>#{idx+1}</b> of 5<br><br>"
                f"<b>💡 ML Enhancement:</b><br>"
                f"Machine learning optimization improved precision<br>"
                f"Dynamic threshold adjustment and pattern learning"
            )
            after_hover_texts.append(after_hover)

        fig_optimization = go.Figure()

        fig_optimization.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=before_precision,
            name='Before ML',
            mode='lines+markers',
            line=dict(color=colors['danger'], width=3),
            marker=dict(size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=before_hover_texts
        ))

        fig_optimization.add_trace(go.Scatter(
            x=list(range(1, 6)),
            y=after_precision,
            name='After ML',
            mode='lines+markers',
            line=dict(color=colors['success'], width=3),
            marker=dict(size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=after_hover_texts
        ))

        fig_optimization.update_layout(
            title="Precision Improvement Post-ML Optimization",
            xaxis_title="Rule Rank",
            yaxis_title="Precision",
            height=350,
            yaxis=dict(range=[0.7, 1.0])
        )

        st.plotly_chart(fig_optimization, use_container_width=True, key="ml_optimization_impact")

    # ML Feature Importance for Rules
    st.markdown("### 🔍 ML Feature Importance Analysis")

    feat_col1, feat_col2 = st.columns(2)

    with feat_col1:
        st.markdown("#### Top Features for Fraud Detection")

        ml_features = [
            'Transaction Amount Deviation',
            'Time-of-Day Anomaly',
            'Geographic Risk Score',
            'Account Behavior Change',
            'Counterparty Risk',
            'Velocity Metrics'
        ]
        ml_importance = [0.28, 0.22, 0.18, 0.14, 0.11, 0.07]

        # Enhanced hover texts for feature importance
        feature_hover_texts = []

        for idx, (feature, importance) in enumerate(zip(ml_features, ml_importance)):
            rank = idx + 1

            # Assess importance level
            if importance > 0.25:
                status = "🔴 CRITICAL FEATURE"
                status_color = "#ef4444"
                insight = "Dominant predictor - highest impact on fraud detection"
            elif importance > 0.15:
                status = "🟡 HIGH IMPORTANCE"
                status_color = "#f59e0b"
                insight = "Strong predictor - significant detection contribution"
            elif importance > 0.10:
                status = "🟢 MODERATE IMPORTANCE"
                status_color = "#10b981"
                insight = "Valuable predictor - meaningful contribution"
            else:
                status = "🔵 SUPPORTING FEATURE"
                status_color = "#3b82f6"
                insight = "Supporting predictor - complementary signal"

            # Feature-specific descriptions
            descriptions = {
                'Transaction Amount Deviation': "Detects unusual transaction amounts vs. user history",
                'Time-of-Day Anomaly': "Identifies transactions at atypical times",
                'Geographic Risk Score': "Assesses location-based fraud risk",
                'Account Behavior Change': "Monitors changes in account usage patterns",
                'Counterparty Risk': "Evaluates recipient fraud risk profile",
                'Velocity Metrics': "Tracks transaction frequency and speed"
            }

            hover_text = (
                f"<b style='font-size:14px'>{feature}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 Feature Statistics:</b><br>"
                f"• Importance Score: <b>{importance:.1%}</b><br>"
                f"• Rank: <b>#{rank}</b> of 6<br>"
                f"• Contribution: <b>{importance*100:.1f}%</b><br><br>"
                f"<b>🔍 What It Detects:</b><br>"
                f"{descriptions.get(feature, 'ML-learned pattern')}<br><br>"
                f"<b>💡 ML Insight:</b><br>"
                f"{insight}"
            )
            feature_hover_texts.append(hover_text)

        fig_ml_features = go.Figure(go.Bar(
            y=ml_features,
            x=ml_importance,
            orientation='h',
            marker=dict(
                color=ml_importance,
                colorscale='Viridis',
                showscale=False
            ),
            text=[f"{v:.1%}" for v in ml_importance],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=feature_hover_texts
        ))

        fig_ml_features.update_layout(
            title="ML-Learned Feature Importance",
            xaxis_title="Importance Score",
            height=300
        )

        st.plotly_chart(fig_ml_features, use_container_width=True, key="ml_feature_importance_rules")

    with feat_col2:
        st.markdown("#### Rule Synergy Matrix")

        # Show which rules work best together
        rule_synergy = np.array([
            [1.00, 0.85, 0.72, 0.68],
            [0.85, 1.00, 0.78, 0.65],
            [0.72, 0.78, 1.00, 0.82],
            [0.68, 0.65, 0.82, 1.00]
        ])

        top_rules = ['Amount Anomalies', 'Device Fingerprint', 'Geo-Location', 'Behavior Score']

        # Enhanced hover texts for synergy matrix
        synergy_hover_texts = []

        for row_idx in range(len(top_rules)):
            row_hovers = []
            for col_idx in range(len(top_rules)):
                rule1 = top_rules[row_idx]
                rule2 = top_rules[col_idx]
                synergy_score = rule_synergy[row_idx, col_idx]

                if rule1 == rule2:
                    status = "⭐ SELF"
                    status_color = "#6b7280"
                    insight = "Diagonal represents individual rule performance"
                    recommendation = "No action - self-correlation"
                elif synergy_score > 0.80:
                    status = "🔴 VERY HIGH SYNERGY"
                    status_color = "#ef4444"
                    insight = "These rules work exceptionally well together"
                    recommendation = "Maintain both in detection pipeline"
                elif synergy_score > 0.70:
                    status = "🟡 HIGH SYNERGY"
                    status_color = "#f59e0b"
                    insight = "Strong complementary detection capability"
                    recommendation = "Good rule pairing - monitor effectiveness"
                else:
                    status = "🟢 MODERATE SYNERGY"
                    status_color = "#10b981"
                    insight = "Independent but compatible detection signals"
                    recommendation = "Acceptable pairing"

                hover_text = (
                    f"<b style='font-size:14px'>{rule1} × {rule2}</b><br><br>"
                    f"<b style='color:{status_color}'>{status}</b><br><br>"
                    f"<b>📊 Synergy Metrics:</b><br>"
                    f"• Synergy Score: <b>{synergy_score:.2f}</b><br>"
                    f"• Correlation: <b>{synergy_score*100:.0f}%</b><br><br>"
                    f"<b>💡 ML Insight:</b><br>"
                    f"{insight}<br><br>"
                    f"<b>🎯 Recommendation:</b><br>"
                    f"{recommendation}"
                )
                row_hovers.append(hover_text)

            synergy_hover_texts.append(row_hovers)

        fig_synergy = go.Figure(data=go.Heatmap(
            z=rule_synergy,
            x=top_rules,
            y=top_rules,
            colorscale='RdYlGn',
            text=rule_synergy,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Synergy"),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=synergy_hover_texts
        ))

        fig_synergy.update_layout(
            title="ML-Detected Rule Synergies",
            height=300
        )

        st.plotly_chart(fig_synergy, use_container_width=True, key="rule_synergy_matrix")

    st.markdown("---")
    st.caption(f"💡 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Note:** Enhanced analytics with synthetic correlation data")

if __name__ == "__main__":
    render()
    