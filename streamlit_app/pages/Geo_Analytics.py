"""
Behavioral Analytics Page

Geographic fraud analysis and behavioral anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from streamlit_app.theme import apply_master_theme, render_page_header, get_chart_colors
from streamlit_app.ai_recommendations import get_ai_engine, render_ai_insight
from streamlit_app.explainability import get_explainability_engine


# Generate synthetic dataset for visualization
np.random.seed(42)

# VPN/Proxy fraud locations (USA)
usa_vpn_locations = pd.DataFrame({
    'state': ['California', 'Texas', 'Florida', 'New York', 'Illinois', 'Pennsylvania',
              'Ohio', 'Georgia', 'North Carolina', 'Michigan', 'New Jersey', 'Virginia'],
    'lat': [36.7783, 31.9686, 27.6648, 42.1657, 40.6331, 41.2033,
            40.4173, 32.1656, 35.7596, 44.3148, 40.0583, 37.4316],
    'lon': [-119.4179, -99.9018, -81.5158, -74.9481, -89.3985, -77.1945,
            -82.9071, -82.9001, -79.0193, -85.6024, -74.4057, -78.6569],
    'vpn_fraud_count': [145, 98, 87, 132, 76, 54, 48, 65, 52, 43, 89, 58],
    'intensity': [0.9, 0.6, 0.55, 0.8, 0.48, 0.35, 0.3, 0.42, 0.33, 0.27, 0.57, 0.37]
})

# Behavioral anomaly timeline data
behavioral_timeline = pd.DataFrame({
    'time': pd.date_range(start='2024-11-01', periods=48, freq='H'),
    'normal_frequency': np.random.poisson(lam=3, size=48),
    'normal_amount': np.random.normal(loc=150, scale=30, size=48),
    'flagged_frequency': [3] * 40 + [8, 9, 12, 15, 18, 11, 7, 5],
    'flagged_amount': [150] * 40 + [180, 250, 450, 2500, 3200, 1800, 500, 300]
})


def render():
    """Render the Behavioral Analytics page"""

    # Apply theme
    apply_master_theme()

    # Header
    render_page_header(
        title="🗺️ Geographic & Behavioral Analysis",
        subtitle="Location-based fraud patterns and behavioral anomaly detection",
        show_logo=False
    )

    # Get standardized chart colors
    colors = get_chart_colors()

    # VPN/Proxy Fraud Locations (USA)
    st.subheader("🌐 Geolocation Threat Map")
    st.caption("Device locations remotely triggered across USA using VPN/Proxy")

    # Enhanced map hover with explainability
    map_hover_texts = []
    for _, row in usa_vpn_locations.iterrows():
        state = row['state']
        fraud_count = row['vpn_fraud_count']
        intensity = row['intensity']

        # Risk assessment
        if fraud_count > 100:
            risk_level = "🔴 CRITICAL"
            risk_color = "#dc2626"
            recommendation = "Deploy enhanced monitoring team immediately"
        elif fraud_count > 50:
            risk_level = "🟠 HIGH"
            risk_color = "#f59e0b"
            recommendation = "Increase screening for this region"
        elif fraud_count > 20:
            risk_level = "🟡 MODERATE"
            risk_color = "#eab308"
            recommendation = "Standard monitoring with alertness"
        else:
            risk_level = "🟢 LOW"
            risk_color = "#10b981"
            recommendation = "Normal processing procedures"

        # Estimate financial impact
        avg_fraud_loss = 3200  # Average VPN/proxy fraud loss
        total_risk = fraud_count * avg_fraud_loss

        hover_text = (
            f"<b style='font-size:16px'>{state}</b><br><br>"
            f"<b style='color:{risk_color}'>{risk_level} RISK</b><br><br>"
            f"<b>📊 Threat Metrics:</b><br>"
            f"• VPN/Proxy Fraud Cases: <b>{fraud_count}</b><br>"
            f"• Intensity Score: <b>{intensity:.2f}</b><br>"
            f"• National Rank: <b>#{usa_vpn_locations[usa_vpn_locations['vpn_fraud_count'] >= fraud_count].shape[0]}</b><br><br>"
            f"<b>💰 Financial Impact:</b><br>"
            f"• Est. Total Exposure: <b>${total_risk:,}</b><br>"
            f"• Avg per Case: <b>${avg_fraud_loss:,}</b><br><br>"
            f"<b>🎯 Recommendation:</b><br>"
            f"{recommendation}"
        )
        map_hover_texts.append(hover_text)

    fig_usa_map = go.Figure(go.Scattergeo(
        lon=usa_vpn_locations['lon'],
        lat=usa_vpn_locations['lat'],
        text=usa_vpn_locations['state'],
        mode='markers',
        marker=dict(
            size=usa_vpn_locations['vpn_fraud_count'] / 3,
            color=usa_vpn_locations['intensity'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Intensity"),
            line=dict(width=1, color='white'),
            sizemode='diameter'
        ),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=map_hover_texts
    ))

    fig_usa_map.update_layout(
        geo=dict(
            scope='usa',
            projection_type='albers usa',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastlinecolor='rgb(204, 204, 204)',
            showlakes=True,
            lakecolor='rgb(255, 255, 255)',
            showsubunits=True,
            showcountries=True,
            resolution=50,
            projection=dict(
                type='albers usa'
            )
        ),
        height=600,
        title="VPN/Proxy Fraud Activity Across USA"
    )

    st.plotly_chart(fig_usa_map, use_container_width=True)

    # Top states table
    col1, col2, col3 = st.columns(3)
    top_states = usa_vpn_locations.nlargest(3, 'vpn_fraud_count')

    with col1:
        st.metric("Highest VPN Fraud", top_states.iloc[0]['state'],
                 f"{top_states.iloc[0]['vpn_fraud_count']} cases")
    with col2:
        st.metric("Second Highest", top_states.iloc[1]['state'],
                 f"{top_states.iloc[1]['vpn_fraud_count']} cases")
    with col3:
        st.metric("Third Highest", top_states.iloc[2]['state'],
                 f"{top_states.iloc[2]['vpn_fraud_count']} cases")

    st.markdown("---")

    # Behavioral Anomaly Timeline
    st.subheader("🧠 Anomaly Detection Timeline")
    st.caption("Normal behavior baseline vs flagged transaction patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Transaction Frequency Analysis**")

        fig_behavior_freq = go.Figure()

        # Enhanced hover for normal frequency
        normal_freq_hover = [
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Normal Frequency:</b> {freq} txn/hour<br><br>"
            f"<b>💡 Meaning:</b> Expected transaction rate<br>"
            f"<b>📊 Status:</b> Baseline behavior pattern<br>"
            f"<b>✅ Assessment:</b> Legitimate customer activity"
            for time, freq in zip(behavioral_timeline['time'], behavioral_timeline['normal_frequency'])
        ]

        fig_behavior_freq.add_trace(go.Scatter(
            x=behavioral_timeline['time'],
            y=behavioral_timeline['normal_frequency'],
            name='Normal Baseline',
            line=dict(color='#10b981', width=2),
            fill='tonexty',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=normal_freq_hover
        ))

        # Enhanced hover for flagged frequency
        flagged_freq_hover = []
        for time, normal_freq, flagged_freq in zip(
            behavioral_timeline['time'],
            behavioral_timeline['normal_frequency'],
            behavioral_timeline['flagged_frequency']
        ):
            deviation = ((flagged_freq - normal_freq) / normal_freq * 100) if normal_freq > 0 else 0

            if deviation > 200:
                severity = "🔴 CRITICAL"
                alert = "Extreme anomaly - potential account takeover"
            elif deviation > 100:
                severity = "🟠 HIGH"
                alert = "Significant spike - investigate immediately"
            elif deviation > 50:
                severity = "🟡 MODERATE"
                alert = "Unusual activity - monitor closely"
            else:
                severity = "🟢 LOW"
                alert = "Minor deviation - within tolerance"

            hover_text = (
                f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M')}<br>"
                f"<b>Flagged Frequency:</b> {flagged_freq} txn/hour<br>"
                f"<b>Normal Baseline:</b> {normal_freq} txn/hour<br><br>"
                f"<b>📊 Deviation:</b> <b>{deviation:+.1f}%</b><br>"
                f"<b>⚠️ Severity:</b> {severity}<br><br>"
                f"<b>🎯 Alert:</b> {alert}"
            )
            flagged_freq_hover.append(hover_text)

        fig_behavior_freq.add_trace(go.Scatter(
            x=behavioral_timeline['time'],
            y=behavioral_timeline['flagged_frequency'],
            name='Flagged Activity',
            line=dict(color='#ef4444', width=2),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=flagged_freq_hover
        ))

        fig_behavior_freq.update_layout(
            xaxis_title="Time",
            yaxis_title="Transaction Frequency",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_behavior_freq, use_container_width=True)

    with col2:
        st.markdown("**Transaction Amount Analysis**")

        fig_behavior_amount = go.Figure()

        # Enhanced hover for normal amounts
        normal_amt_hover = [
            f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M')}<br>"
            f"<b>Normal Amount:</b> ${amt:,.2f}<br><br>"
            f"<b>💡 Meaning:</b> Typical transaction size<br>"
            f"<b>📊 Profile:</b> Customer's baseline spending<br>"
            f"<b>✅ Assessment:</b> Expected behavior"
            for time, amt in zip(behavioral_timeline['time'], behavioral_timeline['normal_amount'])
        ]

        fig_behavior_amount.add_trace(go.Scatter(
            x=behavioral_timeline['time'],
            y=behavioral_timeline['normal_amount'],
            name='Normal Baseline',
            line=dict(color='#10b981', width=2),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=normal_amt_hover
        ))

        # Enhanced hover for flagged amounts
        flagged_amt_hover = []
        for time, normal_amt, flagged_amt in zip(
            behavioral_timeline['time'],
            behavioral_timeline['normal_amount'],
            behavioral_timeline['flagged_amount']
        ):
            amt_increase = ((flagged_amt - normal_amt) / normal_amt * 100) if normal_amt > 0 else 0

            if amt_increase > 500:
                risk = "🔴 CRITICAL"
                assessment = "Extreme amount spike - likely fraud"
            elif amt_increase > 200:
                risk = "🟠 HIGH"
                assessment = "Unusual high-value transaction"
            elif amt_increase > 100:
                risk = "🟡 MODERATE"
                assessment = "Above normal spending - verify"
            else:
                risk = "🟢 LOW"
                assessment = "Slightly elevated - acceptable variance"

            hover_text = (
                f"<b>Time:</b> {time.strftime('%Y-%m-%d %H:%M')}<br>"
                f"<b>Flagged Amount:</b> ${flagged_amt:,.2f}<br>"
                f"<b>Normal Amount:</b> ${normal_amt:,.2f}<br><br>"
                f"<b>📊 Increase:</b> <b>{amt_increase:+.1f}%</b><br>"
                f"<b>⚠️ Risk Level:</b> {risk}<br><br>"
                f"<b>💡 Assessment:</b> {assessment}<br>"
                f"<b>💰 Potential Loss:</b> ${flagged_amt:,.2f} at risk"
            )
            flagged_amt_hover.append(hover_text)

        fig_behavior_amount.add_trace(go.Scatter(
            x=behavioral_timeline['time'],
            y=behavioral_timeline['flagged_amount'],
            name='Flagged Activity',
            line=dict(color='#ef4444', width=2),
            mode='lines+markers',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=flagged_amt_hover
        ))

        fig_behavior_amount.update_layout(
            xaxis_title="Time",
            yaxis_title="Transaction Amount ($)",
            yaxis_type="log",
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig_behavior_amount, use_container_width=True)

    # Anomaly detection insights
    st.markdown("**🚨 Anomaly Detection Summary:**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frequency Spike", "8→18 txn/hr", delta="125% increase")
    with col2:
        st.metric("Amount Spike", "$150→$3,200", delta="2,033% increase")
    with col3:
        st.metric("Detection Time", "< 5 seconds", delta="Real-time")
    with col4:
        st.metric("Pattern Match", "Known fraud signature", delta="99% confidence")

    st.markdown("---")

    # Combined Multi-Metric Anomaly View
    st.subheader("🤖 Cross-Vector Threat Analysis")
    st.caption("Comprehensive view of device, location, and behavioral inconsistencies")

    # Create sample anomaly data for a suspicious transaction
    anomaly_metrics = pd.DataFrame({
        'Metric': ['Device Fingerprint', 'Geo-Location', 'VPN Usage', 'Typing Speed',
                   'Transaction Time', 'Amount', 'Frequency'],
        'Normal_Range': [100, 100, 0, 68, 12, 150, 3],
        'Current_Value': [0, 15, 100, 23, 3, 15000, 15],
        'Anomaly_Score': [100, 85, 100, 66, 75, 99, 83]
    })

    fig_multi_anomaly = go.Figure()

    fig_multi_anomaly.add_trace(go.Bar(
        x=anomaly_metrics['Metric'],
        y=anomaly_metrics['Anomaly_Score'],
        marker=dict(
            color=anomaly_metrics['Anomaly_Score'],
            colorscale='YlOrRd',
            showscale=True,
            colorbar=dict(title="Anomaly<br>Score")
        ),
        text=anomaly_metrics['Anomaly_Score'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Anomaly Score: %{y}<extra></extra>'
    ))

    fig_multi_anomaly.update_layout(
        xaxis_title="Detection Parameter",
        yaxis_title="Anomaly Score (0-100)",
        height=400,
        yaxis=dict(range=[0, 110])
    )

    st.plotly_chart(fig_multi_anomaly, use_container_width=True)

    st.markdown("---")

    # AI Geographic Pattern Analysis
    st.markdown("## 🤖 AI Geographic Risk Analysis")

    geo_col1, geo_col2 = st.columns(2)

    with geo_col1:
        st.markdown("### 🗺️ Regional Fraud Trends")

        # Get AI insights on geographic patterns
        ai_engine = get_ai_engine()

        top_states = usa_vpn_locations.nlargest(3, 'vpn_fraud_count')

        for idx, row in top_states.iterrows():
            geo_insight = ai_engine.get_pattern_insight(
                pattern_type="geographic",
                pattern_data={
                    "location": row['state'],
                    "vpn_count": int(row['vpn_fraud_count']),
                    "intensity": float(row['intensity']),
                    "trend": "increasing"
                }
            )

            render_ai_insight(
                title=f"{row['state']} - {row['vpn_fraud_count']} VPN Cases",
                recommendation=geo_insight,
                icon="🌍"
            )

    with geo_col2:
        st.markdown("### 🕵️ Behavioral Anomaly Insights")

        # Anomaly pattern insight
        anomaly_insight = ai_engine.get_pattern_insight(
            pattern_type="behavioral",
            pattern_data={
                "device_change": "100% different",
                "location_shift": "6,147 miles",
                "vpn_detected": True,
                "typing_speed_variance": "66%",
                "time_anomaly": "2 AM activity"
            }
        )

        render_ai_insight(
            title="Account Takeover Pattern Detected",
            recommendation=anomaly_insight,
            icon="🚨"
        )

        # Temporal pattern insight
        temporal_insight = ai_engine.get_pattern_insight(
            pattern_type="temporal",
            pattern_data={
                "peak_hours": "2-4 AM",
                "normal_hours": "9 AM - 6 PM",
                "fraud_spike": "250% above baseline",
                "day_pattern": "Weekdays higher risk"
            }
        )

        render_ai_insight(
            title="Temporal Fraud Patterns",
            recommendation=temporal_insight,
            icon="⏰"
        )

    st.markdown("---")
    st.caption(f"💡 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | **Note:** Geographic and behavioral analytics with synthetic data")

if __name__ == "__main__":
    render()
    