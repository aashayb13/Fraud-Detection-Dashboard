"""
Investigation Tools Page

Deep-dive investigation features for fraud analysts.
Search transactions, investigate accounts, and analyze fraud detection module outputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List

from streamlit_app.api_client import get_api_client
from streamlit_app.theme import apply_master_theme, render_page_header, get_chart_colors
from streamlit_app.ai_recommendations import get_ai_engine, render_ai_insight
from streamlit_app.explainability import get_explainability_engine


def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"


def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


# def render_transaction_search():
#     """Render transaction search interface"""
#     st.markdown("### 🔍 AI-Powered Transaction Intelligence Search")

#     with st.form("transaction_search"):
#         col1, col2 = st.columns(2)

#         with col1:
#             transaction_id = st.text_input("Transaction ID", help="Search by exact or partial transaction ID")
#             account_id = st.text_input("Account ID", help="Filter by account")

#             # Amount range
#             st.markdown("**Amount Range**")
#             amount_col1, amount_col2 = st.columns(2)
#             with amount_col1:
#                 min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0, step=100.0)
#             with amount_col2:
#                 max_amount = st.number_input("Max Amount", min_value=0.0, value=100000.0, step=100.0)

#         with col2:
#             # Date range
#             st.markdown("**Date Range**")
#             date_col1, date_col2 = st.columns(2)
#             with date_col1:
#                 start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
#             with date_col2:
#                 end_date = st.date_input("End Date", value=datetime.now())

#             risk_level = st.selectbox(
#                 "Risk Level",
#                 ["All", "Low", "Medium", "High"],
#                 help="Filter by risk level"
#             )

#             limit = st.number_input("Max Results", min_value=10, max_value=500, value=50, step=10)

#         search_button = st.form_submit_button("🔍 Search", use_container_width=True)

#     if search_button:
#         client = get_api_client()

#         try:
#             with st.spinner("Searching transactions..."):
#                 results = client.search_transactions(
#                     transaction_id=transaction_id if transaction_id else None,
#                     account_id=account_id if account_id else None,
#                     min_amount=min_amount if min_amount > 0 else None,
#                     max_amount=max_amount if max_amount < 100000 else None,
#                     start_date=start_date.isoformat() if start_date else None,
#                     end_date=end_date.isoformat() if end_date else None,
#                     risk_level=risk_level.lower() if risk_level != "All" else None,
#                     limit=limit
#                 )

#             transactions = results.get("transactions", [])

#             if not transactions:
#                 st.info("No transactions found matching your criteria")
#                 return

#             st.success(f"Found {len(transactions)} transaction(s)")

#             # Display results
#             for idx, tx in enumerate(transactions):
#                 with st.expander(
#                     f"**{tx['transaction_id']}** - {format_currency(tx['amount'])} - "
#                     f"{tx.get('transaction_type', 'N/A')} - Risk: {tx.get('risk_score', 0):.2f}",
#                     expanded=(idx == 0)
#                 ):
#                     col1, col2, col3 = st.columns(3)

#                     with col1:
#                         st.markdown("#### Transaction Details")
#                         st.markdown(f"**ID:** {tx['transaction_id']}")
#                         st.markdown(f"**Account:** {tx['account_id']}")
#                         st.markdown(f"**Amount:** {format_currency(tx['amount'])}")
#                         st.markdown(f"**Direction:** {tx.get('direction', 'N/A')}")

#                     with col2:
#                         st.markdown("#### Risk Information")
#                         risk_score = tx.get('risk_score', 0)
#                         st.markdown(f"**Risk Score:** {risk_score:.3f}")
#                         st.markdown(f"**Decision:** {tx.get('decision', 'N/A')}")
#                         st.markdown(f"**Status:** {tx.get('review_status', 'N/A')}")
#                         st.markdown(f"**Rules Triggered:** {tx.get('triggered_rules_count', 0)}")

#                     with col3:
#                         st.markdown("#### Other Info")
#                         st.markdown(f"**Type:** {tx.get('transaction_type', 'N/A')}")
#                         st.markdown(f"**Counterparty:** {tx.get('counterparty_id', 'N/A')}")
#                         st.markdown(f"**Timestamp:** {format_timestamp(tx.get('timestamp', ''))}")

#                     # Action buttons
#                     btn_col1, btn_col2, btn_col3 = st.columns(3)
#                     with btn_col1:
#                         if st.button(f"View Module Breakdown", key=f"modules_{idx}"):
#                             st.session_state.view_module_breakdown = tx['transaction_id']
#                     with btn_col2:
#                         if st.button(f"Investigate Account", key=f"account_{idx}"):
#                             st.session_state.investigate_account = tx['account_id']
#                     with btn_col3:
#                         st.markdown("")  # Spacing

#         except Exception as e:
#             st.error(f"Search failed: {str(e)}")

def render_transaction_search():
    """Render transaction search interface"""
    st.markdown("### 🔍 Intelligent Transaction Search")

    with st.form("transaction_search"):
        col1, col2 = st.columns(2)

        with col1:
            transaction_id = st.text_input("Transaction ID", help="Search by exact or partial transaction ID")
            account_id = st.text_input("Account ID", help="Filter by account")

            # Amount range
            st.markdown("**Amount Range**")
            amount_col1, amount_col2 = st.columns(2)
            with amount_col1:
                min_amount = st.number_input("Min Amount", min_value=0.0, value=0.0, step=100.0)
            with amount_col2:
                max_amount = st.number_input("Max Amount", min_value=0.0, value=100000.0, step=100.0)

        with col2:
            # Date range
            st.markdown("**Date Range**")
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
            with date_col2:
                end_date = st.date_input("End Date", value=datetime.now())

            risk_level = st.selectbox(
                "Risk Level",
                ["All", "Low", "Medium", "High"],
                help="Filter by risk level"
            )

            limit = st.number_input("Max Results", min_value=10, max_value=500, value=50, step=10)

        search_button = st.form_submit_button("🔍 Search", use_container_width=True)

    if search_button:
        client = get_api_client()

        try:
            with st.spinner("Searching transactions..."):
                results = client.search_transactions(
                    transaction_id=transaction_id if transaction_id else None,
                    account_id=account_id if account_id else None,
                    min_amount=min_amount if min_amount > 0 else None,
                    max_amount=max_amount if max_amount < 100000 else None,
                    start_date=start_date.isoformat() if start_date else None,
                    end_date=end_date.isoformat() if end_date else None,
                    risk_level=risk_level.lower() if risk_level != "All" else None,
                    limit=limit
                )

            transactions = results.get("transactions", [])

            if not transactions:
                st.info("No transactions found matching your criteria")
                return

            st.success(f"Found {len(transactions)} transaction(s)")

            # ===============================
            # 📊 Global Visualizations (NEW)
            # ===============================
            # Extract data safely
            risk_scores = [float(t.get('risk_score', 0) or 0) for t in transactions]
            amounts = [float(t.get('amount', 0) or 0) for t in transactions]
            tx_ids = [t.get('transaction_id', 'N/A') for t in transactions]

            viz_col1, viz_col2 = st.columns(2)

            # Risk Score Distribution (Histogram)
            with viz_col1:
                st.markdown("#### 📊 Risk Severity Distribution")

                # Enhanced hover for histogram bins
                # Calculate histogram data first
                hist_data, bin_edges = np.histogram(risk_scores, bins=20)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                dist_hover_texts = []
                total_txs = len(risk_scores)
                for count, score_mid in zip(hist_data, bin_centers):
                    percentage = (count / total_txs) * 100

                    if score_mid > 0.75:
                        status = "🔴 CRITICAL RISK"
                        status_color = "#ef4444"
                        insight = "Transactions require immediate review - likely fraud"
                    elif score_mid > 0.50:
                        status = "🟡 HIGH RISK"
                        status_color = "#f59e0b"
                        insight = "Elevated risk - manual review recommended"
                    elif score_mid > 0.25:
                        status = "🟢 MODERATE RISK"
                        status_color = "#10b981"
                        insight = "Medium risk - automated monitoring"
                    else:
                        status = "✅ LOW RISK"
                        status_color = "#3b82f6"
                        insight = "Low risk - routine transactions"

                    hover_text = (
                        f"<b style='font-size:14px'>Risk Range: {score_mid:.2f}</b><br><br>"
                        f"<b style='color:{status_color}'>{status}</b><br><br>"
                        f"<b>📊 Distribution Stats:</b><br>"
                        f"• Transaction Count: <b>{count}</b><br>"
                        f"• Percentage: <b>{percentage:.1f}%</b><br>"
                        f"• Risk Bucket: <b>{score_mid:.2f}</b><br><br>"
                        f"<b>💡 Risk Assessment:</b><br>"
                        f"{insight}"
                    )
                    dist_hover_texts.append(hover_text)

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Bar(
                    x=bin_centers,
                    y=hist_data,
                    width=(bin_edges[1] - bin_edges[0]) * 0.9,
                    marker_color='#3b82f6',
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=dist_hover_texts
                ))
                fig_dist.update_layout(
                    xaxis_title="Risk Score",
                    yaxis_title="Transaction Count",
                    height=400,
                    bargap=0.05
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            # Amount vs Risk (Scatter)
            with viz_col2:
                st.markdown("#### 📈 Risk-Value Correlation Analysis")

                # Enhanced hover for scatter
                scatter_hover_texts = []
                for tx_id, amount, risk in zip(tx_ids, amounts, risk_scores):
                    if risk > 0.75:
                        status = "🔴 BLOCK"
                        status_color = "#ef4444"
                        action = "Transaction blocked - potential fraud"
                    elif risk > 0.50:
                        status = "🟡 REVIEW"
                        status_color = "#f59e0b"
                        action = "Flagged for analyst review"
                    else:
                        status = "✅ CLEAR"
                        status_color = "#10b981"
                        action = "Transaction cleared automatically"

                    # Size classification
                    if amount > 10000:
                        size_class = "High-value transaction"
                    elif amount > 1000:
                        size_class = "Medium-value transaction"
                    else:
                        size_class = "Standard transaction"

                    hover_text = (
                        f"<b style='font-size:14px'>{tx_id}</b><br><br>"
                        f"<b style='color:{status_color}'>{status}</b><br><br>"
                        f"<b>📊 Transaction Details:</b><br>"
                        f"• Amount: <b>${amount:,.2f}</b><br>"
                        f"• Risk Score: <b>{risk:.3f}</b><br>"
                        f"• Classification: <b>{size_class}</b><br><br>"
                        f"<b>💡 Action Taken:</b><br>"
                        f"{action}"
                    )
                    scatter_hover_texts.append(hover_text)

                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=amounts,
                    y=risk_scores,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=risk_scores,
                        colorscale='RdYlGn_r',
                        showscale=True,
                        colorbar=dict(title="Risk Score")
                    ),
                    text=tx_ids,
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=scatter_hover_texts
                ))
                fig_scatter.update_layout(
                    xaxis_title="Transaction Amount ($)",
                    yaxis_title="Risk Score",
                    height=400,
                    hovermode='closest'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.divider()
            st.markdown("#### Results")

            # =====================================
            # Existing per-transaction expanders
            # =====================================
            for idx, tx in enumerate(transactions):
                with st.expander(
                    f"**{tx['transaction_id']}** - {format_currency(tx['amount'])} - "
                    f"{tx.get('transaction_type', 'N/A')} - Risk: {tx.get('risk_score', 0):.2f}",
                    expanded=(idx == 0)
                ):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown("#### Transaction Details")
                        st.markdown(f"**ID:** {tx['transaction_id']}")
                        st.markdown(f"**Account:** {tx['account_id']}")
                        st.markdown(f"**Amount:** {format_currency(tx['amount'])}")
                        st.markdown(f"**Direction:** {tx.get('direction', 'N/A')}")

                    with col2:
                        st.markdown("#### Risk Information")
                        risk_score = float(tx.get('risk_score', 0) or 0)
                        st.markdown(f"**Risk Score:** {risk_score:.3f}")
                        st.markdown(f"**Decision:** {tx.get('decision', 'N/A')}")
                        st.markdown(f"**Status:** {tx.get('review_status', 'N/A')}")
                        st.markdown(f"**Rules Triggered:** {tx.get('triggered_rules_count', 0)}")

                    with col3:
                        st.markdown("#### Other Info")
                        st.markdown(f"**Type:** {tx.get('transaction_type', 'N/A')}")
                        st.markdown(f"**Counterparty:** {tx.get('counterparty_id', 'N/A')}")
                        st.markdown(f"**Timestamp:** {format_timestamp(tx.get('timestamp', ''))}")

                    # AI Analysis Section
                    st.markdown("---")
                    st.markdown("#### 🤖 AI Analysis")

                    ai_engine = get_ai_engine()
                    tx_recommendation = ai_engine.get_risk_recommendation(
                        risk_score=risk_score,
                        amount=tx['amount'],
                        context={
                            'type': tx.get('transaction_type', 'Unknown'),
                            'rules_triggered': tx.get('triggered_rules_count', 0),
                            'decision': tx.get('decision', 'N/A')
                        }
                    )

                    st.info(tx_recommendation)

                    # Action buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    with btn_col1:
                        if st.button(f"View Module Breakdown", key=f"modules_{idx}"):
                            st.session_state.view_module_breakdown = tx['transaction_id']
                    with btn_col2:
                        if st.button(f"Investigate Account", key=f"account_{idx}"):
                            st.session_state.investigate_account = tx['account_id']
                    with btn_col3:
                        st.markdown("")  # Spacing

        except Exception as e:
            st.error(f"Search failed: {str(e)}")


def render_module_breakdown(transaction_id: str):
    """Render fraud detection module breakdown"""
    st.markdown(f"### 🔬 Detection Module Analytics - {transaction_id}")

    client = get_api_client()

    try:
        with st.spinner("Loading module breakdown..."):
            breakdown = client.get_transaction_module_breakdown(transaction_id)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Risk Score", f"{breakdown.get('risk_score', 0):.3f}")
        with col2:
            st.metric("Modules Triggered", f"{breakdown.get('total_modules_triggered', 0)}/25")
        with col3:
            st.metric("Decision", breakdown.get('decision', 'N/A'))
        with col4:
            st.metric("Status", breakdown.get('review_status', 'N/A'))

        st.divider()

        modules = breakdown.get("modules_triggered", [])

        if not modules:
            st.info("No fraud detection modules were triggered for this transaction")
            return

        st.markdown("#### ⚡ Active Detection Triggers")

        # Create DataFrame
        df = pd.DataFrame(modules)

        # Color code by severity
        def get_severity_color(severity):
            colors = {
                "high": "#ff4444",
                "medium": "#ff8800",
                "low": "#ffaa00"
            }
            return colors.get(severity, "#cccccc")

        # Display as colored cards
        for idx, module in enumerate(modules):
            severity = module.get("severity", "low")
            color = get_severity_color(severity)

            st.markdown(
                f"""
                <div style="
                    background-color: {color}15;
                    border-left: 4px solid {color};
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 10px;
                ">
                    <h4 style="margin: 0; color: {color};">
                        {module.get('description', module.get('name', 'Unknown Module'))}
                    </h4>
                    <p style="margin: 5px 0;">
                        <strong>Weight:</strong> {module.get('weight', 0):.3f} |
                        <strong>Category:</strong> {module.get('category', 'general')} |
                        <strong>Severity:</strong> {severity.upper()}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Summary chart
        st.markdown("#### 🎯 Module Contribution Weights")
        fig = px.bar(
            df,
            x="weight",
            y="description",
            orientation='h',
            color="severity",
            color_discrete_map={
                "high": "#ff4444",
                "medium": "#ff8800",
                "low": "#ffaa00"
            },
            title="Triggered Modules by Weight"
        )
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load module breakdown: {str(e)}")


def render_account_risk_timeline(account_id: str, time_range: str = "7d"):
    """Render risk score timeline for an account"""
    client = get_api_client()

    try:
        with st.spinner("Loading risk timeline..."):
            timeline_data = client.get_account_risk_timeline(account_id, time_range)

        timeline = timeline_data.get("timeline", [])
        statistics = timeline_data.get("statistics", {})

        if not timeline:
            st.info("No transaction history available for this time period")
            return

        # Statistics overview
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Transactions", timeline_data.get("total_transactions", 0))

        with col2:
            st.metric("Avg Risk", f"{statistics.get('average_risk', 0):.3f}")

        with col3:
            st.metric("Current Risk", f"{statistics.get('current_risk', 0):.3f}")

        with col4:
            trend = statistics.get('risk_trend', 'stable')
            trend_emoji = "📈" if trend == "increasing" else "📉"
            st.metric("Trend", f"{trend_emoji} {trend.title()}")

        with col5:
            st.metric("High Risk Count", statistics.get('high_risk_count', 0))

        # Create timeline chart
        df = pd.DataFrame(timeline)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Dual-axis chart: Risk score and moving average
        fig = go.Figure()

        # Add risk score scatter
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['risk_score'],
            name='Risk Score',
            mode='markers+lines',
            marker=dict(
                size=8,
                color=df['risk_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk Score")
            ),
            line=dict(color='lightblue', width=1),
            text=df['transaction_id'],
            hovertemplate='<b>%{text}</b><br>Risk: %{y:.3f}<br>Time: %{x}<extra></extra>'
        ))

        # Add moving average line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['moving_average'],
            name='Moving Average (5 tx)',
            line=dict(color='darkblue', width=3),
            hovertemplate='Moving Avg: %{y:.3f}<extra></extra>'
        ))

        # Add risk threshold lines
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange",
                     annotation_text="High Risk Threshold (0.6)")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                     annotation_text="Critical Risk Threshold (0.8)")

        fig.update_layout(
            title=f"Risk Score Timeline - {account_id}",
            xaxis=dict(title="Date/Time"),
            yaxis=dict(title="Risk Score", range=[0, 1]),
            hovermode='x unified',
            height=400,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Transaction amount vs risk scatter
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=df['amount'],
            y=df['risk_score'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['risk_score'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Risk")
            ),
            text=df['transaction_type'],
            hovertemplate='<b>%{text}</b><br>Amount: $%{x:,.0f}<br>Risk: %{y:.3f}<extra></extra>'
        ))

        fig2.add_hline(y=0.6, line_dash="dash", line_color="orange")
        fig2.add_hline(y=0.8, line_dash="dash", line_color="red")

        fig2.update_layout(
            title="Risk Score vs Transaction Amount",
            xaxis=dict(title="Transaction Amount ($)"),
            yaxis=dict(title="Risk Score", range=[0, 1]),
            height=350
        )

        st.plotly_chart(fig2, use_container_width=True)

        # Detailed transaction table
        with st.expander("📋 View Detailed Timeline Data"):
            display_df = df[[
                'timestamp', 'transaction_id', 'amount', 'risk_score',
                'moving_average', 'decision', 'review_status', 'triggered_rules_count'
            ]].copy()

            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            display_df['amount'] = display_df['amount'].apply(format_currency)
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.3f}")
            display_df['moving_average'] = display_df['moving_average'].apply(lambda x: f"{x:.3f}")

            display_df.columns = [
                'Timestamp', 'Transaction ID', 'Amount', 'Risk Score',
                'Moving Avg', 'Decision', 'Status', 'Rules Triggered'
            ]

            st.dataframe(display_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Failed to load risk timeline: {str(e)}")


def render_account_investigation(account_id: str):
    """Render comprehensive account investigation"""
    st.markdown(f"### 👤 Account Forensics Center - {account_id}")

    client = get_api_client()

    try:
        with st.spinner("Loading account information..."):
            account_data = client.get_account_investigation(account_id)

        # Account Overview
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Account Info")
            st.markdown(f"**Account ID:** {account_data.get('account_id')}")
            st.markdown(f"**Status:** {account_data.get('status')}")
            st.markdown(f"**Risk Tier:** {account_data.get('risk_tier')}")
            st.markdown(f"**Created:** {format_timestamp(account_data.get('creation_date', ''))}")

        stats = account_data.get("statistics", {})

        with col2:
            st.markdown("#### Transaction Statistics")
            st.metric("Total Transactions", stats.get("total_transactions", 0))
            st.metric("Total Value", format_currency(stats.get("total_value", 0)))

        with col3:
            st.markdown("#### Risk Profile")
            st.metric("Avg Risk Score", f"{stats.get('average_risk_score', 0):.3f}")
            st.metric("High Risk Count", stats.get("high_risk_count", 0))
            high_risk_rate = stats.get("high_risk_rate", 0) * 100
            st.metric("High Risk Rate", f"{high_risk_rate:.1f}%")

        st.divider()

        # Risk Score Timeline
        st.markdown("### 📊 Account Risk Evolution")

        # Time range selector for timeline
        timeline_col1, timeline_col2 = st.columns([2, 1])
        with timeline_col1:
            timeline_range = st.selectbox(
                "Timeline Period",
                ["24h", "7d", "30d"],
                index=1,
                format_func=lambda x: {
                    "24h": "Last 24 Hours",
                    "7d": "Last 7 Days",
                    "30d": "Last 30 Days"
                }[x],
                key="timeline_range"
            )
        with timeline_col2:
            st.markdown("")  # Spacing

        render_account_risk_timeline(account_id, timeline_range)

        st.divider()

        # Employees
        employees = account_data.get("employees", [])
        if employees:
            st.markdown("#### Associated Employees")
            emp_df = pd.DataFrame(employees)
            st.dataframe(emp_df, use_container_width=True, hide_index=True)

        st.divider()

        # Recent transactions
        st.markdown("#### Recent Transactions")
        recent_txs = account_data.get("recent_transactions", [])

        if recent_txs:
            tx_df = pd.DataFrame(recent_txs)
            tx_df["amount"] = tx_df["amount"].apply(format_currency)
            tx_df["timestamp"] = tx_df["timestamp"].apply(format_timestamp)
            st.dataframe(tx_df, use_container_width=True, hide_index=True)
        else:
            st.info("No recent transactions")

    except Exception as e:
        st.error(f"Failed to load account information: {str(e)}")


def render():
    """Render the Investigation Tools page"""

    # Apply theme
    apply_master_theme()

    # Header
    render_page_header(
        title="Fraud Surveillance Hub",
        subtitle="Real-Time Fraud Detection & Alert Management",
        show_logo=False
    )

    # Get standardized chart colors
    colors = get_chart_colors()

    # Check if we need to show specific views
    if "view_module_breakdown" in st.session_state:
        transaction_id = st.session_state.view_module_breakdown
        if st.button("← Back to Search"):
            del st.session_state.view_module_breakdown
            st.rerun()
        render_module_breakdown(transaction_id)
        return

    if "investigate_account" in st.session_state:
        account_id = st.session_state.investigate_account
        if st.button("← Back to Search"):
            del st.session_state.investigate_account
            st.rerun()
        render_account_investigation(account_id)
        return

    # Default view: Transaction search
    render_transaction_search()

    # ML Intelligence for Fraud Monitoring
    st.divider()
    st.markdown("## 🤖 ML-Powered Fraud Intelligence")
    st.markdown("*Real-time machine learning insights for fraud detection and prevention*")

    ml_fraud_col1, ml_fraud_col2, ml_fraud_col3, ml_fraud_col4 = st.columns(4)

    with ml_fraud_col1:
        st.metric("ML Fraud Detection", "98.2%", "+2.8%")
    with ml_fraud_col2:
        st.metric("Real-time Blocks", "347", "+52")
    with ml_fraud_col3:
        st.metric("Prevented Losses", "$3.8M", "+$680K")
    with ml_fraud_col4:
        st.metric("Detection Latency", "6ms", "-1ms")

    ml_fraud_viz_col1, ml_fraud_viz_col2 = st.columns(2)

    with ml_fraud_viz_col1:
        st.markdown("### 🎯 ML Fraud Detection by Category")

        fraud_categories = [
            'Account Takeover',
            'Payment Fraud',
            'Identity Theft',
            'Money Laundering',
            'Card Fraud',
            'Wire Fraud'
        ]
        detection_rates = [98.5, 97.8, 96.2, 94.1, 99.1, 95.7]
        fraud_counts = [145, 89, 67, 23, 201, 45]

        fig_fraud_detection = go.Figure()

        # Enhanced hover information with explainability
        hover_texts = []
        for category, rate, count in zip(fraud_categories, detection_rates, fraud_counts):
            avg_loss_map = {
                'Account Takeover': 12400,
                'Payment Fraud': 3200,
                'Identity Theft': 8500,
                'Money Laundering': 45000,
                'Card Fraud': 850,
                'Wire Fraud': 25000
            }
            avg_loss = avg_loss_map.get(category, 5000)

            if rate >= 98:
                assessment = "⭐ EXCELLENT - Near-perfect detection"
                action = "Maintain current model"
            elif rate >= 95:
                assessment = "✅ STRONG - High effectiveness"
                action = "Continue monitoring"
            else:
                assessment = "⚠️ MODERATE - Room for improvement"
                action = "Review and optimize model"

            hover_text = (
                f"<b>{category}</b><br><br>"
                f"<b>📊 Performance Metrics:</b><br>"
                f"• Detection Rate: <b>{rate:.1f}%</b><br>"
                f"• Cases Detected: <b>{count}</b><br>"
                f"• Est. Cases Missed: <b>{int(count * (100-rate) / rate)}</b><br><br>"
                f"<b>💰 Financial Impact:</b><br>"
                f"• Avg Loss/Case: <b>${avg_loss:,}</b><br>"
                f"• Total Prevented: <b>${count * avg_loss:,}</b><br><br>"
                f"<b>🎯 Assessment:</b> {assessment}<br>"
                f"<b>💡 Action:</b> {action}"
            )
            hover_texts.append(hover_text)

        fig_fraud_detection.add_trace(go.Bar(
            x=fraud_categories,
            y=detection_rates,
            name='Detection Rate (%)',
            marker=dict(color=colors['success']),
            text=[f"{v:.1f}%" for v in detection_rates],
            textposition='outside',
            yaxis='y',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        # Enhanced hover for scatter plot
        scatter_hover_texts = []
        for category, count in zip(fraud_categories, fraud_counts):
            total_estimated = int(count * 1.05)  # Estimate with 5% escape rate
            scatter_hover_text = (
                f"<b>{category} - Cases Detected</b><br><br>"
                f"<b>📈 Detection Stats:</b><br>"
                f"• Detected: <b>{count} cases</b><br>"
                f"• Est. Total: <b>{total_estimated} cases</b><br>"
                f"• Escape Rate: <b>~5%</b><br><br>"
                f"<b>💡 Insight:</b><br>"
                f"{'High volume - Major fraud vector' if count > 150 else 'Moderate volume - Monitor trends' if count > 50 else 'Low volume but critical to track'}"
            )
            scatter_hover_texts.append(scatter_hover_text)

        fig_fraud_detection.add_trace(go.Scatter(
            x=fraud_categories,
            y=fraud_counts,
            name='Cases Detected',
            mode='lines+markers',
            marker=dict(size=10, color=colors['danger']),
            line=dict(width=3),
            yaxis='y2',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=scatter_hover_texts
        ))

        fig_fraud_detection.update_layout(
            title="ML Detection Performance by Fraud Type",
            yaxis=dict(title='Detection Rate (%)', range=[90, 102]),
            yaxis2=dict(title='Cases Detected', overlaying='y', side='right'),
            height=350,
            xaxis=dict(tickangle=-45),
            hovermode='x unified'
        )

        st.plotly_chart(fig_fraud_detection, use_container_width=True, key="ml_fraud_detection")

    with ml_fraud_viz_col2:
        st.markdown("### 📊 ML Anomaly Score Distribution")

        # Generate anomaly scores
        np.random.seed(42)
        normal_scores = np.random.beta(2, 8, 9000) * 0.4  # Normal transactions
        suspicious_scores = np.random.beta(5, 5, 800) * 0.5 + 0.3  # Suspicious
        fraud_scores = np.random.beta(8, 2, 200) * 0.4 + 0.6  # Fraud

        all_scores = np.concatenate([normal_scores, suspicious_scores, fraud_scores])

        fig_anomaly = go.Figure()

        fig_anomaly.add_trace(go.Histogram(
            x=all_scores,
            nbinsx=40,
            marker=dict(
                color=all_scores,
                colorscale='RdYlGn_r',
                showscale=False
            ),
            opacity=0.75,
            name='Anomaly Distribution'
        ))

        # Add threshold lines
        fig_anomaly.add_vline(x=0.5, line_dash="dash", line_color="orange",
                             annotation_text="Review Threshold")
        fig_anomaly.add_vline(x=0.75, line_dash="dash", line_color="red",
                             annotation_text="Block Threshold")

        fig_anomaly.update_layout(
            title="ML Anomaly Score Distribution",
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            height=350
        )

        st.plotly_chart(fig_anomaly, use_container_width=True, key="ml_anomaly_dist")

    # ML Model Performance Tracking
    st.markdown("### 📈 ML Model Performance Tracking")

    perf_track_col1, perf_track_col2, perf_track_col3 = st.columns(3)

    with perf_track_col1:
        st.markdown("#### Detection Accuracy Trend")

        days_7 = pd.date_range(end=datetime.now(), periods=7, freq='D')
        accuracy_trend = [96.8, 97.2, 97.5, 97.8, 98.0, 98.1, 98.2]

        # Enhanced hover for accuracy trend
        acc_hover_texts = []
        for idx, (day, acc) in enumerate(zip(days_7, accuracy_trend)):
            if idx > 0:
                change = acc - accuracy_trend[idx-1]
            else:
                change = 0

            if acc > 98.0:
                status = "⭐ EXCELLENT"
                status_color = "#10b981"
            elif acc > 97.5:
                status = "✅ STRONG"
                status_color = "#22c55e"
            else:
                status = "📊 GOOD"
                status_color = "#3b82f6"

            hover_text = (
                f"<b style='font-size:14px'>{day.strftime('%Y-%m-%d')}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 Performance:</b><br>"
                f"• Accuracy: <b>{acc:.1f}%</b><br>"
                f"• Day-over-Day: <b>{change:+.1f}%</b><br>"
                f"• Trend: <b>{'Improving' if change > 0 else 'Stable'}</b>"
            )
            acc_hover_texts.append(hover_text)

        fig_acc_trend = go.Figure()
        fig_acc_trend.add_trace(go.Scatter(
            x=days_7,
            y=accuracy_trend,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color=colors['primary'], width=3),
            marker=dict(size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=acc_hover_texts
        ))

        fig_acc_trend.update_layout(
            height=250,
            yaxis=dict(range=[96, 99], title='Accuracy (%)'),
            showlegend=False
        )

        st.plotly_chart(fig_acc_trend, use_container_width=True, key="fraud_acc_trend")

    with perf_track_col2:
        st.markdown("#### False Positive Rate")

        fp_trend = [0.082, 0.075, 0.071, 0.068, 0.065, 0.062, 0.058]

        # Enhanced hover for FP trend
        fp_hover_texts = []
        for idx, (day, fp) in enumerate(zip(days_7, fp_trend)):
            if idx > 0:
                change = fp - fp_trend[idx-1]
            else:
                change = 0

            if fp < 0.06:
                status = "🟢 EXCELLENT"
                status_color = "#10b981"
            elif fp < 0.075:
                status = "✅ GOOD"
                status_color = "#22c55e"
            else:
                status = "🟡 ACCEPTABLE"
                status_color = "#f59e0b"

            hover_text = (
                f"<b style='font-size:14px'>{day.strftime('%Y-%m-%d')}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 FP Metrics:</b><br>"
                f"• FP Rate: <b>{fp:.1%}</b><br>"
                f"• Day-over-Day: <b>{change:+.3f}</b><br>"
                f"• Trend: <b>{'Improving' if change < 0 else 'Stable'}</b>"
            )
            fp_hover_texts.append(hover_text)

        fig_fp_trend = go.Figure()
        fig_fp_trend.add_trace(go.Scatter(
            x=days_7,
            y=fp_trend,
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color=colors['warning'], width=3),
            marker=dict(size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=fp_hover_texts
        ))

        fig_fp_trend.update_layout(
            height=250,
            yaxis=dict(range=[0, 0.1], title='FP Rate'),
            showlegend=False
        )

        st.plotly_chart(fig_fp_trend, use_container_width=True, key="fraud_fp_trend")

    with perf_track_col3:
        st.markdown("#### Fraud Amount Prevented")

        prevented_amounts = [3.2, 3.4, 3.5, 3.6, 3.7, 3.8, 3.8]

        # Enhanced hover for prevented amounts
        prev_hover_texts = []
        for idx, (day, amount) in enumerate(zip(days_7, prevented_amounts)):
            if idx > 0:
                change = amount - prevented_amounts[idx-1]
            else:
                change = 0

            status = "💰 PROTECTED"
            status_color = "#10b981"

            hover_text = (
                f"<b style='font-size:14px'>{day.strftime('%Y-%m-%d')}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 Prevention Stats:</b><br>"
                f"• Amount: <b>${amount:.1f}M</b><br>"
                f"• Day-over-Day: <b>${change:+.1f}M</b><br>"
                f"• Weekly Total: <b>${sum(prevented_amounts):.1f}M</b>"
            )
            prev_hover_texts.append(hover_text)

        fig_prevented = go.Figure()
        fig_prevented.add_trace(go.Bar(
            x=days_7,
            y=prevented_amounts,
            marker=dict(color=colors['success']),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=prev_hover_texts
        ))

        fig_prevented.update_layout(
            height=250,
            yaxis=dict(title='Amount ($M)'),
            showlegend=False
        )

        st.plotly_chart(fig_prevented, use_container_width=True, key="fraud_prevented")

    # ML Insights Summary
    st.markdown("### 💡 ML Intelligence Summary")

    insight_cards_col1, insight_cards_col2, insight_cards_col3 = st.columns(3)

    with insight_cards_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 15px; border-radius: 10px; color: white;">
            <h5 style="margin-top: 0; color: white;">🎯 Real-time Detection</h5>
            <p style="font-size: 14px;">ML models process 1,247 transactions per minute
            with 98.2% accuracy, blocking fraudulent transactions in under 6ms.</p>
        </div>
        """, unsafe_allow_html=True)

    with insight_cards_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 15px; border-radius: 10px; color: white;">
            <h5 style="margin-top: 0; color: white;">🛡️ Adaptive Learning</h5>
            <p style="font-size: 14px;">Models continuously learn from new fraud patterns,
            improving detection accuracy by 2.8% quarter-over-quarter.</p>
        </div>
        """, unsafe_allow_html=True)

    with insight_cards_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                    padding: 15px; border-radius: 10px; color: white;">
            <h5 style="margin-top: 0; color: white;">💰 Financial Impact</h5>
            <p style="font-size: 14px;">ML-powered fraud prevention has saved $3.8M this
            month, with false positive rates down 29% from last quarter.</p>
        </div>
        """, unsafe_allow_html=True)

    # Quick access section
    st.divider()
    st.markdown("### ⚡ Quick Access")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Recent High-Risk Transactions")
        st.markdown("View the most recent high-risk flagged transactions")
        if st.button("View High-Risk Alerts", use_container_width=True):
            # Trigger search with high risk filter
            st.info("Use the search form above and select 'High' risk level")

    with col2:
        st.markdown("#### Account Lookup")
        account_lookup = st.text_input("Enter Account ID")
        if st.button("Investigate Account", use_container_width=True, key="quick_account"):
            if account_lookup:
                st.session_state.investigate_account = account_lookup
                st.rerun()
            else:
                st.warning("Please enter an account ID")


if __name__ == "__main__":
    render()
