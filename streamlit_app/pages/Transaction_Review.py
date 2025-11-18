"""
Transaction Review Detail Page

Comprehensive view of transaction risk scoring workflow showing:
- How auto-clear vs manual review decisions are made
- Complete rule evaluation with triggered rules highlighted
- Risk score calculation with visual breakdowns
- Decision thresholds and critical level assignment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Dict, Any, List

from streamlit_app.api_client import get_api_client
from streamlit_app.theme import get_chart_colors
from streamlit_app.explainability import get_explainability_engine


def render_workflow_diagram():
    """Render the transaction processing workflow diagram"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📊 Transaction Processing Workflow</h3>
    </div>
    """, unsafe_allow_html=True)

    # Create a Sankey diagram to show the workflow - Blue theme
    fig = go.Figure(data=[go.Sankey(
        node = dict(
            pad = 15,
            thickness = 20,
            line = dict(color = "white", width = 0.5),
            label = [
                "Incoming Transaction",  # 0
                "Rule Engine Check",  # 1
                "Risk Score Calculation",  # 2
                "Threshold Comparison",  # 3
                "Auto-Cleared (< 0.3)",  # 4
                "Manual Review (0.3-0.6)",  # 5
                "High Priority Review (> 0.6)",  # 6
                "Critical Review (> 0.8)"  # 7
            ],
            color = ["#1e3a8a", "#2563eb", "#3b82f6", "#60a5fa", "#28A745", "#FFC107", "#FF5722", "#DC3545"],
            x = [0.1, 0.3, 0.5, 0.7, 0.95, 0.95, 0.95, 0.95],
            y = [0.5, 0.5, 0.5, 0.5, 0.2, 0.45, 0.7, 0.9]
        ),
        link = dict(
            source = [0, 1, 2, 3, 3, 3, 3],
            target = [1, 2, 3, 4, 5, 6, 7],
            value = [100, 100, 100, 30, 30, 30, 10],
            color = ["#93c5fd", "#93c5fd", "#93c5fd", "#28A745", "#FFC107", "#FF5722", "#DC3545"]
        )
    )])

    fig.update_layout(
        title="Transaction Flow: From Receipt to Decision",
        font=dict(size=12),
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Workflow steps explanation
    with st.expander("📖 Workflow Step Details", expanded=False):
        st.markdown("""
        **Step 1: Incoming Transaction**
        - Transaction received and basic validation performed
        - Transaction details extracted (amount, type, counterparty, etc.)

        **Step 2: Rule Engine Check**
        - Transaction evaluated against ALL configured fraud detection rules
        - Rules checked include:
          - Geographic fraud (high-risk countries, unexpected routing)
          - Account takeover (recent phone changes, unverified changes)
          - Transaction patterns (velocity, amount anomalies, odd hours)
          - Payroll fraud (account changes, suspicious sources)
          - Money laundering (chain detection, layering patterns)

        **Step 3: Risk Score Calculation**
        - Each triggered rule contributes its weight to the total score
        - Total weight is normalized to 0-1 scale
        - Formula: Risk Score = Sum(Triggered Rule Weights) / Sum(All Rule Weights)

        **Step 4: Threshold Comparison**
        - Risk score compared against configured thresholds:
          - **< 0.3**: Auto-cleared (Low Risk)
          - **0.3 - 0.6**: Manual Review Required (Medium Risk)
          - **0.6 - 0.8**: High Priority Review (High Risk)
          - **> 0.8**: Critical Priority Review (Critical Risk)

        **Step 5: Decision Assignment**
        - Transaction assigned to appropriate queue based on risk level
        - Critical level determines review priority and urgency
        """)


def render_transaction_card(transaction: Dict[str, Any]):
    """Render transaction details card"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">💳 Transaction Details</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Transaction ID", transaction['transaction_id'])
        st.caption(f"Time: {format_timestamp(transaction['timestamp'])}")

    with col2:
        st.metric("Amount", f"${transaction['amount']:,.2f}")
        st.caption(f"Type: {transaction['transaction_type']}")

    with col3:
        direction_emoji = "🔴" if transaction['direction'] == 'debit' else "🟢"
        st.metric("Direction", f"{direction_emoji} {transaction['direction'].upper()}")
        st.caption(f"Counterparty: {transaction.get('counterparty_id', 'N/A')}")

    with col4:
        st.metric("Account", transaction['account_id'])
        st.caption(f"Description: {transaction.get('description', 'N/A')[:30]}")


def render_rule_evaluation(assessment: Dict[str, Any], all_rules: List[Dict[str, Any]]):
    """Render detailed rule evaluation showing all rules checked"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">🔍 Rule Evaluation Results</h3>
    </div>
    """, unsafe_allow_html=True)

    triggered_rules = assessment.get('triggered_rules', {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rules Checked", len(all_rules))

    with col2:
        st.metric("Rules Triggered", len(triggered_rules),
                 delta=f"{(len(triggered_rules)/max(len(all_rules), 1)*100):.1f}%",
                 delta_color="inverse")

    with col3:
        total_weight = sum(rule.get('weight', 0) for rule in triggered_rules.values())
        st.metric("Total Weight", f"{total_weight:.2f}")

    with col4:
        max_weight = max([rule.get('weight', 0) for rule in triggered_rules.values()], default=0)
        st.metric("Highest Weight", f"{max_weight:.2f}")

    st.divider()

    # Categorize rules
    rule_categories = {
        "Geographic Fraud": [],
        "Account Takeover": [],
        "Transaction Patterns": [],
        "Payroll Fraud": [],
        "Odd Hours Activity": [],
        "Money Laundering": [],
        "Other": []
    }

    for rule in all_rules:
        rule_name = rule['name']
        is_triggered = rule_name in triggered_rules
        rule_data = triggered_rules.get(rule_name, rule)

        # Categorize based on rule name
        if any(x in rule_name.lower() for x in ['country', 'geographic', 'routing', 'foreign']):
            category = "Geographic Fraud"
        elif any(x in rule_name.lower() for x in ['phone', 'takeover', 'device']):
            category = "Account Takeover"
        elif any(x in rule_name.lower() for x in ['velocity', 'amount', 'deviation', 'threshold']):
            category = "Transaction Patterns"
        elif 'payroll' in rule_name.lower():
            category = "Payroll Fraud"
        elif 'odd_hours' in rule_name.lower() or 'weekend' in rule_name.lower():
            category = "Odd Hours Activity"
        elif any(x in rule_name.lower() for x in ['chain', 'layering', 'mule', 'reversal']):
            category = "Money Laundering"
        else:
            category = "Other"

        rule_categories[category].append({
            'name': rule_name,
            'description': rule_data.get('description', 'No description'),
            'weight': rule_data.get('weight', 0),
            'triggered': is_triggered
        })

    # Display rules by category
    for category, rules in rule_categories.items():
        if not rules:
            continue

        triggered_count = sum(1 for r in rules if r['triggered'])

        with st.expander(f"**{category}** ({triggered_count}/{len(rules)} triggered)",
                        expanded=(triggered_count > 0)):

            # Sort by triggered status, then by weight
            rules.sort(key=lambda x: (not x['triggered'], -x['weight']))

            for rule in rules:
                if rule['triggered']:
                    st.markdown(f"""
                    <div style='background-color: #fee; padding: 10px; border-left: 4px solid #d32f2f; margin-bottom: 10px; border-radius: 5px;'>
                        <strong>🔴 {rule['name']}</strong> (Weight: {rule['weight']:.1f})<br/>
                        <span style='color: #666;'>{rule['description']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #f0f0f0; padding: 10px; border-left: 4px solid #4caf50; margin-bottom: 10px; border-radius: 5px;'>
                        <strong>✅ {rule['name']}</strong> (Weight: {rule['weight']:.1f})<br/>
                        <span style='color: #666;'>{rule['description']}</span>
                    </div>
                    """, unsafe_allow_html=True)


def render_risk_score_calculation(assessment: Dict[str, Any], all_rules: List[Dict[str, Any]]):
    """Render visual risk score calculation breakdown"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📈 Risk Score Calculation</h3>
    </div>
    """, unsafe_allow_html=True)

    triggered_rules = assessment.get('triggered_rules', {})
    risk_score = assessment['risk_score']

    # Calculate components
    total_triggered_weight = sum(rule.get('weight', 0) for rule in triggered_rules.values())
    total_possible_weight = sum(rule.get('weight', 0) for rule in all_rules)

    # Create visualization of score calculation
    col1, col2 = st.columns([2, 1])

    with col1:
        # Waterfall chart showing weight accumulation
        if triggered_rules:
            rule_names = []
            weights = []

            for name, rule in triggered_rules.items():
                rule_names.append(name[:30] + "..." if len(name) > 30 else name)
                weights.append(rule.get('weight', 0))

            # Sort by weight
            sorted_data = sorted(zip(rule_names, weights), key=lambda x: x[1], reverse=True)
            rule_names, weights = zip(*sorted_data) if sorted_data else ([], [])

            # Enhanced waterfall hover with explainability
            waterfall_hover_texts = []
            cumulative_score = 0
            for rule_name, weight in zip(rule_names, weights):
                cumulative_score += weight
                contribution_pct = (weight / risk_score * 100) if risk_score > 0 else 0

                # Assess impact
                if weight > 0.15:
                    impact = "🔴 MAJOR CONTRIBUTOR"
                    note = "This rule significantly raises the risk score"
                elif weight > 0.08:
                    impact = "🟠 HIGH IMPACT"
                    note = "Notable contribution to overall risk"
                elif weight > 0.04:
                    impact = "🟡 MODERATE"
                    note = "Meaningful but not dominant factor"
                else:
                    impact = "🟢 MINOR"
                    note = "Small incremental contribution"

                hover_text = (
                    f"<b style='font-size:14px'>{rule_name}</b><br><br>"
                    f"<b>📊 Contribution:</b><br>"
                    f"• Weight Added: <b>+{weight:.3f}</b><br>"
                    f"• Share of Total: <b>{contribution_pct:.1f}%</b><br>"
                    f"• Cumulative Score: <b>{cumulative_score:.3f}</b><br><br>"
                    f"<b style='color:#dc2626'>{impact}</b><br>"
                    f"<b>💡 Meaning:</b> {note}<br><br>"
                    f"<b>🎯 Impact:</b> Without this rule, score would be {cumulative_score - weight:.3f}"
                )
                waterfall_hover_texts.append(hover_text)

            # Add final score hover
            final_hover = (
                f"<b style='font-size:14px'>Final Risk Score</b><br><br>"
                f"<b>📊 Score:</b> <b>{risk_score:.3f}</b><br><br>"
                f"<b>🎯 Decision:</b><br>"
                f"{'🔴 HIGH RISK - Manual Review Required' if risk_score >= 0.8 else '🟠 MEDIUM RISK - Review Needed' if risk_score >= 0.6 else '🟡 LOW RISK - Quick Check' if risk_score >= 0.3 else '🟢 AUTO-CLEARED - Low Risk'}<br><br>"
                f"<b>📊 Composition:</b><br>"
                f"• Number of Rules: <b>{len(weights)}</b><br>"
                f"• Total Weight: <b>{sum(weights):.3f}</b><br>"
                f"• Avg per Rule: <b>{sum(weights)/len(weights):.3f}</b>" if weights else ""
            )
            waterfall_hover_texts.append(final_hover)

            fig = go.Figure(go.Waterfall(
                name = "Risk Score",
                orientation = "v",
                measure = ["relative"] * len(weights) + ["total"],
                x = list(rule_names) + ["Final Risk Score"],
                textposition = "outside",
                text = [f"+{w:.2f}" for w in weights] + [f"{risk_score:.3f}"],
                y = list(weights) + [risk_score],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                hovertemplate='%{customdata}<extra></extra>',
                customdata=waterfall_hover_texts
            ))

            fig.update_layout(
                title = "Risk Score Accumulation by Rule",
                showlegend = False,
                height = 400,
                yaxis_title = "Weight Contribution"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("✅ No rules triggered - Clean transaction")

    with col2:
        st.markdown("#### Calculation Formula")
        st.markdown(f"""
        **Triggered Rules Weight:**
        {total_triggered_weight:.2f}

        **Total Possible Weight:**
        {total_possible_weight:.2f}

        **Normalization:**
        ```
        Risk Score =
          {total_triggered_weight:.2f} / {total_possible_weight:.2f}
          = {risk_score:.4f}
        ```

        **Rounded Score:**
        **{risk_score:.2f}**
        """)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "", 'font': {'size': 40}},
            gauge = {
                'axis': {'range': [None, 1], 'tickwidth': 1},
                'bar': {'color': get_risk_color(risk_score)},
                'steps': [
                    {'range': [0, 0.3], 'color': '#E8F5E9'},
                    {'range': [0.3, 0.6], 'color': '#FFF9C4'},
                    {'range': [0.6, 0.8], 'color': '#FFCCBC'},
                    {'range': [0.8, 1], 'color': '#FFCDD2'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.6
                }
            }
        ))

        fig.update_layout(height=250, margin=dict(l=20, r=20, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_threshold_comparison(assessment: Dict[str, Any]):
    """Render threshold comparison visualization"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">🎯 Threshold Comparison & Decision Logic</h3>
    </div>
    """, unsafe_allow_html=True)

    risk_score = assessment['risk_score']
    decision = assessment['decision']

    # Determine risk level and color
    if risk_score < 0.3:
        risk_level = "LOW RISK"
        risk_color = "#28A745"
        critical_level = "Auto-Cleared"
    elif risk_score < 0.6:
        risk_level = "MEDIUM RISK"
        risk_color = "#FFC107"
        critical_level = "Manual Review Required"
    elif risk_score < 0.8:
        risk_level = "HIGH RISK"
        risk_color = "#FF5722"
        critical_level = "High Priority Review"
    else:
        risk_level = "CRITICAL RISK"
        risk_color = "#DC3545"
        critical_level = "Critical Priority Review"

    # Create horizontal bar showing score position
    fig = go.Figure()

    # Add threshold zones
    fig.add_trace(go.Bar(
        y=['Risk Level'],
        x=[0.3],
        name='Auto-Clear Zone',
        orientation='h',
        marker=dict(color='#28A745'),
        text=['Auto-Clear<br>(< 0.3)'],
        textposition='inside',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Bar(
        y=['Risk Level'],
        x=[0.3],
        name='Manual Review Zone',
        orientation='h',
        marker=dict(color='#FFC107'),
        text=['Manual Review<br>(0.3 - 0.6)'],
        textposition='inside',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Bar(
        y=['Risk Level'],
        x=[0.2],
        name='High Priority Zone',
        orientation='h',
        marker=dict(color='#FF5722'),
        text=['High Priority<br>(0.6 - 0.8)'],
        textposition='inside',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Bar(
        y=['Risk Level'],
        x=[0.2],
        name='Critical Zone',
        orientation='h',
        marker=dict(color='#DC3545'),
        text=['Critical<br>(> 0.8)'],
        textposition='inside',
        hoverinfo='skip'
    ))

    # Add marker for current score
    fig.add_trace(go.Scatter(
        x=[risk_score],
        y=['Risk Level'],
        mode='markers+text',
        marker=dict(size=20, color='black', symbol='diamond'),
        text=[f'Score: {risk_score:.2f}'],
        textposition='top center',
        name='Current Transaction',
        hoverinfo='text',
        hovertext=f'Risk Score: {risk_score:.2f}<br>{risk_level}'
    ))

    fig.update_layout(
        barmode='stack',
        height=200,
        showlegend=False,
        xaxis=dict(
            title='Risk Score',
            range=[0, 1],
            tickvals=[0, 0.3, 0.6, 0.8, 1.0]
        ),
        yaxis=dict(showticklabels=False),
        margin=dict(l=0, r=0, t=20, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Decision explanation
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(f"""
        <div style='background-color: {risk_color}20; padding: 20px; border-radius: 10px; border: 3px solid {risk_color}; text-align: center;'>
            <h2 style='color: {risk_color}; margin: 0;'>{risk_level}</h2>
            <h3 style='margin: 10px 0;'>Risk Score: {risk_score:.2f}</h3>
            <h4 style='margin: 10px 0;'>Decision: {decision.replace('_', ' ').title()}</h4>
            <p style='font-size: 18px; font-weight: bold; margin: 10px 0;'>{critical_level}</p>
        </div>
        """, unsafe_allow_html=True)


def render_decision_explanation(assessment: Dict[str, Any]):
    """Render detailed explanation of why decision was made"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">💡 Decision Explanation</h3>
    </div>
    """, unsafe_allow_html=True)

    risk_score = assessment['risk_score']
    decision = assessment['decision']
    triggered_rules = assessment.get('triggered_rules', {})

    # Build explanation
    explanation_parts = []

    if risk_score < 0.3:
        explanation_parts.append(f"✅ **Transaction Auto-Cleared**: Risk score ({risk_score:.2f}) is below the auto-approve threshold (0.3).")
        if len(triggered_rules) == 0:
            explanation_parts.append("- No fraud detection rules were triggered")
            explanation_parts.append("- Transaction characteristics match normal patterns")
        else:
            explanation_parts.append(f"- Only {len(triggered_rules)} minor rule(s) triggered with low combined weight")
            explanation_parts.append("- Risk level too low to warrant manual review")

    elif risk_score < 0.6:
        explanation_parts.append(f"⚠️ **Manual Review Required**: Risk score ({risk_score:.2f}) is in the manual review range (0.3 - 0.6).")
        explanation_parts.append(f"- {len(triggered_rules)} fraud detection rule(s) triggered")
        explanation_parts.append("- Risk level requires human review for final decision")
        explanation_parts.append("- Transaction should be reviewed before approval")

    elif risk_score < 0.8:
        explanation_parts.append(f"🔴 **High Priority Review**: Risk score ({risk_score:.2f}) indicates high fraud risk (0.6 - 0.8).")
        explanation_parts.append(f"- {len(triggered_rules)} significant fraud indicators detected")
        explanation_parts.append("- Multiple risk factors present")
        explanation_parts.append("- **Immediate review recommended** - high fraud likelihood")

    else:
        explanation_parts.append(f"🚨 **CRITICAL Priority Review**: Risk score ({risk_score:.2f}) indicates critical fraud risk (> 0.8).")
        explanation_parts.append(f"- {len(triggered_rules)} major fraud indicators triggered")
        explanation_parts.append("- Severe risk factors detected")
        explanation_parts.append("- **URGENT REVIEW REQUIRED** - Very high fraud probability")
        explanation_parts.append("- Consider blocking transaction pending review")

    # Add key risk factors
    if triggered_rules:
        explanation_parts.append("\n**Key Risk Factors:**")
        # Sort by weight
        sorted_rules = sorted(
            triggered_rules.items(),
            key=lambda x: x[1].get('weight', 0),
            reverse=True
        )
        for name, rule in sorted_rules[:5]:  # Top 5
            weight = rule.get('weight', 0)
            description = rule.get('description', name)
            explanation_parts.append(f"- [{weight:.1f}] {description}")

    # Display explanation
    for part in explanation_parts:
        st.markdown(part)

    # Review recommendations
    st.markdown("---")
    st.markdown("#### 👨‍💼 Recommended Actions")

    if risk_score < 0.3:
        st.success("✅ Safe to approve - No action required")
    elif risk_score < 0.6:
        st.warning("""
        ⚠️ **Review Actions:**
        1. Verify transaction details with customer
        2. Check for any unusual patterns
        3. Approve if details confirm legitimacy
        4. Reject if suspicious elements found
        """)
    elif risk_score < 0.8:
        st.error("""
        🔴 **High Priority Review Actions:**
        1. **Contact customer immediately** for verification
        2. Review recent account activity for compromise indicators
        3. Check triggered rules for specific fraud patterns
        4. Consider temporary hold on transaction
        5. Escalate to fraud specialist if needed
        """)
    else:
        st.error("""
        🚨 **CRITICAL Review Actions:**
        1. **BLOCK transaction immediately** pending review
        2. **Contact customer urgently** through verified channels
        3. Review complete account history
        4. Check for account takeover indicators
        5. **Escalate to senior fraud analyst immediately**
        6. Consider account suspension if fraud confirmed
        7. File SAR if required by regulations
        """)


def format_timestamp(timestamp_str):
    """Format ISO timestamp to readable format"""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp_str


def get_risk_color(risk_score):
    """Get color based on risk score"""
    if risk_score < 0.3:
        return "#28A745"
    elif risk_score < 0.6:
        return "#FFC107"
    elif risk_score < 0.8:
        return "#FF5722"
    else:
        return "#DC3545"


def render_audit_trail(transaction: Dict[str, Any], assessment: Dict[str, Any]):
    """Render comprehensive audit trail with timeline and event history"""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">📜 Audit Trail & Decision History</h3>
    </div>
    """, unsafe_allow_html=True)

    # Generate mock audit events - in production, fetch from database
    base_time = datetime.fromisoformat(transaction['timestamp'].replace('Z', '+00:00'))

    audit_events = [
        {
            "timestamp": base_time,
            "event_type": "system",
            "event": "Transaction Received",
            "description": f"Transaction {transaction['transaction_id']} received from {transaction.get('account_id', 'Unknown')}",
            "actor": "System",
            "details": {
                "amount": transaction['amount'],
                "type": transaction['transaction_type'],
                "direction": transaction['direction']
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(milliseconds=50),
            "event_type": "system",
            "event": "Rule Evaluation Started",
            "description": "Transaction entered fraud detection pipeline",
            "actor": "Fraud Engine",
            "details": {
                "total_rules_checked": 20,
                "evaluation_mode": "real-time"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(milliseconds=120),
            "event_type": "rule_trigger",
            "event": "Rule Triggered",
            "description": f"{len(assessment['triggered_rules'])} fraud detection rules triggered",
            "actor": "Fraud Engine",
            "details": assessment['triggered_rules']
        },
        {
            "timestamp": base_time + pd.Timedelta(milliseconds=150),
            "event_type": "system",
            "event": "Risk Score Calculated",
            "description": f"Risk score computed: {assessment['risk_score']:.3f}",
            "actor": "Risk Scoring Engine",
            "details": {
                "risk_score": assessment['risk_score'],
                "total_weight": sum(r['weight'] for r in assessment['triggered_rules'].values()),
                "decision": assessment['decision']
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(milliseconds=180),
            "event_type": "decision",
            "event": "Decision Made",
            "description": f"Transaction routed to {assessment['decision'].replace('_', ' ').title()}",
            "actor": "Decision Engine",
            "details": {
                "decision": assessment['decision'],
                "risk_level": "HIGH" if assessment['risk_score'] > 0.6 else "MEDIUM" if assessment['risk_score'] > 0.3 else "LOW",
                "requires_review": assessment['decision'] == "manual_review"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(minutes=5),
            "event_type": "analyst_action",
            "event": "Assigned to Analyst",
            "description": "Transaction assigned to fraud analyst for review",
            "actor": "System",
            "details": {
                "analyst_id": "analyst_001",
                "analyst_name": "Sarah Chen",
                "queue": "High Priority Queue"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(minutes=12),
            "event_type": "analyst_action",
            "event": "Review Started",
            "description": "Analyst began transaction review",
            "actor": "Sarah Chen (analyst_001)",
            "details": {
                "session_id": "review_session_123",
                "ip_address": "10.0.1.45"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(minutes=15),
            "event_type": "note",
            "event": "Note Added",
            "description": "Analyst added review notes",
            "actor": "Sarah Chen (analyst_001)",
            "details": {
                "note": "Verified with customer via phone. Customer confirmed legitimate international wire transfer for business expansion. Provided invoice and contracts.",
                "note_type": "investigation"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(minutes=18),
            "event_type": "analyst_action",
            "event": "Customer Contact",
            "description": "Customer verification call completed",
            "actor": "Sarah Chen (analyst_001)",
            "details": {
                "contact_method": "phone",
                "phone_number": "***-***-1234",
                "verification_status": "confirmed",
                "customer_response": "Legitimate transaction confirmed"
            }
        },
        {
            "timestamp": base_time + pd.Timedelta(minutes=22),
            "event_type": "approval",
            "event": "Transaction Approved",
            "description": "Transaction approved after manual review",
            "actor": "Sarah Chen (analyst_001)",
            "details": {
                "approval_reason": "Customer verified transaction as legitimate business expense",
                "supporting_docs": ["invoice_INV-2025-001.pdf", "contract_signed.pdf"],
                "final_decision": "APPROVED"
            }
        }
    ]

    # Timeline Visualization
    st.markdown("#### 🕐 Event Timeline")

    # Create timeline chart
    fig = go.Figure()

    # Color mapping for event types - Blue theme
    color_map = {
        "system": "#1e3a8a",
        "rule_trigger": "#3b82f6",
        "decision": "#60a5fa",
        "analyst_action": "#93c5fd",
        "note": "#667eea",
        "approval": "#2563eb"
    }

    # Enhanced hover for timeline events
    for event in audit_events:
        event_type_labels = {
            "system": "⚙️ SYSTEM EVENT",
            "rule_trigger": "🔔 RULE TRIGGER",
            "decision": "🎯 DECISION",
            "analyst_action": "👤 ANALYST ACTION",
            "note": "📝 NOTE",
            "approval": "✅ APPROVAL"
        }

        event_label = event_type_labels.get(event['event_type'], "📌 EVENT")
        event_color = color_map.get(event['event_type'], '#666666')

        # Build enhanced hover
        hover_text = (
            f"<b style='font-size:14px'>{event['event']}</b><br><br>"
            f"<b style='color:{event_color}'>{event_label}</b><br><br>"
            f"<b>📊 Event Details:</b><br>"
            f"• Time: <b>{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}</b><br>"
            f"• Actor: <b>{event['actor']}</b><br>"
            f"• Type: <b>{event['event_type'].replace('_', ' ').title()}</b><br><br>"
            f"<b>💡 Description:</b><br>"
            f"{event['description']}"
        )

        fig.add_trace(go.Scatter(
            x=[event['timestamp']],
            y=[event['event']],
            mode='markers+text',
            marker=dict(
                size=15,
                color=color_map.get(event['event_type'], '#666666'),
                line=dict(width=2, color='white')
            ),
            text=[f"{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}"],
            textposition="top center",
            textfont=dict(size=9),
            name=event['event'],
            hovertemplate='%{customdata}<extra></extra>',
            customdata=[hover_text]
        ))

    fig.update_layout(
        title="Transaction Processing Timeline",
        xaxis_title="Time",
        yaxis_title="Event",
        height=400,
        showlegend=False,
        hovermode='closest',
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed Event Log
    st.markdown("#### 📋 Detailed Event Log")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["All Events", "System Events", "Analyst Actions", "Rule Triggers"])

    with tab1:
        # All events table
        for i, event in enumerate(audit_events):
            event_color = color_map.get(event['event_type'], '#666666')

            with st.expander(
                f"**{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}** - {event['event']} ({event['actor']})",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([1, 3])

                with col1:
                    st.markdown(f"""
                    **Event Type:**
                    <span style='color: {event_color}; font-weight: bold;'>●</span> {event['event_type'].replace('_', ' ').title()}

                    **Actor:**
                    {event['actor']}

                    **Timestamp:**
                    {event['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"**Description:**\n\n{event['description']}")

                    if event['details']:
                        st.markdown("**Details:**")
                        if isinstance(event['details'], dict):
                            for key, value in event['details'].items():
                                if isinstance(value, dict):
                                    st.json(value)
                                else:
                                    st.markdown(f"- **{key.replace('_', ' ').title()}:** {value}")

    with tab2:
        # System events only
        system_events = [e for e in audit_events if e['event_type'] == 'system']
        if system_events:
            for event in system_events:
                st.markdown(f"""
                **{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}** - {event['event']}
                *{event['description']}*
                """)
        else:
            st.info("No system events")

    with tab3:
        # Analyst actions only
        analyst_events = [e for e in audit_events if e['event_type'] in ['analyst_action', 'note', 'approval']]
        if analyst_events:
            for event in analyst_events:
                icon = "✅" if event['event_type'] == 'approval' else "📝" if event['event_type'] == 'note' else "👤"
                st.markdown(f"""
                {icon} **{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}** - {event['event']}
                *Actor: {event['actor']}*
                {event['description']}
                """)
                if 'note' in event['details']:
                    st.info(f"💬 {event['details']['note']}")
        else:
            st.info("No analyst actions yet")

    with tab4:
        # Rule triggers only
        rule_events = [e for e in audit_events if e['event_type'] == 'rule_trigger']
        if rule_events:
            for event in rule_events:
                st.markdown(f"**{event['timestamp'].strftime('%H:%M:%S.%f')[:-3]}** - {event['event']}")
                st.markdown(f"*{event['description']}*")

                if isinstance(event['details'], dict):
                    st.markdown("**Triggered Rules:**")
                    for rule_name, rule_data in event['details'].items():
                        st.markdown(f"- 🔴 **{rule_name}** (Weight: {rule_data['weight']:.1f})")
                        st.markdown(f"  *{rule_data['description']}*")
        else:
            st.info("No rules triggered")

    # Analyst Performance Metrics
    st.markdown("---")
    st.markdown("#### 📊 Review Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Review Time", "22 min", delta="-3 min vs avg")

    with col2:
        st.metric("Analyst Actions", "5 actions", delta="+2 vs avg")

    with col3:
        st.metric("Customer Contact", "1 call", delta="Required")

    with col4:
        st.metric("Final Decision", "APPROVED", delta="After verification")


def get_mock_all_rules():
    """Get all configured rules for display (mock data for demonstration)"""
    # In production, this would fetch from the API/rules engine
    return [
        # Geographic rules
        {"name": "payment_to_high_risk_country", "description": "Payment routed to high-risk or sanctioned country", "weight": 3.5},
        {"name": "unexpected_country_routing", "description": "Payment routed to unexpected country based on vendor history", "weight": 2.5},
        {"name": "domestic_to_foreign_switch", "description": "Domestic-only vendor suddenly paid through foreign account", "weight": 3.0},
        {"name": "first_international_payment", "description": "First international payment from account", "weight": 1.5},

        # Account takeover rules
        {"name": "immediate_transfer_after_phone_change_1h", "description": "Outgoing transfer within 1 hour(s) of phone change - critical account takeover alert", "weight": 5.0},
        {"name": "phone_change_before_transfer_24h", "description": "Outgoing transfer within 24 hours of phone/device change - possible account takeover", "weight": 3.5},
        {"name": "large_transfer_after_phone_change_5000", "description": "Large transfer (>=$5,000.00) within 48h of phone change - high-risk takeover", "weight": 4.0},

        # Transaction patterns
        {"name": "amount_exceeds_10000", "description": "Transaction amount exceeds $10,000.00", "weight": 2.0},
        {"name": "velocity_5_in_24h", "description": "More than 5 transactions in 24 hours", "weight": 1.5},
        {"name": "amount_deviation_3x", "description": "Transaction amount deviates from average by 3x", "weight": 2.0},
        {"name": "new_counterparty", "description": "Transaction with a new counterparty", "weight": 1.0},

        # Payroll rules
        {"name": "payroll_recent_account_change", "description": "Payroll transaction to bank account changed within 30 days", "weight": 3.0},
        {"name": "payroll_unverified_account_change", "description": "Payroll transaction to account with unverified banking information changes", "weight": 4.0},
        {"name": "payroll_suspicious_change_source", "description": "Account changed via email/phone request rather than secure portal", "weight": 3.5},

        # Odd hours rules
        {"name": "odd_hours_transaction", "description": "Transaction initiated during odd hours (22:00 - 06:00)", "weight": 2.0},
        {"name": "large_odd_hours_transaction_5000", "description": "Large transaction (>= $5,000.00) initiated during odd hours - elevated fraud risk", "weight": 3.5},
        {"name": "odd_hours_pattern_deviation", "description": "Transaction at odd hours deviates significantly from customer's normal activity pattern", "weight": 4.0},

        # Money laundering rules
        {"name": "suspicious_chain_70", "description": "Suspicious transaction chain detected (threshold: 0.7)", "weight": 2.0},
        {"name": "credit_refund_transfer_chain_1", "description": "Credit-Refund-Transfer chain detected (min 1 chains)", "weight": 2.5},
        {"name": "layering_pattern_1", "description": "Layering pattern detected - multiple small credits consolidated (min 1 patterns)", "weight": 2.0},
        {"name": "money_mule_72h", "description": "Money mule pattern detected: 5+ small incoming payments (avg ≤$500.00), 70%+ flow-through, transferred within 48h", "weight": 2.0},
    ]


def render():
    """Main render function for Transaction Review Detail page"""

    st.set_page_config(page_title="Transaction Review Detail", page_icon="🔍", layout="wide")

    # Get standardized chart colors
    colors = get_chart_colors()

    # Header with blue theme
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin: 0;">🔍 Transaction Review Detail</h1>
        <p style="color: #e0e7ff; font-size: 18px; margin: 10px 0 0 0;">
            <strong>Comprehensive Analysis: Auto-Clear vs Manual Review Decision Process</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Transaction selector
    col1, col2 = st.columns([3, 1])
    with col1:
        transaction_id = st.text_input("Enter Transaction ID to review:", placeholder="TX_000001")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Load Transaction", use_container_width=True)

    if transaction_id or search_button:
        # In production, fetch from API
        # For now, show with mock data
        with st.spinner("Loading transaction details..."):
            # Mock data - in production, replace with API call
            transaction = {
                "transaction_id": transaction_id or "TX_000001",
                "amount": 15750.00,
                "transaction_type": "WIRE",
                "direction": "debit",
                "timestamp": "2025-01-15T23:45:30",
                "account_id": "ACC_0001",
                "counterparty_id": "COUNTER_0042",
                "description": "International wire transfer"
            }

            assessment = {
                "assessment_id": "RISK_000001",
                "transaction_id": transaction_id or "TX_000001",
                "risk_score": 0.72,
                "decision": "manual_review",
                "triggered_rules": {
                    "large_odd_hours_transaction_5000": {
                        "description": "Large transaction (>= $5,000.00) initiated during odd hours - elevated fraud risk",
                        "weight": 3.5
                    },
                    "amount_exceeds_10000": {
                        "description": "Transaction amount exceeds $10,000.00",
                        "weight": 2.0
                    },
                    "odd_hours_pattern_deviation": {
                        "description": "Transaction at odd hours deviates significantly from customer's normal activity pattern",
                        "weight": 4.0
                    },
                    "first_international_payment": {
                        "description": "First international payment from account",
                        "weight": 1.5
                    }
                }
            }

            all_rules = get_mock_all_rules()

        # Render workflow diagram
        render_workflow_diagram()
        st.divider()

        # Render transaction details
        render_transaction_card(transaction)
        st.divider()

        # Render rule evaluation
        render_rule_evaluation(assessment, all_rules)
        st.divider()

        # Render risk score calculation
        render_risk_score_calculation(assessment, all_rules)
        st.divider()

        # Render threshold comparison
        render_threshold_comparison(assessment)
        st.divider()

        # Render decision explanation
        render_decision_explanation(assessment)

        # Render audit trail
        st.divider()
        render_audit_trail(transaction, assessment)

        # Action buttons
        st.divider()
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;">
            <h3 style="color: white; margin: 0;">🎬 Review Actions</h3>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("✅ Approve Transaction", use_container_width=True):
                st.success("Transaction approved!")

        with col2:
            if st.button("❌ Reject Transaction", use_container_width=True):
                st.error("Transaction rejected!")

        with col3:
            if st.button("⏸️ Hold for Investigation", use_container_width=True):
                st.warning("Transaction placed on hold")

        with col4:
            if st.button("📧 Contact Customer", use_container_width=True):
                st.info("Customer verification initiated")

    else:
        st.info("👆 Enter a transaction ID above to view detailed risk analysis")

        # ML Intelligence Section
        st.markdown("---")
        st.markdown("## 🤖 ML-Powered Transaction Intelligence")
        st.markdown("*Real-time machine learning insights for transaction screening*")

        ml_metrics_col1, ml_metrics_col2, ml_metrics_col3, ml_metrics_col4 = st.columns(4)

        with ml_metrics_col1:
            st.metric("ML Confidence", "89.7%", "+3.2%")
        with ml_metrics_col2:
            st.metric("Auto-Cleared Today", "11,915", "+245")
        with ml_metrics_col3:
            st.metric("ML Accuracy", "94.1%", "+1.5%")
        with ml_metrics_col4:
            st.metric("Avg Processing", "8ms", "-2ms")

        ml_viz_col1, ml_viz_col2 = st.columns(2)

        with ml_viz_col1:
            st.markdown("### 🎯 ML Risk Score Distribution")

            # Generate risk score distribution
            np.random.seed(42)
            risk_scores = np.concatenate([
                np.random.beta(2, 8, 8000) * 0.5,      # Low risk (auto-cleared)
                np.random.beta(5, 5, 1500) * 0.6 + 0.2, # Medium risk
                np.random.beta(8, 2, 500) * 0.4 + 0.6   # High risk
            ])

            fig_risk_dist = go.Figure()

            # Create histogram with color coding
            counts, bins = np.histogram(risk_scores, bins=30)
            colors_bins = [colors['success'] if b < 0.3 else colors['warning'] if b < 0.7 else colors['danger']
                          for b in bins[:-1]]

            # Enhanced histogram hover with explainability
            histogram_hover_texts = []
            total_txns = len(risk_scores)
            for bin_start, count in zip(bins[:-1], counts):
                bin_end = bins[list(bins[:-1]).index(bin_start) + 1]
                bin_mid = (bin_start + bin_end) / 2
                pct_of_total = (count / total_txns * 100) if total_txns > 0 else 0

                # Determine risk category
                if bin_mid < 0.3:
                    category = "🟢 LOW RISK"
                    decision = "Auto-Cleared"
                    action = "No manual review needed"
                    color_label = "Green Zone"
                elif bin_mid < 0.7:
                    category = "🟡 MEDIUM RISK"
                    decision = "Manual Review"
                    action = "Analyst review required"
                    color_label = "Yellow Zone"
                else:
                    category = "🔴 HIGH RISK"
                    decision = "Priority Review"
                    action = "Immediate investigation needed"
                    color_label = "Red Zone"

                hover_text = (
                    f"<b style='font-size:14px'>Risk Score Range: {bin_start:.2f} - {bin_end:.2f}</b><br><br>"
                    f"<b>📊 Volume:</b><br>"
                    f"• Transactions: <b>{count}</b><br>"
                    f"• Share of Total: <b>{pct_of_total:.1f}%</b><br><br>"
                    f"<b>{category}</b><br>"
                    f"• Zone: <b>{color_label}</b><br>"
                    f"• Decision: <b>{decision}</b><br>"
                    f"• Action: {action}<br><br>"
                    f"<b>💡 Meaning:</b><br>"
                    f"{'These transactions are automatically approved with minimal risk' if bin_mid < 0.3 else 'These transactions need human review to verify legitimacy' if bin_mid < 0.7 else 'These transactions have strong fraud indicators and need priority attention'}"
                )
                histogram_hover_texts.append(hover_text)

            fig_risk_dist.add_trace(go.Bar(
                x=bins[:-1],
                y=counts,
                marker=dict(color=colors_bins),
                name='Risk Distribution',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=histogram_hover_texts
            ))

            # Add threshold lines
            fig_risk_dist.add_vline(x=0.3, line_dash="dash", line_color="orange",
                                   annotation_text="Auto-Clear Threshold")
            fig_risk_dist.add_vline(x=0.7, line_dash="dash", line_color="red",
                                   annotation_text="High Risk Threshold")

            fig_risk_dist.update_layout(
                title="Transaction Risk Score Distribution (Last 24 Hours)",
                xaxis_title="ML Risk Score",
                yaxis_title="Number of Transactions",
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig_risk_dist, use_container_width=True, key="txn_risk_dist")

        with ml_viz_col2:
            st.markdown("### 🔍 Top ML Feature Contributions")

            features_txn = [
                'Transaction Amount',
                'Time of Day',
                'Counterparty History',
                'Location Consistency',
                'Account Age',
                'Recent Activity Pattern',
                'Device Fingerprint',
                'Behavioral Score'
            ]
            importance_txn = [0.32, 0.18, 0.15, 0.12, 0.10, 0.07, 0.04, 0.02]

            # Enhanced feature importance hover with explainability
            feature_descriptions = {
                'Transaction Amount': {
                    'desc': 'How unusual is the transaction amount compared to normal behavior',
                    'example': 'A $10,000 transaction from someone who usually spends $50',
                    'impact': 'Most important factor'
                },
                'Time of Day': {
                    'desc': 'Whether the transaction occurs at an unusual time',
                    'example': '3 AM transaction from someone who normally transacts during business hours',
                    'impact': 'Strong indicator'
                },
                'Counterparty History': {
                    'desc': 'Trust level and history with the recipient',
                    'example': 'First-time recipient vs established trusted contact',
                    'impact': 'Significant factor'
                },
                'Location Consistency': {
                    'desc': 'Whether transaction location matches user patterns',
                    'example': 'Transaction from foreign country when user is local',
                    'impact': 'Important signal'
                },
                'Account Age': {
                    'desc': 'How long the account has been active',
                    'example': 'Brand new account vs 5-year-old account',
                    'impact': 'Moderate factor'
                },
                'Recent Activity Pattern': {
                    'desc': 'Consistency with recent transaction behavior',
                    'example': 'Sudden burst of activity after dormant period',
                    'impact': 'Moderate signal'
                },
                'Device Fingerprint': {
                    'desc': 'Recognition of device used for transaction',
                    'example': 'New device vs recognized trusted device',
                    'impact': 'Supporting factor'
                },
                'Behavioral Score': {
                    'desc': 'Overall behavioral biometric analysis',
                    'example': 'Typing patterns, mouse movements, navigation flow',
                    'impact': 'Minor factor'
                }
            }

            feature_hover_texts = []
            for feature, importance in zip(features_txn, importance_txn):
                info = feature_descriptions[feature]

                # Determine significance
                if importance >= 0.20:
                    significance = "🔴 CRITICAL"
                    weight_desc = "Dominant driver of risk score"
                elif importance >= 0.10:
                    significance = "🟠 HIGH"
                    weight_desc = "Major contributor to risk assessment"
                elif importance >= 0.05:
                    significance = "🟡 MODERATE"
                    weight_desc = "Meaningful but not decisive"
                else:
                    significance = "🟢 LOW"
                    weight_desc = "Supporting evidence only"

                hover_text = (
                    f"<b style='font-size:14px'>{feature}</b><br><br>"
                    f"<b>📊 Importance:</b> <b>{importance:.1%}</b><br>"
                    f"<b>{significance}</b><br>"
                    f"{weight_desc}<br><br>"
                    f"<b>💡 What It Measures:</b><br>"
                    f"{info['desc']}<br><br>"
                    f"<b>📝 Example:</b><br>"
                    f"{info['example']}<br><br>"
                    f"<b>🎯 Impact Level:</b> {info['impact']}"
                )
                feature_hover_texts.append(hover_text)

            fig_features_txn = go.Figure(go.Bar(
                y=features_txn,
                x=importance_txn,
                orientation='h',
                marker=dict(
                    color=importance_txn,
                    colorscale='Blues',
                    showscale=False
                ),
                text=[f"{v:.1%}" for v in importance_txn],
                textposition='outside',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=feature_hover_texts
            ))

            fig_features_txn.update_layout(
                title="ML Feature Importance for Risk Scoring",
                xaxis_title="Contribution to Risk Score",
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig_features_txn, use_container_width=True, key="txn_feature_importance")

        # ML Performance Insights
        st.markdown("### 📊 ML Model Performance Metrics")

        perf_col1, perf_col2, perf_col3 = st.columns(3)

        with perf_col1:
            st.markdown("####Precision & Recall")

            precision_data = [0.945, 0.932, 0.928, 0.941, 0.938, 0.943, 0.949]
            recall_data = [0.912, 0.905, 0.898, 0.915, 0.908, 0.918, 0.923]
            days_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

            # Enhanced hover for precision
            precision_hover_texts = []
            for day, prec, rec in zip(days_labels, precision_data, recall_data):
                f1 = 2 * (prec * rec) / (prec + rec)
                hover_text = (
                    f"<b style='font-size:14px'>{day}</b><br><br>"
                    f"<b style='color:#3b82f6'>📊 PRECISION</b><br><br>"
                    f"<b>Metrics:</b><br>"
                    f"• Precision: <b>{prec:.1%}</b><br>"
                    f"• Recall: <b>{rec:.1%}</b><br>"
                    f"• F1 Score: <b>{f1:.1%}</b><br><br>"
                    f"<b>💡 Meaning:</b><br>"
                    f"{prec*100:.1f}% of flagged transactions are actually fraud"
                )
                precision_hover_texts.append(hover_text)

            # Enhanced hover for recall
            recall_hover_texts = []
            for day, prec, rec in zip(days_labels, precision_data, recall_data):
                f1 = 2 * (prec * rec) / (prec + rec)
                hover_text = (
                    f"<b style='font-size:14px'>{day}</b><br><br>"
                    f"<b style='color:#10b981'>📊 RECALL</b><br><br>"
                    f"<b>Metrics:</b><br>"
                    f"• Recall: <b>{rec:.1%}</b><br>"
                    f"• Precision: <b>{prec:.1%}</b><br>"
                    f"• F1 Score: <b>{f1:.1%}</b><br><br>"
                    f"<b>💡 Meaning:</b><br>"
                    f"{rec*100:.1f}% of actual fraud is caught"
                )
                recall_hover_texts.append(hover_text)

            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=days_labels, y=precision_data, name='Precision',
                                       line=dict(color=colors['primary'], width=3), mode='lines+markers',
                                       hovertemplate='%{customdata}<extra></extra>',
                                       customdata=precision_hover_texts))
            fig_pr.add_trace(go.Scatter(x=days_labels, y=recall_data, name='Recall',
                                       line=dict(color=colors['success'], width=3), mode='lines+markers',
                                       hovertemplate='%{customdata}<extra></extra>',
                                       customdata=recall_hover_texts))

            fig_pr.update_layout(height=250, yaxis=dict(range=[0.85, 1.0]), showlegend=True)
            st.plotly_chart(fig_pr, use_container_width=True, key="txn_pr_metrics")

        with perf_col2:
            st.markdown("#### False Positive Rate")

            fp_rate = [0.068, 0.072, 0.075, 0.065, 0.070, 0.062, 0.058]

            # Enhanced hover for FP rate
            fp_hover_texts = []
            for idx, (day, fp) in enumerate(zip(days_labels, fp_rate)):
                week_avg = sum(fp_rate) / len(fp_rate)
                if fp < 0.06:
                    status = "🟢 EXCELLENT"
                elif fp < 0.07:
                    status = "✅ GOOD"
                else:
                    status = "🟡 ACCEPTABLE"

                hover_text = (
                    f"<b style='font-size:14px'>{day}</b><br><br>"
                    f"<b>{status}</b><br><br>"
                    f"<b>📊 FP Metrics:</b><br>"
                    f"• FP Rate: <b>{fp:.1%}</b><br>"
                    f"• Week Avg: <b>{week_avg:.1%}</b><br>"
                    f"• vs Avg: <b>{(fp-week_avg)*100:+.1f}pp</b>"
                )
                fp_hover_texts.append(hover_text)

            fig_fp = go.Figure()
            fig_fp.add_trace(go.Scatter(x=days_labels, y=fp_rate, name='FP Rate',
                                       fill='tozeroy', line=dict(color=colors['danger'], width=3),
                                       hovertemplate='%{customdata}<extra></extra>',
                                       customdata=fp_hover_texts))

            fig_fp.update_layout(height=250, yaxis=dict(range=[0, 0.1]), showlegend=False)
            st.plotly_chart(fig_fp, use_container_width=True, key="txn_fp_rate")

        with perf_col3:
            st.markdown("#### Processing Throughput")

            throughput = [1180, 1205, 1190, 1225, 1210, 1247, 1265]

            # Enhanced hover for throughput
            throughput_hover_texts = []
            for idx, (day, tput) in enumerate(zip(days_labels, throughput)):
                week_avg = sum(throughput) / len(throughput)
                change = ((tput - throughput[idx-1]) / throughput[idx-1] * 100) if idx > 0 else 0

                hover_text = (
                    f"<b style='font-size:14px'>{day}</b><br><br>"
                    f"<b style='color:#3b82f6'>⚡ THROUGHPUT</b><br><br>"
                    f"<b>📊 Performance:</b><br>"
                    f"• Throughput: <b>{tput} tx/min</b><br>"
                    f"• Week Avg: <b>{week_avg:.0f} tx/min</b><br>"
                    f"• vs Previous: <b>{change:+.1f}%</b>"
                )
                throughput_hover_texts.append(hover_text)

            fig_throughput = go.Figure()
            fig_throughput.add_trace(go.Bar(x=days_labels, y=throughput,
                                           marker=dict(color=colors['info']),
                                           hovertemplate='%{customdata}<extra></extra>',
                                           customdata=throughput_hover_texts))

            fig_throughput.update_layout(height=250, yaxis_title="Transactions/Min")
            st.plotly_chart(fig_throughput, use_container_width=True, key="txn_throughput")

        # Show example
        st.markdown("---")
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;">
            <h3 style="color: white; margin: 0;">📚 Example Use Cases</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **Low Risk Example**
            - TX_000100
            - $250 ACH transfer
            - 2 PM business hours
            - Known counterparty
            - Risk Score: 0.15
            - ✅ Auto-cleared
            """)

        with col2:
            st.markdown("""
            **Medium Risk Example**
            - TX_000200
            - $5,500 wire transfer
            - New counterparty
            - Risk Score: 0.45
            - ⚠️ Manual review
            """)

        with col3:
            st.markdown("""
            **High Risk Example**
            - TX_000300
            - $25,000 international wire
            - 2 AM transaction time
            - Recent phone change
            - Risk Score: 0.85
            - 🚨 Critical review
            """)


if __name__ == "__main__":
    render()
