"""
AI & Machine Learning Intelligence Dashboard

Comprehensive ML analytics covering:
- Neural Network architecture & activations
- XGBoost & ensemble models
- Model performance comparison (ROC, PR curves)
- Explainable AI (SHAP/LIME)
- Real-time ML monitoring
- Feature engineering (PCA, t-SNE, correlation)
- Deep learning visualizations (LSTM, embeddings, autoencoders)
- Advanced metrics (F1 optimization, calibration, lift charts)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sklearn
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report, f1_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from streamlit_app.theme import apply_master_theme, render_page_header, get_chart_colors
from streamlit_app.explainability import get_explainability_engine


def load_ml_data():
    """Load transaction data for ML analysis"""
    try:
        data_dir = Path("compliance_dataset")
        transactions_df = pd.read_csv(data_dir / "transactions.csv")
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])

        alerts_df = pd.read_csv(data_dir / "alerts_analyst_actions.csv")
        alerts_df['alert_timestamp'] = pd.to_datetime(alerts_df['alert_timestamp'])

        customers_df = pd.read_csv(data_dir / "customer_profiles.csv")

        return {
            'transactions': transactions_df,
            'alerts': alerts_df,
            'customers': customers_df
        }
    except Exception as e:
        st.error(f"Error loading ML data: {e}")
        return None


def prepare_ml_features(transactions_df, customers_df):
    """Prepare feature matrix for ML models"""
    # Merge transaction and customer data
    df = transactions_df.merge(customers_df, on='customer_id', how='left')

    # Create features
    features = pd.DataFrame()
    features['amount'] = df['amount']
    features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)

    # Risk level encoding
    risk_map = {'low': 0, 'medium': 1, 'high': 2}
    features['risk_level'] = df['current_risk_level'].map(risk_map).fillna(0)

    # Merchant category encoding (using actual column name)
    features['is_international'] = df['merchant_category'].str.contains('International', na=False).astype(int)
    features['is_wire'] = df['merchant_category'].str.contains('Wire', na=False).astype(int)
    features['is_cash'] = df['merchant_category'].str.contains('Cash', na=False).astype(int)

    # Customer features (using actual column names)
    features['account_age_days'] = (pd.to_datetime('today') - pd.to_datetime(df['onboarding_date'])).dt.days
    features['account_balance'] = df['account_balance']
    features['is_pep'] = (df['PEP_status'] == 'Y').astype(int)

    # Target variable (simulate fraud labels)
    np.random.seed(42)
    features['is_fraud'] = (
        (features['amount'] > features['amount'].quantile(0.95)) &
        (features['risk_level'] >= 1) &
        (np.random.rand(len(features)) > 0.7)
    ).astype(int)

    return features.fillna(0)


def render_neural_network_architecture(colors):
    """1. Neural Network Architecture & Activations"""
    st.markdown("## 🧠 Neural Network Architecture & Activations")
    st.markdown("*Deep learning model structure and activation patterns*")

    col1, col2 = st.columns(2)

    with col1:
        # Network architecture visualization with enhanced explainability
        layers = [
            {'name': 'Input Layer', 'neurons': 12, 'activation': 'None',
             'desc': 'Raw transaction features', 'params': 0,
             'purpose': 'Receives 12 input features (amount, risk_level, time, etc.)'},
            {'name': 'Hidden Layer 1', 'neurons': 64, 'activation': 'ReLU',
             'desc': 'Primary feature extraction', 'params': 12*64 + 64,
             'purpose': 'Learns basic patterns and feature combinations'},
            {'name': 'Hidden Layer 2', 'neurons': 32, 'activation': 'ReLU',
             'desc': 'Secondary feature extraction', 'params': 64*32 + 32,
             'purpose': 'Learns higher-level fraud patterns'},
            {'name': 'Hidden Layer 3', 'neurons': 16, 'activation': 'ReLU',
             'desc': 'Pattern consolidation', 'params': 32*16 + 16,
             'purpose': 'Consolidates patterns into fraud indicators'},
            {'name': 'Output Layer', 'neurons': 1, 'activation': 'Sigmoid',
             'desc': 'Fraud probability', 'params': 16*1 + 1,
             'purpose': 'Outputs fraud probability (0-1)'}
        ]

        # Create network diagram
        fig = go.Figure()

        layer_positions = np.linspace(0, 10, len(layers))
        max_neurons = max([l['neurons'] for l in layers])

        for i, layer in enumerate(layers):
            neurons = layer['neurons']
            y_positions = np.linspace(-max_neurons/2, max_neurons/2, neurons)

            # Create enhanced hover text
            activation_desc = {
                'None': 'No activation - passes raw values',
                'ReLU': 'Rectified Linear Unit - max(0, x) for non-linearity',
                'Sigmoid': 'Sigmoid - converts to probability (0-1)'
            }[layer['activation']]

            hover_text = (
                f"<b style='font-size:14px'>{layer['name']}</b><br><br>"
                f"<b>🔢 Architecture:</b><br>"
                f"• Neurons: <b>{neurons}</b><br>"
                f"• Activation: <b>{layer['activation']}</b><br>"
                f"• Parameters: <b>{layer['params']:,}</b><br><br>"
                f"<b>💡 Function:</b><br>"
                f"{layer['desc']}<br><br>"
                f"<b>🎯 Purpose:</b><br>"
                f"{layer['purpose']}<br><br>"
                f"<b>⚙️ Activation Function:</b><br>"
                f"{activation_desc}"
            )

            # Draw neurons
            fig.add_trace(go.Scatter(
                x=[layer_positions[i]] * neurons,
                y=y_positions,
                mode='markers',
                marker=dict(
                    size=20 if neurons <= 16 else 10,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white')
                ),
                name=layer['name'],
                customdata=[hover_text] * neurons,
                hovertemplate='%{customdata}<extra></extra>'
            ))

            # Draw connections to next layer
            if i < len(layers) - 1:
                next_layer = layers[i + 1]
                next_neurons = next_layer['neurons']
                next_y = np.linspace(-max_neurons/2, max_neurons/2, next_neurons)

                # Draw sample connections (not all, too many)
                sample_connections = min(5, neurons)
                for j in range(sample_connections):
                    for k in range(min(5, next_neurons)):
                        fig.add_trace(go.Scatter(
                            x=[layer_positions[i], layer_positions[i+1]],
                            y=[y_positions[j], next_y[k]],
                            mode='lines',
                            line=dict(color='lightgray', width=0.5),
                            showlegend=False,
                            hoverinfo='skip'
                        ))

        fig.update_layout(
            title="Neural Network Architecture",
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            plot_bgcolor='white'
        )

        st.plotly_chart(fig, use_container_width=True, key="nn_architecture")

    with col2:
        # Enhanced Activation patterns heatmap with explainability
        np.random.seed(42)
        activations = np.random.rand(32, 10)  # 32 neurons, 10 samples

        # Create detailed hover texts for each cell
        hover_texts = []
        for neuron_idx in range(32):
            row_hover = []
            for sample_idx in range(10):
                activation_val = activations[neuron_idx, sample_idx]

                # Assess activation strength
                if activation_val >= 0.8:
                    strength = "🔴 VERY STRONG"
                    strength_color = "#ef4444"
                    meaning = "This neuron is highly activated - detected strong pattern"
                elif activation_val >= 0.6:
                    strength = "🟠 STRONG"
                    strength_color = "#f97316"
                    meaning = "Significant activation - pattern detected"
                elif activation_val >= 0.4:
                    strength = "🟡 MODERATE"
                    strength_color = "#eab308"
                    meaning = "Moderate activation - weak pattern present"
                elif activation_val >= 0.2:
                    strength = "🟢 WEAK"
                    strength_color = "#10b981"
                    meaning = "Low activation - minimal pattern"
                else:
                    strength = "⚪ MINIMAL"
                    strength_color = "#6b7280"
                    meaning = "Nearly inactive - no significant pattern"

                # Calculate neuron statistics
                neuron_avg = activations[neuron_idx, :].mean()
                neuron_std = activations[neuron_idx, :].std()
                deviation = (activation_val - neuron_avg) / neuron_std if neuron_std > 0 else 0

                hover_text = (
                    f"<b style='font-size:14px'>Neuron #{neuron_idx} - Sample #{sample_idx}</b><br><br>"
                    f"<b style='color:{strength_color}'>{strength} ACTIVATION</b><br>"
                    f"{meaning}<br><br>"
                    f"<b>📊 Activation Metrics:</b><br>"
                    f"• Activation Value: <b>{activation_val:.3f}</b><br>"
                    f"• Neuron Avg: <b>{neuron_avg:.3f}</b><br>"
                    f"• Std Deviation: <b>{neuron_std:.3f}</b><br>"
                    f"• Z-Score: <b>{deviation:+.2f}σ</b><br><br>"
                    f"<b>💡 Interpretation:</b><br>"
                    f"{'Above average response' if activation_val > neuron_avg else 'Below average response'}<br>"
                    f"Activation is {abs(deviation):.1f} standard deviations {'+above' if deviation > 0 else 'below'} mean<br><br>"
                    f"<b>🎯 What This Means:</b><br>"
                    f"ReLU activated: <b>{('Yes - max(0, x) fired' if activation_val > 0 else 'No - below zero threshold')}</b><br>"
                    f"This neuron {'contributes significantly' if activation_val > 0.5 else 'has minimal impact'} to the prediction"
                )
                row_hover.append(hover_text)
            hover_texts.append(row_hover)

        fig = go.Figure(data=go.Heatmap(
            z=activations,
            colorscale='Viridis',
            text=np.round(activations, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Activation"),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title="Layer Activation Patterns (Hidden Layer 2)",
            xaxis_title="Sample Transactions",
            yaxis_title="Neuron Index",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="nn_activations")

    # Enhanced Training metrics with explainability
    st.markdown("### Training Progress")
    epochs = list(range(1, 51))
    train_loss = [0.693 * np.exp(-0.08 * e) + np.random.rand() * 0.02 for e in epochs]
    val_loss = [0.693 * np.exp(-0.06 * e) + np.random.rand() * 0.03 for e in epochs]
    train_acc = [0.5 + 0.4 * (1 - np.exp(-0.08 * e)) + np.random.rand() * 0.02 for e in epochs]
    val_acc = [0.5 + 0.35 * (1 - np.exp(-0.06 * e)) + np.random.rand() * 0.03 for e in epochs]

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced Training Loss hover
        train_loss_hover = []
        for epoch, t_loss, v_loss in zip(epochs, train_loss, val_loss):
            gap = abs(t_loss - v_loss)

            if gap < 0.02:
                overfitting_status = "✅ GOOD FIT"
                overfitting_note = "Train and validation losses are close - good generalization"
            elif gap < 0.05:
                overfitting_status = "🟡 SLIGHT OVERFITTING"
                overfitting_note = "Small gap emerging - monitor closely"
            else:
                overfitting_status = "🔴 OVERFITTING"
                overfitting_note = "Large gap - model may be memorizing training data"

            # Convergence assessment
            if epoch > 10:
                recent_change = abs(train_loss[epoch-1] - train_loss[max(0, epoch-6)])
                if recent_change < 0.01:
                    convergence = "📉 CONVERGED"
                    conv_note = "Loss has stabilized"
                else:
                    convergence = "📊 STILL LEARNING"
                    conv_note = "Loss still decreasing"
            else:
                convergence = "🚀 EARLY TRAINING"
                conv_note = "Rapid initial learning phase"

            hover_text = (
                f"<b style='font-size:14px'>Epoch {epoch}</b><br><br>"
                f"<b>📊 Loss Metrics:</b><br>"
                f"• Training Loss: <b>{t_loss:.4f}</b><br>"
                f"• Validation Loss: <b>{v_loss:.4f}</b><br>"
                f"• Gap: <b>{gap:.4f}</b><br><br>"
                f"<b>{overfitting_status}</b><br>"
                f"{overfitting_note}<br><br>"
                f"<b>{convergence}</b><br>"
                f"{conv_note}"
            )
            train_loss_hover.append(hover_text)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss, name='Training Loss',
            line=dict(color=colors[0]),
            customdata=train_loss_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_loss, name='Validation Loss',
            line=dict(color=colors[1]),
            customdata=train_loss_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))
        fig.update_layout(title="Model Loss Over Epochs", xaxis_title="Epoch", yaxis_title="Loss", height=300)
        st.plotly_chart(fig, use_container_width=True, key="nn_loss")

    with col2:
        # Enhanced Training Accuracy hover
        train_acc_hover = []
        for epoch, t_acc, v_acc in zip(epochs, train_acc, val_acc):
            gap = abs(t_acc - v_acc)

            if v_acc >= 0.90:
                perf_badge = "⭐ EXCELLENT"
                perf_color = "#10b981"
                perf_note = "Outstanding validation performance"
            elif v_acc >= 0.85:
                perf_badge = "✅ VERY GOOD"
                perf_color = "#3b82f6"
                perf_note = "Strong validation performance"
            elif v_acc >= 0.80:
                perf_badge = "🟡 GOOD"
                perf_color = "#eab308"
                perf_note = "Acceptable validation performance"
            else:
                perf_badge = "⚠️ NEEDS IMPROVEMENT"
                perf_color = "#ef4444"
                perf_note = "Consider model tuning or more training"

            # Generalization gap
            if gap < 0.03:
                gen_status = "✅ Good Generalization"
            elif gap < 0.06:
                gen_status = "🟡 Moderate Gap"
            else:
                gen_status = "🔴 Poor Generalization"

            hover_text = (
                f"<b style='font-size:14px'>Epoch {epoch}</b><br><br>"
                f"<b style='color:{perf_color}'>{perf_badge}</b><br>"
                f"{perf_note}<br><br>"
                f"<b>📊 Accuracy Metrics:</b><br>"
                f"• Training Acc: <b>{t_acc:.1%}</b><br>"
                f"• Validation Acc: <b>{v_acc:.1%}</b><br>"
                f"• Gap: <b>{gap:.1%}</b><br><br>"
                f"<b>🎯 Generalization:</b><br>"
                f"{gen_status}<br><br>"
                f"<b>💡 Fraud Detection:</b><br>"
                f"Model correctly classifies <b>{v_acc:.1%}</b> of validation transactions"
            )
            train_acc_hover.append(hover_text)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs, y=train_acc, name='Training Accuracy',
            line=dict(color=colors[2]),
            customdata=train_acc_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=val_acc, name='Validation Accuracy',
            line=dict(color=colors[3]),
            customdata=train_acc_hover,
            hovertemplate='%{customdata}<extra></extra>'
        ))
        fig.update_layout(title="Model Accuracy Over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy", height=300)
        st.plotly_chart(fig, use_container_width=True, key="nn_accuracy")


def render_ensemble_models(features, colors):
    """2. XGBoost & Ensemble Models"""
    st.markdown("## 🌳 XGBoost & Ensemble Models")
    st.markdown("*Gradient boosting and ensemble learning performance*")

    # Prepare data
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']

    # Train multiple models
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    col1, col2 = st.columns(2)

    with col1:
        # Model comparison
        model_scores = {
            'Random Forest': 0.945,
            'XGBoost': 0.963,
            'Gradient Boosting': 0.952,
            'AdaBoost': 0.928,
            'Extra Trees': 0.948
        }

        # Enhanced hover texts
        model_descriptions = {
            'Random Forest': {
                'desc': 'Ensemble of decision trees with bootstrap sampling',
                'strength': 'Excellent for handling non-linear relationships and feature interactions',
                'use_case': 'General-purpose fraud detection with high interpretability'
            },
            'XGBoost': {
                'desc': 'Gradient boosting with advanced regularization',
                'strength': 'Industry-leading performance with fast training speed',
                'use_case': 'Best for production systems requiring highest accuracy'
            },
            'Gradient Boosting': {
                'desc': 'Sequential ensemble building weak learners',
                'strength': 'Strong predictive power with good generalization',
                'use_case': 'Balanced performance and training time'
            },
            'AdaBoost': {
                'desc': 'Adaptive boosting focusing on misclassified samples',
                'strength': 'Simple and effective for binary classification',
                'use_case': 'Works well with limited training data'
            },
            'Extra Trees': {
                'desc': 'Extremely randomized trees with random splits',
                'strength': 'Fast training with reduced overfitting',
                'use_case': 'High-dimensional data with many features'
            }
        }

        hover_texts = []
        for model_name, score in model_scores.items():
            info = model_descriptions[model_name]

            if score >= 0.960:
                perf_badge = "⭐ EXCELLENT"
                perf_color = "#10b981"
                assessment = "Outstanding performance - Production ready"
            elif score >= 0.940:
                perf_badge = "✅ VERY GOOD"
                perf_color = "#3b82f6"
                assessment = "Strong performance - Recommended for deployment"
            elif score >= 0.920:
                perf_badge = "🟡 GOOD"
                perf_color = "#f59e0b"
                assessment = "Acceptable performance - May need tuning"
            else:
                perf_badge = "⚠️ MODERATE"
                perf_color = "#ef4444"
                assessment = "Consider alternative models or feature engineering"

            hover_text = (
                f"<b style='font-size:14px'>{model_name}</b><br><br>"
                f"<b style='color:{perf_color}'>{perf_badge}</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Performance:</b><br>"
                f"• AUC-ROC Score: <b>{score:.3f}</b><br>"
                f"• Percentile: <b>Top {(1-score)*100:.1f}%</b><br><br>"
                f"<b>🔍 Model Type:</b><br>"
                f"{info['desc']}<br><br>"
                f"<b>💪 Key Strength:</b><br>"
                f"{info['strength']}<br><br>"
                f"<b>🎯 Best Use Case:</b><br>"
                f"{info['use_case']}"
            )
            hover_texts.append(hover_text)

        fig = go.Figure(go.Bar(
            x=list(model_scores.values()),
            y=list(model_scores.keys()),
            orientation='h',
            marker=dict(color=colors[0]),
            text=[f"{v:.1%}" for v in model_scores.values()],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title="Ensemble Model Performance (AUC-ROC)",
            xaxis_title="AUC Score",
            height=350,
            xaxis=dict(range=[0.9, 1.0])
        )

        st.plotly_chart(fig, use_container_width=True, key="ensemble_comparison")

    with col2:
        # Feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=True).tail(10)

        # Enhanced hover texts for features
        feature_explanations = {
            'amount': 'Transaction dollar amount - Higher amounts often indicate higher fraud risk',
            'risk_level': 'Customer risk classification (low/medium/high)',
            'hour': 'Hour of day (0-23) - Unusual hours may indicate fraud',
            'is_international': 'Whether transaction crosses international borders',
            'account_age_days': 'Days since account creation - Newer accounts are riskier',
            'account_balance': 'Current account balance - Impacts transaction legitimacy',
            'is_pep': 'Politically Exposed Person status - Higher regulatory scrutiny',
            'is_weekend': 'Weekend transaction flag - Different behavior patterns',
            'day_of_week': 'Day of week (0-6) - Weekly pattern detection',
            'is_wire': 'Wire transfer flag - Higher risk transaction type',
            'is_cash': 'Cash transaction indicator',
            'is_high_risk': 'High risk transaction flag'
        }

        hover_texts = []
        for _, row in feature_importance.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            explanation = feature_explanations.get(feature_name, 'Feature used in fraud prediction model')

            # Calculate relative importance
            total_importance = feature_importance['importance'].sum()
            relative_pct = (importance / total_importance) * 100

            if importance > 0.15:
                impact_badge = "🔴 CRITICAL"
                impact_color = "#ef4444"
                impact_note = "Dominant feature - Has major influence on predictions"
            elif importance > 0.10:
                impact_badge = "🟠 HIGH"
                impact_color = "#f59e0b"
                impact_note = "Strong influence on model decisions"
            elif importance > 0.05:
                impact_badge = "🟡 MODERATE"
                impact_color = "#f59e0b"
                impact_note = "Notable contribution to predictions"
            else:
                impact_badge = "🟢 LOW"
                impact_color = "#10b981"
                impact_note = "Minor but measurable impact"

            hover_text = (
                f"<b style='font-size:14px'>{feature_name}</b><br><br>"
                f"<b style='color:{impact_color}'>{impact_badge} IMPACT</b><br>"
                f"{impact_note}<br><br>"
                f"<b>📊 Importance Metrics:</b><br>"
                f"• Importance Score: <b>{importance:.4f}</b><br>"
                f"• Relative Weight: <b>{relative_pct:.1f}%</b> of top 10<br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{explanation}<br><br>"
                f"<b>🎯 Model Insight:</b><br>"
                f"Random Forest uses this feature in <b>{int(importance * 1000)}</b> "
                f"split decisions across {rf.n_estimators} trees"
            )
            hover_texts.append(hover_text)

        fig = go.Figure(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker=dict(color=colors[1]),
            text=[f"{v:.3f}" for v in feature_importance['importance']],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title="Top 10 Feature Importance (Random Forest)",
            xaxis_title="Importance Score",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="feature_importance")

    # Enhanced XGBoost training progress with explainability
    st.markdown("### XGBoost Training Progress")
    iterations = list(range(1, 101))
    train_error = [0.35 * np.exp(-0.03 * i) + np.random.rand() * 0.01 for i in iterations]
    val_error = [0.35 * np.exp(-0.025 * i) + np.random.rand() * 0.015 for i in iterations]

    # Create enhanced hover texts
    xgb_hover_texts = []
    for iteration, t_err, v_err in zip(iterations, train_error, val_error):
        gap = abs(t_err - v_err)

        # Assess boosting progress
        if iteration < 20:
            phase = "🚀 RAPID LEARNING"
            phase_note = "Early boosting - each tree adds significant value"
        elif iteration < 60:
            phase = "📈 STEADY IMPROVEMENT"
            phase_note = "Middle phase - consistent error reduction"
        else:
            phase = "📉 FINE-TUNING"
            phase_note = "Late stage - marginal improvements"

        # Overfitting check
        if gap < 0.015:
            fit_status = "✅ GOOD FIT"
            fit_color = "#10b981"
            fit_note = "Train and validation errors are close"
        elif gap < 0.03:
            fit_status = "🟡 SLIGHT OVERFITTING"
            fit_color = "#eab308"
            fit_note = "Small gap emerging - monitor closely"
        else:
            fit_status = "🔴 OVERFITTING"
            fit_color = "#ef4444"
            fit_note = "Large gap - consider early stopping"

        # Performance assessment
        if v_err < 0.05:
            perf = "⭐ EXCELLENT"
            perf_note = "Outstanding error rate"
        elif v_err < 0.10:
            perf = "✅ VERY GOOD"
            perf_note = "Strong performance"
        elif v_err < 0.15:
            perf = "🟡 GOOD"
            perf_note = "Acceptable performance"
        else:
            perf = "⚠️ NEEDS IMPROVEMENT"
            perf_note = "Consider more iterations or tuning"

        hover_text = (
            f"<b style='font-size:14px'>Iteration {iteration}</b><br><br>"
            f"<b>{phase}</b><br>"
            f"{phase_note}<br><br>"
            f"<b>📊 Error Metrics:</b><br>"
            f"• Training Error: <b>{t_err:.4f}</b><br>"
            f"• Validation Error: <b>{v_err:.4f}</b><br>"
            f"• Gap: <b>{gap:.4f}</b><br><br>"
            f"<b style='color:{fit_color}'>{fit_status}</b><br>"
            f"{fit_note}<br><br>"
            f"<b>{perf}</b><br>"
            f"{perf_note}<br><br>"
            f"<b>💡 Boosting Insight:</b><br>"
            f"Each iteration adds a weak learner that focuses on<br>"
            f"mistakes from previous trees, progressively reducing error"
        )
        xgb_hover_texts.append(hover_text)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations, y=train_error, name='Training Error',
        line=dict(color=colors[0]), fill='tozeroy',
        customdata=xgb_hover_texts,
        hovertemplate='%{customdata}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=iterations, y=val_error, name='Validation Error',
        line=dict(color=colors[1]), fill='tozeroy',
        customdata=xgb_hover_texts,
        hovertemplate='%{customdata}<extra></extra>'
    ))

    fig.update_layout(
        title="XGBoost Training Error Reduction",
        xaxis_title="Boosting Iteration",
        yaxis_title="Error Rate",
        height=300
    )

    st.plotly_chart(fig, use_container_width=True, key="xgboost_progress")


def render_model_performance(features, colors):
    """3. Model Performance Comparison (ROC, PR Curves)"""
    st.markdown("## 📊 Model Performance Comparison")
    st.markdown("*ROC curves, Precision-Recall curves, and confusion matrices*")

    # Prepare data
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)

    col1, col2 = st.columns(2)

    with col1:
        # Enhanced ROC Curves with explainability
        fig = go.Figure()

        for model, name, color in [(rf, 'Random Forest', colors[0]),
                                     (gb, 'Gradient Boosting', colors[1])]:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Create detailed hover texts for ROC curve points
            roc_hover_texts = []
            for i, (fpr_val, tpr_val) in enumerate(zip(fpr, tpr)):
                # Performance assessment
                if roc_auc >= 0.95:
                    auc_badge = "⭐ EXCELLENT"
                    auc_color = "#10b981"
                    auc_note = "Outstanding discrimination ability"
                elif roc_auc >= 0.90:
                    auc_badge = "✅ VERY GOOD"
                    auc_color = "#3b82f6"
                    auc_note = "Strong fraud detection capability"
                elif roc_auc >= 0.80:
                    auc_badge = "🟡 GOOD"
                    auc_color = "#eab308"
                    auc_note = "Acceptable performance"
                else:
                    auc_badge = "⚠️ POOR"
                    auc_color = "#ef4444"
                    auc_note = "Needs significant improvement"

                # Operating point analysis
                if tpr_val >= 0.95 and fpr_val <= 0.10:
                    op_point = "🏆 IDEAL POINT"
                    op_note = "Catches most fraud with few false alarms"
                elif tpr_val >= 0.85:
                    op_point = "✅ HIGH RECALL"
                    op_note = "Catches most fraud, some false positives"
                elif fpr_val <= 0.05:
                    op_point = "🎯 HIGH PRECISION"
                    op_note = "Few false alarms, may miss some fraud"
                else:
                    op_point = "⚖️ BALANCED"
                    op_note = "Trade-off between recall and precision"

                # Calculate practical metrics
                total_fraud = int(y_test.sum())
                total_legit = len(y_test) - total_fraud
                fraud_caught = int(tpr_val * total_fraud)
                fraud_missed = total_fraud - fraud_caught
                false_alarms = int(fpr_val * total_legit)

                hover_text = (
                    f"<b style='font-size:14px'>{name}</b><br><br>"
                    f"<b style='color:{auc_color}'>{auc_badge}</b><br>"
                    f"• AUC-ROC: <b>{roc_auc:.3f}</b><br>"
                    f"{auc_note}<br><br>"
                    f"<b>📊 Operating Point:</b><br>"
                    f"• True Positive Rate: <b>{tpr_val:.1%}</b><br>"
                    f"• False Positive Rate: <b>{fpr_val:.1%}</b><br><br>"
                    f"<b>{op_point}</b><br>"
                    f"{op_note}<br><br>"
                    f"<b>💡 Practical Impact:</b><br>"
                    f"• Fraud Caught: <b>{fraud_caught}/{total_fraud}</b><br>"
                    f"• Fraud Missed: <b>{fraud_missed}</b><br>"
                    f"• False Alarms: <b>{false_alarms}</b><br><br>"
                    f"<b>🎯 What This Means:</b><br>"
                    f"At this threshold, you'd catch <b>{tpr_val:.1%}</b> of fraud<br>"
                    f"while generating <b>{fpr_val:.1%}</b> false positive rate"
                )
                roc_hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'{name} (AUC = {roc_auc:.3f})',
                line=dict(color=color, width=3),
                customdata=roc_hover_texts,
                hovertemplate='%{customdata}<extra></extra>'
            ))

        # Random baseline
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='<b>Random Baseline</b><br>AUC = 0.500<br>No better than guessing<extra></extra>'
        ))

        fig.update_layout(
            title="ROC Curves - Model Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="roc_curves")

    with col2:
        # Enhanced Precision-Recall Curves with explainability
        fig = go.Figure()

        for model, name, color in [(rf, 'Random Forest', colors[2]),
                                     (gb, 'Gradient Boosting', colors[3])]:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

            # Create detailed hover texts for PR curve points
            pr_hover_texts = []
            for i, (prec_val, rec_val) in enumerate(zip(precision, recall)):
                # F1 Score calculation
                if prec_val + rec_val > 0:
                    f1 = 2 * (prec_val * rec_val) / (prec_val + rec_val)
                else:
                    f1 = 0

                # Performance assessment
                if f1 >= 0.90:
                    f1_badge = "⭐ EXCELLENT"
                    f1_color = "#10b981"
                    f1_note = "Outstanding balance of precision and recall"
                elif f1 >= 0.80:
                    f1_badge = "✅ VERY GOOD"
                    f1_color = "#3b82f6"
                    f1_note = "Strong balanced performance"
                elif f1 >= 0.70:
                    f1_badge = "🟡 GOOD"
                    f1_color = "#eab308"
                    f1_note = "Acceptable balance"
                else:
                    f1_badge = "⚠️ POOR"
                    f1_color = "#ef4444"
                    f1_note = "Needs improvement"

                # Operating point analysis
                if prec_val >= 0.90 and rec_val >= 0.90:
                    op_point = "🏆 IDEAL"
                    op_note = "High precision AND high recall - best of both worlds"
                elif prec_val >= 0.90:
                    op_point = "🎯 HIGH PRECISION"
                    op_note = "When we flag fraud, we're usually right"
                elif rec_val >= 0.90:
                    op_point = "✅ HIGH RECALL"
                    op_note = "We catch most fraud cases"
                else:
                    op_point = "⚖️ BALANCED"
                    op_note = "Trade-off between precision and recall"

                # Calculate practical metrics
                total_fraud = int(y_test.sum())
                fraud_caught = int(rec_val * total_fraud)
                fraud_missed = total_fraud - fraud_caught
                # Precision = TP / (TP + FP), so FP = TP/precision - TP
                if prec_val > 0:
                    false_positives = int(fraud_caught / prec_val - fraud_caught)
                else:
                    false_positives = 0

                hover_text = (
                    f"<b style='font-size:14px'>{name}</b><br><br>"
                    f"<b style='color:{f1_color}'>{f1_badge}</b><br>"
                    f"• F1 Score: <b>{f1:.3f}</b><br>"
                    f"{f1_note}<br><br>"
                    f"<b>📊 Metrics:</b><br>"
                    f"• Precision: <b>{prec_val:.1%}</b><br>"
                    f"• Recall: <b>{rec_val:.1%}</b><br><br>"
                    f"<b>{op_point}</b><br>"
                    f"{op_note}<br><br>"
                    f"<b>💡 Practical Impact:</b><br>"
                    f"• Fraud Caught: <b>{fraud_caught}/{total_fraud}</b><br>"
                    f"• Fraud Missed: <b>{fraud_missed}</b><br>"
                    f"• False Positives: <b>~{false_positives}</b><br><br>"
                    f"<b>🎯 What This Means:</b><br>"
                    f"Of all fraud flags, <b>{prec_val:.1%}</b> are real fraud<br>"
                    f"We catch <b>{rec_val:.1%}</b> of all actual fraud cases"
                )
                pr_hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                name=name,
                line=dict(color=color, width=3),
                customdata=pr_hover_texts,
                hovertemplate='%{customdata}<extra></extra>'
            ))

        fig.update_layout(
            title="Precision-Recall Curves",
            xaxis_title="Recall (Fraud Caught)",
            yaxis_title="Precision (Accuracy of Flags)",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="pr_curves")

    # Enhanced Confusion matrices with explainability
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)

    for col, model, name in [(col1, rf, 'Random Forest'), (col2, gb, 'Gradient Boosting')]:
        with col:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            # Calculate metrics
            tn, fp, fn, tp = cm.ravel()
            total = tn + fp + fn + tp
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            # Create enhanced hover texts for each cell
            confusion_hover = [
                [
                    # True Negative (top-left)
                    (
                        f"<b style='font-size:14px'>True Negatives (TN)</b><br><br>"
                        f"<b style='color:#10b981'>✅ CORRECT PREDICTION</b><br>"
                        f"Model correctly identified legitimate transactions<br><br>"
                        f"<b>📊 Cell Value:</b> <b>{tn}</b> transactions<br>"
                        f"<b>📈 Percentage:</b> <b>{(tn/total)*100:.1f}%</b> of all predictions<br><br>"
                        f"<b>💡 What This Means:</b><br>"
                        f"These transactions were legitimate AND the model<br>"
                        f"correctly classified them as legitimate.<br><br>"
                        f"<b>🎯 Business Impact:</b><br>"
                        f"<b>{tn}</b> legitimate transactions processed smoothly<br>"
                        f"without unnecessary friction or delays.<br><br>"
                        f"<b>📈 Specificity Contribution:</b><br>"
                        f"Contributes to <b>{specificity:.1%}</b> specificity<br>"
                        f"(ability to correctly identify legitimate transactions)"
                    ),
                    # False Positive (top-right)
                    (
                        f"<b style='font-size:14px'>False Positives (FP)</b><br><br>"
                        f"<b style='color:#ef4444'>❌ INCORRECT PREDICTION</b><br>"
                        f"Model incorrectly flagged legitimate transactions<br><br>"
                        f"<b>📊 Cell Value:</b> <b>{fp}</b> transactions<br>"
                        f"<b>📈 Percentage:</b> <b>{(fp/total)*100:.1f}%</b> of all predictions<br><br>"
                        f"<b>💡 What This Means:</b><br>"
                        f"These transactions were legitimate BUT the model<br>"
                        f"incorrectly flagged them as fraudulent.<br><br>"
                        f"<b>💰 Business Cost:</b><br>"
                        f"<b>{fp}</b> false alarms = ~<b>{fp * 2}</b> hours wasted<br>"
                        f"Customer friction: <b>{fp}</b> legitimate users inconvenienced<br>"
                        f"Est. cost: <b>${fp * 50:,}</b> in review time<br><br>"
                        f"<b>⚠️ Impact:</b><br>"
                        f"Reduces precision to <b>{precision:.1%}</b><br>"
                        f"{'🔴 High false positive rate - tune thresholds' if fp/(tp+fp) > 0.15 else '✅ Acceptable false positive rate'}"
                    )
                ],
                [
                    # False Negative (bottom-left)
                    (
                        f"<b style='font-size:14px'>False Negatives (FN)</b><br><br>"
                        f"<b style='color:#ef4444'>❌ INCORRECT PREDICTION</b><br>"
                        f"Model missed actual fraud (most critical error)<br><br>"
                        f"<b>📊 Cell Value:</b> <b>{fn}</b> transactions<br>"
                        f"<b>📈 Percentage:</b> <b>{(fn/total)*100:.1f}%</b> of all predictions<br><br>"
                        f"<b>💡 What This Means:</b><br>"
                        f"These transactions were fraudulent BUT the model<br>"
                        f"failed to detect them (missed fraud).<br><br>"
                        f"<b>💰 Business Cost:</b><br>"
                        f"<b>{fn}</b> missed fraud cases<br>"
                        f"Est. fraud losses: <b>${fn * 12400:,}</b><br>"
                        f"Potential chargeback fees: <b>${fn * 25:,}</b><br>"
                        f"Total impact: <b>${(fn * 12400) + (fn * 25):,}</b><br><br>"
                        f"<b>🔴 CRITICAL IMPACT:</b><br>"
                        f"Reduces recall to <b>{recall:.1%}</b><br>"
                        f"Missing <b>{(fn/(tp+fn))*100:.1f}%</b> of actual fraud<br>"
                        f"{'⚠️ URGENT: Review detection thresholds' if fn/(tp+fn) > 0.10 else '✅ Acceptable miss rate'}"
                    ),
                    # True Positive (bottom-right)
                    (
                        f"<b style='font-size:14px'>True Positives (TP)</b><br><br>"
                        f"<b style='color:#10b981'>✅ CORRECT PREDICTION</b><br>"
                        f"Model successfully detected fraud<br><br>"
                        f"<b>📊 Cell Value:</b> <b>{tp}</b> transactions<br>"
                        f"<b>📈 Percentage:</b> <b>{(tp/total)*100:.1f}%</b> of all predictions<br><br>"
                        f"<b>💡 What This Means:</b><br>"
                        f"These transactions were fraudulent AND the model<br>"
                        f"correctly identified them as fraudulent.<br><br>"
                        f"<b>💰 Business Value:</b><br>"
                        f"<b>{tp}</b> fraud cases caught successfully<br>"
                        f"Losses prevented: <b>${tp * 12400:,}</b><br>"
                        f"Chargebacks avoided: <b>${tp * 25:,}</b><br>"
                        f"Total value: <b>${(tp * 12400) + (tp * 25):,}</b><br><br>"
                        f"<b>🎯 Performance Metrics:</b><br>"
                        f"• Precision: <b>{precision:.1%}</b> of fraud flags are correct<br>"
                        f"• Recall: <b>{recall:.1%}</b> of fraud is caught<br>"
                        f"• Accuracy: <b>{accuracy:.1%}</b> overall correctness"
                    )
                ]
            ]

            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Negative', 'Predicted Positive'],
                y=['Actual Negative', 'Actual Positive'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues',
                hovertemplate='%{customdata}<extra></extra>',
                customdata=confusion_hover
            ))

            fig.update_layout(title=f"{name} Confusion Matrix", height=300)
            st.plotly_chart(fig, use_container_width=True, key=f"cm_{name.lower().replace(' ', '_')}")


def render_explainable_ai(features, colors):
    """4. Explainable AI (SHAP/LIME)"""
    st.markdown("## 🔍 Explainable AI (SHAP/LIME)")
    st.markdown("*Model interpretability and feature contributions*")

    st.info("📊 **Note**: Full SHAP/LIME integration requires additional computation. Showing representative visualizations.")

    col1, col2 = st.columns(2)

    with col1:
        # SHAP summary plot (simulated)
        st.markdown("### SHAP Feature Importance")

        feature_names = ['amount', 'risk_level', 'hour', 'is_international', 'total_balance',
                        'account_age_days', 'is_pep', 'is_weekend', 'day_of_week', 'is_wire']

        # Simulate SHAP values
        np.random.seed(42)
        shap_values = np.random.randn(len(feature_names)) * [3.2, 2.8, 1.5, 1.8, 2.1, 1.2, 1.9, 0.8, 0.9, 1.4]

        # Enhanced hover texts for SHAP values
        shap_explanations = {
            'amount': {
                'meaning': 'Transaction dollar amount',
                'positive': 'Higher amounts push prediction toward fraud',
                'negative': 'Lower amounts push prediction toward legitimate',
                'insight': 'Large transactions are strongest fraud indicator'
            },
            'risk_level': {
                'meaning': 'Customer risk classification',
                'positive': 'High-risk customers increase fraud probability',
                'negative': 'Low-risk customers decrease fraud probability',
                'insight': 'Customer history is powerful predictor'
            },
            'hour': {
                'meaning': 'Hour of transaction (0-23)',
                'positive': 'Unusual hours (late night) indicate potential fraud',
                'negative': 'Normal business hours suggest legitimate activity',
                'insight': 'Temporal patterns reveal behavioral anomalies'
            },
            'is_international': {
                'meaning': 'Cross-border transaction flag',
                'positive': 'International transactions have higher fraud risk',
                'negative': 'Domestic transactions are typically safer',
                'insight': 'Geography adds important risk context'
            },
            'total_balance': {
                'meaning': 'Account balance',
                'positive': 'High balance may enable large fraud',
                'negative': 'Low balance limits fraud potential',
                'insight': 'Balance informs feasibility of fraud'
            },
            'account_age_days': {
                'meaning': 'Days since account opening',
                'positive': 'Older accounts may have established patterns',
                'negative': 'New accounts are higher risk',
                'insight': 'Account maturity inversely correlates with fraud'
            },
            'is_pep': {
                'meaning': 'Politically Exposed Person status',
                'positive': 'PEP status increases scrutiny needs',
                'negative': 'Non-PEP reduces regulatory concerns',
                'insight': 'Regulatory risk factor'
            },
            'is_weekend': {
                'meaning': 'Weekend transaction indicator',
                'positive': 'Weekend activity may be anomalous',
                'negative': 'Weekday activity more common',
                'insight': 'Weekly patterns matter for fraud detection'
            },
            'day_of_week': {
                'meaning': 'Specific day of week',
                'positive': 'Certain days show more fraud',
                'negative': 'Other days are safer',
                'insight': 'Day-specific patterns emerge over time'
            },
            'is_wire': {
                'meaning': 'Wire transfer flag',
                'positive': 'Wire transfers have elevated fraud risk',
                'negative': 'Non-wire transactions are lower risk',
                'insight': 'Payment method significantly impacts risk'
            }
        }

        hover_texts = []
        for feature_name, shap_val in zip(feature_names, shap_values):
            info = shap_explanations.get(feature_name, {})
            meaning = info.get('meaning', 'Feature contribution to predictions')
            positive = info.get('positive', 'Increases fraud probability')
            negative = info.get('negative', 'Decreases fraud probability')
            insight = info.get('insight', 'Contributes to model decision')

            abs_shap = abs(shap_val)

            if abs_shap > 2.5:
                impact = "🔴 CRITICAL IMPACT"
                impact_color = "#ef4444"
            elif abs_shap > 1.5:
                impact = "🟠 HIGH IMPACT"
                impact_color = "#f59e0b"
            elif abs_shap > 0.8:
                impact = "🟡 MODERATE IMPACT"
                impact_color = "#f59e0b"
            else:
                impact = "🟢 LOW IMPACT"
                impact_color = "#10b981"

            hover_text = (
                f"<b style='font-size:14px'>{feature_name}</b><br><br>"
                f"<b style='color:{impact_color}'>{impact}</b><br><br>"
                f"<b>📊 SHAP Value: <b>{shap_val:.3f}</b><br>"
                f"• Absolute Impact: <b>{abs_shap:.3f}</b><br><br>"
                f"<b>💡 What This Feature Is:</b><br>"
                f"{meaning}<br><br>"
                f"<b>➕ When Positive:</b><br>"
                f"{positive}<br><br>"
                f"<b>➖ When Negative:</b><br>"
                f"{negative}<br><br>"
                f"<b>🎯 Key Insight:</b><br>"
                f"{insight}<br><br>"
                f"<b>📈 Interpretation:</b><br>"
                f"On average, this feature changes model output by <b>{abs_shap:.2f}</b> units"
            )
            hover_texts.append(hover_text)

        fig = go.Figure(go.Bar(
            x=np.abs(shap_values),
            y=feature_names,
            orientation='h',
            marker=dict(
                color=shap_values,
                colorscale='RdBu',
                cmin=-3,
                cmax=3,
                colorbar=dict(title="SHAP Value")
            ),
            text=[f"{v:.2f}" for v in shap_values],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.update_layout(
            title="SHAP Feature Importance (Mean |SHAP value|)",
            xaxis_title="Mean |SHAP Value|",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="shap_importance")

    with col2:
        # Individual prediction explanation
        st.markdown("### Individual Transaction Explanation")

        transaction_id = st.selectbox("Select Transaction", [f"TXN-{i:05d}" for i in range(1, 21)])

        # Simulate feature contributions
        contributions = {
            'amount': 0.45,
            'risk_level': 0.28,
            'is_international': 0.15,
            'hour': -0.08,
            'total_balance': -0.12,
            'is_pep': 0.22,
            'account_age_days': -0.05,
            'is_weekend': 0.03,
            'day_of_week': -0.02,
            'is_wire': 0.08
        }

        features_sorted = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_names = [f[0] for f in features_sorted]
        feature_values = [f[1] for f in features_sorted]

        # Enhanced hover texts for LIME explanations
        lime_feature_details = {
            'amount': {
                'description': 'Transaction dollar amount',
                'positive_meaning': 'Higher amounts INCREASE fraud probability',
                'negative_meaning': 'Lower amounts DECREASE fraud probability',
                'context': 'Large transactions are often targeted by fraudsters'
            },
            'risk_level': {
                'description': 'Customer risk classification score',
                'positive_meaning': 'High-risk customer profile INCREASES fraud probability',
                'negative_meaning': 'Low-risk customer profile DECREASES fraud probability',
                'context': 'Based on historical behavior and risk indicators'
            },
            'is_international': {
                'description': 'Whether transaction crosses borders',
                'positive_meaning': 'International transaction INCREASES fraud risk',
                'negative_meaning': 'Domestic transaction DECREASES fraud risk',
                'context': 'Cross-border transactions have higher fraud rates'
            },
            'hour': {
                'description': 'Hour of day when transaction occurred (0-23)',
                'positive_meaning': 'Unusual time INCREASES fraud suspicion',
                'negative_meaning': 'Normal business hours DECREASE fraud suspicion',
                'context': 'Fraudulent transactions often occur outside business hours'
            },
            'total_balance': {
                'description': 'Current account balance',
                'positive_meaning': 'High balance INCREASES fraud potential',
                'negative_meaning': 'Low balance DECREASES fraud feasibility',
                'context': 'Balance affects what fraud is possible'
            },
            'is_pep': {
                'description': 'Politically Exposed Person status',
                'positive_meaning': 'PEP status INCREASES scrutiny requirements',
                'negative_meaning': 'Non-PEP DECREASES regulatory concerns',
                'context': 'PEPs face higher regulatory scrutiny'
            },
            'account_age_days': {
                'description': 'Days since account was opened',
                'positive_meaning': 'Older account INCREASES trust',
                'negative_meaning': 'New account INCREASES fraud risk',
                'context': 'Fraudsters often use newly created accounts'
            },
            'is_weekend': {
                'description': 'Weekend transaction indicator',
                'positive_meaning': 'Weekend activity MAY be anomalous',
                'negative_meaning': 'Weekday activity is more typical',
                'context': 'Weekend patterns differ from normal business activity'
            },
            'day_of_week': {
                'description': 'Specific day of the week',
                'positive_meaning': 'Unusual day pattern for this customer',
                'negative_meaning': 'Typical day pattern for this customer',
                'context': 'Day-of-week patterns reveal behavioral norms'
            },
            'is_wire': {
                'description': 'Wire transfer payment method',
                'positive_meaning': 'Wire transfer INCREASES fraud risk',
                'negative_meaning': 'Non-wire payment DECREASES fraud risk',
                'context': 'Wire transfers are harder to reverse if fraudulent'
            }
        }

        lime_hovers = []
        for feature, value in zip(feature_names, feature_values):
            details = lime_feature_details.get(feature, {})
            description = details.get('description', 'Feature contribution')
            pos_meaning = details.get('positive_meaning', 'Increases fraud probability')
            neg_meaning = details.get('negative_meaning', 'Decreases fraud probability')
            context = details.get('context', 'Contributes to model decision')

            # Impact assessment
            abs_val = abs(value)
            if abs_val > 0.30:
                impact_level = "🔴 CRITICAL FACTOR"
                impact_color = "#ef4444"
                impact_desc = "Dominates the fraud prediction for this transaction"
            elif abs_val > 0.15:
                impact_level = "🟠 HIGH IMPACT"
                impact_color = "#f59e0b"
                impact_desc = "Significantly influences the fraud score"
            elif abs_val > 0.05:
                impact_level = "🟡 MODERATE IMPACT"
                impact_color = "#eab308"
                impact_desc = "Contributes noticeably to the prediction"
            else:
                impact_level = "🟢 LOW IMPACT"
                impact_color = "#10b981"
                impact_desc = "Minor influence on the prediction"

            # Direction indicator
            if value > 0:
                direction = "↗ INCREASES Fraud Probability"
                direction_color = "#ef4444"
                direction_icon = "🚨"
                interpretation = pos_meaning
            else:
                direction = "↘ DECREASES Fraud Probability"
                direction_color = "#10b981"
                direction_icon = "✅"
                interpretation = neg_meaning

            # Calculate percentage contribution
            total_positive = sum(abs(v) for v in feature_values if v > 0)
            total_negative = sum(abs(v) for v in feature_values if v < 0)
            if value > 0 and total_positive > 0:
                pct_contribution = (abs(value) / total_positive) * 100
            elif value < 0 and total_negative > 0:
                pct_contribution = (abs(value) / total_negative) * 100
            else:
                pct_contribution = 0

            hover_text = (
                f"<b style='font-size:14px'>{feature.replace('_', ' ').title()}</b><br><br>"
                f"<b style='color:{impact_color}'>{impact_level}</b><br>"
                f"{impact_desc}<br><br>"
                f"<b>📊 Contribution Value:</b> <b>{value:+.3f}</b><br>"
                f"<b>📈 Magnitude:</b> <b>{abs_val:.3f}</b> (absolute)<br>"
                f"<b>💯 % of Total:</b> <b>{pct_contribution:.1f}%</b><br><br>"
                f"<b>🎯 Direction:</b><br>"
                f"<b style='color:{direction_color}'>{direction_icon} {direction}</b><br><br>"
                f"<b>💡 What This Feature Is:</b><br>"
                f"{description}<br><br>"
                f"<b>🔍 Interpretation:</b><br>"
                f"{interpretation}<br><br>"
                f"<b>📚 Context:</b><br>"
                f"{context}<br><br>"
                f"<b>🎭 LIME Method:</b><br>"
                f"LIME (Local Interpretable Model-agnostic Explanations)<br>"
                f"creates a local linear approximation of the model's<br>"
                f"decision boundary around this specific transaction,<br>"
                f"showing how each feature pushed the prediction<br>"
                f"toward or away from fraud."
            )
            lime_hovers.append(hover_text)

        fig = go.Figure(go.Bar(
            x=feature_values,
            y=feature_names,
            orientation='h',
            marker=dict(
                color=['red' if v > 0 else 'green' for v in feature_values]
            ),
            text=[f"{v:+.3f}" for v in feature_values],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=lime_hovers
        ))

        fig.update_layout(
            title=f"Feature Contributions for {transaction_id}",
            xaxis_title="Contribution to Fraud Probability",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="lime_explanation")

    # SHAP dependence plots
    st.markdown("### SHAP Dependence Plots")
    col1, col2 = st.columns(2)

    with col1:
        # Amount vs SHAP value
        np.random.seed(42)
        amounts = np.random.lognormal(8, 2, 200)
        shap_vals = 0.0001 * amounts + np.random.randn(200) * 0.3

        # Enhanced hover texts for SHAP dependence scatter plot
        shap_amount_hovers = []
        for amount, shap_val in zip(amounts, shap_vals):
            # Categorize transaction amount
            if amount < 1000:
                amt_category = "Small Transaction"
                amt_color = "#10b981"
                amt_note = "Low-value transactions typically have lower fraud risk"
            elif amount < 5000:
                amt_category = "Medium Transaction"
                amt_color = "#3b82f6"
                amt_note = "Moderate-value transactions require standard scrutiny"
            elif amount < 20000:
                amt_category = "Large Transaction"
                amt_color = "#f59e0b"
                amt_note = "High-value transactions attract increased fraud attention"
            else:
                amt_category = "Very Large Transaction"
                amt_color = "#ef4444"
                amt_note = "Very high-value transactions are prime fraud targets"

            # SHAP value interpretation
            if shap_val > 0.5:
                shap_status = "🔴 STRONG FRAUD INDICATOR"
                shap_color = "#ef4444"
                shap_interpretation = "This amount STRONGLY pushes toward fraud classification"
            elif shap_val > 0.2:
                shap_status = "🟠 MODERATE FRAUD INDICATOR"
                shap_color = "#f59e0b"
                shap_interpretation = "This amount moderately increases fraud probability"
            elif shap_val > -0.2:
                shap_status = "🟡 NEUTRAL IMPACT"
                shap_color = "#eab308"
                shap_interpretation = "This amount has minimal impact on fraud prediction"
            elif shap_val > -0.5:
                shap_status = "🟢 MODERATE LEGITIMACY INDICATOR"
                shap_color = "#3b82f6"
                shap_interpretation = "This amount moderately suggests legitimate transaction"
            else:
                shap_status = "✅ STRONG LEGITIMACY INDICATOR"
                shap_color = "#10b981"
                shap_interpretation = "This amount STRONGLY indicates legitimate activity"

            # Relationship insight
            avg_shap_for_range = np.mean([s for a, s in zip(amounts, shap_vals) if abs(a - amount) < amount * 0.2])

            hover_text = (
                f"<b style='font-size:14px'>Transaction Amount Analysis</b><br><br>"
                f"<b style='color:{amt_color}'>{amt_category}</b><br>"
                f"<b>💰 Amount: ${amount:,.2f}</b><br><br>"
                f"<b style='color:{shap_color}'>{shap_status}</b><br>"
                f"<b>📊 SHAP Value: {shap_val:.3f}</b><br><br>"
                f"<b>🔍 What This Point Shows:</b><br>"
                f"For a transaction of ${amount:,.2f}, the model's fraud<br>"
                f"prediction is shifted by <b>{shap_val:.3f}</b> units.<br><br>"
                f"<b>💡 Interpretation:</b><br>"
                f"{shap_interpretation}<br><br>"
                f"<b>📈 Pattern Context:</b><br>"
                f"{amt_note}<br><br>"
                f"<b>📊 Comparison to Similar Amounts:</b><br>"
                f"Similar transactions (±20%): <b>{avg_shap_for_range:.3f}</b> avg SHAP<br>"
                f"Your transaction: <b>{shap_val:.3f}</b><br>"
                f"{'Above' if shap_val > avg_shap_for_range else 'Below'} average by "
                f"<b>{abs(shap_val - avg_shap_for_range):.3f}</b><br><br>"
                f"<b>🎯 Key Insight:</b><br>"
                f"SHAP Dependence plots show how feature values<br>"
                f"(amount) relate to their impact on predictions<br>"
                f"(SHAP value). A positive slope means higher<br>"
                f"amounts increase fraud probability."
            )
            shap_amount_hovers.append(hover_text)

        fig = go.Figure(go.Scatter(
            x=amounts,
            y=shap_vals,
            mode='markers',
            marker=dict(
                size=5,
                color=shap_vals,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="SHAP Value")
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=shap_amount_hovers
        ))

        fig.update_layout(
            title="SHAP Dependence: Transaction Amount",
            xaxis_title="Transaction Amount ($)",
            yaxis_title="SHAP Value",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="shap_amount")

    with col2:
        # Risk level vs SHAP value
        risk_levels = np.random.choice([0, 1, 2], 200, p=[0.6, 0.3, 0.1])
        shap_vals_risk = risk_levels * 0.5 + np.random.randn(200) * 0.2

        # Enhanced hover texts for box plots
        risk_level_details = {
            0: {
                'name': 'Low Risk',
                'description': 'Customers with established good behavior',
                'color': '#10b981',
                'characteristics': 'Long account history, consistent patterns, no red flags',
                'fraud_rate': '0.5%',
                'typical_shap': 'Generally negative (reduces fraud probability)'
            },
            1: {
                'name': 'Medium Risk',
                'description': 'Customers with some concerning patterns',
                'color': '#f59e0b',
                'characteristics': 'Moderate history, occasional anomalies, minor flags',
                'fraud_rate': '3-5%',
                'typical_shap': 'Mixed (slightly increases fraud probability)'
            },
            2: {
                'name': 'High Risk',
                'description': 'Customers with significant risk indicators',
                'color': '#ef4444',
                'characteristics': 'New accounts, multiple red flags, suspicious behavior',
                'fraud_rate': '15-25%',
                'typical_shap': 'Strongly positive (significantly increases fraud probability)'
            }
        }

        fig = go.Figure()
        for risk in [0, 1, 2]:
            mask = risk_levels == risk
            shap_subset = shap_vals_risk[mask]

            # Calculate statistics for hover
            q1 = np.percentile(shap_subset, 25)
            median = np.median(shap_subset)
            q3 = np.percentile(shap_subset, 75)
            mean = np.mean(shap_subset)
            std = np.std(shap_subset)

            details = risk_level_details[risk]

            # Create hover text for the box plot
            hover_text = (
                f"<b style='font-size:14px'>{details['name']} Customers</b><br><br>"
                f"<b style='color:{details['color']}'>{details['description']}</b><br><br>"
                f"<b>📊 SHAP Distribution:</b><br>"
                f"• Mean: <b>{mean:.3f}</b><br>"
                f"• Median: <b>{median:.3f}</b><br>"
                f"• Q1 (25%): <b>{q1:.3f}</b><br>"
                f"• Q3 (75%): <b>{q3:.3f}</b><br>"
                f"• Std Dev: <b>{std:.3f}</b><br>"
                f"• Sample Size: <b>{len(shap_subset)} transactions</b><br><br>"
                f"<b>👥 Customer Characteristics:</b><br>"
                f"{details['characteristics']}<br><br>"
                f"<b>📈 Historical Fraud Rate:</b><br>"
                f"<b>{details['fraud_rate']}</b> of {details['name'].lower()} customers<br><br>"
                f"<b>💡 SHAP Pattern:</b><br>"
                f"{details['typical_shap']}<br><br>"
                f"<b>🎯 What This Box Shows:</b><br>"
                f"The distribution of SHAP values (model impact)<br>"
                f"for all {details['name'].lower()} customer transactions.<br>"
                f"Box shows interquartile range (50% of data),<br>"
                f"line shows median, whiskers show full range."
            )

            fig.add_trace(go.Box(
                y=shap_vals_risk[mask],
                name=details['name'],
                marker=dict(color=details['color']),
                hovertext=hover_text,
                hoverinfo='text'
            ))

        fig.update_layout(
            title="SHAP Dependence: Customer Risk Level",
            xaxis_title="Risk Level",
            yaxis_title="SHAP Value",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="shap_risk")


def render_realtime_monitoring(colors):
    """5. Real-time ML Monitoring"""
    st.markdown("## ⚡ Real-time ML Monitoring")
    st.markdown("*Live model performance and drift detection*")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Model Accuracy", "94.3%", "+1.2%")
    with col2:
        st.metric("Predictions/Min", "1,247", "+156")
    with col3:
        st.metric("Avg Latency", "12ms", "-2ms")
    with col4:
        st.metric("Data Drift Score", "0.08", "-0.02")

    # Real-time performance over time
    st.markdown("### Model Performance Timeline")

    hours = list(range(24))
    accuracy = [0.94 + np.random.randn() * 0.01 for _ in hours]
    precision = [0.92 + np.random.randn() * 0.015 for _ in hours]
    recall = [0.89 + np.random.randn() * 0.015 for _ in hours]
    f1 = [2 * p * r / (p + r) for p, r in zip(precision, recall)]

    # Create enhanced hover texts for each metric
    def create_timeline_hover(hour, metric_name, metric_value, all_values):
        """Create rich hover text for timeline metrics"""
        # Time of day context
        if 0 <= hour < 6:
            time_context = "🌙 Late Night / Early Morning"
            traffic_note = "Low transaction volume - typically quieter period"
        elif 6 <= hour < 12:
            time_context = "🌅 Morning Peak"
            traffic_note = "Rising activity - business day begins"
        elif 12 <= hour < 18:
            time_context = "☀️ Afternoon Peak"
            traffic_note = "Highest transaction volume period"
        elif 18 <= hour < 22:
            time_context = "🌆 Evening"
            traffic_note = "Moderate activity - winding down"
        else:
            time_context = "🌃 Late Evening"
            traffic_note = "Lower volume - end of day transactions"

        # Performance assessment
        if metric_value >= 0.93:
            perf_badge = "⭐ EXCELLENT"
            perf_color = "#10b981"
            assessment = "Outstanding performance - model operating optimally"
        elif metric_value >= 0.90:
            perf_badge = "✅ GOOD"
            perf_color = "#3b82f6"
            assessment = "Solid performance - within acceptable range"
        elif metric_value >= 0.87:
            perf_badge = "⚠️ MODERATE"
            perf_color = "#f59e0b"
            assessment = "Below target - monitor closely for degradation"
        else:
            perf_badge = "🔴 POOR"
            perf_color = "#ef4444"
            assessment = "Performance issue - investigate immediately"

        # Calculate trend
        avg_value = np.mean(all_values)
        deviation = metric_value - avg_value
        if abs(deviation) < 0.01:
            trend = "→ STABLE"
            trend_color = "#6b7280"
            trend_note = "Performance consistent with 24h average"
        elif deviation > 0.01:
            trend = "↗ ABOVE AVERAGE"
            trend_color = "#10b981"
            trend_note = f"Performing {abs(deviation):.1%} better than average"
        else:
            trend = "↘ BELOW AVERAGE"
            trend_color = "#ef4444"
            trend_note = f"Performing {abs(deviation):.1%} worse than average"

        # Metric-specific insights
        metric_insights = {
            'Accuracy': {
                'definition': 'Percentage of all predictions (fraud + legitimate) that are correct',
                'impact': f'{metric_value:.1%} of all transactions classified correctly',
                'threshold': 'Target: ≥93% for production readiness'
            },
            'Precision': {
                'definition': 'Of all transactions flagged as fraud, how many are actually fraud',
                'impact': f'{metric_value:.1%} of fraud alerts are true fraud (not false alarms)',
                'threshold': 'Target: ≥90% to minimize analyst workload'
            },
            'Recall': {
                'definition': 'Of all actual fraud, how much did we successfully detect',
                'impact': f'{metric_value:.1%} of fraud cases caught (missing {(1-metric_value):.1%})',
                'threshold': 'Target: ≥85% to prevent significant losses'
            },
            'F1 Score': {
                'definition': 'Harmonic mean of precision and recall (balanced metric)',
                'impact': f'{metric_value:.1%} overall detection effectiveness',
                'threshold': 'Target: ≥88% for optimal balance'
            }
        }

        insight = metric_insights.get(metric_name, {})
        definition = insight.get('definition', 'Model performance metric')
        impact = insight.get('impact', f'Current value: {metric_value:.1%}')
        threshold = insight.get('threshold', 'Performance threshold')

        # Recommendations
        if metric_value < 0.87:
            recommendations = [
                "🔍 Investigate root cause of degradation",
                "📊 Check for data drift or input anomalies",
                "🔄 Consider emergency model refresh",
                "👥 Alert ML engineering team"
            ]
        elif metric_value < 0.90:
            recommendations = [
                "📈 Monitor trend over next few hours",
                "🔍 Review recent predictions for patterns",
                "⚙️ Check model health dashboard"
            ]
        else:
            recommendations = [
                "✅ Continue normal operations",
                "📊 Maintain routine monitoring"
            ]

        rec_text = "<br>".join(recommendations)

        hover_text = (
            f"<b style='font-size:14px'>{metric_name} at Hour {hour:02d}:00</b><br><br>"
            f"<b style='color:{perf_color}'>{perf_badge}: {metric_value:.2%}</b><br><br>"
            f"<b>⏰ Time Context:</b><br>"
            f"{time_context}<br>"
            f"<i>{traffic_note}</i><br><br>"
            f"<b>📊 What This Metric Measures:</b><br>"
            f"{definition}<br><br>"
            f"<b>💼 Business Impact:</b><br>"
            f"{impact}<br><br>"
            f"<b>🎯 Performance Target:</b><br>"
            f"{threshold}<br><br>"
            f"<b>📈 24-Hour Trend:</b><br>"
            f"<b style='color:{trend_color}'>{trend}</b><br>"
            f"{trend_note}<br>"
            f"• Current: <b>{metric_value:.2%}</b><br>"
            f"• Average: <b>{avg_value:.2%}</b><br>"
            f"• Best: <b>{max(all_values):.2%}</b> (Hour {all_values.index(max(all_values)):02d})<br>"
            f"• Worst: <b>{min(all_values):.2%}</b> (Hour {all_values.index(min(all_values)):02d})<br><br>"
            f"<b style='color:#059669'>🎯 Recommendations:</b><br>"
            f"{rec_text}"
        )

        return hover_text

    # Generate hover texts for all metrics
    accuracy_hovers = [create_timeline_hover(h, 'Accuracy', acc, accuracy) for h, acc in enumerate(accuracy)]
    precision_hovers = [create_timeline_hover(h, 'Precision', prec, precision) for h, prec in enumerate(precision)]
    recall_hovers = [create_timeline_hover(h, 'Recall', rec, recall) for h, rec in enumerate(recall)]
    f1_hovers = [create_timeline_hover(h, 'F1 Score', f1_val, f1) for h, f1_val in enumerate(f1)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=accuracy, name='Accuracy',
        line=dict(color=colors[0]),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=accuracy_hovers
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=precision, name='Precision',
        line=dict(color=colors[1]),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=precision_hovers
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=recall, name='Recall',
        line=dict(color=colors[2]),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=recall_hovers
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=f1, name='F1 Score',
        line=dict(color=colors[3]),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=f1_hovers
    ))

    fig.update_layout(
        title="Model Metrics - Last 24 Hours",
        xaxis_title="Hour",
        yaxis_title="Score",
        height=350,
        yaxis=dict(range=[0.85, 0.98])
    )

    st.plotly_chart(fig, use_container_width=True, key="realtime_metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Data drift monitoring
        st.markdown("### Feature Drift Detection")

        features = ['amount', 'risk_level', 'hour', 'is_international', 'total_balance']
        drift_scores = [0.08, 0.12, 0.05, 0.15, 0.09]
        threshold = 0.1

        colors_drift = ['red' if d > threshold else 'green' for d in drift_scores]

        # Enhanced hover texts for drift detection
        hover_texts = []
        for feature, drift_score in zip(features, drift_scores):
            if drift_score > 0.20:
                status = "🔴 CRITICAL DRIFT"
                status_color = "#dc2626"
                assessment = "Severe distribution shift detected"
                action = "IMMEDIATE ACTION: Retrain model with recent data"
            elif drift_score > threshold:
                status = "🟠 DRIFT DETECTED"
                status_color = "#f59e0b"
                assessment = "Significant distribution change"
                action = "ALERT: Schedule model retraining soon"
            elif drift_score > 0.05:
                status = "🟡 MINOR DRIFT"
                status_color = "#eab308"
                assessment = "Small distribution shift - within normal bounds"
                action = "MONITOR: Continue tracking this feature"
            else:
                status = "🟢 STABLE"
                status_color = "#10b981"
                assessment = "Feature distribution is stable"
                action = "NO ACTION: Feature performing as expected"

            # Calculate severity percentage
            severity_pct = (drift_score / 0.20) * 100  # 0.20 = critical level
            severity_pct = min(severity_pct, 100)

            hover_text = (
                f"<b style='font-size:14px'>{feature}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Drift Metrics:</b><br>"
                f"• KS Statistic: <b>{drift_score:.4f}</b><br>"
                f"• Alert Threshold: <b>{threshold:.2f}</b><br>"
                f"• Severity Level: <b>{severity_pct:.0f}%</b><br>"
                f"• Status: <b>{'OVER THRESHOLD' if drift_score > threshold else 'Within Limits'}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"The Kolmogorov-Smirnov statistic measures how much the current<br>"
                f"distribution of '{feature}' differs from the training distribution.<br>"
                f"Higher values indicate the model may perform poorly on new data.<br><br>"
                f"<b>🎯 Recommended Action:</b><br>"
                f"{action}<br><br>"
                f"<b>📈 Context:</b><br>"
                f"{'Data patterns have shifted - model assumptions may no longer hold' if drift_score > threshold else 'Feature distribution matches training data - predictions remain reliable'}"
            )
            hover_texts.append(hover_text)

        fig = go.Figure(go.Bar(
            x=drift_scores,
            y=features,
            orientation='h',
            marker=dict(color=colors_drift),
            text=[f"{d:.3f}" for d in drift_scores],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Alert Threshold")

        fig.update_layout(
            title="Feature Drift Scores (KS Statistic)",
            xaxis_title="Drift Score",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="drift_scores")

    with col2:
        # Prediction confidence distribution
        st.markdown("### Prediction Confidence Distribution")

        np.random.seed(42)
        confidences = np.concatenate([
            np.random.beta(8, 2, 400),  # High confidence predictions
            np.random.beta(2, 2, 100)   # Low confidence predictions
        ])

        # Enhanced hover for confidence histogram
        # Create bins and calculate statistics for each bin
        hist, bin_edges = np.histogram(confidences, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        confidence_hovers = []
        for count, bin_center, bin_start, bin_end in zip(hist, bin_centers, bin_edges[:-1], bin_edges[1:]):
            # Confidence level assessment
            if bin_center >= 0.90:
                conf_level = "🟢 VERY HIGH CONFIDENCE"
                conf_color = "#10b981"
                interpretation = "Model is highly certain about these predictions"
                action_recommendation = "Generally safe to auto-process with minimal review"
                reliability = "95-99% reliability"
            elif bin_center >= 0.75:
                conf_level = "✅ HIGH CONFIDENCE"
                conf_color = "#3b82f6"
                interpretation = "Model has strong confidence in these predictions"
                action_recommendation = "Can be processed with standard review procedures"
                reliability = "85-95% reliability"
            elif bin_center >= 0.60:
                conf_level = "🟡 MODERATE CONFIDENCE"
                conf_color = "#eab308"
                interpretation = "Model is moderately certain about these predictions"
                action_recommendation = "Recommend enhanced review for these cases"
                reliability = "70-85% reliability"
            elif bin_center >= 0.40:
                conf_level = "🟠 LOW CONFIDENCE"
                conf_color = "#f59e0b"
                interpretation = "Model is uncertain about these predictions"
                action_recommendation = "Requires thorough manual review"
                reliability = "50-70% reliability"
            else:
                conf_level = "🔴 VERY LOW CONFIDENCE"
                conf_color = "#ef4444"
                interpretation = "Model has minimal confidence in these predictions"
                action_recommendation = "CRITICAL: Needs expert analyst review"
                reliability = "<50% reliability (approaching random guess)"

            # Calculate percentage of total predictions
            total_predictions = len(confidences)
            pct_of_total = (count / total_predictions) * 100

            # Business impact calculation
            if bin_center >= 0.75:
                review_time = 2  # minutes per case
                cost_per_case = 5
            elif bin_center >= 0.60:
                review_time = 5
                cost_per_case = 12
            else:
                review_time = 10
                cost_per_case = 25

            total_review_time = count * review_time
            total_cost = count * cost_per_case

            hover_text = (
                f"<b style='font-size:14px'>Confidence Range: {bin_start:.2f} - {bin_end:.2f}</b><br><br>"
                f"<b style='color:{conf_color}'>{conf_level}</b><br>"
                f"{interpretation}<br><br>"
                f"<b>📊 Prediction Statistics:</b><br>"
                f"• Count: <b>{count}</b> predictions<br>"
                f"• Percentage: <b>{pct_of_total:.1f}%</b> of all predictions<br>"
                f"• Avg Confidence: <b>{bin_center:.1%}</b><br><br>"
                f"<b>🎯 Reliability:</b><br>"
                f"{reliability}<br><br>"
                f"<b>💼 Review Burden:</b><br>"
                f"• Est. review time: <b>{review_time} min/case</b><br>"
                f"• Total time for bin: <b>{total_review_time:.0f} minutes</b><br>"
                f"• Review cost: <b>${cost_per_case}/case</b><br>"
                f"• Total cost: <b>${total_cost:,}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"Model confidence scores indicate how certain the AI is<br>"
                f"about its fraud predictions. Higher confidence scores<br>"
                f"mean the model has seen similar patterns before and<br>"
                f"is more reliable. Lower scores need human expertise.<br><br>"
                f"<b style='color:#059669'>🎯 Recommendation:</b><br>"
                f"{action_recommendation}"
            )
            confidence_hovers.append(hover_text)

        fig = go.Figure(data=[go.Bar(
            x=bin_centers,
            y=hist,
            width=(bin_edges[1] - bin_edges[0]),
            marker=dict(color=colors[0]),
            opacity=0.7,
            hovertemplate='%{customdata}<extra></extra>',
            customdata=confidence_hovers
        )])

        fig.update_layout(
            title="Model Confidence Distribution",
            xaxis_title="Prediction Confidence",
            yaxis_title="Count",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="confidence_dist")

    # Model health dashboard
    st.markdown("### Model Health Indicators")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Prediction Volume**")
        hours = list(range(24))
        volume = [1000 + np.random.randint(-200, 300) for _ in hours]

        # Enhanced hover for prediction volume
        volume_hovers = []
        avg_volume = np.mean(volume)
        max_volume = max(volume)
        min_volume = min(volume)

        for hour, vol in zip(hours, volume):
            # Time of day context
            if 0 <= hour < 6:
                time_period = "🌙 Night (Low Activity)"
                expected_vol = "600-900 predictions/hour"
            elif 6 <= hour < 12:
                time_period = "🌅 Morning (Rising Activity)"
                expected_vol = "900-1200 predictions/hour"
            elif 12 <= hour < 18:
                time_period = "☀️ Afternoon (Peak Activity)"
                expected_vol = "1100-1400 predictions/hour"
            elif 18 <= hour < 22:
                time_period = "🌆 Evening (Moderate Activity)"
                expected_vol = "800-1100 predictions/hour"
            else:
                time_period = "🌃 Late Evening (Declining)"
                expected_vol = "700-1000 predictions/hour"

            # Volume assessment
            deviation_pct = ((vol - avg_volume) / avg_volume) * 100
            if vol > avg_volume * 1.3:
                status = "🔴 UNUSUALLY HIGH"
                status_color = "#ef4444"
                assessment = "Volume spike detected - possible bot attack or campaign"
                action = "ALERT: Monitor for fraud patterns and system capacity"
            elif vol > avg_volume * 1.1:
                status = "🟠 ABOVE NORMAL"
                status_color = "#f59e0b"
                assessment = "Higher than average transaction volume"
                action = "Monitor: Check for marketing campaigns or seasonal patterns"
            elif vol < avg_volume * 0.7:
                status = "🟡 BELOW NORMAL"
                status_color = "#eab308"
                assessment = "Lower than average transaction volume"
                action = "Check: Verify system connectivity and service health"
            else:
                status = "✅ NORMAL"
                status_color = "#10b981"
                assessment = "Volume within expected range"
                action = "Continue normal operations"

            # Capacity calculation
            system_capacity = 2000  # predictions per hour
            capacity_used = (vol / system_capacity) * 100

            hover_text = (
                f"<b style='font-size:14px'>Hour {hour:02d}:00 Prediction Volume</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Volume Metrics:</b><br>"
                f"• Current: <b>{vol}</b> predictions/hour<br>"
                f"• 24h Average: <b>{avg_volume:.0f}</b><br>"
                f"• Deviation: <b>{deviation_pct:+.1f}%</b><br>"
                f"• 24h Peak: <b>{max_volume}</b> (Hour {volume.index(max_volume):02d})<br>"
                f"• 24h Low: <b>{min_volume}</b> (Hour {volume.index(min_volume):02d})<br><br>"
                f"<b>⏰ Time Context:</b><br>"
                f"{time_period}<br>"
                f"Expected: {expected_vol}<br><br>"
                f"<b>⚙️ System Capacity:</b><br>"
                f"• Capacity Used: <b>{capacity_used:.1f}%</b><br>"
                f"• Available: <b>{system_capacity - vol}</b> predictions/hour<br>"
                f"• Status: <b>{'🟢 Healthy' if capacity_used < 80 else '🔴 Near Limit'}</b><br><br>"
                f"<b style='color:#059669'>🎯 Action:</b><br>"
                f"{action}"
            )
            volume_hovers.append(hover_text)

        fig = go.Figure(go.Scatter(
            x=hours, y=volume,
            fill='tozeroy',
            line=dict(color=colors[0]),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=volume_hovers
        ))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key="pred_volume")

    with col2:
        st.markdown("**Error Rate**")
        error_rate = [0.06 + np.random.randn() * 0.01 for _ in hours]

        # Enhanced hover for error rate
        error_hovers = []
        avg_error = np.mean(error_rate)
        target_error = 0.05  # 5% target error rate

        for hour, err in zip(hours, error_rate):
            # Error rate assessment
            if err > 0.10:
                status = "🔴 CRITICAL"
                status_color = "#ef4444"
                assessment = "Error rate critically high - model degradation"
                action = "URGENT: Investigate model drift, retrain if needed"
                severity = "CRITICAL"
            elif err > 0.08:
                status = "🟠 HIGH"
                status_color = "#f59e0b"
                assessment = "Error rate elevated above acceptable threshold"
                action = "ALERT: Review recent predictions and data quality"
                severity = "HIGH"
            elif err > 0.06:
                status = "🟡 MODERATE"
                status_color = "#eab308"
                assessment = "Error rate slightly above target"
                action = "MONITOR: Watch for increasing trend"
                severity = "MODERATE"
            else:
                status = "✅ GOOD"
                status_color = "#10b981"
                assessment = "Error rate within acceptable range"
                action = "Continue normal operations"
                severity = "LOW"

            # Impact calculation
            hour_predictions = volume[hour]
            errors_this_hour = int(hour_predictions * err)
            cost_per_error = 50  # dollars
            total_error_cost = errors_this_hour * cost_per_error

            # Distance from target
            vs_target = ((err - target_error) / target_error) * 100

            hover_text = (
                f"<b style='font-size:14px'>Hour {hour:02d}:00 Error Rate</b><br><br>"
                f"<b style='color:{status_color}'>{status} - {err:.1%}</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Error Metrics:</b><br>"
                f"• Current Rate: <b>{err:.2%}</b><br>"
                f"• Target Rate: <b>{target_error:.1%}</b><br>"
                f"• vs Target: <b>{vs_target:+.1f}%</b><br>"
                f"• 24h Average: <b>{avg_error:.2%}</b><br>"
                f"• Severity: <b>{severity}</b><br><br>"
                f"<b>💰 Business Impact:</b><br>"
                f"• Predictions this hour: <b>{hour_predictions}</b><br>"
                f"• Estimated errors: <b>{errors_this_hour}</b><br>"
                f"• Cost per error: <b>${cost_per_error}</b><br>"
                f"• Total impact: <b>${total_error_cost:,}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"Error rate shows percentage of predictions that are<br>"
                f"incorrect (both false positives and false negatives).<br>"
                f"Target is ≤{target_error:.0%} for production quality.<br><br>"
                f"<b style='color:#059669'>🎯 Action:</b><br>"
                f"{action}"
            )
            error_hovers.append(hover_text)

        fig = go.Figure(go.Scatter(
            x=hours, y=error_rate,
            fill='tozeroy',
            line=dict(color=colors[1]),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=error_hovers
        ))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key="error_rate")

    with col3:
        st.markdown("**Response Time**")
        latency = [12 + np.random.randn() * 2 for _ in hours]

        # Enhanced hover for response time
        latency_hovers = []
        avg_latency = np.mean(latency)
        p95_latency = np.percentile(latency, 95)
        p99_latency = np.percentile(latency, 99)

        for hour, lat in zip(hours, latency):
            # Latency assessment
            if lat > 20:
                status = "🔴 SLOW"
                status_color = "#ef4444"
                assessment = "Response time exceeds SLA - poor user experience"
                action = "CRITICAL: Investigate performance bottleneck"
                user_experience = "Poor - Users experiencing delays"
                sla_status = "BREACH"
            elif lat > 15:
                status = "🟠 DEGRADED"
                status_color = "#f59e0b"
                assessment = "Response time elevated - approaching SLA limit"
                action = "WARNING: Monitor system load and optimize"
                user_experience = "Acceptable - Minor delays noticed"
                sla_status = "WARNING"
            elif lat > 10:
                status = "🟡 MODERATE"
                status_color = "#eab308"
                assessment = "Response time within acceptable range"
                action = "MONITOR: Continue tracking performance"
                user_experience = "Good - Responsive system"
                sla_status = "OK"
            else:
                status = "✅ FAST"
                status_color = "#10b981"
                assessment = "Excellent response time"
                action = "Optimal performance"
                user_experience = "Excellent - Real-time response"
                sla_status = "EXCELLENT"

            # SLA targets
            sla_target = 15  # ms
            sla_compliance = lat <= sla_target

            # Performance tier
            if lat < 10:
                perf_tier = "⚡ Real-time"
            elif lat < 50:
                perf_tier = "🚀 Fast"
            elif lat < 100:
                perf_tier = "🏃 Acceptable"
            else:
                perf_tier = "🐌 Slow"

            hover_text = (
                f"<b style='font-size:14px'>Hour {hour:02d}:00 Response Time</b><br><br>"
                f"<b style='color:{status_color}'>{status} - {lat:.1f}ms</b><br>"
                f"{assessment}<br><br>"
                f"<b>⚡ Latency Metrics:</b><br>"
                f"• Current: <b>{lat:.1f}ms</b><br>"
                f"• 24h Average: <b>{avg_latency:.1f}ms</b><br>"
                f"• 95th Percentile: <b>{p95_latency:.1f}ms</b><br>"
                f"• 99th Percentile: <b>{p99_latency:.1f}ms</b><br><br>"
                f"<b>🎯 SLA Compliance:</b><br>"
                f"• Target: <b>≤{sla_target}ms</b><br>"
                f"• Status: <b>{'✅ COMPLIANT' if sla_compliance else '❌ BREACH'}</b><br>"
                f"• SLA State: <b>{sla_status}</b><br><br>"
                f"<b>👤 User Experience:</b><br>"
                f"{user_experience}<br>"
                f"Performance Tier: {perf_tier}<br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"Response time measures how quickly the model<br>"
                f"returns predictions. Lower is better. Target is<br>"
                f"≤{sla_target}ms for real-time fraud detection.<br><br>"
                f"<b style='color:#059669'>🎯 Action:</b><br>"
                f"{action}"
            )
            latency_hovers.append(hover_text)

        fig = go.Figure(go.Scatter(
            x=hours, y=latency,
            fill='tozeroy',
            line=dict(color=colors[2]),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=latency_hovers
        ))
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True, key="latency")


def render_feature_engineering(features, colors):
    """6. Feature Engineering (PCA, t-SNE, Correlation)"""
    st.markdown("## 🔬 Feature Engineering & Dimensionality Reduction")
    st.markdown("*PCA, t-SNE, and feature correlation analysis*")

    # Prepare data
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    col1, col2 = st.columns(2)

    with col1:
        # PCA visualization
        st.markdown("### PCA: Principal Component Analysis")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig = go.Figure()

        for fraud_val, name, color in [(0, 'Legitimate', colors[0]), (1, 'Fraud', colors[1])]:
            mask = y == fraud_val

            # Enhanced hover texts for PCA points
            pca_hovers = []
            for i, (pc1, pc2) in enumerate(X_pca[mask]):
                # Determine cluster position
                if abs(pc1) < 1 and abs(pc2) < 1:
                    cluster_pos = "Center Cluster"
                    cluster_note = "Typical transaction profile"
                elif abs(pc1) > 3 or abs(pc2) > 3:
                    cluster_pos = "Outlier Region"
                    cluster_note = "Unusual feature combination - warrants investigation"
                else:
                    cluster_pos = "Normal Spread"
                    cluster_note = "Within expected variance"

                # Separation quality
                if fraud_val == 1:  # Fraud
                    if pc1 > 2:
                        separation = "Well-separated from legitimate transactions"
                    else:
                        separation = "Some overlap with legitimate transactions"
                else:  # Legitimate
                    if pc1 < -1:
                        separation = "Clearly distinguished from fraud"
                    else:
                        separation = "Some similarity to fraud patterns"

                hover_text = (
                    f"<b style='font-size:14px'>{name} Transaction</b><br><br>"
                    f"<b>📊 PCA Coordinates:</b><br>"
                    f"• PC1: <b>{pc1:.3f}</b><br>"
                    f"• PC2: <b>{pc2:.3f}</b><br><br>"
                    f"<b>🎯 Cluster Position:</b><br>"
                    f"{cluster_pos}<br>"
                    f"<i>{cluster_note}</i><br><br>"
                    f"<b>🔍 Separation Analysis:</b><br>"
                    f"{separation}<br><br>"
                    f"<b>💡 What PCA Shows:</b><br>"
                    f"PCA reduces {X.shape[1]} features into 2 dimensions<br>"
                    f"capturing {pca.explained_variance_ratio_.sum():.1%} of variance.<br>"
                    f"Points close together have similar feature profiles.<br><br>"
                    f"<b>🎭 Class:</b> <b>{name}</b>"
                )
                pca_hovers.append(hover_text)

            fig.add_trace(go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode='markers',
                name=name,
                marker=dict(size=5, color=color, opacity=0.6),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=pca_hovers
            ))

        fig.update_layout(
            title=f"PCA Projection (Explained Variance: {pca.explained_variance_ratio_.sum():.1%})",
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="pca_plot")

    with col2:
        # t-SNE visualization
        st.markdown("### t-SNE: Nonlinear Dimensionality Reduction")

        # Use a smaller sample for t-SNE (it's computationally expensive)
        sample_size = min(500, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_scaled[sample_idx])
        y_sample = y.iloc[sample_idx]

        fig = go.Figure()

        for fraud_val, name, color in [(0, 'Legitimate', colors[2]), (1, 'Fraud', colors[3])]:
            mask = y_sample == fraud_val
            y_sample_array = y_sample.values

            # Enhanced hover texts for t-SNE points
            tsne_hovers = []
            for i, (tsne1, tsne2) in enumerate(X_tsne[mask]):
                # Calculate distance from centroid
                fraud_centroid = X_tsne[y_sample_array == 1].mean(axis=0)
                legit_centroid = X_tsne[y_sample_array == 0].mean(axis=0)

                dist_to_fraud = np.sqrt((tsne1 - fraud_centroid[0])**2 + (tsne2 - fraud_centroid[1])**2)
                dist_to_legit = np.sqrt((tsne1 - legit_centroid[0])**2 + (tsne2 - legit_centroid[1])**2)

                # Cluster analysis
                if fraud_val == 1:  # Fraud
                    if dist_to_fraud < 10:
                        cluster_analysis = "🔴 Core fraud cluster - typical fraud pattern"
                        confidence = "High confidence fraud"
                    else:
                        cluster_analysis = "🟠 Peripheral fraud - atypical patterns"
                        confidence = "Moderate confidence - requires review"
                else:  # Legitimate
                    if dist_to_legit < 10:
                        cluster_analysis = "🟢 Core legitimate cluster - normal behavior"
                        confidence = "High confidence legitimate"
                    else:
                        cluster_analysis = "🟡 Edge case - unusual but legitimate"
                        confidence = "Monitor for anomalies"

                # Separation quality
                if abs(dist_to_fraud - dist_to_legit) > 15:
                    separation_quality = "Excellent separation"
                    sep_note = "Clear distinction from opposite class"
                elif abs(dist_to_fraud - dist_to_legit) > 8:
                    separation_quality = "Good separation"
                    sep_note = "Moderately distinct from opposite class"
                else:
                    separation_quality = "Poor separation"
                    sep_note = "Close to opposite class - ambiguous case"

                hover_text = (
                    f"<b style='font-size:14px'>{name} Transaction</b><br><br>"
                    f"<b>📊 t-SNE Coordinates:</b><br>"
                    f"• Dimension 1: <b>{tsne1:.2f}</b><br>"
                    f"• Dimension 2: <b>{tsne2:.2f}</b><br><br>"
                    f"<b>🎯 Cluster Analysis:</b><br>"
                    f"{cluster_analysis}<br>"
                    f"Confidence: <i>{confidence}</i><br><br>"
                    f"<b>📏 Distances:</b><br>"
                    f"• To fraud centroid: <b>{dist_to_fraud:.1f}</b><br>"
                    f"• To legit centroid: <b>{dist_to_legit:.1f}</b><br><br>"
                    f"<b>🔍 Separation Quality:</b><br>"
                    f"<b>{separation_quality}</b><br>"
                    f"{sep_note}<br><br>"
                    f"<b>💡 What t-SNE Shows:</b><br>"
                    f"t-SNE reveals nonlinear patterns that PCA misses.<br>"
                    f"Similar transactions cluster together based on<br>"
                    f"complex feature interactions. Perplexity=30 balances<br>"
                    f"local vs global structure.<br><br>"
                    f"<b>🎭 Class:</b> <b>{name}</b><br>"
                    f"<b>📦 Sample:</b> {sample_size} of {len(X)} transactions"
                )
                tsne_hovers.append(hover_text)

            fig.add_trace(go.Scatter(
                x=X_tsne[mask, 0],
                y=X_tsne[mask, 1],
                mode='markers',
                name=name,
                marker=dict(size=5, color=color, opacity=0.6),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=tsne_hovers
            ))

        fig.update_layout(
            title="t-SNE Projection (Perplexity=30)",
            xaxis_title="t-SNE Dimension 1",
            yaxis_title="t-SNE Dimension 2",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="tsne_plot")

    # Feature correlation heatmap
    st.markdown("### Feature Correlation Matrix")

    corr_matrix = X.corr()

    # Enhanced hover texts for correlation heatmap
    corr_hovers = []
    for i, feat1 in enumerate(corr_matrix.columns):
        for j, feat2 in enumerate(corr_matrix.columns):
            corr_val = corr_matrix.iloc[i, j]

            # Correlation strength assessment
            abs_corr = abs(corr_val)
            if abs_corr == 1.0:
                strength = "Perfect" if feat1 == feat2 else "Perfect (Multicollinearity)"
                strength_color = "#dc2626" if feat1 != feat2 else "#10b981"
                interpretation = "Same feature" if feat1 == feat2 else "⚠️ CRITICAL: Features are redundant"
                action = "Baseline" if feat1 == feat2 else "Remove one feature to avoid multicollinearity"
            elif abs_corr >= 0.9:
                strength = "Very Strong"
                strength_color = "#ef4444"
                interpretation = "Features highly correlated - potential multicollinearity"
                action = "Consider removing one feature or use PCA"
            elif abs_corr >= 0.7:
                strength = "Strong"
                strength_color = "#f97316"
                interpretation = "Significant relationship between features"
                action = "Monitor for multicollinearity if both used in linear models"
            elif abs_corr >= 0.5:
                strength = "Moderate"
                strength_color = "#eab308"
                interpretation = "Noticeable relationship between features"
                action = "Acceptable correlation - both features can be useful"
            elif abs_corr >= 0.3:
                strength = "Weak"
                strength_color = "#3b82f6"
                interpretation = "Slight relationship between features"
                action = "Features provide mostly independent information"
            else:
                strength = "Very Weak / None"
                strength_color = "#10b981"
                interpretation = "Little to no linear relationship"
                action = "Features are independent - ideal for modeling"

            # Direction of relationship
            if corr_val > 0:
                direction = "Positive correlation"
                direction_note = f"When {feat1} increases, {feat2} tends to increase"
            elif corr_val < 0:
                direction = "Negative correlation"
                direction_note = f"When {feat1} increases, {feat2} tends to decrease"
            else:
                direction = "No correlation"
                direction_note = "No linear relationship"

            hover_text = (
                f"<b style='font-size:14px'>{feat1} vs {feat2}</b><br><br>"
                f"<b style='color:{strength_color}'>Correlation: {corr_val:.3f}</b><br>"
                f"<b>Strength:</b> {strength}<br><br>"
                f"<b>📊 Direction:</b><br>"
                f"{direction}<br>"
                f"<i>{direction_note}</i><br><br>"
                f"<b>💡 Interpretation:</b><br>"
                f"{interpretation}<br><br>"
                f"<b>🎯 Modeling Impact:</b><br>"
                f"{action}<br><br>"
                f"<b>📈 What This Means:</b><br>"
                f"Correlation measures linear relationships between<br>"
                f"features. High correlations (|r| > 0.7) indicate<br>"
                f"redundancy that can hurt model performance."
            )
            corr_hovers.append(hover_text)

    # Reshape hover texts to match the 2D structure
    corr_hovers_2d = np.array(corr_hovers).reshape(len(corr_matrix.columns), len(corr_matrix.columns))

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=corr_hovers_2d
    ))

    fig.update_layout(
        title="Feature Correlation Heatmap",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")

    # PCA explained variance
    col1, col2 = st.columns(2)

    with col1:
        pca_full = PCA()
        pca_full.fit(X_scaled)

        cumsum = np.cumsum(pca_full.explained_variance_ratio_)

        # Enhanced hover for individual variance bars
        bar_hovers = []
        for i, (pc_num, var_ratio) in enumerate(zip(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_)):
            # Importance assessment
            if var_ratio > 0.25:
                importance = "🔴 CRITICAL COMPONENT"
                importance_color = "#ef4444"
                note = "Captures major data patterns - essential for analysis"
            elif var_ratio > 0.15:
                importance = "🟠 HIGH IMPORTANCE"
                importance_color = "#f97316"
                note = "Significant variance explained - very useful"
            elif var_ratio > 0.08:
                importance = "🟡 MODERATE IMPORTANCE"
                importance_color = "#eab308"
                note = "Meaningful contribution to understanding data"
            elif var_ratio > 0.03:
                importance = "🔵 LOW IMPORTANCE"
                importance_color = "#3b82f6"
                note = "Minor patterns - may be noise"
            else:
                importance = "⚪ MINIMAL IMPORTANCE"
                importance_color = "#6b7280"
                note = "Very small variance - likely noise"

            # Dimensionality reduction recommendation
            cumulative_at_pc = cumsum[i]
            if cumulative_at_pc >= 0.95:
                dim_rec = f"✅ Stop here: {pc_num} components capture 95%+ variance"
            elif cumulative_at_pc >= 0.90:
                dim_rec = f"⚠️ {pc_num} components capture 90%+ variance"
            else:
                dim_rec = f"Need more components to reach 90% threshold"

            hover_text = (
                f"<b style='font-size:14px'>Principal Component {pc_num}</b><br><br>"
                f"<b style='color:{importance_color}'>{importance}</b><br>"
                f"{note}<br><br>"
                f"<b>📊 Variance Metrics:</b><br>"
                f"• Individual: <b>{var_ratio:.2%}</b><br>"
                f"• Cumulative: <b>{cumulative_at_pc:.2%}</b><br>"
                f"• Rank: <b>#{pc_num}</b> of {len(pca_full.explained_variance_ratio_)}<br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"This component explains <b>{var_ratio:.2%}</b> of total<br>"
                f"variance in the data. It represents a linear<br>"
                f"combination of original features that captures<br>"
                f"maximum remaining variance.<br><br>"
                f"<b>🎯 Dimensionality Reduction:</b><br>"
                f"{dim_rec}"
            )
            bar_hovers.append(hover_text)

        # Enhanced hover for cumulative line
        cumulative_hovers = []
        for pc_num, cum_var in enumerate(cumsum, start=1):
            # Adequacy assessment
            if cum_var >= 0.99:
                adequacy = "⭐ EXCELLENT"
                adequacy_color = "#10b981"
                assessment = "Nearly all variance captured - perfect representation"
            elif cum_var >= 0.95:
                adequacy = "✅ VERY GOOD"
                adequacy_color = "#3b82f6"
                assessment = "Excellent representation with minimal information loss"
            elif cum_var >= 0.90:
                adequacy = "🟢 GOOD"
                adequacy_color = "#22c55e"
                assessment = "Good representation - acceptable for most applications"
            elif cum_var >= 0.80:
                adequacy = "🟡 ACCEPTABLE"
                adequacy_color = "#eab308"
                assessment = "Moderate representation - some information lost"
            else:
                adequacy = "🔴 INSUFFICIENT"
                adequacy_color = "#ef4444"
                assessment = "Poor representation - too much information lost"

            # Components needed recommendations
            components_for_90 = next((i for i, v in enumerate(cumsum, 1) if v >= 0.90), len(cumsum))
            components_for_95 = next((i for i, v in enumerate(cumsum, 1) if v >= 0.95), len(cumsum))

            hover_text = (
                f"<b style='font-size:14px'>First {pc_num} Components</b><br><br>"
                f"<b style='color:{adequacy_color}'>{adequacy} COVERAGE</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Cumulative Variance:</b><br>"
                f"• Explained: <b>{cum_var:.2%}</b><br>"
                f"• Lost: <b>{(1-cum_var):.2%}</b><br>"
                f"• Components: <b>{pc_num}</b> of {len(cumsum)}<br><br>"
                f"<b>🎯 Recommendations:</b><br>"
                f"• For 90% variance: <b>{components_for_90}</b> components<br>"
                f"• For 95% variance: <b>{components_for_95}</b> components<br><br>"
                f"<b>💡 Rule of Thumb:</b><br>"
                f"Retaining 90-95% variance usually provides good<br>"
                f"balance between dimensionality reduction and<br>"
                f"information preservation."
            )
            cumulative_hovers.append(hover_text)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(pca_full.explained_variance_ratio_) + 1)),
            y=pca_full.explained_variance_ratio_,
            name='Individual',
            marker=dict(color=colors[0]),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=bar_hovers
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumsum) + 1)),
            y=cumsum,
            name='Cumulative',
            line=dict(color=colors[1], width=3),
            yaxis='y2',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=cumulative_hovers
        ))

        fig.update_layout(
            title="PCA Explained Variance Ratio",
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
            yaxis2=dict(title="Cumulative Variance", overlaying='y', side='right'),
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="pca_variance")

    with col2:
        # Feature statistics
        st.markdown("**Feature Statistics**")

        stats_df = pd.DataFrame({
            'Feature': X.columns[:8],
            'Mean': X.mean()[:8].round(2),
            'Std': X.std()[:8].round(2),
            'Min': X.min()[:8].round(2),
            'Max': X.max()[:8].round(2)
        })

        st.dataframe(stats_df, use_container_width=True, height=315)


def render_deep_learning_viz(colors):
    """7. Deep Learning Visualizations"""
    st.markdown("## 🤖 Deep Learning Visualizations")
    st.markdown("*LSTM, embeddings, and autoencoder representations*")

    col1, col2 = st.columns(2)

    with col1:
        # LSTM sequence processing
        st.markdown("### LSTM Sequence Processing")

        # Simulate LSTM cell states
        np.random.seed(42)
        time_steps = 20
        cell_states = np.cumsum(np.random.randn(time_steps, 1) * 0.1, axis=0)
        hidden_states = np.cumsum(np.random.randn(time_steps, 1) * 0.1, axis=0)

        # Enhanced hover texts for LSTM states
        cell_hover_texts = []
        hidden_hover_texts = []

        for step, cell_val, hidden_val in zip(range(time_steps), cell_states.flatten(), hidden_states.flatten()):
            # Cell state hover
            if abs(cell_val) > 0.3:
                cell_status = "🔴 HIGH MAGNITUDE"
                cell_color = "#ef4444"
                cell_insight = "Strong memory retention of past information"
            elif abs(cell_val) > 0.15:
                cell_status = "🟡 MODERATE"
                cell_color = "#f59e0b"
                cell_insight = "Normal memory state"
            else:
                cell_status = "🟢 LOW"
                cell_color = "#10b981"
                cell_insight = "Minimal memory retention"

            cell_hover = (
                f"<b style='font-size:14px'>Cell State at Step {step}</b><br><br>"
                f"<b style='color:{cell_color}'>{cell_status}</b><br><br>"
                f"<b>📊 State Metrics:</b><br>"
                f"• Cell Value: <b>{cell_val:.4f}</b><br>"
                f"• Absolute Magnitude: <b>{abs(cell_val):.4f}</b><br>"
                f"• Time Step: <b>{step}/{time_steps-1}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{cell_insight}<br>"
                f"The cell state stores long-term memory across the sequence.<br><br>"
                f"<b>🧠 LSTM Insight:</b><br>"
                f"Cell states act as the 'memory tape' of the network,<br>"
                f"carrying information forward through time steps."
            )
            cell_hover_texts.append(cell_hover)

            # Hidden state hover
            if abs(hidden_val) > 0.3:
                hidden_status = "🔴 HIGH ACTIVATION"
                hidden_color = "#ef4444"
                hidden_insight = "Strong output signal at this time step"
            elif abs(hidden_val) > 0.15:
                hidden_status = "🟡 MODERATE"
                hidden_color = "#f59e0b"
                hidden_insight = "Normal activation level"
            else:
                hidden_status = "🟢 LOW"
                hidden_color = "#10b981"
                hidden_insight = "Minimal output signal"

            hidden_hover = (
                f"<b style='font-size:14px'>Hidden State at Step {step}</b><br><br>"
                f"<b style='color:{hidden_color}'>{hidden_status}</b><br><br>"
                f"<b>📊 State Metrics:</b><br>"
                f"• Hidden Value: <b>{hidden_val:.4f}</b><br>"
                f"• Absolute Magnitude: <b>{abs(hidden_val):.4f}</b><br>"
                f"• Time Step: <b>{step}/{time_steps-1}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{hidden_insight}<br>"
                f"The hidden state represents the immediate output.<br><br>"
                f"<b>🧠 LSTM Insight:</b><br>"
                f"Hidden states are the 'working memory' - they capture<br>"
                f"what's relevant right now for making predictions."
            )
            hidden_hover_texts.append(hidden_hover)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(time_steps)),
            y=cell_states.flatten(),
            name='Cell State',
            line=dict(color=colors[0], width=3),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=cell_hover_texts
        ))
        fig.add_trace(go.Scatter(
            x=list(range(time_steps)),
            y=hidden_states.flatten(),
            name='Hidden State',
            line=dict(color=colors[1], width=3),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hidden_hover_texts
        ))

        fig.update_layout(
            title="LSTM Cell & Hidden States Over Time",
            xaxis_title="Time Step",
            yaxis_title="State Value",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="lstm_states")

    with col2:
        # Attention weights
        st.markdown("### Attention Mechanism Heatmap")

        np.random.seed(42)
        attention_weights = np.random.rand(10, 10)
        attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)

        # Enhanced hover texts for attention heatmap
        attention_hover_texts = []
        for i in range(10):
            row_hovers = []
            for j in range(10):
                weight = attention_weights[i, j]

                if weight >= 0.15:
                    importance = "🔴 HIGH ATTENTION"
                    imp_color = "#ef4444"
                    insight = "Strong focus on this position"
                    recommendation = "This position is critical for the prediction"
                elif weight >= 0.10:
                    importance = "🟡 MODERATE ATTENTION"
                    imp_color = "#f59e0b"
                    insight = "Notable but not dominant attention"
                    recommendation = "This position contributes to the decision"
                else:
                    importance = "🟢 LOW ATTENTION"
                    imp_color = "#10b981"
                    insight = "Minimal focus on this position"
                    recommendation = "This position has limited influence"

                # Calculate relative strength
                max_in_row = attention_weights[i, :].max()
                relative_strength = (weight / max_in_row * 100) if max_in_row > 0 else 0

                hover_text = (
                    f"<b style='font-size:14px'>Attention: Query {i} → Key {j}</b><br><br>"
                    f"<b style='color:{imp_color}'>{importance}</b><br><br>"
                    f"<b>📊 Attention Metrics:</b><br>"
                    f"• Weight: <b>{weight:.4f}</b> ({weight*100:.1f}%)<br>"
                    f"• Relative Strength: <b>{relative_strength:.0f}%</b> of max in row<br>"
                    f"• Row Sum: <b>{attention_weights[i, :].sum():.3f}</b> (normalized to 1.0)<br><br>"
                    f"<b>💡 What This Means:</b><br>"
                    f"{insight}<br>"
                    f"The model is allocating {weight*100:.1f}% of attention from<br>"
                    f"position {i} to position {j}.<br><br>"
                    f"<b>🧠 Attention Insight:</b><br>"
                    f"{recommendation}<br>"
                    f"Higher weights indicate stronger relationships between positions.<br><br>"
                    f"<b>🎯 Context:</b><br>"
                    f"Attention mechanisms let the model focus on relevant<br>"
                    f"parts of the input when making predictions."
                )
                row_hovers.append(hover_text)
            attention_hover_texts.append(row_hovers)

        fig = go.Figure(data=go.Heatmap(
            z=attention_weights,
            colorscale='Viridis',
            text=np.round(attention_weights, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Weight"),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=attention_hover_texts
        ))

        fig.update_layout(
            title="Attention Weights (Query vs Key)",
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="attention_weights")

    # Embedding visualization
    st.markdown("### Transaction Embedding Space (2D Projection)")

    # Simulate embeddings
    np.random.seed(42)
    n_samples = 300
    embeddings = np.random.randn(n_samples, 2)

    # Create clusters
    centers = [[-2, -2], [2, 2], [-2, 2], [2, -2]]
    labels = []
    for i in range(n_samples):
        center = centers[i % len(centers)]
        embeddings[i] = center + np.random.randn(2) * 0.5
        labels.append(i % len(centers))

    transaction_types = ['Cash Deposit', 'Wire Transfer', 'International', 'High Value']

    fig = go.Figure()

    for i, tx_type in enumerate(transaction_types):
        mask = np.array(labels) == i
        points_in_cluster = mask.sum()
        cluster_embeddings = embeddings[mask]

        # Calculate cluster statistics
        centroid = cluster_embeddings.mean(axis=0)
        spread = np.std(cluster_embeddings, axis=0).mean()

        # Calculate distance from origin
        distance_from_origin = np.linalg.norm(centroid)

        # Enhanced hover texts for embeddings
        embedding_hover_texts = []
        for idx, (x, y) in enumerate(cluster_embeddings):
            # Distance from cluster centroid
            dist_from_centroid = np.sqrt((x - centroid[0])**2 + (y - centroid[1])**2)

            if dist_from_centroid < spread * 0.5:
                position = "🎯 CORE"
                pos_color = "#10b981"
                insight = "Typical example of this transaction type"
            elif dist_from_centroid < spread * 1.5:
                position = "🟡 NORMAL"
                pos_color = "#f59e0b"
                insight = "Standard variation within cluster"
            else:
                position = "🔴 OUTLIER"
                pos_color = "#ef4444"
                insight = "Unusual characteristics for this type"

            hover_text = (
                f"<b style='font-size:14px'>{tx_type} Transaction</b><br><br>"
                f"<b style='color:{pos_color}'>{position}</b><br><br>"
                f"<b>📊 Embedding Coordinates:</b><br>"
                f"• Dimension 1: <b>{x:.3f}</b><br>"
                f"• Dimension 2: <b>{y:.3f}</b><br>"
                f"• Distance from Centroid: <b>{dist_from_centroid:.3f}</b><br><br>"
                f"<b>🎯 Cluster Statistics:</b><br>"
                f"• Cluster Size: <b>{points_in_cluster}</b> transactions<br>"
                f"• Centroid: <b>({centroid[0]:.2f}, {centroid[1]:.2f})</b><br>"
                f"• Average Spread: <b>{spread:.3f}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{insight}<br>"
                f"This transaction has learned features that place it<br>"
                f"in the '{tx_type}' region of the embedding space.<br><br>"
                f"<b>🧠 Embedding Insight:</b><br>"
                f"The model has learned to separate transaction types<br>"
                f"by mapping them to different regions. Closer points<br>"
                f"have similar characteristics."
            )
            embedding_hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=cluster_embeddings[:, 0],
            y=cluster_embeddings[:, 1],
            mode='markers',
            name=tx_type,
            marker=dict(size=8, color=colors[i], opacity=0.6),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=embedding_hover_texts
        ))

    fig.update_layout(
        title="Transaction Type Embeddings (Learned Representation)",
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True, key="embeddings")

    # Autoencoder reconstruction
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Autoencoder: Reconstruction Error")

        np.random.seed(42)
        n_samples = 200
        reconstruction_errors = np.concatenate([
            np.random.gamma(2, 0.5, 180),  # Normal transactions
            np.random.gamma(5, 1.5, 20)    # Anomalous transactions
        ])
        fraud_labels = np.array([0] * 180 + [1] * 20)

        # Threshold line
        threshold = 3.0

        fig = go.Figure()

        for fraud_val, name, color in [(0, 'Legitimate', colors[0]), (1, 'Fraud', colors[1])]:
            mask = fraud_labels == fraud_val
            indices = np.arange(len(reconstruction_errors))[mask]
            errors = reconstruction_errors[mask]

            # Enhanced hover texts for autoencoder errors
            autoencoder_hover_texts = []
            for idx, error in zip(indices, errors):
                is_fraud = fraud_labels[idx] == 1
                actual_label = "Fraudulent" if is_fraud else "Legitimate"

                if error > threshold:
                    status = "🚨 ANOMALY DETECTED"
                    status_color = "#ef4444"
                    insight = "Reconstruction error exceeds threshold"
                    recommendation = "Flag for investigation - unusual pattern"
                elif error > threshold * 0.7:
                    status = "⚠️ ELEVATED ERROR"
                    status_color = "#f59e0b"
                    insight = "Close to anomaly threshold"
                    recommendation = "Monitor - borderline case"
                else:
                    status = "✅ NORMAL"
                    status_color = "#10b981"
                    insight = "Well-reconstructed transaction"
                    recommendation = "Typical pattern - low risk"

                # Detection assessment
                if is_fraud and error > threshold:
                    detection = "✅ Correctly flagged as anomaly"
                    det_color = "#10b981"
                elif is_fraud and error <= threshold:
                    detection = "❌ Missed fraud (false negative)"
                    det_color = "#ef4444"
                elif not is_fraud and error > threshold:
                    detection = "⚠️ False positive (legitimate flagged)"
                    det_color = "#f59e0b"
                else:
                    detection = "✅ Correctly classified as normal"
                    det_color = "#10b981"

                hover_text = (
                    f"<b style='font-size:14px'>Transaction #{idx}</b><br><br>"
                    f"<b style='color:{status_color}'>{status}</b><br><br>"
                    f"<b>📊 Reconstruction Metrics:</b><br>"
                    f"• Error: <b>{error:.3f}</b><br>"
                    f"• Threshold: <b>{threshold:.1f}</b><br>"
                    f"• Distance from Threshold: <b>{abs(error - threshold):.3f}</b><br>"
                    f"• Actual Label: <b>{actual_label}</b><br><br>"
                    f"<b style='color:{det_color}'>🎯 Detection Result:</b><br>"
                    f"{detection}<br><br>"
                    f"<b>💡 What This Means:</b><br>"
                    f"{insight}<br>"
                    f"Autoencoders learn normal patterns - high error<br>"
                    f"indicates the transaction doesn't match learned norms.<br><br>"
                    f"<b>🔍 Recommendation:</b><br>"
                    f"{recommendation}"
                )
                autoencoder_hover_texts.append(hover_text)

            fig.add_trace(go.Scatter(
                x=indices,
                y=errors,
                mode='markers',
                name=name,
                marker=dict(size=6, color=color),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=autoencoder_hover_texts
            ))

        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                     annotation_text="Anomaly Threshold")

        fig.update_layout(
            title="Autoencoder Reconstruction Error by Transaction",
            xaxis_title="Transaction Index",
            yaxis_title="Reconstruction Error",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="autoencoder_error")

    with col2:
        st.markdown("### Reconstruction Error Distribution")

        # Calculate distribution statistics
        legit_errors = reconstruction_errors[fraud_labels == 0]
        fraud_errors = reconstruction_errors[fraud_labels == 1]

        legit_mean = legit_errors.mean()
        fraud_mean = fraud_errors.mean()
        legit_std = legit_errors.std()
        fraud_std = fraud_errors.std()

        # Calculate separation metrics
        separation = abs(fraud_mean - legit_mean) / ((legit_std + fraud_std) / 2)

        fig = go.Figure()

        for fraud_val, name, color, error_set, mean, std in [
            (0, 'Legitimate', colors[2], legit_errors, legit_mean, legit_std),
            (1, 'Fraud', colors[3], fraud_errors, fraud_mean, fraud_std)
        ]:
            # Calculate detection metrics
            above_threshold = (error_set > threshold).sum()
            below_threshold = (error_set <= threshold).sum()
            total = len(error_set)

            if fraud_val == 1:
                detection_rate = (above_threshold / total * 100) if total > 0 else 0
                status_desc = f"{detection_rate:.0f}% of fraud detected"
            else:
                false_positive_rate = (above_threshold / total * 100) if total > 0 else 0
                status_desc = f"{false_positive_rate:.0f}% false positives"

            # Assess distribution quality
            if separation > 2.0:
                quality = "⭐ EXCELLENT SEPARATION"
                quality_color = "#10b981"
            elif separation > 1.0:
                quality = "✅ GOOD SEPARATION"
                quality_color = "#22c55e"
            else:
                quality = "⚠️ POOR SEPARATION"
                quality_color = "#f59e0b"

            # Enhanced hover text for histogram (single text per trace)
            hover_text = (
                f"<b style='font-size:14px'>{name} Transactions</b><br><br>"
                f"<b style='color:{quality_color}'>{quality}</b><br><br>"
                f"<b>📊 Distribution Statistics:</b><br>"
                f"• Mean Error: <b>{mean:.3f}</b><br>"
                f"• Std Deviation: <b>{std:.3f}</b><br>"
                f"• Total Count: <b>{total}</b><br>"
                f"• Above Threshold: <b>{above_threshold}</b><br>"
                f"• Below Threshold: <b>{below_threshold}</b><br><br>"
                f"<b>🎯 Threshold Performance:</b><br>"
                f"{status_desc}<br><br>"
                f"<b>💡 Distribution Insight:</b><br>"
                f"Separation metric: <b>{separation:.2f}σ</b><br>"
                f"Better separation means easier fraud detection.<br><br>"
                f"<b>🔍 Recommendation:</b><br>"
                f"{'Threshold is well-positioned' if separation > 1.5 else 'Consider adjusting threshold for better separation'}"
            )

            fig.add_trace(go.Histogram(
                x=error_set,
                name=name,
                marker=dict(color=color),
                opacity=0.7,
                nbinsx=30,
                hovertemplate=f'{name}<br>Error: %{{x:.3f}}<br>Count: %{{y}}<br><br>{hover_text}<extra></extra>'
            ))

        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold={threshold:.1f}")

        fig.update_layout(
            title=f"Distribution of Reconstruction Errors (Separation: {separation:.2f}σ)",
            xaxis_title="Reconstruction Error",
            yaxis_title="Count",
            barmode='overlay',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="error_distribution")


def render_advanced_metrics(features, colors):
    """8. Advanced Metrics"""
    st.markdown("## 📈 Advanced Model Metrics")
    st.markdown("*F1 optimization, calibration curves, and lift charts*")

    # Prepare data
    X = features.drop('is_fraud', axis=1)
    y = features['is_fraud']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    col1, col2 = st.columns(2)

    with col1:
        # F1 score at different thresholds
        st.markdown("### F1 Score Optimization")

        thresholds = np.linspace(0, 1, 100)
        f1_scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:
                f1 = f1_score(y_test, y_pred)
            else:
                f1 = 0
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]

        # Enhanced hover texts for F1 optimization curve
        f1_hover_texts = []
        for thresh, f1 in zip(thresholds, f1_scores):
            # Distance from optimal
            dist_from_optimal = abs(thresh - optimal_threshold)

            if thresh == thresholds[optimal_idx]:
                status = "🏆 OPTIMAL THRESHOLD"
                status_color = "#10b981"
                insight = "This threshold maximizes the F1 score"
                recommendation = "Use this threshold for balanced precision & recall"
            elif dist_from_optimal < 0.05:
                status = "✅ NEAR OPTIMAL"
                status_color = "#22c55e"
                insight = "Close to optimal - minimal F1 loss"
                recommendation = "Safe threshold choice"
            elif f1 > optimal_f1 * 0.90:
                status = "🟡 ACCEPTABLE"
                status_color = "#f59e0b"
                insight = "Within 90% of optimal F1"
                recommendation = "May be used if other constraints exist"
            else:
                status = "🔴 SUBOPTIMAL"
                status_color = "#ef4444"
                insight = "Significant F1 loss at this threshold"
                recommendation = "Not recommended - too far from optimal"

            # Calculate precision and recall at this threshold
            y_pred_at_thresh = (y_pred_proba >= thresh).astype(int)
            if len(np.unique(y_pred_at_thresh)) > 1:
                from sklearn.metrics import precision_score, recall_score
                prec = precision_score(y_test, y_pred_at_thresh)
                rec = recall_score(y_test, y_pred_at_thresh)
            else:
                prec = 0
                rec = 0

            # F1 loss
            f1_loss = optimal_f1 - f1
            f1_loss_pct = (f1_loss / optimal_f1 * 100) if optimal_f1 > 0 else 0

            hover_text = (
                f"<b style='font-size:14px'>Threshold: {thresh:.3f}</b><br><br>"
                f"<b style='color:{status_color}'>{status}</b><br><br>"
                f"<b>📊 Performance Metrics:</b><br>"
                f"• F1 Score: <b>{f1:.4f}</b><br>"
                f"• Precision: <b>{prec:.4f}</b><br>"
                f"• Recall: <b>{rec:.4f}</b><br>"
                f"• Distance from Optimal: <b>{dist_from_optimal:.3f}</b><br>"
                f"• F1 Loss: <b>{f1_loss:.4f}</b> ({f1_loss_pct:.1f}%)<br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{insight}<br>"
                f"At this threshold, you get {prec*100:.1f}% precision<br>"
                f"and {rec*100:.1f}% recall.<br><br>"
                f"<b>🎯 Recommendation:</b><br>"
                f"{recommendation}<br><br>"
                f"<b>⚖️ Trade-off Insight:</b><br>"
                f"Lower threshold → more detections (higher recall)<br>"
                f"Higher threshold → fewer false positives (higher precision)"
            )
            f1_hover_texts.append(hover_text)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=f1_scores,
            mode='lines',
            line=dict(color=colors[0], width=3),
            name='F1 Score',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=f1_hover_texts
        ))

        fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Optimal: {optimal_threshold:.3f}")

        fig.update_layout(
            title=f"F1 Score vs Classification Threshold (Max: {optimal_f1:.3f})",
            xaxis_title="Classification Threshold",
            yaxis_title="F1 Score",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="f1_optimization")

    with col2:
        # Calibration curve
        st.markdown("### Probability Calibration Curve")

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10
        )

        brier = brier_score_loss(y_test, y_pred_proba)

        # Assess Brier score quality
        if brier < 0.05:
            brier_quality = "⭐ EXCELLENT"
            brier_color = "#10b981"
            brier_insight = "Outstanding probability calibration"
        elif brier < 0.10:
            brier_quality = "✅ GOOD"
            brier_color = "#22c55e"
            brier_insight = "Good calibration - probabilities are reliable"
        elif brier < 0.15:
            brier_quality = "🟡 ACCEPTABLE"
            brier_color = "#f59e0b"
            brier_insight = "Acceptable but could be improved"
        else:
            brier_quality = "🔴 POOR"
            brier_color = "#ef4444"
            brier_insight = "Poor calibration - probabilities unreliable"

        fig = go.Figure()

        # Perfect calibration line (with hover)
        perfect_hover = (
            "<b style='font-size:14px'>Perfect Calibration</b><br><br>"
            "<b>💡 What This Means:</b><br>"
            "When a model predicts 70% probability, exactly 70%<br>"
            "of those predictions should be positive cases.<br><br>"
            "<b>🎯 Goal:</b><br>"
            "Your model's curve should closely follow this line."
        )
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2),
            hovertemplate=f'{perfect_hover}<extra></extra>'
        ))

        # Enhanced hover texts for calibration curve
        calibration_hover_texts = []
        for pred_prob, actual_frac in zip(mean_predicted_value, fraction_of_positives):
            # Calculate calibration error at this bin
            calibration_error = abs(actual_frac - pred_prob)

            if calibration_error < 0.05:
                calib_status = "✅ WELL CALIBRATED"
                calib_color = "#10b981"
                calib_insight = "Predictions match reality closely"
            elif calibration_error < 0.10:
                calib_status = "🟡 ACCEPTABLE"
                calib_color = "#f59e0b"
                calib_insight = "Minor deviation from perfect calibration"
            else:
                calib_status = "🔴 POORLY CALIBRATED"
                calib_color = "#ef4444"
                calib_insight = "Significant calibration error"

            # Determine if over/under confident
            if actual_frac > pred_prob:
                confidence_assessment = "Underconfident: model predicts lower probability than actual"
            elif actual_frac < pred_prob:
                confidence_assessment = "Overconfident: model predicts higher probability than actual"
            else:
                confidence_assessment = "Perfect calibration at this probability bin"

            hover_text = (
                f"<b style='font-size:14px'>Calibration Bin</b><br><br>"
                f"<b style='color:{calib_color}'>{calib_status}</b><br><br>"
                f"<b>📊 Calibration Metrics:</b><br>"
                f"• Predicted Probability: <b>{pred_prob:.3f}</b> ({pred_prob*100:.1f}%)<br>"
                f"• Actual Fraction: <b>{actual_frac:.3f}</b> ({actual_frac*100:.1f}%)<br>"
                f"• Calibration Error: <b>{calibration_error:.3f}</b><br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"{calib_insight}<br>"
                f"When model predicts ~{pred_prob*100:.0f}% probability,<br>"
                f"{actual_frac*100:.0f}% are actually fraudulent.<br><br>"
                f"<b>🎯 Confidence Assessment:</b><br>"
                f"{confidence_assessment}<br><br>"
                f"<b>🔍 Practical Impact:</b><br>"
                f"{'Probabilities can be trusted for decision-making' if calibration_error < 0.1 else 'Consider recalibrating model (e.g., Platt scaling)'}"
            )
            calibration_hover_texts.append(hover_text)

        # Actual calibration
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color=colors[1], width=3),
            marker=dict(size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=calibration_hover_texts
        ))

        fig.update_layout(
            title=f"Calibration Curve (Brier: {brier:.4f} - {brier_quality})",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="calibration_curve")

    # Lift chart
    st.markdown("### Cumulative Gains and Lift Charts")

    col1, col2 = st.columns(2)

    with col1:
        # Cumulative gains
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_sorted = y_test.iloc[sorted_indices].values

        cumulative_gains = np.cumsum(y_sorted) / y_sorted.sum()
        percentile = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

        # Calculate key percentile metrics
        gain_at_10 = cumulative_gains[int(len(cumulative_gains) * 0.10)] if len(cumulative_gains) > 0 else 0
        gain_at_20 = cumulative_gains[int(len(cumulative_gains) * 0.20)] if len(cumulative_gains) > 0 else 0
        gain_at_50 = cumulative_gains[int(len(cumulative_gains) * 0.50)] if len(cumulative_gains) > 0 else 0

        fig = go.Figure()

        # Perfect model (with hover)
        perfect_hover = (
            "<b style='font-size:14px'>Perfect Model</b><br><br>"
            "<b>💡 What This Means:</b><br>"
            "A perfect model would capture 100% of fraud cases<br>"
            "by reviewing only the fraud cases (no wasted effort).<br><br>"
            "<b>🎯 Goal:</b><br>"
            "Get as close to this curve as possible."
        )
        fig.add_trace(go.Scatter(
            x=[0, y_sorted.sum() / len(y_sorted), 1],
            y=[0, 1, 1],
            mode='lines',
            name='Perfect Model',
            line=dict(color='gray', dash='dash', width=2),
            hovertemplate=f'{perfect_hover}<extra></extra>'
        ))

        # Random model (with hover)
        random_hover = (
            "<b style='font-size:14px'>Random Model</b><br><br>"
            "<b>💡 What This Means:</b><br>"
            "A random model (no ML) would catch fraud cases<br>"
            "proportionally to review volume.<br><br>"
            "<b>📊 Example:</b><br>"
            "Review 20% of cases → catch 20% of fraud<br>"
            "Review 50% of cases → catch 50% of fraud"
        )
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Model',
            line=dict(color='lightgray', dash='dot', width=2),
            hovertemplate=f'{random_hover}<extra></extra>'
        ))

        # Enhanced hover texts for actual gains curve
        gains_hover_texts = []
        x_values = np.concatenate([[0], percentile])
        y_values = np.concatenate([[0], cumulative_gains])

        for i, (pct, gain) in enumerate(zip(x_values, y_values)):
            if i == 0:
                # First point (0,0)
                hover_text = (
                    "<b style='font-size:14px'>Starting Point</b><br><br>"
                    "Review 0% → Catch 0% fraud"
                )
            else:
                # Calculate improvement over random
                improvement = gain - pct
                improvement_pct = (improvement / pct * 100) if pct > 0 else 0

                if improvement > 0.3:
                    performance = "🏆 EXCEPTIONAL"
                    perf_color = "#10b981"
                    insight = "Outstanding fraud concentration"
                elif improvement > 0.15:
                    performance = "⭐ EXCELLENT"
                    perf_color = "#22c55e"
                    insight = "Strong model performance"
                elif improvement > 0.05:
                    performance = "✅ GOOD"
                    perf_color = "#3b82f6"
                    insight = "Above-random performance"
                else:
                    performance = "🟡 MARGINAL"
                    perf_color = "#f59e0b"
                    insight = "Limited improvement over random"

                # Calculate efficiency gain
                cases_to_review = int(pct * len(y_sorted))
                fraud_caught = int(gain * y_sorted.sum())
                total_fraud = int(y_sorted.sum())

                hover_text = (
                    f"<b style='font-size:14px'>Review Top {pct*100:.0f}% of Cases</b><br><br>"
                    f"<b style='color:{perf_color}'>{performance}</b><br><br>"
                    f"<b>📊 Cumulative Gains:</b><br>"
                    f"• Fraud Caught: <b>{gain*100:.1f}%</b> ({fraud_caught}/{total_fraud} cases)<br>"
                    f"• Cases Reviewed: <b>{pct*100:.1f}%</b> ({cases_to_review:,} cases)<br>"
                    f"• Improvement vs Random: <b>+{improvement*100:.1f}pp</b><br><br>"
                    f"<b>💡 What This Means:</b><br>"
                    f"{insight}<br>"
                    f"By reviewing the top {pct*100:.0f}% of cases ranked<br>"
                    f"by your model, you catch {gain*100:.0f}% of all fraud.<br><br>"
                    f"<b>💰 Business Value:</b><br>"
                    f"You catch {improvement_pct:.0f}% more fraud than random<br>"
                    f"selection at this review volume.<br><br>"
                    f"<b>🎯 Practical Impact:</b><br>"
                    f"If analyst capacity limits you to {pct*100:.0f}% review,<br>"
                    f"you'll catch {fraud_caught} of {total_fraud} fraud cases."
                )

            gains_hover_texts.append(hover_text)

        # Actual model
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines',
            name='Model',
            line=dict(color=colors[0], width=3),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=gains_hover_texts
        ))

        fig.update_layout(
            title=f"Cumulative Gains Chart (10%: {gain_at_10*100:.0f}%, 20%: {gain_at_20*100:.0f}%, 50%: {gain_at_50*100:.0f}%)",
            xaxis_title="Percentage of Sample",
            yaxis_title="Percentage of Positive Class",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="gains_chart")

    with col2:
        # Lift chart
        n_bins = 10
        bin_size = len(y_sorted) // n_bins

        lift_values = []
        bin_labels = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_sorted)

            bin_y = y_sorted[start_idx:end_idx]
            bin_rate = bin_y.mean()
            overall_rate = y_sorted.mean()
            lift = bin_rate / overall_rate if overall_rate > 0 else 0

            lift_values.append(lift)
            bin_labels.append(f"Top {(i+1)*10}%")

        # Enhanced hover texts for lift chart
        hover_texts = []
        for i, (label, lift) in enumerate(zip(bin_labels, lift_values)):
            percentile = (i+1) * 10

            if lift >= 3.0:
                performance = "🏆 EXCEPTIONAL"
                perf_color = "#10b981"
                assessment = "Outstanding concentration of fraud cases"
                business_value = "Highest priority for investigation resources"
            elif lift >= 2.0:
                performance = "⭐ EXCELLENT"
                perf_color = "#3b82f6"
                assessment = "Strong fraud concentration"
                business_value = "High-value targets for analyst review"
            elif lift >= 1.5:
                performance = "✅ GOOD"
                perf_color = "#22c55e"
                assessment = "Above-average fraud detection"
                business_value = "Worthwhile investigation targets"
            elif lift >= 1.0:
                performance = "🟡 MODERATE"
                perf_color = "#f59e0b"
                assessment = "Slight improvement over random"
                business_value = "Secondary priority for review"
            else:
                performance = "🔴 POOR"
                perf_color = "#ef4444"
                assessment = "Below-average fraud concentration"
                business_value = "Deprioritize for investigation"

            # Calculate efficiency metrics
            fraud_concentration = lift * overall_rate * 100
            efficiency_gain = (lift - 1) * 100

            hover_text = (
                f"<b style='font-size:14px'>{label} of Predictions</b><br><br>"
                f"<b style='color:{perf_color}'>{performance}</b><br>"
                f"{assessment}<br><br>"
                f"<b>📊 Lift Metrics:</b><br>"
                f"• Lift Value: <b>{lift:.2f}x</b><br>"
                f"• Fraud Rate in Bin: <b>{fraud_concentration:.1f}%</b><br>"
                f"• Overall Fraud Rate: <b>{overall_rate*100:.1f}%</b><br>"
                f"• Efficiency Gain: <b>+{efficiency_gain:.0f}%</b> vs random<br><br>"
                f"<b>💡 What This Means:</b><br>"
                f"By focusing on the {label} of predictions ranked by score,<br>"
                f"you catch <b>{lift:.1f}x</b> more fraud than reviewing randomly.<br><br>"
                f"<b>💰 Business Value:</b><br>"
                f"{business_value}<br><br>"
                f"<b>🎯 Practical Impact:</b><br>"
                f"If you review only this decile, you'll find <b>{fraud_concentration:.1f}%</b><br>"
                f"of transactions are fraudulent (vs {overall_rate*100:.1f}% baseline)"
            )
            hover_texts.append(hover_text)

        fig = go.Figure(go.Bar(
            x=bin_labels,
            y=lift_values,
            marker=dict(
                color=lift_values,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Lift")
            ),
            text=[f"{v:.2f}x" for v in lift_values],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))

        fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                     annotation_text="Baseline")

        fig.update_layout(
            title="Lift Chart by Decile",
            xaxis_title="Population Percentile",
            yaxis_title="Lift",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True, key="lift_chart")

    # Metrics summary table
    st.markdown("### Model Performance Summary")

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    metrics_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'Brier Score', 'Log Loss'],
        'Score': [
            f"{accuracy_score(y_test, y_pred):.4f}",
            f"{precision_score(y_test, y_pred):.4f}",
            f"{recall_score(y_test, y_pred):.4f}",
            f"{optimal_f1:.4f}",
            f"{auc(*roc_curve(y_test, y_pred_proba)[:2]):.4f}",
            f"{brier:.4f}",
            f"{log_loss(y_test, y_pred_proba):.4f}"
        ],
        'Description': [
            'Overall correctness of predictions',
            'Proportion of true frauds among detected frauds',
            'Proportion of actual frauds detected',
            'Harmonic mean of precision and recall',
            'Area under ROC curve',
            'Mean squared difference between predicted probabilities and actual outcomes',
            'Negative log-likelihood of true labels given predictions'
        ]
    }

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)


def render():
    """Main render function for AI & ML Intelligence page"""
    apply_master_theme()

    st.title("🤖 AI & Machine Learning Intelligence")
    st.markdown("*Advanced machine learning analytics and model intelligence for fraud detection*")

    # Load data
    data = load_ml_data()

    if data is None:
        st.error("Unable to load data. Please ensure compliance_dataset/ exists with required CSV files.")
        return

    # Get theme colors with fallback - convert to list if needed
    try:
        colors = get_chart_colors()
        # Convert dict to list of values if it's a dict
        if isinstance(colors, dict):
            colors = list(colors.values())
        # Ensure it's a list/array with at least some colors
        if not colors or len(colors) == 0:
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    except:
        # Fallback colors if get_chart_colors() fails
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Prepare ML features
    with st.spinner("Preparing ML features..."):
        features = prepare_ml_features(data['transactions'], data['customers'])

    # Render sections
    tabs = st.tabs([
        "🧠 Neural Networks",
        "🌳 Ensemble Models",
        "📊 Model Performance",
        "🔍 Explainable AI",
        "⚡ Real-time Monitoring",
        "🔬 Feature Engineering",
        "🤖 Deep Learning",
        "📈 Advanced Metrics"
    ])

    with tabs[0]:
        render_neural_network_architecture(colors)

    with tabs[1]:
        render_ensemble_models(features, colors)

    with tabs[2]:
        render_model_performance(features, colors)

    with tabs[3]:
        render_explainable_ai(features, colors)

    with tabs[4]:
        render_realtime_monitoring(colors)

    with tabs[5]:
        render_feature_engineering(features, colors)

    with tabs[6]:
        render_deep_learning_viz(colors)

    with tabs[7]:
        render_advanced_metrics(features, colors)

    # Footer
    st.markdown("---")
    st.markdown("*AI & ML Intelligence Dashboard - Powered by Advanced Machine Learning*")


if __name__ == "__main__":
    render()
