# app/services/risk_scoring.py
from typing import Dict, Any, Optional, List
from app.services.rules_engine import RulesEngine, Rule
import json

class RiskScorer:
    def __init__(self, rules_engine: RulesEngine, ml_model=None):
        """
        Initialize a risk scorer with rules engine and optional ML model.
        
        Args:
            rules_engine: Rules engine for rule-based scoring
            ml_model: Optional ML model for predictive scoring
        """
        self.rules_engine = rules_engine
        self.ml_model = ml_model
        self.rule_score_weight = 0.7
        self.ml_score_weight = 0.3
    
    def score_transaction(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a transaction based on rules and optional ML model.
        
        Args:
            transaction: Transaction data
            context: Additional contextual data
            
        Returns:
            Dictionary with risk assessment results
        """
        # Get triggered rules
        triggered_rules = self.rules_engine.evaluate_all(transaction, context)
        
        # Calculate rule-based score (normalized between 0-1)
        total_weight = sum(rule.weight for rule in triggered_rules.values())
        max_possible_weight = sum(rule.weight for rule in self.rules_engine.rules)
        
        # Avoid division by zero if no rules exist
        rule_score = total_weight / max(max_possible_weight, 0.001)
        
        # Add ML score if model exists
        ml_score = 0.0
        ml_explanation = None
        if self.ml_model:
            features = self._extract_features(transaction, context)
            ml_result = self.ml_model.predict(features)
            ml_score = ml_result['score']
            ml_explanation = ml_result.get('explanation')
        
        # Combine scores (weighted)
        if self.ml_model:
            combined_score = (self.rule_score_weight * rule_score + 
                             self.ml_score_weight * ml_score)
        else:
            combined_score = rule_score
        
        # Cap score between 0-1
        combined_score = max(0.0, min(1.0, combined_score))
        
        return {
            "risk_score": combined_score,
            "rule_score": rule_score,
            "triggered_rules": {name: rule.to_dict() for name, rule in triggered_rules.items()},
            "ml_score": ml_score if self.ml_model else None,
            "ml_explanation": ml_explanation,
            "explanation": self._generate_explanation(triggered_rules, ml_score, ml_explanation)
        }
    
    def _extract_features(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and normalize features for ML model."""
        # This would be customized based on your specific model requirements
        features = {
            "amount": transaction.get("amount", 0),
            "account_age_days": context.get("account_age_days", 0),
            "tx_count_24h": context.get("tx_count_last_hours", {}).get(24, 0),
            "amount_deviation": context.get("amount_deviation", 1.0),
            "is_new_counterparty": 1 if context.get("is_new_counterparty", False) else 0,
        }
        return features
    
    def _generate_explanation(self, triggered_rules: Dict[str, Rule], 
                             ml_score: Optional[float],
                             ml_explanation: Optional[str]) -> List[str]:
        """
        Generate human-readable explanation of risk factors.
        
        Args:
            triggered_rules: Dictionary of triggered rules
            ml_score: ML model risk score
            ml_explanation: ML model explanation
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        # Add rule explanations
        for rule in triggered_rules.values():
            explanations.append(rule.description)
        
        # Add ML explanation if available and significant
        if ml_score and ml_score > 0.3:
            if ml_explanation:
                explanations.append(ml_explanation)
            else:
                explanations.append(f"ML model detected unusual patterns (score: {ml_score:.2f})")
        
        return explanations if explanations else ["No risk factors identified"]