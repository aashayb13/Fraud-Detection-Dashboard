# app/services/decision_engine.py
from typing import Dict, Any
from dataclasses import dataclass
from app.services.risk_scoring import RiskScorer
from config.settings import (
    DEFAULT_AUTO_APPROVE_THRESHOLD, 
    DEFAULT_MANUAL_REVIEW_THRESHOLD,
    DEFAULT_HIGH_VALUE_THRESHOLD,
    HOURLY_REVIEW_COST,
    AVG_REVIEW_TIME_MINUTES
)

@dataclass
class ThresholdConfig:
    """Configuration for decision thresholds and cost-benefit analysis."""
    auto_approve_below: float = DEFAULT_AUTO_APPROVE_THRESHOLD
    manual_review_above: float = DEFAULT_MANUAL_REVIEW_THRESHOLD
    high_value_threshold: float = DEFAULT_HIGH_VALUE_THRESHOLD
    
    # Cost-benefit parameters
    hourly_review_cost: float = HOURLY_REVIEW_COST
    avg_review_time_minutes: float = AVG_REVIEW_TIME_MINUTES
    
    def get_review_cost(self) -> float:
        """Calculate the cost of a single transaction review."""
        return (self.hourly_review_cost / 60) * self.avg_review_time_minutes

class DecisionEngine:
    def __init__(self, risk_scorer: RiskScorer, config: ThresholdConfig = None):
        """
        Initialize decision engine with risk scorer and thresholds.
        
        Args:
            risk_scorer: Risk scoring component
            config: Optional configuration (uses defaults if not provided)
        """
        self.risk_scorer = risk_scorer
        self.config = config or ThresholdConfig()
    
    def evaluate(self, transaction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a transaction and decide on appropriate action.
        
        Args:
            transaction: Transaction data
            context: Additional contextual data
            
        Returns:
            Decision result with risk assessment
        """
        # Get risk assessment
        risk_assessment = self.risk_scorer.score_transaction(transaction, context)
        risk_score = risk_assessment["risk_score"]
        
        # Base decision on score and amount
        decision = "auto_approve"
        review_reason = None
        
        if risk_score > self.config.manual_review_above:
            decision = "manual_review"
            review_reason = "High risk score"
        elif risk_score > self.config.auto_approve_below:
            # Middle zone - consider amount and cost-benefit
            if transaction.get("amount", 0) > self.config.high_value_threshold:
                decision = "manual_review"
                review_reason = f"Transaction amount exceeds high-value threshold (${self.config.high_value_threshold:,.2f})"
            else:
                # Cost-benefit calculation
                review_cost = self.config.get_review_cost()
                expected_loss = transaction.get("amount", 0) * risk_score
                
                if expected_loss > review_cost:
                    decision = "manual_review"
                    review_reason = "Expected loss exceeds review cost"
        
        # Cost-benefit analysis
        review_cost = self.config.get_review_cost()
        expected_loss = transaction.get("amount", 0) * risk_score
        
        return {
            "transaction_id": transaction.get("transaction_id"),
            "decision": decision,
            "review_reason": review_reason,
            "risk_assessment": risk_assessment,
            "cost_benefit": {
                "review_cost_usd": review_cost,
                "expected_loss_usd": expected_loss,
                "net_benefit_of_review_usd": expected_loss - review_cost if decision == "manual_review" else 0,
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update decision engine configuration."""
        if "auto_approve_below" in new_config:
            self.config.auto_approve_below = float(new_config["auto_approve_below"])
        if "manual_review_above" in new_config:
            self.config.manual_review_above = float(new_config["manual_review_above"])
        if "high_value_threshold" in new_config:
            self.config.high_value_threshold = float(new_config["high_value_threshold"])
        if "hourly_review_cost" in new_config:
            self.config.hourly_review_cost = float(new_config["hourly_review_cost"])
        if "avg_review_time_minutes" in new_config:
            self.config.avg_review_time_minutes = float(new_config["avg_review_time_minutes"])