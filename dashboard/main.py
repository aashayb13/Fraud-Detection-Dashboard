# dashboard/main.py
"""
Transaction Monitoring Dashboard

Unified visualization for ALL fraud detection scenarios.
Shows real-time monitoring across all use cases.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database import get_db, Transaction, RiskAssessment, AccountChangeHistory
from app.utils.main import format_currency


class DashboardData:
    """
    Aggregates data from ALL fraud scenarios for dashboard display.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_overview_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get overview statistics across ALL scenarios.

        Returns aggregated metrics for the dashboard.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=time_window_hours)).isoformat()

        # Get all recent transactions
        recent_txs = self.db.query(Transaction).filter(
            Transaction.timestamp > cutoff
        ).all()

        # Get all recent risk assessments
        recent_assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.review_timestamp > cutoff
        ).all()

        total_transactions = len(recent_txs)
        total_value = sum(tx.amount for tx in recent_txs)

        # Count by decision type
        decisions = {"auto_approve": 0, "manual_review": 0, "blocked": 0}
        for assessment in recent_assessments:
            decisions[assessment.decision] = decisions.get(assessment.decision, 0) + 1

        # Average risk score
        avg_risk = (
            sum(a.risk_score for a in recent_assessments) / len(recent_assessments)
            if recent_assessments else 0
        )

        return {
            "time_window_hours": time_window_hours,
            "total_transactions": total_transactions,
            "total_value": total_value,
            "auto_approved": decisions["auto_approve"],
            "manual_review": decisions["manual_review"],
            "blocked": decisions.get("blocked", 0),
            "average_risk_score": avg_risk,
            "review_rate": decisions["manual_review"] / max(total_transactions, 1)
        }

    def get_scenario_breakdown(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Break down activity by fraud scenario type.

        This shows which scenarios are triggering most frequently.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=time_window_hours)).isoformat()

        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.review_timestamp > cutoff
        ).all()

        # Initialize scenario counters
        scenarios = {
            "payroll_fraud": {"count": 0, "total_risk": 0, "high_risk": 0},
            "credit_fraud": {"count": 0, "total_risk": 0, "high_risk": 0},
            "wire_fraud": {"count": 0, "total_risk": 0, "high_risk": 0},
            "other": {"count": 0, "total_risk": 0, "high_risk": 0},
        }

        # Categorize by triggered rules
        for assessment in assessments:
            import json
            triggered = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}

            # Check which scenario rules were triggered
            if any("payroll" in rule for rule in triggered.keys()):
                category = "payroll_fraud"
            elif any("credit" in rule for rule in triggered.keys()):
                category = "credit_fraud"
            elif any("wire" in rule for rule in triggered.keys()):
                category = "wire_fraud"
            else:
                category = "other"

            scenarios[category]["count"] += 1
            scenarios[category]["total_risk"] += assessment.risk_score
            if assessment.risk_score > 0.6:
                scenarios[category]["high_risk"] += 1

        # Calculate averages
        for scenario in scenarios.values():
            if scenario["count"] > 0:
                scenario["avg_risk"] = scenario["total_risk"] / scenario["count"]
            else:
                scenario["avg_risk"] = 0

        return scenarios

    def get_top_triggered_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently triggered rules across ALL scenarios.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

        assessments = self.db.query(RiskAssessment).filter(
            RiskAssessment.review_timestamp > cutoff
        ).all()

        # Count rule triggers
        rule_counts = {}
        for assessment in assessments:
            import json
            triggered = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}

            for rule_name, rule_info in triggered.items():
                if rule_name not in rule_counts:
                    rule_counts[rule_name] = {
                        "name": rule_name,
                        "description": rule_info.get("description", ""),
                        "count": 0,
                        "weight": rule_info.get("weight", 0)
                    }
                rule_counts[rule_name]["count"] += 1

        # Sort by count and return top N
        sorted_rules = sorted(
            rule_counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )

        return sorted_rules[:limit]

    def get_manual_review_queue(self) -> List[Dict[str, Any]]:
        """
        Get transactions pending manual review across ALL scenarios.
        """
        pending = self.db.query(RiskAssessment).filter(
            RiskAssessment.decision == "manual_review",
            RiskAssessment.review_status == "pending"
        ).order_by(RiskAssessment.risk_score.desc()).all()

        queue = []
        for assessment in pending:
            tx = self.db.query(Transaction).filter(
                Transaction.transaction_id == assessment.transaction_id
            ).first()

            if tx:
                import json
                triggered = json.loads(assessment.triggered_rules) if assessment.triggered_rules else {}

                queue.append({
                    "assessment_id": assessment.assessment_id,
                    "transaction_id": tx.transaction_id,
                    "amount": tx.amount,
                    "transaction_type": tx.transaction_type,
                    "risk_score": assessment.risk_score,
                    "triggered_rules": list(triggered.keys()),
                    "timestamp": tx.timestamp
                })

        return queue

    def get_recent_account_changes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent account changes (for payroll fraud monitoring).
        """
        changes = self.db.query(AccountChangeHistory).order_by(
            AccountChangeHistory.timestamp.desc()
        ).limit(limit).all()

        return [
            {
                "change_id": change.change_id,
                "employee_id": change.employee_id,
                "change_type": change.change_type,
                "change_source": change.change_source,
                "verified": change.verified,
                "flagged": change.flagged_as_suspicious,
                "timestamp": change.timestamp
            }
            for change in changes
        ]


def print_dashboard_summary():
    """
    Print a text-based dashboard summary.

    In production, this would be a web interface showing:
    - Real-time metrics across all fraud scenarios
    - Interactive charts and graphs
    - Manual review queue
    - Drill-down by scenario type
    """
    db = next(get_db())

    try:
        dashboard = DashboardData(db)

        print("\n" + "="*80)
        print("TRANSACTION MONITORING DASHBOARD")
        print("="*80)

        # Overview stats
        stats = dashboard.get_overview_stats()
        print(f"\n>> Last 24 Hours Overview:")
        print(f"  Total Transactions: {stats['total_transactions']}")
        print(f"  Total Value: {format_currency(stats['total_value'])}")
        print(f"  Auto-Approved: {stats['auto_approved']} ({stats['auto_approved']/max(stats['total_transactions'],1)*100:.1f}%)")
        print(f"  Manual Review: {stats['manual_review']} ({stats['review_rate']*100:.1f}%)")
        print(f"  Average Risk Score: {stats['average_risk_score']:.2f}")

        # Scenario breakdown
        scenarios = dashboard.get_scenario_breakdown()
        print(f"\n>> Activity by Fraud Scenario:")
        for name, data in scenarios.items():
            if data['count'] > 0:
                print(f"  {name.replace('_', ' ').title()}:")
                print(f"    Transactions: {data['count']}, Avg Risk: {data['avg_risk']:.2f}, High Risk: {data['high_risk']}")

        # Top triggered rules
        top_rules = dashboard.get_top_triggered_rules(5)
        print(f"\n>> Most Triggered Rules:")
        for i, rule in enumerate(top_rules, 1):
            print(f"  {i}. {rule['description']} (triggered {rule['count']} times)")

        # Review queue
        queue = dashboard.get_manual_review_queue()
        print(f"\n>> Manual Review Queue: {len(queue)} items")
        for item in queue[:5]:  # Show top 5
            print(f"  - {item['transaction_id']}: {format_currency(item['amount'])} (Risk: {item['risk_score']:.2f})")

        print("\n" + "="*80)

    finally:
        db.close()


if __name__ == "__main__":
    print_dashboard_summary()
