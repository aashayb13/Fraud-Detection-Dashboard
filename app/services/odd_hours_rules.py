# app/services/odd_hours_rules.py
"""
Odd hours transaction fraud detection rules.

This module implements detection rules for the following fraud scenario:
Large or significant transactions initiated at unusual hours - particularly
during the middle of the night or outside a customer's normal activity window.

Common fraud patterns:
1. Fraudster initiates large transfers late at night to delay detection
2. Account takeover leads to unusual transaction timing
3. Insider fraud conducted during off-hours to avoid oversight
4. Automated/bot-driven fraud that doesn't follow human patterns

Risk indicators:
- Transactions during late night/early morning (10 PM - 6 AM)
- Large amounts transferred at odd hours
- Timing deviates significantly from customer's historical patterns
- Weekend or holiday transactions that are unusual for the account
"""

from typing import Dict, Any, List
from app.services.rules_engine import Rule
from config.settings import (
    ODD_HOURS_START,
    ODD_HOURS_END,
    ODD_HOURS_LARGE_TRANSACTION_THRESHOLD,
    ODD_HOURS_VERY_LARGE_THRESHOLD,
    ODD_HOURS_DEVIATION_FROM_PATTERN,
    ODD_HOURS_MIN_HISTORICAL_TRANSACTIONS
)


def create_odd_hours_transaction_rule(
    rule_name: str = None,
    weight: float = 2.0
) -> Rule:
    """
    Detect transactions occurring during odd hours (late night/early morning).

    Flags transactions during the configured odd hours window (default 10 PM - 6 AM).
    This is a baseline indicator that requires combination with other factors for
    strong fraud detection.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object for odd hours transaction detection
    """
    name = rule_name or "odd_hours_transaction"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        return context.get("is_odd_hours", False)

    return Rule(
        name=name,
        description=f"Transaction initiated during odd hours ({ODD_HOURS_START:02d}:00 - {ODD_HOURS_END:02d}:00)",
        condition_func=condition,
        weight=weight
    )


def create_large_odd_hours_transaction_rule(
    threshold: float = ODD_HOURS_LARGE_TRANSACTION_THRESHOLD,
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect large transactions during odd hours.

    Combination of significant amount and unusual timing is a strong fraud indicator.
    Legitimate large transactions are typically conducted during business hours with
    proper oversight.

    Args:
        threshold: Minimum transaction amount to trigger (default from settings)
        rule_name: Optional custom rule name
        weight: Rule importance weight (elevated due to combined risk factors)

    Returns:
        Rule object for large odd hours transaction detection
    """
    name = rule_name or f"large_odd_hours_transaction_{int(threshold)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        amount = abs(transaction.get("amount", 0))

        return is_odd_hours and amount >= threshold

    return Rule(
        name=name,
        description=f"Large transaction (>= ${threshold:,.2f}) initiated during odd hours - elevated fraud risk",
        condition_func=condition,
        weight=weight
    )


def create_very_large_odd_hours_transaction_rule(
    threshold: float = ODD_HOURS_VERY_LARGE_THRESHOLD,
    rule_name: str = None,
    weight: float = 5.0
) -> Rule:
    """
    Detect very large transactions during odd hours.

    Critical risk: Very large amounts combined with odd timing require immediate review.
    Extremely rare for legitimate business to conduct major transfers at night.

    Args:
        threshold: Minimum transaction amount to trigger (default from settings)
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - critical indicator)

    Returns:
        Rule object for very large odd hours transaction detection
    """
    name = rule_name or f"very_large_odd_hours_transaction_{int(threshold)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        amount = abs(transaction.get("amount", 0))

        return is_odd_hours and amount >= threshold

    return Rule(
        name=name,
        description=f"Very large transaction (>= ${threshold:,.2f}) initiated during odd hours - CRITICAL fraud risk",
        condition_func=condition,
        weight=weight
    )


def create_deviates_from_pattern_rule(
    rule_name: str = None,
    weight: float = 4.0
) -> Rule:
    """
    Detect transactions at odd hours that deviate from customer's normal pattern.

    High risk: Customer typically transacts during business hours, but current
    transaction is at an unusual time. This pattern deviation suggests possible
    account compromise or unauthorized access.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight (high - strong behavioral anomaly)

    Returns:
        Rule object for pattern deviation detection
    """
    name = rule_name or "odd_hours_pattern_deviation"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Check if transaction deviates from historical pattern
        deviates = context.get("deviates_from_pattern", False)

        # Also check that we have sufficient history to make this determination
        has_history = context.get("historical_transaction_count", 0) >= ODD_HOURS_MIN_HISTORICAL_TRANSACTIONS

        return deviates and has_history

    return Rule(
        name=name,
        description="Transaction at odd hours deviates significantly from customer's normal activity pattern",
        condition_func=condition,
        weight=weight
    )


def create_unusual_hour_transaction_rule(
    rule_name: str = None,
    weight: float = 3.0
) -> Rule:
    """
    Detect transactions at hours when the customer rarely transacts.

    Identifies transactions at specific hours that are unusual for this customer,
    even if not technically "odd hours". For example, a customer who typically
    transacts at 2 PM making a transaction at 7 AM.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object for unusual hour detection
    """
    name = rule_name or "unusual_hour_transaction"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        return context.get("hour_is_unusual", False)

    return Rule(
        name=name,
        description="Transaction at hour when customer rarely transacts - behavioral anomaly",
        condition_func=condition,
        weight=weight
    )


def create_weekend_odd_hours_transaction_rule(
    min_amount: float = ODD_HOURS_LARGE_TRANSACTION_THRESHOLD,
    rule_name: str = None,
    weight: float = 4.5
) -> Rule:
    """
    Detect large transactions during weekend odd hours.

    Very high risk: Combination of weekend + odd hours + large amount.
    Legitimate businesses rarely conduct significant transactions during
    weekend nights when staff is unavailable.

    Args:
        min_amount: Minimum transaction amount to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - multiple risk factors)

    Returns:
        Rule object for weekend odd hours detection
    """
    name = rule_name or f"weekend_odd_hours_large_transaction_{int(min_amount)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        is_weekend = context.get("is_weekend", False)
        amount = abs(transaction.get("amount", 0))

        return is_odd_hours and is_weekend and amount >= min_amount

    return Rule(
        name=name,
        description=f"Large transaction (>= ${min_amount:,.2f}) during weekend odd hours - multiple risk factors",
        condition_func=condition,
        weight=weight
    )


def create_new_counterparty_odd_hours_rule(
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect transactions to new counterparties during odd hours.

    High risk: First-time recipient combined with unusual timing suggests
    possible fraud. Legitimate businesses typically establish new vendor
    relationships during business hours.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object for new counterparty at odd hours detection
    """
    name = rule_name or "new_counterparty_odd_hours"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        is_new_counterparty = context.get("is_new_counterparty", False)

        return is_odd_hours and is_new_counterparty

    return Rule(
        name=name,
        description="Transaction to new counterparty during odd hours - unusual timing for new relationship",
        condition_func=condition,
        weight=weight
    )


def create_first_odd_hours_transaction_rule(
    rule_name: str = None,
    weight: float = 2.5
) -> Rule:
    """
    Detect accounts with no recent odd hours activity making odd hours transactions.

    Moderate risk: First odd hours transaction in recent period (7 days) suggests
    a change in behavior that warrants additional scrutiny.

    Args:
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object for first odd hours transaction detection
    """
    name = rule_name or "first_odd_hours_transaction_recent"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)

        # Get count of recent odd hours transactions (excluding current)
        recent_odd_hours_count = context.get("recent_odd_hours_transaction_count", 0)

        # Trigger if this is odd hours but no recent history of odd hours transactions
        return is_odd_hours and recent_odd_hours_count == 0

    return Rule(
        name=name,
        description="First odd hours transaction in recent period - behavioral change detected",
        condition_func=condition,
        weight=weight
    )


def create_repeated_odd_hours_transactions_rule(
    min_count: int = 3,
    min_total_amount: float = 10000.0,
    rule_name: str = None,
    weight: float = 3.5
) -> Rule:
    """
    Detect multiple odd hours transactions in recent period with significant total amount.

    High risk: Pattern of repeated odd hours activity with substantial amounts
    suggests systematic fraud or money movement designed to avoid detection.

    Args:
        min_count: Minimum number of recent odd hours transactions
        min_total_amount: Minimum total amount across recent odd hours transactions
        rule_name: Optional custom rule name
        weight: Rule importance weight

    Returns:
        Rule object for repeated odd hours transactions detection
    """
    name = rule_name or f"repeated_odd_hours_transactions_{min_count}tx_{int(min_total_amount)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)

        if not is_odd_hours:
            return False

        recent_odd_hours_count = context.get("recent_odd_hours_transaction_count", 0)
        recent_odd_hours_total = context.get("recent_odd_hours_total_amount", 0)

        return recent_odd_hours_count >= min_count and recent_odd_hours_total >= min_total_amount

    return Rule(
        name=name,
        description=f"Multiple odd hours transactions ({min_count}+) with total >= ${min_total_amount:,.2f} - systematic odd hours activity",
        condition_func=condition,
        weight=weight
    )


def create_outgoing_transfer_odd_hours_rule(
    min_amount: float = 1000.0,
    rule_name: str = None,
    weight: float = 4.0
) -> Rule:
    """
    Detect outgoing transfers during odd hours.

    High risk: Outgoing transfers (debits) at unusual times are more concerning
    than credits, as they represent potential fund exfiltration. Combined with
    odd timing, this is a strong fraud indicator.

    Args:
        min_amount: Minimum transfer amount to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight (high - fund exfiltration risk)

    Returns:
        Rule object for outgoing odd hours transfer detection
    """
    name = rule_name or f"outgoing_transfer_odd_hours_{int(min_amount)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        is_outgoing = transaction.get("direction", "").lower() == "debit"
        amount = abs(transaction.get("amount", 0))

        return is_odd_hours and is_outgoing and amount >= min_amount

    return Rule(
        name=name,
        description=f"Outgoing transfer (>= ${min_amount:,.2f}) during odd hours - fund exfiltration risk",
        condition_func=condition,
        weight=weight
    )


def create_international_transfer_odd_hours_rule(
    min_amount: float = 5000.0,
    rule_name: str = None,
    weight: float = 5.0
) -> Rule:
    """
    Detect international transfers during odd hours.

    Critical risk: International wire transfers at odd hours are extremely high risk.
    Combines cross-border complexity with unusual timing, making recovery difficult
    if fraudulent.

    Args:
        min_amount: Minimum transfer amount to trigger
        rule_name: Optional custom rule name
        weight: Rule importance weight (very high - critical risk)

    Returns:
        Rule object for international odd hours transfer detection
    """
    name = rule_name or f"international_transfer_odd_hours_{int(min_amount)}"

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        is_odd_hours = context.get("is_odd_hours", False)
        amount = abs(transaction.get("amount", 0))

        # Check if international transfer
        tx_type = transaction.get("transaction_type", "").lower()
        is_international = (
            "international" in tx_type or
            "wire" in tx_type or
            "swift" in tx_type
        )

        # Could also check country code if available
        counterparty_country = transaction.get("counterparty_country", "US")
        is_cross_border = counterparty_country != "US"

        return is_odd_hours and (is_international or is_cross_border) and amount >= min_amount

    return Rule(
        name=name,
        description=f"International transfer (>= ${min_amount:,.2f}) during odd hours - CRITICAL cross-border fraud risk",
        condition_func=condition,
        weight=weight
    )


def initialize_odd_hours_rules() -> List[Rule]:
    """
    Initialize all odd hours fraud detection rules.

    Returns a comprehensive set of rules for detecting suspicious transactions
    occurring at unusual times, with varying severity levels based on amount,
    timing deviation, and other risk factors.

    Returns:
        List of configured Rule objects for odd hours fraud detection
    """
    return [
        # Critical - highest priority
        create_very_large_odd_hours_transaction_rule(weight=5.0),
        create_international_transfer_odd_hours_rule(weight=5.0),

        # High priority - strong indicators
        create_weekend_odd_hours_transaction_rule(weight=4.5),
        create_deviates_from_pattern_rule(weight=4.0),
        create_outgoing_transfer_odd_hours_rule(weight=4.0),
        create_large_odd_hours_transaction_rule(weight=3.5),
        create_new_counterparty_odd_hours_rule(weight=3.5),
        create_repeated_odd_hours_transactions_rule(weight=3.5),

        # Medium priority - contextual indicators
        create_unusual_hour_transaction_rule(weight=3.0),
        create_first_odd_hours_transaction_rule(weight=2.5),

        # Lower priority - baseline indicator
        create_odd_hours_transaction_rule(weight=2.0),
    ]


def get_high_security_odd_hours_rules() -> List[Rule]:
    """
    Get a conservative rule set for high-security environments.

    Uses lower thresholds to catch more potential fraud at the cost of
    increased false positives.

    Returns:
        List of Rule objects with conservative thresholds
    """
    return [
        # Very strict thresholds
        create_very_large_odd_hours_transaction_rule(threshold=10000.0, weight=5.0),
        create_international_transfer_odd_hours_rule(min_amount=2500.0, weight=5.0),
        create_weekend_odd_hours_transaction_rule(min_amount=2500.0, weight=4.5),
        create_deviates_from_pattern_rule(weight=4.0),
        create_outgoing_transfer_odd_hours_rule(min_amount=500.0, weight=4.0),
        create_large_odd_hours_transaction_rule(threshold=2500.0, weight=3.5),
        create_new_counterparty_odd_hours_rule(weight=3.5),
        create_repeated_odd_hours_transactions_rule(min_count=2, min_total_amount=5000.0, weight=3.5),
        create_unusual_hour_transaction_rule(weight=3.0),
        create_first_odd_hours_transaction_rule(weight=2.5),
        create_odd_hours_transaction_rule(weight=2.0),
    ]


def get_balanced_odd_hours_rules() -> List[Rule]:
    """
    Get a balanced rule set for standard fraud detection.

    Balances fraud detection with false positive management.
    Uses default thresholds from settings.

    Returns:
        List of Rule objects with balanced thresholds
    """
    return initialize_odd_hours_rules()


def get_permissive_odd_hours_rules() -> List[Rule]:
    """
    Get a permissive rule set for low false-positive tolerance.

    Only triggers on high-confidence fraud patterns with elevated thresholds.

    Returns:
        List of Rule objects with permissive thresholds
    """
    return [
        # Higher thresholds, critical indicators only
        create_very_large_odd_hours_transaction_rule(threshold=50000.0, weight=5.0),
        create_international_transfer_odd_hours_rule(min_amount=25000.0, weight=5.0),
        create_weekend_odd_hours_transaction_rule(min_amount=25000.0, weight=4.5),
        create_deviates_from_pattern_rule(weight=4.0),
        create_outgoing_transfer_odd_hours_rule(min_amount=10000.0, weight=4.0),
        create_repeated_odd_hours_transactions_rule(min_count=5, min_total_amount=50000.0, weight=3.5),
    ]
