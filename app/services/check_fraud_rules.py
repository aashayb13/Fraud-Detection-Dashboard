"""
Check Fraud Detection Rules

This module contains fraud detection rules specific to check transactions,
including detection of duplicate check deposits and other check-related fraud patterns.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
from sqlalchemy.orm import Session

from .rules_engine import Rule


# Constants for duplicate check detection
DEFAULT_DUPLICATE_CHECK_WINDOW_DAYS = 90  # Look back 90 days for duplicates
DEFAULT_DUPLICATE_WEIGHT = 4.0  # High severity - clear fraud indicator


def create_duplicate_check_rule(
    db: Session,
    lookback_days: int = DEFAULT_DUPLICATE_CHECK_WINDOW_DAYS,
    weight: float = DEFAULT_DUPLICATE_WEIGHT
) -> Rule:
    """
    Detect duplicate check deposits - same check deposited multiple times.

    This rule identifies when a single physical check is deposited more than once,
    which is a common type of check fraud. The detection is based on matching:
    - Check number
    - Check amount
    - Source routing/account information (if available)

    Args:
        db: Database session for querying historical transactions
        lookback_days: Number of days to look back for duplicate checks (default: 90)
        weight: Risk weight for this rule (default: 4.0 - high severity)

    Returns:
        Rule object configured for duplicate check detection
    """

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Check if this transaction is a duplicate check deposit.

        Args:
            transaction: Current transaction being evaluated
            context: Context containing check history and patterns

        Returns:
            True if duplicate check detected, False otherwise
        """
        # Only evaluate check deposits
        if not is_check_deposit(transaction):
            return False

        # Extract check information from transaction
        check_info = extract_check_info(transaction)
        if not check_info:
            return False  # No check info available, cannot detect duplicates

        # Get duplicate check information from context
        duplicate_checks = context.get("duplicate_checks", [])

        if duplicate_checks:
            # Store detailed information about duplicates in context
            context["duplicate_check_details"] = {
                "current_check_number": check_info.get("check_number"),
                "current_amount": check_info.get("amount"),
                "current_account": transaction.get("account_id"),
                "duplicate_count": len(duplicate_checks),
                "previous_deposits": duplicate_checks,
                "fraud_type": "duplicate_check_deposit"
            }
            return True

        return False

    return Rule(
        name="duplicate_check_deposit",
        description=f"Same check deposited multiple times within {lookback_days} days",
        condition_func=condition,
        weight=weight
    )


def create_rapid_check_sequence_rule(
    min_checks_per_hour: int = 5,
    min_total_amount: float = 5000.0,
    weight: float = 3.0
) -> Rule:
    """
    Detect rapid sequences of check deposits which may indicate fraud.

    Fraudsters sometimes deposit multiple fraudulent checks in quick succession
    before the checks can be verified and bounced.

    Args:
        min_checks_per_hour: Minimum number of checks to trigger (default: 5)
        min_total_amount: Minimum total amount to trigger (default: $5,000)
        weight: Risk weight for this rule (default: 3.0)

    Returns:
        Rule object configured for rapid check sequence detection
    """

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if account shows rapid check deposit sequence."""
        if not is_check_deposit(transaction):
            return False

        # Get rapid check metrics from context
        checks_last_hour = context.get("check_count_1h", 0)
        amount_last_hour = context.get("check_amount_1h", 0.0)

        if checks_last_hour >= min_checks_per_hour and amount_last_hour >= min_total_amount:
            context["rapid_check_details"] = {
                "checks_per_hour": checks_last_hour,
                "total_amount": amount_last_hour,
                "fraud_type": "rapid_check_sequence"
            }
            return True

        return False

    return Rule(
        name="rapid_check_sequence",
        description=f"Rapid check deposits: {min_checks_per_hour}+ checks in 1 hour totaling ${min_total_amount:,.2f}+",
        condition_func=condition,
        weight=weight
    )


def create_check_amount_mismatch_rule(
    max_deviation_percent: float = 5.0,
    weight: float = 3.5
) -> Rule:
    """
    Detect checks where the deposited amount differs from historical deposits of the same check.

    This can indicate check alteration fraud where the amount is changed.

    Args:
        max_deviation_percent: Maximum allowed deviation percentage (default: 5%)
        weight: Risk weight for this rule (default: 3.5)

    Returns:
        Rule object configured for check amount mismatch detection
    """

    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if the check amount differs from previous deposits of same check."""
        if not is_check_deposit(transaction):
            return False

        check_info = extract_check_info(transaction)
        if not check_info:
            return False

        # Check for amount mismatches in context
        amount_mismatch = context.get("check_amount_mismatch")

        if amount_mismatch:
            context["amount_mismatch_details"] = {
                "current_amount": check_info.get("amount"),
                "previous_amount": amount_mismatch.get("previous_amount"),
                "deviation_percent": amount_mismatch.get("deviation_percent"),
                "fraud_type": "check_amount_alteration"
            }
            return True

        return False

    return Rule(
        name="check_amount_mismatch",
        description=f"Check amount differs by more than {max_deviation_percent}% from previous deposits",
        condition_func=condition,
        weight=weight
    )


# Helper functions

def is_check_deposit(transaction: Dict[str, Any]) -> bool:
    """
    Determine if a transaction is a check deposit.

    Args:
        transaction: Transaction dictionary

    Returns:
        True if transaction is a check deposit, False otherwise
    """
    tx_type = transaction.get("transaction_type", "").upper()
    direction = transaction.get("direction", "").lower()

    # Check deposits are incoming (credit) transactions of type CHECK or DEPOSIT
    return (
        direction == "credit" and
        tx_type in ["CHECK", "CHECK_DEPOSIT", "DEPOSIT", "REMOTE_DEPOSIT", "MOBILE_DEPOSIT"]
    )


def extract_check_info(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract check-specific information from transaction metadata.

    Check information is stored in the tx_metadata field as JSON.
    Expected fields:
    - check_number: The check number
    - check_amount: Amount on the check
    - routing_number: Bank routing number from the check
    - account_number: Account number from the check
    - payee: Payee name on the check
    - drawer: Person/entity who wrote the check
    - check_date: Date on the check

    Args:
        transaction: Transaction dictionary

    Returns:
        Dictionary with check information, or empty dict if not available
    """
    metadata_str = transaction.get("tx_metadata", "{}")

    try:
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
    except json.JSONDecodeError:
        return {}

    check_info = {}

    # Extract check-specific fields
    if "check_number" in metadata:
        check_info["check_number"] = metadata["check_number"]
    if "check_amount" in metadata:
        check_info["amount"] = float(metadata["check_amount"])
    if "routing_number" in metadata:
        check_info["routing_number"] = metadata["routing_number"]
    if "account_number" in metadata:
        check_info["account_number"] = metadata["account_number"]
    if "payee" in metadata:
        check_info["payee"] = metadata["payee"]
    if "drawer" in metadata:
        check_info["drawer"] = metadata["drawer"]
    if "check_date" in metadata:
        check_info["check_date"] = metadata["check_date"]

    return check_info


def initialize_check_fraud_rules(db: Session) -> List[Rule]:
    """
    Initialize all check fraud detection rules.

    This function creates and returns all check-related fraud detection rules
    that should be loaded into the rules engine.

    Args:
        db: Database session

    Returns:
        List of Rule objects for check fraud detection
    """
    rules = [
        # Core duplicate check detection
        create_duplicate_check_rule(db),

        # Additional check fraud patterns
        create_rapid_check_sequence_rule(),
        create_check_amount_mismatch_rule(),
    ]

    return rules
