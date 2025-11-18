# app/services/beneficiary_fraud_rules.py
"""
Comprehensive beneficiary fraud detection rules.

This module combines detection for two major fraud scenarios:

1. RAPID BENEFICIARY ADDITION FRAUD
   - Compromised administrator accounts rapidly adding many new beneficiaries
   - Followed by fraudulent payments to those beneficiaries
   - Common pattern in scripted/automated fraud attacks

2. VENDOR IMPERSONATION / BEC ATTACKS
   - Supplier's bank details are changed via impersonation
   - Payment is sent to fraudulent account shortly after change
   - One of the most common Business Email Compromise attack patterns
"""

from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.models.database import Beneficiary, BeneficiaryChangeHistory, Transaction
from app.services.rules_engine import Rule
from config.settings import (
    # Rapid addition settings
    BENEFICIARY_RAPID_ADDITION_THRESHOLD,
    BENEFICIARY_RAPID_ADDITION_WINDOW_HOURS,
    BENEFICIARY_BULK_ADDITION_THRESHOLD,
    BENEFICIARY_BULK_ADDITION_WINDOW_HOURS,
    BENEFICIARY_RECENT_ADDITION_HOURS,
    BENEFICIARY_NEW_BENEFICIARY_PAYMENT_RATIO,
    # Vendor impersonation settings
    BENEFICIARY_SAME_DAY_PAYMENT_HOURS,
    BENEFICIARY_CRITICAL_CHANGE_WINDOW_DAYS,
    BENEFICIARY_SUSPICIOUS_CHANGE_WINDOW_DAYS,
    BENEFICIARY_RAPID_CHANGE_THRESHOLD,
    BENEFICIARY_RAPID_CHANGE_WINDOW_DAYS,
    BENEFICIARY_HIGH_VALUE_THRESHOLD,
    BENEFICIARY_NEW_VENDOR_DAYS
)
import json


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_beneficiary_payment(transaction: Dict[str, Any]) -> bool:
    """Check if transaction is a payment to a beneficiary/vendor."""
    tx_type = transaction.get("transaction_type", "").lower()
    description = transaction.get("description", "").lower()
    direction = transaction.get("direction", "").lower()

    # Must be an outgoing payment
    if direction != "debit":
        return False

    return (
        tx_type in ["ach_debit", "wire_transfer", "payment", "vendor_payment", "supplier_payment"] or
        any(keyword in description for keyword in ["payment", "invoice", "vendor", "supplier", "contractor"])
    )


def get_beneficiary_from_transaction(db: Session, transaction: Dict[str, Any]) -> Beneficiary:
    """Get beneficiary record from transaction data."""
    # Try to get beneficiary from tx_metadata first
    tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
    if tx_metadata:
        if isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except:
                tx_metadata = {}

        beneficiary_id = tx_metadata.get("beneficiary_id")
        if beneficiary_id:
            return db.query(Beneficiary).filter(Beneficiary.beneficiary_id == beneficiary_id).first()

    # Fallback: try to match by counterparty_id
    counterparty_id = transaction.get("counterparty_id")
    if counterparty_id:
        # Try exact match first
        beneficiary = db.query(Beneficiary).filter(Beneficiary.beneficiary_id == counterparty_id).first()
        if beneficiary:
            return beneficiary
        # Try counterparty_id field
        beneficiary = db.query(Beneficiary).filter(Beneficiary.counterparty_id == counterparty_id).first()
        if beneficiary:
            return beneficiary

    return None


# =============================================================================
# RAPID ADDITION FRAUD DETECTION RULES
# =============================================================================

def create_rapid_beneficiary_addition_rule(
    db: Session,
    threshold: int = BENEFICIARY_RAPID_ADDITION_THRESHOLD,
    window_hours: int = BENEFICIARY_RAPID_ADDITION_WINDOW_HOURS,
    weight: float = 2.5
) -> Rule:
    """
    Detect when many beneficiaries are added rapidly to an account.

    This pattern suggests:
    - Compromised administrator account
    - Scripted/automated fraud onboarding
    - Preparation for fraudulent payments

    Args:
        db: Database session
        threshold: Minimum number of beneficiaries to trigger alert
        window_hours: Time window to check for rapid additions
        weight: Rule weight for risk scoring

    Returns:
        Rule object for rapid beneficiary addition detection
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Check context for rapid beneficiary additions
        recent_beneficiaries_count = context.get(f"beneficiaries_added_{window_hours}h", 0)

        if recent_beneficiaries_count >= threshold:
            # Add detailed information to context
            context["rapid_beneficiary_addition_detected"] = True
            context["rapid_beneficiary_count"] = recent_beneficiaries_count
            context["rapid_beneficiary_window_hours"] = window_hours
            return True

        return False

    return Rule(
        name=f"rapid_beneficiary_addition_{window_hours}h",
        description=f"{threshold}+ beneficiaries added within {window_hours} hours",
        condition_func=condition,
        weight=weight
    )


def create_bulk_beneficiary_addition_rule(
    db: Session,
    threshold: int = BENEFICIARY_BULK_ADDITION_THRESHOLD,
    window_hours: int = BENEFICIARY_BULK_ADDITION_WINDOW_HOURS,
    weight: float = 3.0
) -> Rule:
    """
    Detect bulk/scripted beneficiary additions (higher threshold, longer window).

    This pattern indicates large-scale scripted fraud preparation.

    Args:
        db: Database session
        threshold: Minimum number of beneficiaries for bulk detection
        window_hours: Time window to check for bulk additions
        weight: Rule weight for risk scoring

    Returns:
        Rule object for bulk beneficiary addition detection
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Check context for bulk beneficiary additions
        recent_beneficiaries_count = context.get(f"beneficiaries_added_{window_hours}h", 0)

        if recent_beneficiaries_count >= threshold:
            context["bulk_beneficiary_addition_detected"] = True
            context["bulk_beneficiary_count"] = recent_beneficiaries_count
            context["bulk_beneficiary_window_hours"] = window_hours
            return True

        return False

    return Rule(
        name=f"bulk_beneficiary_addition_{window_hours}h",
        description=f"{threshold}+ beneficiaries added (bulk/scripted) within {window_hours} hours",
        condition_func=condition,
        weight=weight
    )


def create_payment_to_new_beneficiary_rule(
    db: Session,
    recent_hours: int = BENEFICIARY_RECENT_ADDITION_HOURS,
    weight: float = 2.0
) -> Rule:
    """
    Detect payments to recently added beneficiaries.

    When combined with rapid addition patterns, this confirms the fraud scenario.

    Args:
        db: Database session
        recent_hours: Hours to consider beneficiary as "newly added"
        weight: Rule weight for risk scoring

    Returns:
        Rule object for detecting payments to new beneficiaries
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing transactions (payments)
        if transaction.get("direction") != "debit":
            return False

        # Check if paying to a recently added beneficiary
        is_new_beneficiary = context.get("is_new_beneficiary", False)
        beneficiary_age_hours = context.get("beneficiary_age_hours")

        if is_new_beneficiary and beneficiary_age_hours is not None and beneficiary_age_hours <= recent_hours:
            context["payment_to_new_beneficiary_detected"] = True
            context["beneficiary_age_hours"] = beneficiary_age_hours
            return True

        return False

    return Rule(
        name=f"payment_to_new_beneficiary_{recent_hours}h",
        description=f"Payment to beneficiary added within {recent_hours} hours",
        condition_func=condition,
        weight=weight
    )


def create_high_new_beneficiary_payment_ratio_rule(
    db: Session,
    min_ratio: float = BENEFICIARY_NEW_BENEFICIARY_PAYMENT_RATIO,
    window_hours: int = 24,
    weight: float = 2.5
) -> Rule:
    """
    Detect when most recent payments go to newly added beneficiaries.

    High ratio of payments to new beneficiaries suggests coordinated fraud.

    Args:
        db: Database session
        min_ratio: Minimum ratio of payments to new beneficiaries (0.0-1.0)
        window_hours: Time window to analyze payment patterns
        weight: Rule weight for risk scoring

    Returns:
        Rule object for detecting high new beneficiary payment ratio
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing transactions
        if transaction.get("direction") != "debit":
            return False

        # Check the ratio of payments to new beneficiaries
        new_beneficiary_payment_ratio = context.get(f"new_beneficiary_payment_ratio_{window_hours}h", 0.0)
        new_beneficiary_payment_count = context.get(f"new_beneficiary_payment_count_{window_hours}h", 0)

        if new_beneficiary_payment_ratio >= min_ratio and new_beneficiary_payment_count >= 3:
            context["high_new_beneficiary_ratio_detected"] = True
            context["new_beneficiary_payment_ratio"] = new_beneficiary_payment_ratio
            context["new_beneficiary_payment_count"] = new_beneficiary_payment_count
            return True

        return False

    return Rule(
        name=f"high_new_beneficiary_payment_ratio_{window_hours}h",
        description=f"{int(min_ratio*100)}%+ of payments to newly added beneficiaries",
        condition_func=condition,
        weight=weight
    )


def create_same_source_bulk_addition_rule(
    db: Session,
    min_count: int = 5,
    window_hours: int = 24,
    weight: float = 3.5
) -> Rule:
    """
    Detect multiple beneficiaries added from the same source/IP in short time.

    Strong indicator of scripted/automated fraud when many beneficiaries
    are added from the same IP address or user.

    Args:
        db: Database session
        min_count: Minimum number of beneficiaries from same source
        window_hours: Time window to check
        weight: Rule weight for risk scoring

    Returns:
        Rule object for detecting same-source bulk additions
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Check for beneficiaries added from same IP
        same_ip_count = context.get(f"beneficiaries_same_ip_{window_hours}h", 0)
        same_user_count = context.get(f"beneficiaries_same_user_{window_hours}h", 0)
        same_source_ip = context.get("same_source_ip")
        same_source_user = context.get("same_source_user")

        if same_ip_count >= min_count or same_user_count >= min_count:
            context["same_source_bulk_addition_detected"] = True
            context["same_source_beneficiary_count"] = max(same_ip_count, same_user_count)
            if same_ip_count >= min_count:
                context["same_source_type"] = "ip_address"
                context["same_source_value"] = same_source_ip
            else:
                context["same_source_type"] = "user"
                context["same_source_value"] = same_source_user
            return True

        return False

    return Rule(
        name=f"same_source_bulk_addition_{window_hours}h",
        description=f"{min_count}+ beneficiaries from same IP/user within {window_hours} hours",
        condition_func=condition,
        weight=weight
    )


def create_unverified_beneficiary_payment_rule(
    db: Session,
    weight: float = 1.5
) -> Rule:
    """
    Detect payments to unverified beneficiaries.

    Legitimate systems typically require beneficiary verification before payments.
    Skipping verification suggests fraudulent intent.

    Args:
        db: Database session
        weight: Rule weight for risk scoring

    Returns:
        Rule object for detecting payments to unverified beneficiaries
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing transactions
        if transaction.get("direction") != "debit":
            return False

        # Check if beneficiary is unverified
        is_beneficiary_verified = context.get("is_beneficiary_verified", True)

        if not is_beneficiary_verified:
            context["unverified_beneficiary_payment_detected"] = True
            return True

        return False

    return Rule(
        name="payment_to_unverified_beneficiary",
        description="Payment to unverified beneficiary",
        condition_func=condition,
        weight=weight
    )


# =============================================================================
# VENDOR IMPERSONATION / BEC DETECTION RULES
# =============================================================================

def create_same_day_payment_after_change_rule(db: Session, weight: float = 5.0) -> Rule:
    """
    Detect payments made on the same day as beneficiary account change.

    This is a CRITICAL risk indicator - the most common pattern in vendor impersonation fraud.
    Attackers often request changes and immediate payment to minimize detection window.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        beneficiary = get_beneficiary_from_transaction(db, transaction)
        if not beneficiary:
            return False

        # Check for changes within the same day (24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=BENEFICIARY_SAME_DAY_PAYMENT_HOURS)
        cutoff_iso = cutoff_time.isoformat()

        same_day_changes = db.query(BeneficiaryChangeHistory).filter(
            BeneficiaryChangeHistory.beneficiary_id == beneficiary.beneficiary_id,
            BeneficiaryChangeHistory.change_type.in_(["account_number", "routing_number", "bank_name"]),
            BeneficiaryChangeHistory.timestamp > cutoff_iso
        ).all()

        if same_day_changes:
            hours_since_change = []
            for change in same_day_changes:
                change_time = datetime.fromisoformat(change.timestamp)
                hours_diff = (datetime.utcnow() - change_time).total_seconds() / 3600
                hours_since_change.append(hours_diff)

            context["same_day_beneficiary_changes"] = [
                {
                    "change_id": change.change_id,
                    "timestamp": change.timestamp,
                    "change_type": change.change_type,
                    "verified": change.verified,
                    "change_source": change.change_source,
                    "hours_ago": hours_diff
                }
                for change, hours_diff in zip(same_day_changes, hours_since_change)
            ]
            context["min_hours_since_change"] = min(hours_since_change)
            return True

        return False

    return Rule(
        name="beneficiary_same_day_payment",
        description=f"Payment to beneficiary within {BENEFICIARY_SAME_DAY_PAYMENT_HOURS} hours of account change (CRITICAL)",
        condition_func=condition,
        weight=weight
    )


def create_recent_account_change_payment_rule(db: Session, weight: float = 4.0) -> Rule:
    """
    Detect payments to beneficiaries with recent account changes.

    Payments within 7 days of an account change are high-risk, as attackers
    typically act quickly after compromising vendor communications.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        beneficiary = get_beneficiary_from_transaction(db, transaction)
        if not beneficiary:
            return False

        # Check for recent changes within critical window
        cutoff_date = datetime.utcnow() - timedelta(days=BENEFICIARY_CRITICAL_CHANGE_WINDOW_DAYS)
        cutoff_iso = cutoff_date.isoformat()

        recent_changes = db.query(BeneficiaryChangeHistory).filter(
            BeneficiaryChangeHistory.beneficiary_id == beneficiary.beneficiary_id,
            BeneficiaryChangeHistory.change_type.in_(["account_number", "routing_number", "bank_name"]),
            BeneficiaryChangeHistory.timestamp > cutoff_iso
        ).all()

        if recent_changes:
            context["recent_beneficiary_changes"] = [
                {
                    "change_id": change.change_id,
                    "timestamp": change.timestamp,
                    "change_type": change.change_type,
                    "verified": change.verified,
                    "change_source": change.change_source,
                    "requestor_name": change.requestor_name,
                    "requestor_email": change.requestor_email
                }
                for change in recent_changes
            ]
            return True

        return False

    return Rule(
        name="beneficiary_recent_account_change",
        description=f"Payment to beneficiary with account changed within {BENEFICIARY_CRITICAL_CHANGE_WINDOW_DAYS} days",
        condition_func=condition,
        weight=weight
    )


def create_unverified_beneficiary_change_rule(db: Session, weight: float = 4.5) -> Rule:
    """
    Detect payments to beneficiaries with unverified account changes.

    Account changes that haven't been properly verified (callback, in-person, etc.)
    are extremely high risk for fraud.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        # Check if we already found recent changes
        recent_changes = context.get("recent_beneficiary_changes", [])
        same_day_changes = context.get("same_day_beneficiary_changes", [])

        all_changes = recent_changes + same_day_changes
        if not all_changes:
            return False

        # Check if any recent changes are unverified
        unverified = [c for c in all_changes if not c.get("verified", False)]

        if unverified:
            context["unverified_beneficiary_changes_count"] = len(unverified)
            context["unverified_beneficiary_changes"] = unverified
            return True

        return False

    return Rule(
        name="beneficiary_unverified_account_change",
        description="Payment to beneficiary with unverified banking information changes",
        condition_func=condition,
        weight=weight
    )


def create_suspicious_change_source_rule(db: Session, weight: float = 3.5) -> Rule:
    """
    Detect beneficiary account changes from suspicious sources.

    Email and phone requests are the primary vectors for vendor impersonation.
    Legitimate changes typically come through authenticated portals or ERP systems.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        recent_changes = context.get("recent_beneficiary_changes", [])
        same_day_changes = context.get("same_day_beneficiary_changes", [])

        all_changes = recent_changes + same_day_changes
        if not all_changes:
            return False

        # Email and phone requests are high-risk for BEC attacks
        suspicious_sources = ["email_request", "phone_request", "fax"]
        suspicious = [
            c for c in all_changes
            if c.get("change_source") in suspicious_sources
        ]

        if suspicious:
            context["suspicious_beneficiary_change_sources"] = [
                c.get("change_source") for c in suspicious
            ]
            context["suspicious_change_details"] = suspicious
            return True

        return False

    return Rule(
        name="beneficiary_suspicious_change_source",
        description="Beneficiary account changed via email/phone/fax request (BEC risk)",
        condition_func=condition,
        weight=weight
    )


def create_first_payment_after_change_rule(db: Session, weight: float = 3.0) -> Rule:
    """
    Detect if this is the first payment after a beneficiary account change.

    The first payment to a new account is when fraud is most likely to succeed,
    before the legitimate vendor realizes they haven't been paid.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        beneficiary = get_beneficiary_from_transaction(db, transaction)
        if not beneficiary:
            return False

        # Check if there were any account changes since last payment
        if not beneficiary.last_payment_date:
            # No previous payment - could be new vendor
            return False

        last_payment = datetime.fromisoformat(beneficiary.last_payment_date)

        changes_since_payment = db.query(BeneficiaryChangeHistory).filter(
            BeneficiaryChangeHistory.beneficiary_id == beneficiary.beneficiary_id,
            BeneficiaryChangeHistory.change_type.in_(["account_number", "routing_number"]),
            BeneficiaryChangeHistory.timestamp > last_payment.isoformat()
        ).first()

        if changes_since_payment:
            context["first_payment_after_beneficiary_change"] = True
            context["last_payment_date"] = beneficiary.last_payment_date
            context["days_since_last_payment"] = (datetime.utcnow() - last_payment).days
            return True

        return False

    return Rule(
        name="beneficiary_first_payment_after_change",
        description="First payment to beneficiary after account information change",
        condition_func=condition,
        weight=weight
    )


def create_high_value_payment_rule(weight: float = 2.5) -> Rule:
    """
    Flag high-value payments to beneficiaries for additional scrutiny.

    Large payments combined with recent account changes are prime targets for fraud.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        amount = transaction.get("amount", 0)
        if amount >= BENEFICIARY_HIGH_VALUE_THRESHOLD:
            context["high_value_beneficiary_payment"] = True
            context["payment_amount"] = amount
            return True

        return False

    return Rule(
        name="beneficiary_high_value_payment",
        description=f"High-value payment to beneficiary (>= ${BENEFICIARY_HIGH_VALUE_THRESHOLD:,.2f})",
        condition_func=condition,
        weight=weight
    )


def create_rapid_account_changes_rule(db: Session, weight: float = 3.5) -> Rule:
    """
    Detect multiple beneficiary account changes in a short period.

    Multiple changes could indicate:
    - Repeated compromise attempts
    - Testing by attackers
    - Confusion that fraudsters can exploit
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        beneficiary = get_beneficiary_from_transaction(db, transaction)
        if not beneficiary:
            return False

        # Check for multiple changes in the window
        cutoff_date = datetime.utcnow() - timedelta(days=BENEFICIARY_RAPID_CHANGE_WINDOW_DAYS)
        cutoff_iso = cutoff_date.isoformat()

        change_count = db.query(BeneficiaryChangeHistory).filter(
            BeneficiaryChangeHistory.beneficiary_id == beneficiary.beneficiary_id,
            BeneficiaryChangeHistory.change_type.in_(["account_number", "routing_number", "bank_name"]),
            BeneficiaryChangeHistory.timestamp > cutoff_iso
        ).count()

        if change_count >= BENEFICIARY_RAPID_CHANGE_THRESHOLD:
            context["rapid_beneficiary_changes_count"] = change_count
            context["rapid_changes_window_days"] = BENEFICIARY_RAPID_CHANGE_WINDOW_DAYS
            return True

        return False

    return Rule(
        name="beneficiary_rapid_account_changes",
        description=f"Multiple beneficiary account changes ({BENEFICIARY_RAPID_CHANGE_THRESHOLD}+) within {BENEFICIARY_RAPID_CHANGE_WINDOW_DAYS} days",
        condition_func=condition,
        weight=weight
    )


def create_new_beneficiary_payment_rule(db: Session, weight: float = 2.0) -> Rule:
    """
    Detect payments to newly registered beneficiaries.

    New vendors with no payment history deserve extra scrutiny,
    especially if combined with other risk factors.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        beneficiary = get_beneficiary_from_transaction(db, transaction)
        if not beneficiary:
            return False

        # Check if beneficiary is newly registered
        registration_date = datetime.fromisoformat(beneficiary.registration_date)
        days_since_registration = (datetime.utcnow() - registration_date).days

        if days_since_registration <= BENEFICIARY_NEW_VENDOR_DAYS:
            context["new_beneficiary"] = True
            context["days_since_registration"] = days_since_registration
            context["total_payments_to_beneficiary"] = beneficiary.total_payments_received
            return True

        return False

    return Rule(
        name="beneficiary_new_vendor_payment",
        description=f"Payment to newly registered beneficiary (within {BENEFICIARY_NEW_VENDOR_DAYS} days)",
        condition_func=condition,
        weight=weight
    )


def create_weekend_change_rule(db: Session, weight: float = 2.0) -> Rule:
    """
    Detect beneficiary account changes made during weekends.

    Weekend changes are unusual for legitimate business operations and may indicate
    fraud attempts when AP staff is unavailable to verify.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        recent_changes = context.get("recent_beneficiary_changes", [])
        same_day_changes = context.get("same_day_beneficiary_changes", [])

        all_changes = recent_changes + same_day_changes
        if not all_changes:
            return False

        weekend_changes = []
        for change in all_changes:
            change_time = datetime.fromisoformat(change["timestamp"])
            # Saturday = 5, Sunday = 6
            if change_time.weekday() >= 5:
                weekend_changes.append(change)

        if weekend_changes:
            context["weekend_beneficiary_changes_count"] = len(weekend_changes)
            return True

        return False

    return Rule(
        name="beneficiary_weekend_account_change",
        description="Beneficiary account change made during weekend before payment",
        condition_func=condition,
        weight=weight
    )


def create_off_hours_change_rule(db: Session, weight: float = 1.5) -> Rule:
    """
    Detect beneficiary account changes made during off-hours.

    Changes made between 10 PM and 6 AM are unusual for legitimate business requests.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_beneficiary_payment(transaction):
            return False

        recent_changes = context.get("recent_beneficiary_changes", [])
        same_day_changes = context.get("same_day_beneficiary_changes", [])

        all_changes = recent_changes + same_day_changes
        if not all_changes:
            return False

        off_hours_changes = []
        for change in all_changes:
            change_time = datetime.fromisoformat(change["timestamp"])
            hour = change_time.hour
            # Off hours: 22:00 - 06:00
            if hour >= 22 or hour < 6:
                off_hours_changes.append(change)

        if off_hours_changes:
            context["off_hours_beneficiary_changes_count"] = len(off_hours_changes)
            return True

        return False

    return Rule(
        name="beneficiary_off_hours_account_change",
        description="Beneficiary account change made during off-hours (10 PM - 6 AM)",
        condition_func=condition,
        weight=weight
    )


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_beneficiary_fraud_rules(db: Session) -> List[Rule]:
    """
    Initialize all beneficiary fraud detection rules.

    Combines rules for both fraud scenarios:
    1. Rapid beneficiary addition fraud (compromised admin)
    2. Vendor impersonation/BEC attacks (changed bank details)

    Returns:
        List of configured Rule objects for comprehensive beneficiary fraud detection
    """
    return [
        # Rapid Addition Fraud Detection
        create_rapid_beneficiary_addition_rule(db),
        create_bulk_beneficiary_addition_rule(db),
        create_payment_to_new_beneficiary_rule(db),
        create_high_new_beneficiary_payment_ratio_rule(db),
        create_same_source_bulk_addition_rule(db),
        create_unverified_beneficiary_payment_rule(db),

        # Vendor Impersonation / BEC Detection
        create_same_day_payment_after_change_rule(db),
        create_recent_account_change_payment_rule(db),
        create_unverified_beneficiary_change_rule(db),
        create_suspicious_change_source_rule(db),
        create_first_payment_after_change_rule(db),
        create_high_value_payment_rule(),
        create_rapid_account_changes_rule(db),
        create_new_beneficiary_payment_rule(db),
        create_weekend_change_rule(db),
        create_off_hours_change_rule(db),
    ]
