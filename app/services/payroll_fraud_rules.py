# app/services/payroll_fraud_rules.py
"""
Payroll fraud detection rules for identifying suspicious direct deposit rerouting.

This module implements detection rules for the following scenario:
An employee's pay is redirected to a different bank account after a deceptive
"HR" or account-update request.
"""
from typing import Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.services.rules_engine import Rule
from app.models.database import Employee, AccountChangeHistory
from config.settings import (
    PAYROLL_SUSPICIOUS_CHANGE_WINDOW_DAYS,
    PAYROLL_RAPID_CHANGE_THRESHOLD,
    PAYROLL_RAPID_CHANGE_WINDOW_DAYS,
    PAYROLL_VERIFICATION_REQUIRED_THRESHOLD
)
import json


def is_payroll_transaction(transaction: Dict[str, Any]) -> bool:
    """Check if transaction is a payroll transaction."""
    tx_type = transaction.get("transaction_type", "").lower()
    description = transaction.get("description", "").lower()

    return (
        tx_type in ["ach_credit", "direct_deposit", "payroll"] or
        any(keyword in description for keyword in ["payroll", "salary", "wages", "direct deposit"])
    )


def get_employee_from_transaction(db: Session, transaction: Dict[str, Any]) -> Employee:
    """Get employee record from transaction data."""
    # Try to get employee from tx_metadata first
    tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
    if tx_metadata:
        if isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except:
                tx_metadata = {}

        employee_id = tx_metadata.get("employee_id")
        if employee_id:
            return db.query(Employee).filter(Employee.employee_id == employee_id).first()

    # Fallback: try to match by account
    account_id = transaction.get("account_id")
    if account_id:
        return db.query(Employee).filter(Employee.account_id == account_id).first()

    return None


def create_recent_account_change_rule(db: Session, weight: float = 3.0) -> Rule:
    """
    Detect payroll transactions to recently changed bank accounts.

    This is a high-risk indicator: if an account was changed within 30 days
    before a payroll transaction, it could indicate fraud.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        employee = get_employee_from_transaction(db, transaction)
        if not employee:
            return False

        # Check for recent account changes
        cutoff_date = datetime.utcnow() - timedelta(days=PAYROLL_SUSPICIOUS_CHANGE_WINDOW_DAYS)
        cutoff_iso = cutoff_date.isoformat()

        recent_changes = db.query(AccountChangeHistory).filter(
            AccountChangeHistory.employee_id == employee.employee_id,
            AccountChangeHistory.change_type.in_(["account_number", "routing_number"]),
            AccountChangeHistory.timestamp > cutoff_iso
        ).all()

        if recent_changes:
            # Add to context for other rules to use
            context["recent_account_changes"] = [
                {
                    "change_id": change.change_id,
                    "timestamp": change.timestamp,
                    "change_type": change.change_type,
                    "verified": change.verified,
                    "change_source": change.change_source
                }
                for change in recent_changes
            ]
            return True

        return False

    return Rule(
        name="payroll_recent_account_change",
        description=f"Payroll transaction to bank account changed within {PAYROLL_SUSPICIOUS_CHANGE_WINDOW_DAYS} days",
        condition_func=condition,
        weight=weight
    )


def create_unverified_account_change_rule(db: Session, weight: float = 4.0) -> Rule:
    """
    Detect payroll to accounts with unverified changes.

    Extremely high risk: account changes that weren't properly verified
    through official channels.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        # Check if we already found recent changes
        recent_changes = context.get("recent_account_changes", [])
        if not recent_changes:
            return False

        # Check if any of the recent changes are unverified
        unverified = [c for c in recent_changes if not c.get("verified", False)]

        if unverified:
            context["unverified_changes_count"] = len(unverified)
            return True

        return False

    return Rule(
        name="payroll_unverified_account_change",
        description="Payroll transaction to account with unverified banking information changes",
        condition_func=condition,
        weight=weight
    )


def create_suspicious_change_source_rule(db: Session, weight: float = 3.5) -> Rule:
    """
    Detect account changes from suspicious sources.

    Email/phone requests are more susceptible to social engineering
    compared to authenticated employee portal changes.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        recent_changes = context.get("recent_account_changes", [])
        if not recent_changes:
            return False

        suspicious_sources = ["email_request", "phone_request"]
        suspicious = [
            c for c in recent_changes
            if c.get("change_source") in suspicious_sources
        ]

        if suspicious:
            context["suspicious_change_sources"] = [c.get("change_source") for c in suspicious]
            return True

        return False

    return Rule(
        name="payroll_suspicious_change_source",
        description="Account changed via email/phone request rather than secure portal",
        condition_func=condition,
        weight=weight
    )


def create_rapid_account_changes_rule(db: Session, weight: float = 3.0) -> Rule:
    """
    Detect multiple account changes in a short period.

    Multiple changes could indicate an attacker trying different approaches
    or testing the system.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        employee = get_employee_from_transaction(db, transaction)
        if not employee:
            return False

        # Check for multiple changes in the window
        cutoff_date = datetime.utcnow() - timedelta(days=PAYROLL_RAPID_CHANGE_WINDOW_DAYS)
        cutoff_iso = cutoff_date.isoformat()

        change_count = db.query(AccountChangeHistory).filter(
            AccountChangeHistory.employee_id == employee.employee_id,
            AccountChangeHistory.change_type.in_(["account_number", "routing_number", "bank_name"]),
            AccountChangeHistory.timestamp > cutoff_iso
        ).count()

        if change_count >= PAYROLL_RAPID_CHANGE_THRESHOLD:
            context["rapid_changes_count"] = change_count
            context["rapid_changes_window_days"] = PAYROLL_RAPID_CHANGE_WINDOW_DAYS
            return True

        return False

    return Rule(
        name="payroll_rapid_account_changes",
        description=f"Multiple account changes ({PAYROLL_RAPID_CHANGE_THRESHOLD}+) within {PAYROLL_RAPID_CHANGE_WINDOW_DAYS} days",
        condition_func=condition,
        weight=weight
    )


def create_first_payroll_after_change_rule(db: Session, weight: float = 2.5) -> Rule:
    """
    Detect if this is the first payroll after an account change.

    First payroll to a new account deserves extra scrutiny.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        employee = get_employee_from_transaction(db, transaction)
        if not employee:
            return False

        # Check if there were any account changes since last payroll
        if not employee.last_payroll_date:
            # No previous payroll recorded, could be new employee or first in system
            return False

        last_payroll = datetime.fromisoformat(employee.last_payroll_date)

        changes_since_payroll = db.query(AccountChangeHistory).filter(
            AccountChangeHistory.employee_id == employee.employee_id,
            AccountChangeHistory.change_type.in_(["account_number", "routing_number"]),
            AccountChangeHistory.timestamp > last_payroll.isoformat()
        ).first()

        if changes_since_payroll:
            context["first_payroll_after_change"] = True
            context["last_payroll_date"] = employee.last_payroll_date
            return True

        return False

    return Rule(
        name="payroll_first_after_account_change",
        description="First payroll transaction after account information change",
        condition_func=condition,
        weight=weight
    )


def create_high_value_payroll_rule(weight: float = 2.0) -> Rule:
    """
    Flag high-value payroll transactions for additional scrutiny.

    Large payroll amounts combined with other risk factors warrant review.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        amount = transaction.get("amount", 0)
        if amount >= PAYROLL_VERIFICATION_REQUIRED_THRESHOLD:
            context["high_value_payroll"] = True
            context["payroll_amount"] = amount
            return True

        return False

    return Rule(
        name="payroll_high_value",
        description=f"High-value payroll transaction (>= ${PAYROLL_VERIFICATION_REQUIRED_THRESHOLD:,.2f})",
        condition_func=condition,
        weight=weight
    )


def create_weekend_account_change_rule(db: Session, weight: float = 2.0) -> Rule:
    """
    Detect account changes made during weekends.

    Weekend changes are unusual for legitimate HR processes and may indicate
    fraud attempts when HR staff is unavailable to verify.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        recent_changes = context.get("recent_account_changes", [])
        if not recent_changes:
            return False

        weekend_changes = []
        for change in recent_changes:
            change_time = datetime.fromisoformat(change["timestamp"])
            # Saturday = 5, Sunday = 6
            if change_time.weekday() >= 5:
                weekend_changes.append(change)

        if weekend_changes:
            context["weekend_changes_count"] = len(weekend_changes)
            return True

        return False

    return Rule(
        name="payroll_weekend_account_change",
        description="Account change made during weekend before payroll",
        condition_func=condition,
        weight=weight
    )


def create_off_hours_account_change_rule(db: Session, weight: float = 1.5) -> Rule:
    """
    Detect account changes made during off-hours (night time).

    Changes made between 10 PM and 6 AM are unusual for legitimate requests.
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        if not is_payroll_transaction(transaction):
            return False

        recent_changes = context.get("recent_account_changes", [])
        if not recent_changes:
            return False

        off_hours_changes = []
        for change in recent_changes:
            change_time = datetime.fromisoformat(change["timestamp"])
            hour = change_time.hour
            # Off hours: 22:00 - 06:00
            if hour >= 22 or hour < 6:
                off_hours_changes.append(change)

        if off_hours_changes:
            context["off_hours_changes_count"] = len(off_hours_changes)
            return True

        return False

    return Rule(
        name="payroll_off_hours_account_change",
        description="Account change made during off-hours (10 PM - 6 AM)",
        condition_func=condition,
        weight=weight
    )


def initialize_payroll_fraud_rules(db: Session) -> List[Rule]:
    """
    Initialize all payroll fraud detection rules.

    Returns:
        List of configured Rule objects for payroll fraud detection
    """
    return [
        create_recent_account_change_rule(db),
        create_unverified_account_change_rule(db),
        create_suspicious_change_source_rule(db),
        create_rapid_account_changes_rule(db),
        create_first_payroll_after_change_rule(db),
        create_high_value_payroll_rule(),
        create_weekend_account_change_rule(db),
        create_off_hours_account_change_rule(db),
    ]
