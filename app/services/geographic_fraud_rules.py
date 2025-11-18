# app/services/geographic_fraud_rules.py
"""
Geographic fraud detection rules for identifying suspicious payment routing.

This module implements detection rules for the following scenario:
A vendor who was always paid domestically is suddenly paid through an overseas
or high-risk bank account.
"""
from typing import Dict, Any, List, Set, Optional
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from app.services.rules_engine import Rule
from app.models.database import Transaction
from config.settings import (
    GEOGRAPHIC_LOOKBACK_DAYS,
    MIN_HISTORICAL_TRANSACTIONS
)


# OFAC and high-risk countries based on international sanctions,
# money laundering concerns, and terrorism financing risks
HIGH_RISK_COUNTRIES: Set[str] = {
    # OFAC Sanctioned Countries
    "IR",  # Iran
    "KP",  # North Korea
    "SY",  # Syria
    "CU",  # Cuba

    # High-risk jurisdictions for money laundering (FATF)
    "AF",  # Afghanistan
    "MM",  # Myanmar (Burma)
    "YE",  # Yemen
    "VE",  # Venezuela
    "ZW",  # Zimbabwe

    # Countries with weak AML/CFT controls
    "HT",  # Haiti
    "SD",  # Sudan
    "SS",  # South Sudan
    "SO",  # Somalia
    "LY",  # Libya

    # Common tax havens and offshore centers
    "VG",  # British Virgin Islands
    "KY",  # Cayman Islands
    "BZ",  # Belize
    "PA",  # Panama
    "LI",  # Liechtenstein
    "MC",  # Monaco
}

# Secondary concern countries (elevated risk but not highest)
ELEVATED_RISK_COUNTRIES: Set[str] = {
    "RU",  # Russia
    "BY",  # Belarus
    "PK",  # Pakistan
    "BD",  # Bangladesh
    "NG",  # Nigeria
    "KE",  # Kenya
    "UA",  # Ukraine (due to ongoing conflict)
    "GH",  # Ghana
    "VN",  # Vietnam
    "ID",  # Indonesia
}


def extract_country_from_transaction(transaction: Dict[str, Any]) -> Optional[str]:
    """
    Extract country code from transaction metadata.

    Args:
        transaction: Transaction data dictionary

    Returns:
        ISO 3166-1 alpha-2 country code (e.g., "US", "GB") or None
    """
    # Try to get from tx_metadata
    tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
    if not tx_metadata:
        return None

    if isinstance(tx_metadata, str):
        try:
            tx_metadata = json.loads(tx_metadata)
        except json.JSONDecodeError:
            return None

    # Check various possible fields
    country = tx_metadata.get("country") or \
              tx_metadata.get("country_code") or \
              tx_metadata.get("bank_country") or \
              tx_metadata.get("destination_country")

    if country:
        # Normalize to uppercase 2-letter code
        return str(country).upper()[:2]

    return None


def get_vendor_payment_history(
    db: Session,
    counterparty_id: str,
    lookback_days: int = None,
    exclude_transaction_id: str = None
) -> List[Dict[str, Any]]:
    """
    Get historical payment data for a vendor/counterparty.

    Args:
        db: Database session
        counterparty_id: Vendor/counterparty identifier
        lookback_days: Number of days to look back (default from config)
        exclude_transaction_id: Transaction ID to exclude (e.g., current transaction)

    Returns:
        List of historical transactions
    """
    if lookback_days is None:
        lookback_days = GEOGRAPHIC_LOOKBACK_DAYS

    cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
    cutoff_iso = cutoff_date.isoformat()

    # Get all outgoing transactions to this counterparty
    query = db.query(Transaction).filter(
        and_(
            Transaction.counterparty_id == counterparty_id,
            Transaction.direction == "debit",  # Outgoing payments
            Transaction.timestamp > cutoff_iso
        )
    )

    # Exclude current transaction if specified
    if exclude_transaction_id:
        query = query.filter(Transaction.transaction_id != exclude_transaction_id)

    historical_txs = query.order_by(Transaction.timestamp.desc()).all()

    return [
        {
            "transaction_id": tx.transaction_id,
            "timestamp": tx.timestamp,
            "amount": tx.amount,
            "transaction_type": tx.transaction_type,
            "tx_metadata": tx.tx_metadata,
            "country": extract_country_from_transaction({
                "tx_metadata": tx.tx_metadata
            })
        }
        for tx in historical_txs
    ]


def analyze_vendor_country_pattern(
    historical_payments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze vendor's historical payment country patterns.

    Args:
        historical_payments: List of historical payment transactions

    Returns:
        Dictionary containing country pattern analysis
    """
    if not historical_payments:
        return {
            "has_history": False,
            "total_payments": 0,
            "countries_used": set(),
            "primary_country": None,
            "is_domestic_only": False,
            "country_distribution": {}
        }

    countries = []
    for payment in historical_payments:
        country = payment.get("country")
        if country:
            countries.append(country)

    if not countries:
        return {
            "has_history": True,
            "total_payments": len(historical_payments),
            "countries_used": set(),
            "primary_country": None,
            "is_domestic_only": False,
            "country_distribution": {},
            "no_country_data": True
        }

    # Count occurrences of each country
    country_counts = {}
    for country in countries:
        country_counts[country] = country_counts.get(country, 0) + 1

    # Determine primary country (most frequent)
    primary_country = max(country_counts, key=country_counts.get)
    unique_countries = set(countries)

    # Check if vendor is domestic-only (assuming "US" is domestic)
    is_domestic_only = unique_countries == {"US"}

    # Calculate distribution percentages
    total_with_country = len(countries)
    country_distribution = {
        country: count / total_with_country
        for country, count in country_counts.items()
    }

    return {
        "has_history": True,
        "total_payments": len(historical_payments),
        "countries_used": unique_countries,
        "primary_country": primary_country,
        "is_domestic_only": is_domestic_only,
        "country_distribution": country_distribution,
        "country_counts": country_counts,
        "no_country_data": False
    }


def create_high_risk_country_rule(db: Session, weight: float = 3.5) -> Rule:
    """
    Detect payments to high-risk countries.

    Flags transactions routed to countries with:
    - Active sanctions (OFAC)
    - High money laundering risk (FATF)
    - Weak AML/CFT controls

    Args:
        db: Database session
        weight: Rule importance weight (default 3.5 - high risk)

    Returns:
        Rule object
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing payments
        if transaction.get("direction") != "debit":
            return False

        country = extract_country_from_transaction(transaction)
        if not country:
            return False

        if country in HIGH_RISK_COUNTRIES:
            context["payment_country"] = country
            context["risk_category"] = "high_risk_sanctioned"
            return True

        if country in ELEVATED_RISK_COUNTRIES:
            context["payment_country"] = country
            context["risk_category"] = "elevated_risk"
            # Lower the weight for elevated risk
            context["adjusted_weight"] = weight * 0.7
            return True

        return False

    return Rule(
        name="payment_to_high_risk_country",
        description="Payment routed to high-risk or sanctioned country",
        condition_func=condition,
        weight=weight
    )


def create_unexpected_country_routing_rule(db: Session, weight: float = 2.5) -> Rule:
    """
    Detect payments to unexpected countries based on vendor history.

    Flags when a vendor who was always paid to one country (typically domestic)
    is suddenly paid through a different country's bank account.

    Args:
        db: Database session
        weight: Rule importance weight (default 2.5)

    Returns:
        Rule object
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing payments
        if transaction.get("direction") != "debit":
            return False

        country = extract_country_from_transaction(transaction)
        if not country:
            return False

        counterparty_id = transaction.get("counterparty_id")
        if not counterparty_id:
            return False

        # Get vendor's payment history (exclude current transaction)
        transaction_id = transaction.get("transaction_id")
        historical_payments = get_vendor_payment_history(
            db, counterparty_id, exclude_transaction_id=transaction_id
        )

        # Need sufficient history to establish pattern
        if len(historical_payments) < MIN_HISTORICAL_TRANSACTIONS:
            context["insufficient_history"] = True
            return False

        # Analyze country pattern
        pattern = analyze_vendor_country_pattern(historical_payments)

        if pattern.get("no_country_data"):
            return False

        # Check if current country matches historical pattern
        primary_country = pattern.get("primary_country")
        countries_used = pattern.get("countries_used", set())

        # Flag if paying to a new country
        if country not in countries_used:
            context["vendor_country_history"] = {
                "primary_country": primary_country,
                "historical_countries": list(countries_used),
                "total_historical_payments": pattern["total_payments"],
                "is_domestic_only_vendor": pattern.get("is_domestic_only", False)
            }
            context["current_country"] = country
            context["country_change_detected"] = True

            # Higher weight if vendor was domestic-only
            if pattern.get("is_domestic_only"):
                context["was_domestic_only"] = True
                context["adjusted_weight"] = weight * 1.5

            return True

        # Also flag if country is rarely used (< 10% of payments)
        country_usage = pattern.get("country_distribution", {}).get(country, 0)
        if country_usage < 0.1 and country != primary_country:
            context["vendor_country_history"] = {
                "primary_country": primary_country,
                "historical_countries": list(countries_used),
                "total_historical_payments": pattern["total_payments"]
            }
            context["current_country"] = country
            context["rarely_used_country"] = True
            context["country_usage_percentage"] = round(country_usage * 100, 1)
            return True

        return False

    return Rule(
        name="unexpected_country_routing",
        description="Payment routed to unexpected country based on vendor history",
        condition_func=condition,
        weight=weight
    )


def create_domestic_to_foreign_switch_rule(db: Session, weight: float = 3.0) -> Rule:
    """
    Detect when a consistently domestic vendor suddenly receives foreign payment.

    This is a specific high-risk pattern where a vendor who has ONLY been paid
    domestically (e.g., always US) suddenly receives payment from overseas.

    Args:
        db: Database session
        weight: Rule importance weight (default 3.0 - high risk)

    Returns:
        Rule object
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing payments
        if transaction.get("direction") != "debit":
            return False

        country = extract_country_from_transaction(transaction)
        if not country:
            return False

        # Only flag foreign payments
        if country == "US":
            return False

        counterparty_id = transaction.get("counterparty_id")
        if not counterparty_id:
            return False

        # Get vendor's payment history (exclude current transaction)
        transaction_id = transaction.get("transaction_id")
        historical_payments = get_vendor_payment_history(
            db, counterparty_id, exclude_transaction_id=transaction_id
        )

        # Need sufficient history
        if len(historical_payments) < MIN_HISTORICAL_TRANSACTIONS:
            return False

        # Analyze country pattern
        pattern = analyze_vendor_country_pattern(historical_payments)

        # Check if vendor was domestic-only
        if pattern.get("is_domestic_only"):
            context["domestic_to_foreign_switch"] = True
            context["historical_payment_count"] = pattern["total_payments"]
            context["new_foreign_country"] = country
            context["severity"] = "high"
            return True

        return False

    return Rule(
        name="domestic_to_foreign_switch",
        description="Domestic-only vendor suddenly paid through foreign account",
        condition_func=condition,
        weight=weight
    )


def create_multiple_countries_rapid_rule(db: Session, weight: float = 2.0) -> Rule:
    """
    Detect payments to the same vendor through multiple countries in short timeframe.

    If a vendor receives payments from multiple different countries within a short
    period (e.g., 30 days), this could indicate account compromise or routing fraud.

    Args:
        db: Database session
        weight: Rule importance weight (default 2.0)

    Returns:
        Rule object
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing payments
        if transaction.get("direction") != "debit":
            return False

        country = extract_country_from_transaction(transaction)
        if not country:
            return False

        counterparty_id = transaction.get("counterparty_id")
        if not counterparty_id:
            return False

        # Get recent payment history (last 30 days, exclude current transaction)
        transaction_id = transaction.get("transaction_id")
        recent_payments = get_vendor_payment_history(
            db, counterparty_id, lookback_days=30, exclude_transaction_id=transaction_id
        )

        if len(recent_payments) < 2:
            return False

        # Get unique countries in recent payments
        recent_countries = set()
        for payment in recent_payments:
            payment_country = payment.get("country")
            if payment_country:
                recent_countries.add(payment_country)

        # Flag if 3+ different countries used in 30 days
        if len(recent_countries) >= 3:
            context["multiple_countries_rapid"] = True
            context["countries_count"] = len(recent_countries)
            context["countries_list"] = list(recent_countries)
            context["timeframe_days"] = 30
            return True

        return False

    return Rule(
        name="multiple_countries_rapid",
        description="Vendor paid through multiple countries in short timeframe",
        condition_func=condition,
        weight=weight
    )


def create_first_international_payment_rule(weight: float = 1.5) -> Rule:
    """
    Flag first-ever international payment from an account.

    The very first international payment from an account deserves scrutiny,
    especially if the account has been active domestically for a while.

    Args:
        weight: Rule importance weight (default 1.5)

    Returns:
        Rule object
    """
    def condition(transaction: Dict[str, Any], context: Dict[str, Any]) -> bool:
        # Only check outgoing payments
        if transaction.get("direction") != "debit":
            return False

        country = extract_country_from_transaction(transaction)
        if not country or country == "US":
            return False

        # Check context for account history
        first_international = context.get("is_first_international_payment", False)

        if first_international:
            context["first_international_payment"] = True
            context["destination_country"] = country
            return True

        return False

    return Rule(
        name="first_international_payment",
        description="First international payment from account",
        condition_func=condition,
        weight=weight
    )


def initialize_geographic_fraud_rules(db: Session) -> List[Rule]:
    """
    Initialize all geographic fraud detection rules.

    Returns:
        List of configured Rule objects for geographic fraud detection
    """
    return [
        create_high_risk_country_rule(db),
        create_unexpected_country_routing_rule(db),
        create_domestic_to_foreign_switch_rule(db),
        create_multiple_countries_rapid_rule(db),
        create_first_international_payment_rule(),
    ]
