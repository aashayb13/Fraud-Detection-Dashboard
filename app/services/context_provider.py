# app/services/context_provider.py
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import json
import datetime
from app.models.database import Transaction, Account, Employee, AccountChangeHistory, Beneficiary, Blacklist, DeviceSession, VPNProxyIP, HighRiskLocation, BehavioralBiometric, FraudFlag, FraudComplaint, MerchantProfile, AccountLimit
from app.services.chain_analyzer import ChainAnalyzer

class ContextProvider:
    def __init__(self, db: Session, enable_chain_analysis: bool = True):
        """
        Initialize context provider with database session.

        Args:
            db: SQLAlchemy database session
            enable_chain_analysis: Whether to enable chain analysis (default True)
        """
        self.db = db
        self.enable_chain_analysis = enable_chain_analysis
        self.chain_analyzer = ChainAnalyzer(db) if enable_chain_analysis else None
    
    def get_transaction_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather contextual information about the account and transaction history.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Context dictionary with historical data
        """
        context = {}
        account_id = transaction.get("account_id")
        
        if not account_id:
            return context
        
        # Get account information
        account = self.db.query(Account).filter(Account.account_id == account_id).first()
        if account:
            # Calculate account age
            creation_date = datetime.datetime.fromisoformat(account.creation_date)
            account_age = (datetime.datetime.utcnow() - creation_date).days
            context["account_age_days"] = account_age
            context["risk_tier"] = account.risk_tier
            
        # Get transaction history
        self._add_transaction_history(context, account_id, transaction)
        
        # Check if counterparty is new
        context["is_new_counterparty"] = self._is_new_counterparty(
            account_id,
            transaction.get("counterparty_id")
        )

        # Add money mule detection context
        self._add_money_mule_context(context, account_id, transaction)

        # Add beneficiary fraud detection context
        self._add_beneficiary_context(context, account_id, transaction)

        # Add chain analysis if enabled
        if self.enable_chain_analysis and self.chain_analyzer:
            chain_analysis = self.chain_analyzer.analyze_transaction_chains(
                account_id, transaction
            )
            context["chain_analysis"] = chain_analysis

        # Add account takeover detection context
        self._add_account_takeover_context(context, account_id, transaction)

        # Add odd hours transaction detection context
        self._add_odd_hours_context(context, account_id, transaction)

        # Add geographic context
        geographic_context = self.get_geographic_context(transaction)
        context.update(geographic_context)

        # Add blacklist detection context
        self._add_blacklist_context(context, transaction)

        # Add device fingerprinting context
        self._add_device_fingerprint_context(context, account_id, transaction)

        # Add VPN/proxy detection context
        self._add_vpn_proxy_context(context, transaction)

        # Add geo-location fraud detection context
        self._add_geolocation_context(context, account_id, transaction)

        # Add behavioral biometrics fraud detection context
        self._add_behavioral_biometric_context(context, account_id, transaction)

        # Add recipient relationship analysis context
        self._add_recipient_relationship_context(context, account_id, transaction)

        # Add social trust score context
        self._add_social_trust_score_context(context, account_id, transaction)

        # Add account age fraud detection context
        self._add_account_age_context(context, account_id, transaction)

        # Add high-risk transaction times fraud detection context
        self._add_high_risk_transaction_times_context(context, account_id, transaction)

        # Add past fraudulent behavior flags detection context
        self._add_past_fraud_flags_context(context, account_id, transaction)

        # Add location-inconsistent transactions detection context
        self._add_location_inconsistent_transactions_context(context, account_id, transaction)

        # Add normalized transaction amount detection context
        self._add_normalized_transaction_amount_context(context, account_id, transaction)

        # Add transaction context anomalies detection context
        self._add_transaction_context_anomalies_context(context, account_id, transaction)

        # Add fraud complaints count detection context
        self._add_fraud_complaints_count_context(context, account_id, transaction)

        # Add merchant category mismatch detection context
        self._add_merchant_category_mismatch_context(context, account_id, transaction)

        # Add user daily limit exceeded detection context
        self._add_user_daily_limit_exceeded_context(context, account_id, transaction)

        # Add recent high-value transaction flags detection context
        self._add_recent_high_value_transaction_flags_context(context, account_id, transaction)

        return context
    
    def _add_transaction_history(self, context: Dict[str, Any],
                                account_id: str,
                                current_tx: Dict[str, Any]) -> None:
        """Add transaction history data to context."""
        # Transaction velocity for different time windows
        timeframes = [1, 6, 24, 168]  # hours (1h, 6h, 24h, 1 week)
        context["tx_count_last_hours"] = {}
        context["small_deposit_count"] = {}

        now = datetime.datetime.utcnow()
        for hours in timeframes:
            time_threshold = (now - datetime.timedelta(hours=hours)).isoformat()

            count = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.timestamp > time_threshold
            ).count()

            context["tx_count_last_hours"][hours] = count

            # Count small deposits (≤ $2.00) for fraud detection
            small_deposit_count = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.timestamp > time_threshold,
                Transaction.amount > 0,
                Transaction.amount <= 2.0,
                Transaction.transaction_type.in_(["ACH", "WIRE", "DEPOSIT", "CREDIT"])
            ).count()

            # Include current transaction if it's a small deposit
            current_amount = current_tx.get("amount", 0)
            current_type = current_tx.get("transaction_type", "").upper()
            if (0 < current_amount <= 2.0 and
                current_type in ["ACH", "WIRE", "DEPOSIT", "CREDIT"]):
                small_deposit_count += 1

            context["small_deposit_count"][hours] = small_deposit_count
        
        # Calculate average transaction amount for this type
        tx_type = current_tx.get("transaction_type")
        if tx_type:
            # Get transactions of same type in last 90 days
            ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()
            
            similar_txs = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.transaction_type == tx_type,
                Transaction.timestamp > ninety_days_ago
            ).all()
            
            if similar_txs:
                amounts = [tx.amount for tx in similar_txs]
                avg_amount = sum(amounts) / len(amounts)
                context["avg_transaction_amount"] = avg_amount
                
                # Calculate standard deviation
                import math
                variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
                std_dev = math.sqrt(variance)
                
                # Calculate deviation of current transaction
                current_amount = current_tx.get("amount", 0)
                if std_dev > 0:
                    context["amount_deviation"] = abs(current_amount - avg_amount) / std_dev
                else:
                    # If all historical amounts are identical, use ratio
                    context["amount_deviation"] = abs(current_amount / max(avg_amount, 0.01))
            else:
                # First transaction of this type
                context["avg_transaction_amount"] = 0
                context["amount_deviation"] = 5.0  # High deviation for first transaction
    
    def _is_new_counterparty(self, account_id: str, counterparty_id: str) -> bool:
        """Check if this is a new counterparty for this account."""
        if not counterparty_id:
            return False

        # Look for previous transactions with this counterparty
        previous_tx = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.counterparty_id == counterparty_id
        ).first()

        return previous_tx is None

    def _add_money_mule_context(self, context: Dict[str, Any],
                                account_id: str,
                                current_tx: Dict[str, Any]) -> None:
        """
        Add money mule detection context.

        Money mule pattern: Multiple small incoming payments quickly followed by outgoing transfers.
        """
        now = datetime.datetime.utcnow()

        # Analyze patterns over different time windows
        time_windows = [24, 72, 168]  # 1 day, 3 days, 1 week (hours)

        for hours in time_windows:
            time_threshold = (now - datetime.timedelta(hours=hours)).isoformat()

            # Get incoming transactions (credits)
            incoming_txs = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.direction == "credit",
                Transaction.timestamp > time_threshold
            ).all()

            # Get outgoing transactions (debits)
            outgoing_txs = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.direction == "debit",
                Transaction.timestamp > time_threshold
            ).all()

            # Calculate metrics
            incoming_count = len(incoming_txs)
            outgoing_count = len(outgoing_txs)
            incoming_total = sum(tx.amount for tx in incoming_txs)
            outgoing_total = sum(tx.amount for tx in outgoing_txs)

            # Store in context
            context[f"incoming_count_{hours}h"] = incoming_count
            context[f"outgoing_count_{hours}h"] = outgoing_count
            context[f"incoming_total_{hours}h"] = incoming_total
            context[f"outgoing_total_{hours}h"] = outgoing_total

            # Calculate average incoming transaction amount (for "many small" detection)
            if incoming_count > 0:
                avg_incoming = incoming_total / incoming_count
                context[f"avg_incoming_amount_{hours}h"] = avg_incoming
            else:
                context[f"avg_incoming_amount_{hours}h"] = 0

            # Calculate flow-through ratio (how much incoming is sent out)
            if incoming_total > 0:
                flow_through_ratio = outgoing_total / incoming_total
                context[f"flow_through_ratio_{hours}h"] = flow_through_ratio
            else:
                context[f"flow_through_ratio_{hours}h"] = 0

        # Calculate average time between incoming and outgoing (velocity of moving money)
        # For recent 7-day window
        week_ago = (now - datetime.timedelta(days=7)).isoformat()

        recent_incoming = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.direction == "credit",
            Transaction.timestamp > week_ago
        ).order_by(Transaction.timestamp).all()

        recent_outgoing = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.direction == "debit",
            Transaction.timestamp > week_ago
        ).order_by(Transaction.timestamp).all()

        # Calculate average time from incoming to next outgoing
        if recent_incoming and recent_outgoing:
            time_gaps = []
            for incoming in recent_incoming:
                incoming_time = datetime.datetime.fromisoformat(incoming.timestamp)
                # Find next outgoing after this incoming
                for outgoing in recent_outgoing:
                    outgoing_time = datetime.datetime.fromisoformat(outgoing.timestamp)
                    if outgoing_time > incoming_time:
                        gap_hours = (outgoing_time - incoming_time).total_seconds() / 3600
                        time_gaps.append(gap_hours)
                        break

            if time_gaps:
                avg_time_to_transfer = sum(time_gaps) / len(time_gaps)
                context["avg_hours_to_transfer"] = avg_time_to_transfer
            else:
                context["avg_hours_to_transfer"] = None
        else:
            context["avg_hours_to_transfer"] = None

    def _add_beneficiary_context(self, context: Dict[str, Any],
                                  account_id: str,
                                  current_tx: Dict[str, Any]) -> None:
        """
        Add beneficiary fraud detection context.

        Detects rapid addition of many beneficiaries followed by payments.
        """
        now = datetime.datetime.utcnow()
        counterparty_id = current_tx.get("counterparty_id")

        # Analyze beneficiary additions over time windows
        time_windows = [24, 72, 168]  # 1 day, 3 days, 1 week (hours)

        for hours in time_windows:
            time_threshold = (now - datetime.timedelta(hours=hours)).isoformat()

            # Count beneficiaries added in this window
            beneficiaries_added = self.db.query(Beneficiary).filter(
                Beneficiary.account_id == account_id,
                Beneficiary.registration_date > time_threshold,
                Beneficiary.status == "active"
            ).all()

            context[f"beneficiaries_added_{hours}h"] = len(beneficiaries_added)

            # Track beneficiaries from same IP address
            if beneficiaries_added:
                ip_counts = {}
                user_counts = {}

                for beneficiary in beneficiaries_added:
                    if beneficiary.ip_address:
                        ip_counts[beneficiary.ip_address] = ip_counts.get(beneficiary.ip_address, 0) + 1
                    if beneficiary.added_by:
                        user_counts[beneficiary.added_by] = user_counts.get(beneficiary.added_by, 0) + 1

                # Find most common IP and user
                if ip_counts:
                    most_common_ip = max(ip_counts.items(), key=lambda x: x[1])
                    context[f"beneficiaries_same_ip_{hours}h"] = most_common_ip[1]
                    context["same_source_ip"] = most_common_ip[0]
                else:
                    context[f"beneficiaries_same_ip_{hours}h"] = 0

                if user_counts:
                    most_common_user = max(user_counts.items(), key=lambda x: x[1])
                    context[f"beneficiaries_same_user_{hours}h"] = most_common_user[1]
                    context["same_source_user"] = most_common_user[0]
                else:
                    context[f"beneficiaries_same_user_{hours}h"] = 0

            # Count payments to newly added beneficiaries in this window
            new_beneficiary_ids = [b.counterparty_id for b in beneficiaries_added if b.counterparty_id]

            if new_beneficiary_ids:
                payments_to_new = self.db.query(Transaction).filter(
                    Transaction.account_id == account_id,
                    Transaction.direction == "debit",
                    Transaction.counterparty_id.in_(new_beneficiary_ids),
                    Transaction.timestamp > time_threshold
                ).count()

                total_payments = self.db.query(Transaction).filter(
                    Transaction.account_id == account_id,
                    Transaction.direction == "debit",
                    Transaction.timestamp > time_threshold
                ).count()

                context[f"new_beneficiary_payment_count_{hours}h"] = payments_to_new

                if total_payments > 0:
                    context[f"new_beneficiary_payment_ratio_{hours}h"] = payments_to_new / total_payments
                else:
                    context[f"new_beneficiary_payment_ratio_{hours}h"] = 0.0
            else:
                context[f"new_beneficiary_payment_count_{hours}h"] = 0
                context[f"new_beneficiary_payment_ratio_{hours}h"] = 0.0

        # Check if current transaction is to a recently added beneficiary
        if counterparty_id:
            beneficiary = self.db.query(Beneficiary).filter(
                Beneficiary.account_id == account_id,
                Beneficiary.counterparty_id == counterparty_id,
                Beneficiary.status == "active"
            ).first()

            if beneficiary:
                # Calculate beneficiary age
                added_time = datetime.datetime.fromisoformat(beneficiary.registration_date)
                beneficiary_age_hours = (now - added_time).total_seconds() / 3600

                context["is_new_beneficiary"] = beneficiary_age_hours <= 48  # Less than 48 hours
                context["beneficiary_age_hours"] = beneficiary_age_hours
                context["is_beneficiary_verified"] = beneficiary.verified
                context["beneficiary_addition_source"] = beneficiary.addition_source
                context["beneficiary_flagged"] = beneficiary.flagged_as_suspicious
            else:
                # No beneficiary record - might be first transaction
                context["is_new_beneficiary"] = False
                context["is_beneficiary_verified"] = True  # Assume verified if no record
                context["beneficiary_age_hours"] = None

    def _add_account_takeover_context(self, context: Dict[str, Any],
                                       account_id: str,
                                       current_tx: Dict[str, Any]) -> None:
        """
        Add account takeover detection context.

        Account takeover pattern:
        - Phone number or device changes occur
        - Followed by suspicious outgoing transfers shortly after
        - This prevents legitimate user from getting security alerts
        """
        now = datetime.datetime.utcnow()

        # Check for recent phone/SIM/device changes (within last 48 hours)
        time_windows = [1, 6, 24, 48]  # hours

        for hours in time_windows:
            time_threshold = (now - datetime.timedelta(hours=hours)).isoformat()

            # Query for phone/device changes
            phone_changes = self.db.query(AccountChangeHistory).filter(
                AccountChangeHistory.account_id == account_id,
                AccountChangeHistory.change_type.in_(["phone", "device", "sim", "phone_number"]),
                AccountChangeHistory.timestamp > time_threshold
            ).all()

            context[f"phone_changes_count_{hours}h"] = len(phone_changes)

            if phone_changes:
                # Store details of most recent change in this window
                most_recent = max(phone_changes, key=lambda c: c.timestamp)
                context[f"most_recent_phone_change_{hours}h"] = {
                    "timestamp": most_recent.timestamp,
                    "change_type": most_recent.change_type,
                    "change_source": most_recent.change_source,
                    "verified": most_recent.verified,
                    "old_value": most_recent.old_value,
                    "new_value": most_recent.new_value
                }

                # Check if change was flagged as suspicious
                suspicious_changes = [c for c in phone_changes if c.flagged_as_suspicious]
                context[f"suspicious_phone_changes_{hours}h"] = len(suspicious_changes)

                # Check if changes were unverified
                unverified_changes = [c for c in phone_changes if not c.verified]
                context[f"unverified_phone_changes_{hours}h"] = len(unverified_changes)

        # For outgoing transfers analysis - check if current transaction is outgoing
        is_outgoing = current_tx.get("direction") == "debit"
        context["is_outgoing_transfer"] = is_outgoing

        # If this is an outgoing transfer, calculate time since most recent phone change
        if is_outgoing and context.get("phone_changes_count_48h", 0) > 0:
            # Find the most recent phone change across all windows
            all_changes = self.db.query(AccountChangeHistory).filter(
                AccountChangeHistory.account_id == account_id,
                AccountChangeHistory.change_type.in_(["phone", "device", "sim", "phone_number"])
            ).order_by(AccountChangeHistory.timestamp.desc()).first()

            if all_changes:
                change_time = datetime.datetime.fromisoformat(all_changes.timestamp)
                current_time = datetime.datetime.fromisoformat(
                    current_tx.get("timestamp", now.isoformat())
                )
                hours_since_change = (current_time - change_time).total_seconds() / 3600
                context["hours_since_phone_change"] = hours_since_change

                # Store if this is first outgoing transfer after phone change
                is_first_outgoing = self.db.query(Transaction).filter(
                    Transaction.account_id == account_id,
                    Transaction.direction == "debit",
                    Transaction.timestamp > all_changes.timestamp,
                    Transaction.timestamp < current_tx.get("timestamp", now.isoformat())
                ).count() == 0

                context["is_first_transfer_after_phone_change"] = is_first_outgoing

    def _add_odd_hours_context(self, context: Dict[str, Any],
                                account_id: str,
                                current_tx: Dict[str, Any]) -> None:
        """
        Add odd hours transaction detection context.

        Detects large transactions occurring at unusual times:
        - Outside normal business hours (late night/early morning)
        - Outside the customer's typical transaction timing patterns
        """
        from config.settings import (
            ODD_HOURS_START,
            ODD_HOURS_END,
            ODD_HOURS_LOOKBACK_DAYS,
            ODD_HOURS_MIN_HISTORICAL_TRANSACTIONS
        )

        now = datetime.datetime.utcnow()

        # Get transaction timestamp
        tx_timestamp_str = current_tx.get("timestamp", now.isoformat())
        tx_timestamp = datetime.datetime.fromisoformat(tx_timestamp_str)

        # Extract hour of day (0-23)
        tx_hour = tx_timestamp.hour
        context["transaction_hour"] = tx_hour

        # Check if transaction is during odd hours (10 PM - 6 AM by default)
        is_odd_hours = False
        if ODD_HOURS_START > ODD_HOURS_END:
            # Wraps around midnight (e.g., 22:00 to 06:00)
            is_odd_hours = tx_hour >= ODD_HOURS_START or tx_hour < ODD_HOURS_END
        else:
            # Does not wrap (e.g., 01:00 to 05:00)
            is_odd_hours = ODD_HOURS_START <= tx_hour < ODD_HOURS_END

        context["is_odd_hours"] = is_odd_hours
        context["odd_hours_window"] = f"{ODD_HOURS_START:02d}:00 - {ODD_HOURS_END:02d}:00"

        # Check if it's a weekend
        # Monday=0, Sunday=6
        is_weekend = tx_timestamp.weekday() >= 5
        context["is_weekend"] = is_weekend

        # Analyze historical transaction timing patterns
        lookback_date = (now - datetime.timedelta(days=ODD_HOURS_LOOKBACK_DAYS)).isoformat()

        # Get historical transactions for this account
        historical_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > lookback_date
        ).all()

        if len(historical_txs) >= ODD_HOURS_MIN_HISTORICAL_TRANSACTIONS:
            # Analyze timing patterns
            odd_hours_count = 0
            business_hours_count = 0
            weekend_count = 0
            hour_distribution = [0] * 24  # Count per hour

            for tx in historical_txs:
                tx_time = datetime.datetime.fromisoformat(tx.timestamp)
                tx_h = tx_time.hour

                hour_distribution[tx_h] += 1

                # Check if odd hours
                if ODD_HOURS_START > ODD_HOURS_END:
                    if tx_h >= ODD_HOURS_START or tx_h < ODD_HOURS_END:
                        odd_hours_count += 1
                    else:
                        business_hours_count += 1
                else:
                    if ODD_HOURS_START <= tx_h < ODD_HOURS_END:
                        odd_hours_count += 1
                    else:
                        business_hours_count += 1

                # Check if weekend
                if tx_time.weekday() >= 5:
                    weekend_count += 1

            total_count = len(historical_txs)

            # Calculate ratios
            context["historical_odd_hours_ratio"] = odd_hours_count / total_count if total_count > 0 else 0
            context["historical_business_hours_ratio"] = business_hours_count / total_count if total_count > 0 else 0
            context["historical_weekend_ratio"] = weekend_count / total_count if total_count > 0 else 0
            context["historical_transaction_count"] = total_count
            context["hour_distribution"] = hour_distribution

            # Determine if current transaction is unusual based on historical pattern
            # If customer typically transacts during business hours, odd hours transaction is unusual
            context["deviates_from_pattern"] = (
                is_odd_hours and
                context["historical_business_hours_ratio"] > 0.8  # 80%+ of transactions during business hours
            )

            # Check if this specific hour is unusual for the customer
            current_hour_historical_count = hour_distribution[tx_hour]
            avg_hourly_count = total_count / 24

            # If this hour has significantly fewer historical transactions
            context["hour_is_unusual"] = (
                current_hour_historical_count < (avg_hourly_count * 0.5) and
                total_count >= 10  # Need enough data
            )

        else:
            # Not enough historical data
            context["historical_transaction_count"] = len(historical_txs)
            context["insufficient_history"] = True

        # Check for other odd hours transactions in recent period (last 7 days)
        recent_lookback = (now - datetime.timedelta(days=7)).isoformat()
        recent_odd_hours_txs = []

        for tx in self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > recent_lookback
        ).all():
            tx_time = datetime.datetime.fromisoformat(tx.timestamp)
            tx_h = tx_time.hour

            # Check if odd hours
            if ODD_HOURS_START > ODD_HOURS_END:
                if tx_h >= ODD_HOURS_START or tx_h < ODD_HOURS_END:
                    recent_odd_hours_txs.append(tx)
            else:
                if ODD_HOURS_START <= tx_h < ODD_HOURS_END:
                    recent_odd_hours_txs.append(tx)

        context["recent_odd_hours_transaction_count"] = len(recent_odd_hours_txs)

        # If this is an odd hours transaction, calculate total amount in recent odd hours
        if is_odd_hours and recent_odd_hours_txs:
            recent_odd_hours_total = sum(abs(tx.amount) for tx in recent_odd_hours_txs)
            context["recent_odd_hours_total_amount"] = recent_odd_hours_total

    def get_payroll_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get payroll-specific context for fraud detection.

        Args:
            transaction: Transaction data

        Returns:
            Context dictionary with payroll-related information
        """
        context = {}

        # Try to get employee information
        employee = self._get_employee_from_transaction(transaction)
        if not employee:
            return context

        context["employee_id"] = employee.employee_id
        context["employee_name"] = employee.name
        context["employment_status"] = employee.employment_status

        # Get account change history
        account_changes = self.db.query(AccountChangeHistory).filter(
            AccountChangeHistory.employee_id == employee.employee_id
        ).order_by(AccountChangeHistory.timestamp.desc()).all()

        if account_changes:
            context["total_account_changes"] = len(account_changes)

            # Most recent change
            most_recent = account_changes[0]
            context["most_recent_change"] = {
                "timestamp": most_recent.timestamp,
                "change_type": most_recent.change_type,
                "change_source": most_recent.change_source,
                "verified": most_recent.verified,
                "flagged_as_suspicious": most_recent.flagged_as_suspicious
            }

            # Count unverified changes
            unverified_count = sum(1 for c in account_changes if not c.verified)
            context["unverified_changes_count"] = unverified_count

            # Count suspicious-source changes
            suspicious_sources = ["email_request", "phone_request"]
            suspicious_count = sum(
                1 for c in account_changes
                if c.change_source in suspicious_sources
            )
            context["suspicious_source_changes_count"] = suspicious_count

        # Get time since last payroll
        if employee.last_payroll_date:
            last_payroll = datetime.datetime.fromisoformat(employee.last_payroll_date)
            days_since = (datetime.datetime.utcnow() - last_payroll).days
            context["days_since_last_payroll"] = days_since
            context["last_payroll_date"] = employee.last_payroll_date

        # Payroll frequency info
        context["payroll_frequency"] = employee.payroll_frequency

        return context

    def _get_employee_from_transaction(self, transaction: Dict[str, Any]) -> Employee:
        """Get employee record from transaction data."""
        # Try tx_metadata first (also check 'metadata' for backward compatibility)
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata:
            if isinstance(tx_metadata, str):
                try:
                    tx_metadata = json.loads(tx_metadata)
                except:
                    tx_metadata = {}

            employee_id = tx_metadata.get("employee_id")
            if employee_id:
                employee = self.db.query(Employee).filter(
                    Employee.employee_id == employee_id
                ).first()
                if employee:
                    return employee

        # Fallback: try by account
        account_id = transaction.get("account_id")
        if account_id:
            return self.db.query(Employee).filter(
                Employee.account_id == account_id
            ).first()

        return None

    def get_check_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get check-specific context for fraud detection.

        This method analyzes check deposit patterns and identifies:
        - Duplicate check deposits (same check deposited multiple times)
        - Rapid check deposit sequences
        - Check amount mismatches (possible alteration)

        Args:
            transaction: Transaction data

        Returns:
            Context dictionary with check-related fraud indicators
        """
        context = {}

        # Extract check information from transaction metadata
        check_info = self._extract_check_info(transaction)
        if not check_info:
            return context  # Not a check transaction or no check data available

        account_id = transaction.get("account_id")
        check_number = check_info.get("check_number")
        check_amount = check_info.get("amount")

        # 1. Check for duplicate check deposits
        if check_number:
            duplicates = self._find_duplicate_checks(
                check_number=check_number,
                check_amount=check_amount,
                routing_number=check_info.get("routing_number"),
                account_number=check_info.get("account_number"),
                exclude_transaction_id=transaction.get("transaction_id")
            )

            if duplicates:
                context["duplicate_checks"] = [
                    {
                        "transaction_id": dup.transaction_id,
                        "account_id": dup.account_id,
                        "timestamp": dup.timestamp,
                        "amount": dup.amount,
                        "check_number": check_number
                    }
                    for dup in duplicates
                ]

        # 2. Count check deposits in the last hour (rapid sequence detection)
        if account_id:
            now = datetime.datetime.utcnow()
            one_hour_ago = (now - datetime.timedelta(hours=1)).isoformat()

            recent_checks = self.db.query(Transaction).filter(
                Transaction.account_id == account_id,
                Transaction.timestamp > one_hour_ago,
                Transaction.direction == "credit",
                Transaction.transaction_type.in_([
                    "CHECK", "CHECK_DEPOSIT", "DEPOSIT",
                    "REMOTE_DEPOSIT", "MOBILE_DEPOSIT"
                ])
            ).all()

            context["check_count_1h"] = len(recent_checks)
            context["check_amount_1h"] = sum(tx.amount for tx in recent_checks)

            # Include current transaction if it's a check deposit
            if self._is_check_deposit(transaction):
                context["check_count_1h"] += 1
                context["check_amount_1h"] += transaction.get("amount", 0)

        # 3. Check for amount mismatches (possible check alteration)
        if check_number and check_amount:
            mismatch = self._check_amount_mismatch(
                check_number=check_number,
                current_amount=check_amount,
                routing_number=check_info.get("routing_number")
            )

            if mismatch:
                context["check_amount_mismatch"] = mismatch

        return context

    def _extract_check_info(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract check-specific information from transaction metadata.

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

    def _is_check_deposit(self, transaction: Dict[str, Any]) -> bool:
        """
        Determine if a transaction is a check deposit.

        Args:
            transaction: Transaction dictionary

        Returns:
            True if transaction is a check deposit, False otherwise
        """
        tx_type = transaction.get("transaction_type", "").upper()
        direction = transaction.get("direction", "").lower()

        return (
            direction == "credit" and
            tx_type in ["CHECK", "CHECK_DEPOSIT", "DEPOSIT", "REMOTE_DEPOSIT", "MOBILE_DEPOSIT"]
        )

    def _find_duplicate_checks(
        self,
        check_number: str,
        check_amount: float = None,
        routing_number: str = None,
        account_number: str = None,
        exclude_transaction_id: str = None,
        lookback_days: int = 90
    ):
        """
        Find previous deposits of the same check.

        A duplicate check is identified by matching:
        - Check number (required)
        - Check amount (if available)
        - Source routing/account numbers (if available)

        Args:
            check_number: The check number to search for
            check_amount: Amount on the check (optional)
            routing_number: Routing number from check (optional)
            account_number: Account number from check (optional)
            exclude_transaction_id: Transaction ID to exclude from results
            lookback_days: How many days to look back (default: 90)

        Returns:
            List of Transaction objects that match (duplicates)
        """
        now = datetime.datetime.utcnow()
        lookback_date = (now - datetime.timedelta(days=lookback_days)).isoformat()

        # Query for check deposits in the lookback period
        query = self.db.query(Transaction).filter(
            Transaction.timestamp > lookback_date,
            Transaction.direction == "credit",
            Transaction.transaction_type.in_([
                "CHECK", "CHECK_DEPOSIT", "DEPOSIT",
                "REMOTE_DEPOSIT", "MOBILE_DEPOSIT"
            ])
        )

        # Exclude the current transaction
        if exclude_transaction_id:
            query = query.filter(Transaction.transaction_id != exclude_transaction_id)

        all_check_txs = query.all()

        # Filter by check number and other attributes in metadata
        duplicates = []
        for tx in all_check_txs:
            try:
                metadata_str = tx.tx_metadata or "{}"
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

                # Check if check number matches
                if metadata.get("check_number") == check_number:
                    # Check for amount match (if provided)
                    if check_amount is not None:
                        tx_check_amount = metadata.get("check_amount")
                        if tx_check_amount is not None:
                            # Amount should match closely (within $0.01 for floating point)
                            if abs(float(tx_check_amount) - check_amount) > 0.01:
                                continue  # Amount doesn't match, not a duplicate

                    # Check for routing number match (if provided)
                    if routing_number is not None:
                        tx_routing = metadata.get("routing_number")
                        if tx_routing is not None and tx_routing != routing_number:
                            continue  # Different bank, might not be duplicate

                    # Check for account number match (if provided)
                    if account_number is not None:
                        tx_account = metadata.get("account_number")
                        if tx_account is not None and tx_account != account_number:
                            continue  # Different account, might not be duplicate

                    # All criteria match - this is a duplicate
                    duplicates.append(tx)

            except (json.JSONDecodeError, ValueError, TypeError):
                # Skip transactions with invalid metadata
                continue

        return duplicates

    def _check_amount_mismatch(
        self,
        check_number: str,
        current_amount: float,
        routing_number: str = None,
        max_deviation_percent: float = 5.0,
        lookback_days: int = 180
    ) -> Dict[str, Any]:
        """
        Check if the check amount differs from previous deposits of the same check.

        This can indicate check alteration fraud.

        Args:
            check_number: The check number
            current_amount: Current transaction amount
            routing_number: Routing number from check (optional)
            max_deviation_percent: Maximum allowed deviation percentage
            lookback_days: How many days to look back

        Returns:
            Dictionary with mismatch details if found, None otherwise
        """
        now = datetime.datetime.utcnow()
        lookback_date = (now - datetime.timedelta(days=lookback_days)).isoformat()

        # Find previous deposits of this check number
        query = self.db.query(Transaction).filter(
            Transaction.timestamp > lookback_date,
            Transaction.direction == "credit",
            Transaction.transaction_type.in_([
                "CHECK", "CHECK_DEPOSIT", "DEPOSIT",
                "REMOTE_DEPOSIT", "MOBILE_DEPOSIT"
            ])
        )

        all_check_txs = query.all()

        # Find matching check numbers
        previous_amounts = []
        for tx in all_check_txs:
            try:
                metadata_str = tx.tx_metadata or "{}"
                metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

                # Check if check number matches
                if metadata.get("check_number") == check_number:
                    # Check routing number if provided
                    if routing_number:
                        tx_routing = metadata.get("routing_number")
                        if tx_routing and tx_routing != routing_number:
                            continue  # Different bank

                    # Get the amount
                    tx_check_amount = metadata.get("check_amount")
                    if tx_check_amount is not None:
                        previous_amounts.append(float(tx_check_amount))

            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Check for amount deviation
        if previous_amounts:
            # Use the most common previous amount as reference
            from collections import Counter
            amount_counts = Counter(previous_amounts)
            most_common_amount = amount_counts.most_common(1)[0][0]

            # Calculate deviation percentage
            if most_common_amount > 0:
                deviation_percent = abs(
                    (current_amount - most_common_amount) / most_common_amount * 100
                )

                if deviation_percent > max_deviation_percent:
                    return {
                        "previous_amount": most_common_amount,
                        "current_amount": current_amount,
                        "deviation_percent": deviation_percent,
                        "occurrences": amount_counts[most_common_amount]
                    }

        return None

    def get_geographic_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get geographic context for international payment fraud detection.

        Args:
            transaction: Transaction data

        Returns:
            Context dictionary with geographic information
        """
        context = {}

        # Only process outgoing payments
        if transaction.get("direction") != "debit":
            return context

        account_id = transaction.get("account_id")
        if not account_id:
            return context

        # Check if this is the first international payment
        # Get all previous outgoing transactions
        all_outgoing = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.direction == "debit"
        ).all()

        # Check if any previous transactions were international
        has_previous_international = False
        for tx in all_outgoing:
            if tx.tx_metadata:
                try:
                    metadata = json.loads(tx.tx_metadata) if isinstance(tx.tx_metadata, str) else tx.tx_metadata
                    country = metadata.get("country") or metadata.get("country_code") or \
                              metadata.get("bank_country") or metadata.get("destination_country")
                    if country and str(country).upper()[:2] != "US":
                        has_previous_international = True
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass

        # Check current transaction country
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        current_country = None
        if tx_metadata:
            if isinstance(tx_metadata, str):
                try:
                    tx_metadata = json.loads(tx_metadata)
                except json.JSONDecodeError:
                    tx_metadata = {}

            current_country = tx_metadata.get("country") or \
                             tx_metadata.get("country_code") or \
                             tx_metadata.get("bank_country") or \
                             tx_metadata.get("destination_country")

        # Flag if this is first international payment
        if current_country and str(current_country).upper()[:2] != "US":
            context["is_first_international_payment"] = not has_previous_international

        return context

    def _add_blacklist_context(self, context: Dict[str, Any],
                                transaction: Dict[str, Any]) -> None:
        """
        Add blacklist detection context.

        Checks if the counterparty, account, or other identifiers are on the blacklist.

        Args:
            context: Context dictionary to update
            transaction: Transaction data
        """
        counterparty_id = transaction.get("counterparty_id")

        # Initialize blacklist flags
        context["is_blacklisted"] = False
        context["blacklist_matches"] = []

        if not counterparty_id:
            return

        # Check if counterparty is blacklisted
        blacklist_entries = self.db.query(Blacklist).filter(
            Blacklist.status == "active",
            Blacklist.entity_value == counterparty_id
        ).all()

        # Also check by entity type if we have metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        # Check for additional identifiers in metadata
        additional_checks = []
        if tx_metadata:
            # Check UPI ID
            upi_id = tx_metadata.get("upi_id")
            if upi_id:
                additional_checks.append(("upi", upi_id))

            # Check merchant ID
            merchant_id = tx_metadata.get("merchant_id")
            if merchant_id:
                additional_checks.append(("merchant", merchant_id))

            # Check routing number
            routing_number = tx_metadata.get("routing_number")
            if routing_number:
                additional_checks.append(("routing_number", routing_number))

            # Check email
            email = tx_metadata.get("email") or tx_metadata.get("recipient_email")
            if email:
                additional_checks.append(("email", email))

            # Check phone
            phone = tx_metadata.get("phone") or tx_metadata.get("recipient_phone")
            if phone:
                additional_checks.append(("phone", phone))

        # Query for additional identifier matches
        for entity_type, entity_value in additional_checks:
            matches = self.db.query(Blacklist).filter(
                Blacklist.status == "active",
                Blacklist.entity_type == entity_type,
                Blacklist.entity_value == entity_value
            ).all()
            blacklist_entries.extend(matches)

        # Process blacklist matches
        if blacklist_entries:
            context["is_blacklisted"] = True
            context["blacklist_matches"] = [
                {
                    "entity_type": entry.entity_type,
                    "entity_value": entry.entity_value,
                    "entity_name": entry.entity_name,
                    "reason": entry.reason,
                    "severity": entry.severity,
                    "added_date": entry.added_date,
                    "source": entry.source
                }
                for entry in blacklist_entries
            ]

            # Get highest severity level
            severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            max_severity = max(
                (entry.severity for entry in blacklist_entries),
                key=lambda s: severity_order.get(s, 0)
            )
            context["blacklist_max_severity"] = max_severity
            context["blacklist_match_count"] = len(blacklist_entries)

    def _add_device_fingerprint_context(self, context: Dict[str, Any],
                                         account_id: str,
                                         transaction: Dict[str, Any]) -> None:
        """
        Add device fingerprinting context for fraud detection.

        Analyzes device mismatches (device ID, browser, OS, IP) compared to historical sessions.

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Extract device information from transaction metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            context["device_info_available"] = False
            return

        # Extract current device info
        current_device_id = tx_metadata.get("device_id")
        current_browser = tx_metadata.get("browser")
        current_os = tx_metadata.get("os")
        current_ip = tx_metadata.get("ip_address")
        current_user_agent = tx_metadata.get("user_agent")

        context["device_info_available"] = True
        context["current_device_id"] = current_device_id
        context["current_browser"] = current_browser
        context["current_os"] = current_os
        context["current_ip"] = current_ip

        # Get historical device sessions for this account (last 90 days)
        now = datetime.datetime.utcnow()
        ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()

        historical_sessions = self.db.query(DeviceSession).filter(
            DeviceSession.account_id == account_id,
            DeviceSession.timestamp > ninety_days_ago
        ).order_by(DeviceSession.timestamp.desc()).all()

        if not historical_sessions:
            # No historical device data - first transaction or new tracking
            context["is_new_device"] = True
            context["device_history_count"] = 0
            context["device_mismatch"] = False
            return

        context["device_history_count"] = len(historical_sessions)

        # Check if current device has been seen before
        device_seen_before = False
        matching_device = None

        if current_device_id:
            for session in historical_sessions:
                if session.device_id == current_device_id:
                    device_seen_before = True
                    matching_device = session
                    break

        context["device_seen_before"] = device_seen_before
        context["is_new_device"] = not device_seen_before

        # Analyze device mismatches
        mismatches = []

        # Get most common device attributes from history
        device_ids = [s.device_id for s in historical_sessions if s.device_id]
        browsers = [s.browser for s in historical_sessions if s.browser]
        os_list = [s.os for s in historical_sessions if s.os]
        ips = [s.ip_address for s in historical_sessions if s.ip_address]

        # Check device ID mismatch
        if current_device_id and device_ids:
            if current_device_id not in device_ids:
                mismatches.append({
                    "attribute": "device_id",
                    "current": current_device_id,
                    "expected": device_ids[0] if device_ids else None,
                    "severity": "high"
                })

        # Check browser mismatch
        if current_browser and browsers:
            # Get most common browser
            from collections import Counter
            browser_counts = Counter(browsers)
            most_common_browser = browser_counts.most_common(1)[0][0]

            if current_browser != most_common_browser:
                # Check if this browser has been used before
                if current_browser not in browsers:
                    mismatches.append({
                        "attribute": "browser",
                        "current": current_browser,
                        "expected": most_common_browser,
                        "severity": "medium"
                    })

        # Check OS mismatch
        if current_os and os_list:
            from collections import Counter
            os_counts = Counter(os_list)
            most_common_os = os_counts.most_common(1)[0][0]

            if current_os != most_common_os:
                if current_os not in os_list:
                    mismatches.append({
                        "attribute": "os",
                        "current": current_os,
                        "expected": most_common_os,
                        "severity": "high"
                    })

        # Check IP address mismatch (new IP)
        if current_ip and ips:
            if current_ip not in ips:
                # New IP address - check how many unique IPs historically
                unique_ips = len(set(ips))
                context["unique_ips_historical"] = unique_ips

                # If user typically uses 1-2 IPs, a new one is more suspicious
                if unique_ips <= 2:
                    mismatches.append({
                        "attribute": "ip_address",
                        "current": current_ip,
                        "expected": ips[0] if ips else None,
                        "severity": "medium"
                    })

        context["device_mismatches"] = mismatches
        context["device_mismatch"] = len(mismatches) > 0
        context["device_mismatch_count"] = len(mismatches)

        # Calculate device mismatch severity score
        if mismatches:
            severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            max_mismatch_severity = max(
                (m["severity"] for m in mismatches),
                key=lambda s: severity_scores.get(s, 0)
            )
            context["device_mismatch_max_severity"] = max_mismatch_severity

        # Check if device has been flagged as suspicious
        if matching_device and matching_device.flagged_as_suspicious:
            context["device_flagged_suspicious"] = True
            context["device_suspicious_reason"] = matching_device.suspicious_reason
        else:
            context["device_flagged_suspicious"] = False

        # Calculate time since last seen (for known devices)
        if matching_device:
            last_seen_time = datetime.datetime.fromisoformat(matching_device.last_seen)
            hours_since_last_seen = (now - last_seen_time).total_seconds() / 3600
            context["hours_since_device_last_seen"] = hours_since_last_seen
            context["device_session_count"] = matching_device.session_count
            context["device_is_trusted"] = matching_device.is_trusted_device

    def _add_vpn_proxy_context(self, context: Dict[str, Any],
                                transaction: Dict[str, Any]) -> None:
        """
        Add VPN/proxy detection context for fraud detection.

        Flags transactions from masked IP addresses (VPN, proxy, Tor, datacenter IPs).

        Args:
            context: Context dictionary to update
            transaction: Transaction data
        """
        # Extract IP address from transaction metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            context["vpn_proxy_check_available"] = False
            return

        # Get current IP address
        current_ip = tx_metadata.get("ip_address")

        if not current_ip:
            context["vpn_proxy_check_available"] = False
            return

        context["vpn_proxy_check_available"] = True
        context["transaction_ip"] = current_ip

        # Initialize VPN/proxy flags
        context["is_vpn_or_proxy"] = False
        context["vpn_proxy_matches"] = []

        # Check against VPN/proxy IP database
        # 1. Check for exact IP match
        exact_matches = self.db.query(VPNProxyIP).filter(
            VPNProxyIP.status == "active",
            VPNProxyIP.ip_address == current_ip
        ).all()

        # 2. Check for subnet/range matches
        # Note: For production, you'd want proper CIDR matching using ipaddress module
        # This is a simplified check
        subnet_matches = self.db.query(VPNProxyIP).filter(
            VPNProxyIP.status == "active",
            VPNProxyIP.subnet.isnot(None)
        ).all()

        # Simple subnet checking (in production, use ipaddress.ip_address and ip_network)
        range_matches = []
        for entry in subnet_matches:
            if entry.subnet and current_ip.startswith(entry.subnet.split('/')[0].rsplit('.', 1)[0]):
                range_matches.append(entry)

        all_matches = exact_matches + range_matches

        # Check metadata for VPN/proxy indicators
        # Some detection services add flags to metadata
        vpn_detected = tx_metadata.get("is_vpn", False)
        proxy_detected = tx_metadata.get("is_proxy", False)
        tor_detected = tx_metadata.get("is_tor", False)
        datacenter_detected = tx_metadata.get("is_datacenter", False)
        hosting_detected = tx_metadata.get("is_hosting", False)

        metadata_indicators = []
        if vpn_detected:
            metadata_indicators.append("vpn")
        if proxy_detected:
            metadata_indicators.append("proxy")
        if tor_detected:
            metadata_indicators.append("tor")
        if datacenter_detected:
            metadata_indicators.append("datacenter")
        if hosting_detected:
            metadata_indicators.append("hosting")

        context["metadata_vpn_proxy_indicators"] = metadata_indicators

        # Process database matches
        if all_matches:
            context["is_vpn_or_proxy"] = True
            context["vpn_proxy_matches"] = [
                {
                    "service_type": entry.service_type,
                    "service_name": entry.service_name,
                    "provider": entry.provider,
                    "risk_level": entry.risk_level,
                    "is_residential_proxy": entry.is_residential_proxy,
                    "is_mobile_proxy": entry.is_mobile_proxy,
                    "confidence": entry.confidence,
                    "country": entry.country,
                    "source": entry.source
                }
                for entry in all_matches
            ]

            # Get highest risk level
            risk_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            max_risk = max(
                (entry.risk_level for entry in all_matches),
                key=lambda r: risk_order.get(r, 0)
            )
            context["vpn_proxy_max_risk_level"] = max_risk

            # Get highest confidence score
            max_confidence = max(entry.confidence for entry in all_matches)
            context["vpn_proxy_max_confidence"] = max_confidence

            # Identify service types detected
            service_types = list(set(entry.service_type for entry in all_matches if entry.service_type))
            context["vpn_proxy_service_types"] = service_types

            # Check for residential/mobile proxies (harder to detect, more sophisticated)
            has_residential = any(entry.is_residential_proxy for entry in all_matches)
            has_mobile = any(entry.is_mobile_proxy for entry in all_matches)
            context["is_residential_proxy"] = has_residential
            context["is_mobile_proxy"] = has_mobile

            context["vpn_proxy_match_count"] = len(all_matches)

        # Check metadata indicators even if no database match
        elif metadata_indicators:
            context["is_vpn_or_proxy"] = True
            context["vpn_proxy_detection_source"] = "metadata"
            context["vpn_proxy_service_types"] = metadata_indicators

        # Additional context from metadata
        if tx_metadata.get("connection_type"):
            context["connection_type"] = tx_metadata.get("connection_type")

        # ISP information (can help identify datacenter/hosting IPs)
        if tx_metadata.get("isp"):
            context["isp"] = tx_metadata.get("isp")

            # Common datacenter/hosting ISP indicators
            datacenter_keywords = ["amazon", "aws", "google cloud", "azure", "digitalocean",
                                   "linode", "ovh", "hetzner", "vultr", "rackspace"]
            isp_lower = tx_metadata.get("isp", "").lower()

            is_datacenter_isp = any(keyword in isp_lower for keyword in datacenter_keywords)
            if is_datacenter_isp and not context["is_vpn_or_proxy"]:
                context["is_vpn_or_proxy"] = True
                context["vpn_proxy_detection_source"] = "datacenter_isp"
                context["vpn_proxy_service_types"] = ["datacenter"]
                context["vpn_proxy_max_risk_level"] = "medium"

    def _add_geolocation_context(self, context: Dict[str, Any],
                                  account_id: str,
                                  transaction: Dict[str, Any]) -> None:
        """
        Add geo-location fraud detection context.

        Identifies transactions from unusual or high-risk geolocations by:
        - Checking against high-risk countries/cities database
        - Analyzing user's historical location patterns
        - Detecting impossible travel (e.g., US then China in 1 hour)

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Extract location from transaction metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            context["geolocation_check_available"] = False
            return

        # Get current location info
        current_country = tx_metadata.get("country") or tx_metadata.get("country_code")
        current_city = tx_metadata.get("city")
        current_region = tx_metadata.get("region") or tx_metadata.get("state")
        current_continent = tx_metadata.get("continent")
        current_ip = tx_metadata.get("ip_address")
        current_lat = tx_metadata.get("latitude")
        current_lon = tx_metadata.get("longitude")

        if not current_country:
            context["geolocation_check_available"] = False
            return

        context["geolocation_check_available"] = True
        context["transaction_country"] = current_country
        context["transaction_city"] = current_city
        context["transaction_region"] = current_region

        # Initialize flags
        context["is_high_risk_location"] = False
        context["high_risk_location_matches"] = []

        # 1. Check against high-risk locations database
        # Check country-level match
        country_matches = self.db.query(HighRiskLocation).filter(
            HighRiskLocation.status == "active",
            HighRiskLocation.country_code == current_country.upper()[:2]
        ).all()

        # Check city-level match (more specific)
        city_matches = []
        if current_city:
            city_matches = self.db.query(HighRiskLocation).filter(
                HighRiskLocation.status == "active",
                HighRiskLocation.city == current_city
            ).all()

        all_location_matches = country_matches + city_matches

        if all_location_matches:
            context["is_high_risk_location"] = True
            context["high_risk_location_matches"] = [
                {
                    "location_type": "city" if match.city else "country",
                    "country_code": match.country_code,
                    "country_name": match.country_name,
                    "city": match.city,
                    "risk_level": match.risk_level,
                    "risk_category": match.risk_category,
                    "risk_score": match.risk_score,
                    "is_sanctioned": match.is_sanctioned,
                    "is_embargoed": match.is_embargoed,
                    "has_high_fraud_rate": match.has_high_fraud_rate,
                    "has_high_cybercrime_rate": match.has_high_cybercrime_rate,
                    "requires_manual_review": match.requires_manual_review,
                    "requires_enhanced_verification": match.requires_enhanced_verification,
                    "block_by_default": match.block_by_default,
                    "reason": match.reason
                }
                for match in all_location_matches
            ]

            # Get highest risk level
            risk_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            max_risk = max(
                (match.risk_level for match in all_location_matches),
                key=lambda r: risk_order.get(r, 0)
            )
            context["location_max_risk_level"] = max_risk

            # Get highest risk score
            max_risk_score = max(match.risk_score for match in all_location_matches)
            context["location_max_risk_score"] = max_risk_score

            # Check for specific risk types
            context["location_is_sanctioned"] = any(m.is_sanctioned for m in all_location_matches)
            context["location_is_embargoed"] = any(m.is_embargoed for m in all_location_matches)
            context["location_has_high_fraud_rate"] = any(m.has_high_fraud_rate for m in all_location_matches)

            # Action recommendations
            context["location_requires_manual_review"] = any(m.requires_manual_review for m in all_location_matches)
            context["location_requires_enhanced_verification"] = any(m.requires_enhanced_verification for m in all_location_matches)
            context["location_block_by_default"] = any(m.block_by_default for m in all_location_matches)

        # 2. Analyze historical location patterns
        now = datetime.datetime.utcnow()
        ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()

        # Get historical device sessions with location data (last 90 days)
        historical_sessions = self.db.query(DeviceSession).filter(
            DeviceSession.account_id == account_id,
            DeviceSession.timestamp > ninety_days_ago,
            DeviceSession.ip_country.isnot(None)
        ).order_by(DeviceSession.timestamp.desc()).all()

        if historical_sessions:
            # Get unique countries from history
            historical_countries = [s.ip_country for s in historical_sessions if s.ip_country]
            unique_countries = list(set(historical_countries))

            context["historical_countries_count"] = len(unique_countries)
            context["historical_countries"] = unique_countries

            # Check if current country is new
            is_new_country = current_country.upper()[:2] not in [c.upper()[:2] for c in unique_countries]
            context["is_new_country"] = is_new_country

            # If mostly from one country, flag deviation
            if historical_countries:
                from collections import Counter
                country_counts = Counter(historical_countries)
                most_common_country, count = country_counts.most_common(1)[0]
                primary_country_percentage = (count / len(historical_countries)) * 100

                context["primary_country"] = most_common_country
                context["primary_country_percentage"] = primary_country_percentage

                # If 80%+ transactions from one country, current location elsewhere is suspicious
                if primary_country_percentage >= 80:
                    if current_country.upper()[:2] != most_common_country.upper()[:2]:
                        context["deviates_from_primary_country"] = True
                    else:
                        context["deviates_from_primary_country"] = False
                else:
                    context["deviates_from_primary_country"] = False

            # 3. Impossible travel detection
            # Check if there's a recent transaction from a different location
            recent_sessions = [s for s in historical_sessions if s.ip_country and s.ip_city]

            if recent_sessions and current_lat and current_lon:
                # Get most recent session
                last_session = recent_sessions[0]

                # Extract last location coordinates from metadata if available
                try:
                    last_session_metadata = json.loads(last_session.user_agent) if last_session.user_agent else {}
                    last_lat = last_session_metadata.get("latitude")
                    last_lon = last_session_metadata.get("longitude")
                    last_country = last_session.ip_country
                    last_city = last_session.ip_city

                    if last_lat and last_lon:
                        # Calculate time difference
                        last_time = datetime.datetime.fromisoformat(last_session.timestamp)
                        current_time = datetime.datetime.fromisoformat(
                            transaction.get("timestamp", now.isoformat())
                        )
                        time_diff_hours = (current_time - last_time).total_seconds() / 3600

                        # Calculate distance (simple Haversine formula)
                        import math
                        lat1, lon1 = float(last_lat), float(last_lon)
                        lat2, lon2 = float(current_lat), float(current_lon)

                        # Radius of Earth in km
                        R = 6371

                        # Convert to radians
                        lat1_rad = math.radians(lat1)
                        lat2_rad = math.radians(lat2)
                        delta_lat = math.radians(lat2 - lat1)
                        delta_lon = math.radians(lon2 - lon1)

                        # Haversine formula
                        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        distance_km = R * c

                        context["distance_from_last_transaction_km"] = distance_km
                        context["hours_since_last_transaction"] = time_diff_hours

                        # Check for impossible travel
                        # Average commercial flight speed ~800 km/h
                        # Allow for 900 km/h to account for time zones, etc.
                        max_possible_speed = 900  # km/h

                        if time_diff_hours > 0:
                            required_speed = distance_km / time_diff_hours
                            context["required_travel_speed_kmh"] = required_speed

                            # Flag if speed would need to exceed max possible
                            if required_speed > max_possible_speed and distance_km > 100:
                                context["impossible_travel_detected"] = True
                                context["impossible_travel_details"] = {
                                    "from_location": f"{last_city}, {last_country}",
                                    "to_location": f"{current_city}, {current_country}",
                                    "distance_km": distance_km,
                                    "time_hours": time_diff_hours,
                                    "required_speed_kmh": required_speed,
                                    "max_possible_speed_kmh": max_possible_speed
                                }
                            else:
                                context["impossible_travel_detected"] = False
                        else:
                            context["impossible_travel_detected"] = False

                except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                    # If we can't parse metadata or calculate distance, skip impossible travel check
                    pass
        else:
            # No historical location data
            context["is_new_country"] = True
            context["historical_countries_count"] = 0

    def _add_behavioral_biometric_context(self, context: Dict[str, Any],
                                           account_id: str,
                                           transaction: Dict[str, Any]) -> None:
        """
        Add behavioral biometrics fraud detection context.

        Monitors user interaction patterns (typing speed, mouse movement, etc.)
        and detects deviations from typical behavior indicating account takeover.

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Extract behavioral data from transaction metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            context["behavioral_biometric_check_available"] = False
            return

        # Extract behavioral metrics from metadata
        behavioral_data = tx_metadata.get("behavioral_data") or tx_metadata.get("biometrics")

        if not behavioral_data:
            context["behavioral_biometric_check_available"] = False
            return

        context["behavioral_biometric_check_available"] = True

        # Extract current session behavioral metrics
        current_typing_speed = behavioral_data.get("typing_speed_wpm")
        current_mouse_speed = behavioral_data.get("mouse_speed_px_sec")
        current_key_hold_time = behavioral_data.get("key_hold_time_ms")
        current_key_interval = behavioral_data.get("key_interval_ms")
        current_mouse_smoothness = behavioral_data.get("mouse_smoothness")
        current_click_accuracy = behavioral_data.get("click_accuracy")
        current_actions_per_min = behavioral_data.get("actions_per_minute")
        current_paste_frequency = behavioral_data.get("paste_frequency")
        current_uses_autofill = behavioral_data.get("uses_autofill", False)
        current_uses_shortcuts = behavioral_data.get("uses_shortcuts", False)

        # Store current metrics in context
        context["current_behavioral_metrics"] = {
            "typing_speed_wpm": current_typing_speed,
            "mouse_speed_px_sec": current_mouse_speed,
            "key_hold_time_ms": current_key_hold_time,
            "key_interval_ms": current_key_interval,
            "mouse_smoothness": current_mouse_smoothness,
            "click_accuracy": current_click_accuracy,
            "actions_per_minute": current_actions_per_min,
            "paste_frequency": current_paste_frequency,
            "uses_autofill": current_uses_autofill,
            "uses_shortcuts": current_uses_shortcuts
        }

        # Get historical behavioral baseline (last 90 days of normal behavior)
        now = datetime.datetime.utcnow()
        ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()

        # Get baseline behavioral profiles (excluding anomalous ones)
        baseline_profiles = self.db.query(BehavioralBiometric).filter(
            BehavioralBiometric.account_id == account_id,
            BehavioralBiometric.timestamp > ninety_days_ago,
            BehavioralBiometric.is_anomalous == False,
            BehavioralBiometric.is_baseline == True
        ).all()

        if not baseline_profiles:
            # No baseline - might be new account or first time tracking
            context["has_behavioral_baseline"] = False
            context["behavioral_deviation_detected"] = False
            context["behavioral_profile_count"] = 0
            return

        context["has_behavioral_baseline"] = True
        context["behavioral_profile_count"] = len(baseline_profiles)

        # Calculate baseline averages
        typing_speeds = [p.avg_typing_speed_wpm for p in baseline_profiles if p.avg_typing_speed_wpm is not None]
        mouse_speeds = [p.avg_mouse_speed_px_sec for p in baseline_profiles if p.avg_mouse_speed_px_sec is not None]
        key_hold_times = [p.avg_key_hold_time_ms for p in baseline_profiles if p.avg_key_hold_time_ms is not None]
        key_intervals = [p.avg_key_interval_ms for p in baseline_profiles if p.avg_key_interval_ms is not None]
        mouse_smoothness_values = [p.mouse_movement_smoothness for p in baseline_profiles if p.mouse_movement_smoothness is not None]
        click_accuracies = [p.click_accuracy for p in baseline_profiles if p.click_accuracy is not None]
        actions_per_min = [p.actions_per_minute for p in baseline_profiles if p.actions_per_minute is not None]
        paste_frequencies = [p.paste_frequency for p in baseline_profiles if p.paste_frequency is not None]

        # Calculate historical patterns for autofill and shortcuts
        uses_autofill_count = sum(1 for p in baseline_profiles if p.uses_autofill)
        uses_shortcuts_count = sum(1 for p in baseline_profiles if p.uses_shortcuts)
        total_profiles = len(baseline_profiles)

        # Helper function to calculate mean and std dev
        def calc_stats(values):
            if not values:
                return None, None
            import math
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((x - mean) ** 2 for x in values) / len(values)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0
            return mean, std_dev

        # Calculate baseline statistics
        baseline_typing_mean, baseline_typing_std = calc_stats(typing_speeds)
        baseline_mouse_speed_mean, baseline_mouse_speed_std = calc_stats(mouse_speeds)
        baseline_key_hold_mean, baseline_key_hold_std = calc_stats(key_hold_times)
        baseline_key_interval_mean, baseline_key_interval_std = calc_stats(key_intervals)
        baseline_mouse_smooth_mean, baseline_mouse_smooth_std = calc_stats(mouse_smoothness_values)
        baseline_click_acc_mean, baseline_click_acc_std = calc_stats(click_accuracies)
        baseline_actions_mean, baseline_actions_std = calc_stats(actions_per_min)
        baseline_paste_mean, baseline_paste_std = calc_stats(paste_frequencies)

        # Store baseline in context
        context["behavioral_baseline"] = {
            "typing_speed_mean": baseline_typing_mean,
            "mouse_speed_mean": baseline_mouse_speed_mean,
            "key_hold_time_mean": baseline_key_hold_mean,
            "key_interval_mean": baseline_key_interval_mean,
            "mouse_smoothness_mean": baseline_mouse_smooth_mean,
            "click_accuracy_mean": baseline_click_acc_mean,
            "actions_per_minute_mean": baseline_actions_mean,
            "paste_frequency_mean": baseline_paste_mean,
            "uses_autofill_percentage": (uses_autofill_count / total_profiles) * 100 if total_profiles > 0 else 0,
            "uses_shortcuts_percentage": (uses_shortcuts_count / total_profiles) * 100 if total_profiles > 0 else 0
        }

        # Detect behavioral deviations
        deviations = []
        deviation_threshold = 2.0  # Number of standard deviations

        # Check typing speed deviation
        if current_typing_speed and baseline_typing_mean and baseline_typing_std:
            if baseline_typing_std > 0:
                typing_deviation = abs(current_typing_speed - baseline_typing_mean) / baseline_typing_std
                if typing_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "typing_speed",
                        "current": current_typing_speed,
                        "baseline_mean": baseline_typing_mean,
                        "std_deviations": typing_deviation,
                        "severity": "high" if typing_deviation > 3.0 else "medium"
                    })

        # Check mouse speed deviation
        if current_mouse_speed and baseline_mouse_speed_mean and baseline_mouse_speed_std:
            if baseline_mouse_speed_std > 0:
                mouse_deviation = abs(current_mouse_speed - baseline_mouse_speed_mean) / baseline_mouse_speed_std
                if mouse_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "mouse_speed",
                        "current": current_mouse_speed,
                        "baseline_mean": baseline_mouse_speed_mean,
                        "std_deviations": mouse_deviation,
                        "severity": "medium"
                    })

        # Check key hold time deviation
        if current_key_hold_time and baseline_key_hold_mean and baseline_key_hold_std:
            if baseline_key_hold_std > 0:
                key_hold_deviation = abs(current_key_hold_time - baseline_key_hold_mean) / baseline_key_hold_std
                if key_hold_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "key_hold_time",
                        "current": current_key_hold_time,
                        "baseline_mean": baseline_key_hold_mean,
                        "std_deviations": key_hold_deviation,
                        "severity": "high" if key_hold_deviation > 3.0 else "medium"
                    })

        # Check key interval deviation
        if current_key_interval and baseline_key_interval_mean and baseline_key_interval_std:
            if baseline_key_interval_std > 0:
                key_interval_deviation = abs(current_key_interval - baseline_key_interval_mean) / baseline_key_interval_std
                if key_interval_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "key_interval",
                        "current": current_key_interval,
                        "baseline_mean": baseline_key_interval_mean,
                        "std_deviations": key_interval_deviation,
                        "severity": "high" if key_interval_deviation > 3.0 else "medium"
                    })

        # Check mouse smoothness deviation
        if current_mouse_smoothness and baseline_mouse_smooth_mean and baseline_mouse_smooth_std:
            if baseline_mouse_smooth_std > 0:
                smoothness_deviation = abs(current_mouse_smoothness - baseline_mouse_smooth_mean) / baseline_mouse_smooth_std
                if smoothness_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "mouse_smoothness",
                        "current": current_mouse_smoothness,
                        "baseline_mean": baseline_mouse_smooth_mean,
                        "std_deviations": smoothness_deviation,
                        "severity": "medium"
                    })

        # Check click accuracy deviation
        if current_click_accuracy and baseline_click_acc_mean and baseline_click_acc_std:
            if baseline_click_acc_std > 0:
                accuracy_deviation = abs(current_click_accuracy - baseline_click_acc_mean) / baseline_click_acc_std
                if accuracy_deviation > deviation_threshold:
                    deviations.append({
                        "metric": "click_accuracy",
                        "current": current_click_accuracy,
                        "baseline_mean": baseline_click_acc_mean,
                        "std_deviations": accuracy_deviation,
                        "severity": "medium"
                    })

        # Check autofill/shortcuts usage changes
        autofill_percentage = context["behavioral_baseline"]["uses_autofill_percentage"]
        shortcuts_percentage = context["behavioral_baseline"]["uses_shortcuts_percentage"]

        # If user always uses autofill (80%+) but suddenly doesn't, flag it
        if autofill_percentage >= 80 and not current_uses_autofill:
            deviations.append({
                "metric": "autofill_usage",
                "current": False,
                "baseline_percentage": autofill_percentage,
                "severity": "medium"
            })

        # If user never uses autofill (< 20%) but suddenly does, flag it
        if autofill_percentage <= 20 and current_uses_autofill:
            deviations.append({
                "metric": "autofill_usage",
                "current": True,
                "baseline_percentage": autofill_percentage,
                "severity": "low"
            })

        # Similar for shortcuts
        if shortcuts_percentage >= 80 and not current_uses_shortcuts:
            deviations.append({
                "metric": "keyboard_shortcuts",
                "current": False,
                "baseline_percentage": shortcuts_percentage,
                "severity": "medium"
            })

        # Store deviation results
        context["behavioral_deviations"] = deviations
        context["behavioral_deviation_detected"] = len(deviations) > 0
        context["behavioral_deviation_count"] = len(deviations)

        if deviations:
            # Calculate overall anomaly score
            severity_scores = {"low": 0.3, "medium": 0.6, "high": 0.9}
            anomaly_scores = [severity_scores.get(d.get("severity", "medium"), 0.5) for d in deviations]
            overall_anomaly_score = sum(anomaly_scores) / len(anomaly_scores) if anomaly_scores else 0
            context["behavioral_anomaly_score"] = overall_anomaly_score

            # Get max severity
            severity_order = {"low": 1, "medium": 2, "high": 3}
            max_severity = max(
                (d.get("severity", "medium") for d in deviations),
                key=lambda s: severity_order.get(s, 0)
            )
            context["behavioral_max_severity"] = max_severity

            # Flag high-risk behavioral changes
            high_severity_count = sum(1 for d in deviations if d.get("severity") == "high")
            context["behavioral_high_risk"] = high_severity_count >= 2  # 2+ high severity deviations
        else:
            context["behavioral_anomaly_score"] = 0.0
            context["behavioral_high_risk"] = False

    def _add_recipient_relationship_context(self, context: Dict[str, Any],
                                             account_id: str,
                                             transaction: Dict[str, Any]) -> None:
        """
        Add recipient relationship analysis for fraud detection.

        Evaluates:
        - If recipient is a new contact (first time transacting)
        - Time since last transaction with this recipient
        - Transaction frequency with recipient
        - Dormant relationship activation (e.g., no contact for 12 months, suddenly active)

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        counterparty_id = transaction.get("counterparty_id")

        if not counterparty_id:
            context["recipient_relationship_check_available"] = False
            return

        context["recipient_relationship_check_available"] = True
        context["recipient_id"] = counterparty_id

        now = datetime.datetime.utcnow()
        current_tx_time = datetime.datetime.fromisoformat(
            transaction.get("timestamp", now.isoformat())
        )

        # Get all previous transactions with this counterparty
        previous_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.counterparty_id == counterparty_id,
            Transaction.timestamp < current_tx_time.isoformat()
        ).order_by(Transaction.timestamp.desc()).all()

        # Check if this is a new recipient
        is_new_recipient = len(previous_txs) == 0
        context["is_new_recipient"] = is_new_recipient
        context["previous_transaction_count"] = len(previous_txs)

        if is_new_recipient:
            # New recipient - no historical relationship
            context["days_since_last_transaction"] = None
            context["is_dormant_relationship"] = False
            context["relationship_status"] = "new"
            return

        # Calculate time since last transaction with this recipient
        last_tx = previous_txs[0]  # Most recent
        last_tx_time = datetime.datetime.fromisoformat(last_tx.timestamp)
        time_since_last = current_tx_time - last_tx_time

        days_since_last = time_since_last.days
        hours_since_last = time_since_last.total_seconds() / 3600

        context["days_since_last_transaction"] = days_since_last
        context["hours_since_last_transaction"] = hours_since_last
        context["last_transaction_date"] = last_tx.timestamp
        context["last_transaction_amount"] = last_tx.amount

        # Analyze transaction frequency with this recipient
        if len(previous_txs) > 1:
            # Calculate average time between transactions
            time_gaps = []
            for i in range(len(previous_txs) - 1):
                tx1_time = datetime.datetime.fromisoformat(previous_txs[i].timestamp)
                tx2_time = datetime.datetime.fromisoformat(previous_txs[i + 1].timestamp)
                gap_days = (tx1_time - tx2_time).days
                time_gaps.append(gap_days)

            if time_gaps:
                avg_gap_days = sum(time_gaps) / len(time_gaps)
                context["avg_days_between_transactions"] = avg_gap_days
                context["transaction_frequency"] = "regular" if avg_gap_days <= 30 else "irregular"

                # Calculate standard deviation of gaps
                import math
                if len(time_gaps) > 1:
                    variance = sum((x - avg_gap_days) ** 2 for x in time_gaps) / len(time_gaps)
                    std_dev = math.sqrt(variance)
                    context["transaction_gap_std_dev"] = std_dev

                    # Check if current gap is anomalous
                    if std_dev > 0:
                        gap_deviation = abs(days_since_last - avg_gap_days) / std_dev
                        context["current_gap_deviation"] = gap_deviation

                        # Flag if gap is significantly longer than normal
                        if gap_deviation > 2.0 and days_since_last > avg_gap_days:
                            context["unusually_long_gap"] = True
                            context["gap_deviation_std"] = gap_deviation
                        else:
                            context["unusually_long_gap"] = False
                    else:
                        context["unusually_long_gap"] = False
                else:
                    context["unusually_long_gap"] = False
        else:
            # Only one previous transaction
            context["avg_days_between_transactions"] = days_since_last
            context["transaction_frequency"] = "first_repeat"

        # Classify relationship based on transaction history
        total_txs_with_recipient = len(previous_txs) + 1  # Include current

        if total_txs_with_recipient == 2:
            relationship_status = "new_repeat"  # Second transaction ever
        elif days_since_last <= 30:
            relationship_status = "active"  # Active relationship (< 30 days)
        elif days_since_last <= 90:
            relationship_status = "recent"  # Recent contact (30-90 days)
        elif days_since_last <= 180:
            relationship_status = "inactive"  # Inactive (3-6 months)
        else:
            relationship_status = "dormant"  # Dormant (6+ months)

        context["relationship_status"] = relationship_status

        # Flag dormant relationships (high fraud risk)
        # Dormant = no contact for 6+ months (180 days)
        is_dormant = days_since_last >= 180
        context["is_dormant_relationship"] = is_dormant

        if is_dormant:
            context["dormant_days"] = days_since_last
            context["dormant_risk_level"] = "critical" if days_since_last >= 365 else "high"

        # Analyze amount patterns with this recipient
        previous_amounts = [tx.amount for tx in previous_txs]
        if previous_amounts:
            avg_amount = sum(previous_amounts) / len(previous_amounts)
            max_amount = max(previous_amounts)
            min_amount = min(previous_amounts)

            context["avg_transaction_amount_with_recipient"] = avg_amount
            context["max_transaction_amount_with_recipient"] = max_amount
            context["min_transaction_amount_with_recipient"] = min_amount

            # Check if current amount is unusual for this recipient
            current_amount = transaction.get("amount", 0)

            if len(previous_amounts) > 1:
                import math
                variance = sum((x - avg_amount) ** 2 for x in previous_amounts) / len(previous_amounts)
                std_dev = math.sqrt(variance)

                if std_dev > 0:
                    amount_deviation = abs(current_amount - avg_amount) / std_dev
                    context["amount_deviation_with_recipient"] = amount_deviation

                    # Flag if amount is significantly different
                    if amount_deviation > 2.0:
                        context["unusual_amount_for_recipient"] = True
                        if current_amount > avg_amount:
                            context["unusual_amount_direction"] = "higher_than_normal"
                        else:
                            context["unusual_amount_direction"] = "lower_than_normal"
                    else:
                        context["unusual_amount_for_recipient"] = False
                else:
                    context["unusual_amount_for_recipient"] = False

            # Check if current amount exceeds previous maximum
            if current_amount > max_amount:
                context["exceeds_previous_max"] = True
                context["max_amount_exceeded_by"] = current_amount - max_amount
                context["max_amount_increase_percentage"] = ((current_amount - max_amount) / max_amount * 100) if max_amount > 0 else 0
            else:
                context["exceeds_previous_max"] = False

        # Calculate relationship metrics
        # Get first transaction with this recipient
        first_tx = previous_txs[-1] if previous_txs else None
        if first_tx:
            first_tx_time = datetime.datetime.fromisoformat(first_tx.timestamp)
            relationship_age_days = (current_tx_time - first_tx_time).days
            context["relationship_age_days"] = relationship_age_days

            # Calculate transaction frequency (transactions per month)
            if relationship_age_days > 0:
                txs_per_month = (total_txs_with_recipient / relationship_age_days) * 30
                context["transactions_per_month_with_recipient"] = txs_per_month
            else:
                context["transactions_per_month_with_recipient"] = 0

        # Flag high-risk scenarios
        risk_flags = []

        # 1. Dormant relationship suddenly active
        if is_dormant:
            risk_flags.append("dormant_relationship_reactivated")

        # 2. New recipient with large transaction
        if is_new_recipient and transaction.get("amount", 0) > 10000:
            risk_flags.append("new_recipient_large_amount")

        # 3. Unusual amount for this recipient
        if context.get("unusual_amount_for_recipient"):
            risk_flags.append("unusual_amount_for_recipient")

        # 4. Exceeds previous maximum by significant margin
        if context.get("exceeds_previous_max"):
            if context.get("max_amount_increase_percentage", 0) > 50:  # 50% increase
                risk_flags.append("significant_amount_increase")

        # 5. Very long gap (unusually long)
        if context.get("unusually_long_gap"):
            risk_flags.append("unusually_long_transaction_gap")

        context["recipient_relationship_risk_flags"] = risk_flags
        context["recipient_relationship_risk_count"] = len(risk_flags)
        context["recipient_relationship_high_risk"] = len(risk_flags) >= 2

    def _add_social_trust_score_context(self, context: Dict[str, Any],
                                         account_id: str,
                                         transaction: Dict[str, Any]) -> None:
        """
        Add social trust score for recipient fraud detection.

        Calculates a comprehensive trust score (0-100) based on multiple factors:
        - Presence in beneficiary/contact list
        - Verification status
        - Transaction history length
        - Transaction frequency
        - Relationship age
        - Social signals (mutual connections, endorsements)

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        counterparty_id = transaction.get("counterparty_id")

        if not counterparty_id:
            context["social_trust_score_available"] = False
            return

        context["social_trust_score_available"] = True

        # Initialize trust score components
        trust_factors = {}
        total_score = 0
        max_possible_score = 100

        # Factor 1: Beneficiary Status (25 points)
        beneficiary = self.db.query(Beneficiary).filter(
            Beneficiary.account_id == account_id,
            Beneficiary.counterparty_id == counterparty_id,
            Beneficiary.status == "active"
        ).first()

        if beneficiary:
            beneficiary_score = 0

            # Base points for being in beneficiary list
            beneficiary_score += 10
            trust_factors["in_beneficiary_list"] = True

            # Verification status
            if beneficiary.verified:
                beneficiary_score += 10
                trust_factors["beneficiary_verified"] = True
            else:
                trust_factors["beneficiary_verified"] = False

            # Not flagged as suspicious
            if not beneficiary.flagged_as_suspicious:
                beneficiary_score += 5
                trust_factors["not_flagged_suspicious"] = True
            else:
                trust_factors["not_flagged_suspicious"] = False
                beneficiary_score -= 5  # Penalty for suspicious flag

            trust_factors["beneficiary_score"] = beneficiary_score
            total_score += beneficiary_score
        else:
            trust_factors["in_beneficiary_list"] = False
            trust_factors["beneficiary_verified"] = False
            trust_factors["beneficiary_score"] = 0

        # Factor 2: Transaction History (30 points)
        now = datetime.datetime.utcnow()
        all_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.counterparty_id == counterparty_id
        ).order_by(Transaction.timestamp.desc()).all()

        transaction_history_score = 0

        if all_txs:
            # Number of previous transactions (up to 15 points)
            tx_count = len(all_txs)
            if tx_count >= 10:
                transaction_history_score += 15
            elif tx_count >= 5:
                transaction_history_score += 10
            elif tx_count >= 2:
                transaction_history_score += 5
            else:  # 1 transaction
                transaction_history_score += 2

            trust_factors["transaction_count"] = tx_count

            # Relationship age (up to 10 points)
            first_tx = all_txs[-1]
            first_tx_time = datetime.datetime.fromisoformat(first_tx.timestamp)
            relationship_age_days = (now - first_tx_time).days

            if relationship_age_days >= 365:  # 1+ years
                transaction_history_score += 10
            elif relationship_age_days >= 180:  # 6+ months
                transaction_history_score += 7
            elif relationship_age_days >= 90:  # 3+ months
                transaction_history_score += 5
            elif relationship_age_days >= 30:  # 1+ month
                transaction_history_score += 3
            else:
                transaction_history_score += 1

            trust_factors["relationship_age_days"] = relationship_age_days

            # Transaction recency (up to 5 points)
            # Penalize if last transaction was too long ago
            last_tx = all_txs[0]
            last_tx_time = datetime.datetime.fromisoformat(last_tx.timestamp)
            days_since_last = (now - last_tx_time).days

            if days_since_last <= 30:  # Recent
                transaction_history_score += 5
            elif days_since_last <= 90:
                transaction_history_score += 3
            elif days_since_last <= 180:
                transaction_history_score += 1
            else:  # Dormant - no bonus, potential risk
                transaction_history_score += 0

            trust_factors["days_since_last_transaction"] = days_since_last
        else:
            # New recipient - low trust
            trust_factors["transaction_count"] = 0
            trust_factors["relationship_age_days"] = 0
            transaction_history_score = 0

        trust_factors["transaction_history_score"] = transaction_history_score
        total_score += transaction_history_score

        # Factor 3: Contact List Presence (15 points)
        # Check transaction metadata for contact list indicators
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        contact_score = 0

        if tx_metadata:
            # Check if recipient is in contact list
            in_contact_list = tx_metadata.get("in_contact_list", False)
            if in_contact_list:
                contact_score += 10
                trust_factors["in_contact_list"] = True
            else:
                trust_factors["in_contact_list"] = False

            # Check if recipient's contact info is saved
            has_saved_info = tx_metadata.get("has_saved_email") or tx_metadata.get("has_saved_phone")
            if has_saved_info:
                contact_score += 5
                trust_factors["has_saved_contact_info"] = True
            else:
                trust_factors["has_saved_contact_info"] = False
        else:
            trust_factors["in_contact_list"] = False
            trust_factors["has_saved_contact_info"] = False

        trust_factors["contact_score"] = contact_score
        total_score += contact_score

        # Factor 4: Social Signals (15 points)
        social_score = 0

        if tx_metadata:
            # Mutual connections
            mutual_connections = tx_metadata.get("mutual_connections", 0)
            if mutual_connections >= 5:
                social_score += 8
            elif mutual_connections >= 2:
                social_score += 5
            elif mutual_connections >= 1:
                social_score += 3

            trust_factors["mutual_connections"] = mutual_connections

            # Endorsements or references
            has_endorsements = tx_metadata.get("has_endorsements", False)
            if has_endorsements:
                social_score += 5
                trust_factors["has_endorsements"] = True
            else:
                trust_factors["has_endorsements"] = False

            # Known entity (business, verified organization)
            is_known_entity = tx_metadata.get("is_verified_business", False) or tx_metadata.get("is_registered_business", False)
            if is_known_entity:
                social_score += 2
                trust_factors["is_known_entity"] = True
            else:
                trust_factors["is_known_entity"] = False
        else:
            trust_factors["mutual_connections"] = 0
            trust_factors["has_endorsements"] = False
            trust_factors["is_known_entity"] = False

        trust_factors["social_score"] = social_score
        total_score += social_score

        # Factor 5: Transaction Pattern Consistency (15 points)
        pattern_score = 0

        if all_txs and len(all_txs) >= 3:
            # Analyze amount consistency
            amounts = [tx.amount for tx in all_txs]
            avg_amount = sum(amounts) / len(amounts)

            # Calculate coefficient of variation (std dev / mean)
            import math
            if len(amounts) > 1:
                variance = sum((x - avg_amount) ** 2 for x in amounts) / len(amounts)
                std_dev = math.sqrt(variance)

                if avg_amount > 0:
                    coefficient_of_variation = std_dev / avg_amount

                    # Lower variation = more consistent = higher trust
                    if coefficient_of_variation <= 0.3:  # Very consistent
                        pattern_score += 10
                    elif coefficient_of_variation <= 0.6:  # Moderately consistent
                        pattern_score += 6
                    elif coefficient_of_variation <= 1.0:  # Somewhat consistent
                        pattern_score += 3

                    trust_factors["amount_consistency_score"] = 10 - min(coefficient_of_variation * 10, 10)

            # Transaction frequency consistency (if we have enough data)
            if len(all_txs) >= 5:
                time_gaps = []
                for i in range(len(all_txs) - 1):
                    tx1_time = datetime.datetime.fromisoformat(all_txs[i].timestamp)
                    tx2_time = datetime.datetime.fromisoformat(all_txs[i + 1].timestamp)
                    gap_days = (tx1_time - tx2_time).days
                    time_gaps.append(gap_days)

                if time_gaps:
                    avg_gap = sum(time_gaps) / len(time_gaps)
                    gap_variance = sum((x - avg_gap) ** 2 for x in time_gaps) / len(time_gaps)
                    gap_std = math.sqrt(gap_variance)

                    if avg_gap > 0:
                        gap_cv = gap_std / avg_gap

                        # Consistent timing = higher trust
                        if gap_cv <= 0.5:  # Very regular
                            pattern_score += 5
                        elif gap_cv <= 1.0:  # Moderately regular
                            pattern_score += 3

                        trust_factors["timing_consistency_score"] = 5 - min(gap_cv * 5, 5)
        else:
            trust_factors["amount_consistency_score"] = 0
            trust_factors["timing_consistency_score"] = 0

        trust_factors["pattern_score"] = pattern_score
        total_score += pattern_score

        # Calculate final trust score (0-100)
        trust_score = min(total_score, max_possible_score)

        # Normalize to 0.0-1.0 scale as well
        trust_score_normalized = trust_score / 100.0

        # Classify trust level
        if trust_score >= 80:
            trust_level = "high"
        elif trust_score >= 60:
            trust_level = "medium_high"
        elif trust_score >= 40:
            trust_level = "medium"
        elif trust_score >= 20:
            trust_level = "low"
        else:
            trust_level = "very_low"

        # Store in context
        context["social_trust_score"] = trust_score
        context["social_trust_score_normalized"] = trust_score_normalized
        context["social_trust_level"] = trust_level
        context["social_trust_factors"] = trust_factors

        # Flag low trust recipients
        context["is_low_trust_recipient"] = trust_score < 40
        context["is_very_low_trust_recipient"] = trust_score < 20
        context["is_high_trust_recipient"] = trust_score >= 80

        # Calculate trust deficit for low trust recipients
        if trust_score < 60:
            trust_deficit = 60 - trust_score
            context["trust_deficit"] = trust_deficit
            context["requires_additional_verification"] = trust_deficit >= 20

    def _add_account_age_context(self, context: Dict[str, Any],
                                  account_id: str,
                                  transaction: Dict[str, Any]) -> None:
        """
        Add account age analysis for fraud detection.

        Analyzes account maturity and flags newly created accounts performing
        high-risk transactions. New accounts are prime targets for fraud.

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Get account information (already queried earlier but re-fetch for completeness)
        account = self.db.query(Account).filter(Account.account_id == account_id).first()

        if not account:
            context["account_age_check_available"] = False
            return

        context["account_age_check_available"] = True

        # Calculate account age (also calculated earlier, but ensure we have it)
        creation_date = datetime.datetime.fromisoformat(account.creation_date)
        now = datetime.datetime.utcnow()
        account_age_days = (now - creation_date).days
        account_age_hours = (now - creation_date).total_seconds() / 3600

        context["account_creation_date"] = account.creation_date
        context["account_age_days"] = account_age_days
        context["account_age_hours"] = account_age_hours

        # Classify account maturity
        if account_age_days < 1:
            account_maturity = "brand_new"  # Less than 1 day old
        elif account_age_days < 7:
            account_maturity = "very_new"  # Less than 1 week
        elif account_age_days < 30:
            account_maturity = "new"  # Less than 1 month
        elif account_age_days < 90:
            account_maturity = "young"  # Less than 3 months
        elif account_age_days < 180:
            account_maturity = "maturing"  # 3-6 months
        elif account_age_days < 365:
            account_maturity = "established"  # 6-12 months
        else:
            account_maturity = "mature"  # 1+ years

        context["account_maturity"] = account_maturity

        # Flag new accounts
        is_brand_new = account_age_days < 1
        is_very_new = account_age_days < 7
        is_new = account_age_days < 30

        context["is_brand_new_account"] = is_brand_new
        context["is_very_new_account"] = is_very_new
        context["is_new_account"] = is_new

        # Get all transactions for this account
        all_account_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id
        ).order_by(Transaction.timestamp).all()

        total_txs = len(all_account_txs)
        context["total_account_transactions"] = total_txs

        # Calculate transaction velocity since account creation
        if account_age_days > 0:
            txs_per_day = total_txs / account_age_days
            context["transactions_per_day_since_creation"] = txs_per_day
        else:
            context["transactions_per_day_since_creation"] = total_txs  # Brand new account

        # Analyze transaction patterns for new accounts
        current_amount = transaction.get("amount", 0)
        current_direction = transaction.get("direction", "")

        # Risk flags for new accounts
        risk_flags = []

        # Flag 1: Brand new account (< 24 hours)
        if is_brand_new:
            risk_flags.append("brand_new_account")

        # Flag 2: Very new account with large transaction
        if is_very_new and current_amount > 5000:
            risk_flags.append("very_new_account_large_amount")

        # Flag 3: New account with very large transaction
        if is_new and current_amount > 10000:
            risk_flags.append("new_account_very_large_amount")

        # Flag 4: New account with high transaction velocity
        if is_new and total_txs >= 10:
            risk_flags.append("new_account_high_velocity")

        # Flag 5: Brand new account with outgoing transfer
        if is_brand_new and current_direction == "debit":
            risk_flags.append("brand_new_account_outgoing_transfer")

        # Flag 6: Very new account with international transaction
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if tx_metadata:
            country = tx_metadata.get("country") or tx_metadata.get("country_code")
            if is_very_new and country and country.upper()[:2] != "US":
                risk_flags.append("very_new_account_international")

        # Analyze first transaction timing
        if all_account_txs:
            first_tx = all_account_txs[0]
            first_tx_time = datetime.datetime.fromisoformat(first_tx.timestamp)
            time_to_first_tx = (first_tx_time - creation_date).total_seconds() / 3600  # hours

            context["hours_to_first_transaction"] = time_to_first_tx

            # Flag 7: Immediate first transaction (within 1 hour of account creation)
            if time_to_first_tx < 1:
                risk_flags.append("immediate_first_transaction")

            # Flag 8: First transaction is large
            if first_tx.amount > 5000:
                risk_flags.append("first_transaction_large_amount")

        # Calculate account age vs transaction amount risk score
        # New accounts with large amounts are risky
        account_age_amount_risk = 0

        if current_amount > 0:
            # Risk increases as amount increases and account age decreases
            amount_factor = min(current_amount / 1000, 100)  # Scale amount

            if account_age_days < 1:
                age_multiplier = 10  # Extreme risk
            elif account_age_days < 7:
                age_multiplier = 5  # Very high risk
            elif account_age_days < 30:
                age_multiplier = 3  # High risk
            elif account_age_days < 90:
                age_multiplier = 2  # Moderate risk
            else:
                age_multiplier = 1  # Lower risk

            account_age_amount_risk = amount_factor * age_multiplier

        context["account_age_amount_risk_score"] = min(account_age_amount_risk, 100)

        # Classify risk level based on account age
        if is_brand_new:
            account_age_risk_level = "critical"
        elif is_very_new:
            account_age_risk_level = "high"
        elif is_new:
            account_age_risk_level = "medium"
        elif account_age_days < 90:
            account_age_risk_level = "low"
        else:
            account_age_risk_level = "minimal"

        context["account_age_risk_level"] = account_age_risk_level

        # Store risk flags
        context["account_age_risk_flags"] = risk_flags
        context["account_age_risk_flag_count"] = len(risk_flags)
        context["account_age_high_risk"] = len(risk_flags) >= 2 or is_brand_new

        # Calculate average transaction amount for account
        if all_account_txs:
            amounts = [tx.amount for tx in all_account_txs]
            avg_account_amount = sum(amounts) / len(amounts)
            max_account_amount = max(amounts)

            context["avg_account_transaction_amount"] = avg_account_amount
            context["max_account_transaction_amount"] = max_account_amount

            # Check if current transaction is unusually large for this account
            if current_amount > max_account_amount:
                context["current_exceeds_account_max"] = True
                context["account_max_exceeded_by"] = current_amount - max_account_amount
            else:
                context["current_exceeds_account_max"] = False

            # For new accounts, check if transaction is much larger than average
            if is_new and total_txs >= 3:
                if current_amount > avg_account_amount * 3:  # 3x average
                    risk_flags.append("new_account_amount_spike")
                    context["account_age_risk_flags"] = risk_flags
                    context["account_age_risk_flag_count"] = len(risk_flags)

        # Analyze account activity pattern
        # New accounts with burst activity are suspicious
        if is_new and total_txs > 0:
            # Calculate daily transaction rate
            daily_rate = total_txs / max(account_age_days, 1)

            if daily_rate >= 5:  # 5+ transactions per day
                context["account_high_activity_rate"] = True
                context["account_daily_transaction_rate"] = daily_rate
            else:
                context["account_high_activity_rate"] = False
                context["account_daily_transaction_rate"] = daily_rate
        else:
            context["account_high_activity_rate"] = False

        # Check for account warming pattern
        # Fraudsters often "warm up" accounts with small transactions before fraud
        if is_new and total_txs >= 5:
            small_tx_count = sum(1 for tx in all_account_txs if abs(tx.amount) <= 100)
            small_tx_percentage = (small_tx_count / total_txs) * 100

            # If 50%+ transactions are small, might be warming pattern
            if small_tx_percentage >= 50 and current_amount > 1000:
                risk_flags.append("account_warming_pattern")
                context["account_age_risk_flags"] = risk_flags
                context["account_age_risk_flag_count"] = len(risk_flags)
                context["account_warming_detected"] = True
                context["small_transaction_percentage"] = small_tx_percentage
            else:
                context["account_warming_detected"] = False
        else:
            context["account_warming_detected"] = False

    def _add_high_risk_transaction_times_context(self, context: Dict[str, Any],
                                                   account_id: str,
                                                   transaction: Dict[str, Any]) -> None:
        """
        Add high-risk transaction times detection for fraud analysis.

        Flags transactions occurring during non-business hours, unusual times,
        weekends, holidays, and detects timing pattern anomalies that may
        indicate account takeover or fraudulent activity.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        import calendar
        from typing import List, Tuple

        now = datetime.datetime.utcnow()

        # Get transaction timestamp
        tx_timestamp_str = transaction.get("timestamp", now.isoformat())
        tx_timestamp = datetime.datetime.fromisoformat(tx_timestamp_str)
        tx_amount = abs(float(transaction.get("amount", 0)))

        # Extract time components
        tx_hour = tx_timestamp.hour
        tx_minute = tx_timestamp.minute
        tx_weekday = tx_timestamp.weekday()  # Monday=0, Sunday=6
        tx_day = tx_timestamp.day

        context["transaction_hour"] = tx_hour
        context["transaction_minute"] = tx_minute
        context["transaction_weekday"] = tx_weekday
        context["transaction_day_of_month"] = tx_day

        # Define time risk windows
        time_windows = {
            "deep_night": (0, 5),      # 12 AM - 5 AM (highest risk)
            "early_morning": (5, 7),   # 5 AM - 7 AM
            "morning": (7, 9),         # 7 AM - 9 AM
            "business_hours": (9, 17), # 9 AM - 5 PM (lowest risk)
            "evening": (17, 22),       # 5 PM - 10 PM
            "late_night": (22, 24)     # 10 PM - 12 AM (high risk)
        }

        # Determine current time window
        current_window = None
        for window_name, (start_hour, end_hour) in time_windows.items():
            if start_hour <= tx_hour < end_hour:
                current_window = window_name
                break

        context["time_window"] = current_window

        # Calculate base time risk score (0-100)
        # Deep night and late night have highest base risk
        time_risk_scores = {
            "deep_night": 85,
            "early_morning": 60,
            "morning": 30,
            "business_hours": 10,
            "evening": 25,
            "late_night": 70
        }

        base_time_risk = time_risk_scores.get(current_window, 50)
        context["base_time_risk_score"] = base_time_risk

        # Check weekend
        is_weekend = tx_weekday >= 5
        context["is_weekend"] = is_weekend

        # Check if holiday (US Federal Holidays for 2024-2025)
        def is_holiday(dt: datetime.datetime) -> Tuple[bool, str]:
            """Check if date is a US federal holiday"""
            year = dt.year
            month = dt.month
            day = dt.day

            # Fixed holidays
            fixed_holidays = {
                (1, 1): "New Year's Day",
                (7, 4): "Independence Day",
                (11, 11): "Veterans Day",
                (12, 25): "Christmas Day"
            }

            if (month, day) in fixed_holidays:
                return True, fixed_holidays[(month, day)]

            # MLK Day - 3rd Monday in January
            if month == 1 and tx_weekday == 0:
                if 15 <= day <= 21:
                    return True, "Martin Luther King Jr. Day"

            # Presidents Day - 3rd Monday in February
            if month == 2 and tx_weekday == 0:
                if 15 <= day <= 21:
                    return True, "Presidents Day"

            # Memorial Day - Last Monday in May
            if month == 5 and tx_weekday == 0:
                last_day = calendar.monthrange(year, 5)[1]
                if day > last_day - 7:
                    return True, "Memorial Day"

            # Labor Day - 1st Monday in September
            if month == 9 and tx_weekday == 0:
                if day <= 7:
                    return True, "Labor Day"

            # Columbus Day - 2nd Monday in October
            if month == 10 and tx_weekday == 0:
                if 8 <= day <= 14:
                    return True, "Columbus Day"

            # Thanksgiving - 4th Thursday in November
            if month == 11 and tx_weekday == 3:
                if 22 <= day <= 28:
                    return True, "Thanksgiving Day"

            return False, ""

        is_holiday_flag, holiday_name = is_holiday(tx_timestamp)
        context["is_holiday"] = is_holiday_flag
        context["holiday_name"] = holiday_name if is_holiday_flag else None

        # End of month pattern (fraudsters often target payroll dates)
        is_end_of_month = tx_day >= 28 or tx_day <= 3
        context["is_end_of_month"] = is_end_of_month

        # Get historical transactions for pattern analysis
        lookback_days = 90
        lookback_date = (now - datetime.timedelta(days=lookback_days)).isoformat()

        historical_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > lookback_date
        ).all()

        context["historical_transaction_count_90d"] = len(historical_txs)

        if len(historical_txs) >= 5:  # Need minimum data
            # Analyze hourly patterns
            hour_distribution = [0] * 24
            weekday_distribution = [0] * 7
            weekend_tx_count = 0
            business_hours_count = 0
            non_business_hours_count = 0
            deep_night_count = 0
            holiday_count = 0

            for hist_tx in historical_txs:
                hist_time = datetime.datetime.fromisoformat(hist_tx.timestamp)
                hist_hour = hist_time.hour
                hist_weekday = hist_time.weekday()

                hour_distribution[hist_hour] += 1
                weekday_distribution[hist_weekday] += 1

                # Count weekend transactions
                if hist_weekday >= 5:
                    weekend_tx_count += 1

                # Count business hours vs non-business hours
                if 9 <= hist_hour < 17:
                    business_hours_count += 1
                else:
                    non_business_hours_count += 1

                # Count deep night transactions
                if 0 <= hist_hour < 5:
                    deep_night_count += 1

                # Count holiday transactions
                if is_holiday(hist_time)[0]:
                    holiday_count += 1

            total_hist = len(historical_txs)

            # Calculate pattern ratios
            context["historical_weekend_ratio"] = weekend_tx_count / total_hist
            context["historical_business_hours_ratio"] = business_hours_count / total_hist
            context["historical_non_business_hours_ratio"] = non_business_hours_count / total_hist
            context["historical_deep_night_ratio"] = deep_night_count / total_hist
            context["historical_holiday_ratio"] = holiday_count / total_hist

            # Calculate statistical metrics for timing
            import statistics

            # Convert hour distribution to percentage
            hour_percentages = [(count / total_hist) * 100 for count in hour_distribution]
            current_hour_percentage = hour_percentages[tx_hour]

            context["current_hour_historical_percentage"] = current_hour_percentage
            context["hour_distribution"] = hour_distribution

            # Determine if current time deviates from pattern
            avg_hour_percentage = 100 / 24  # ~4.17%

            # Flag if this hour is historically uncommon (less than 25% of average)
            hour_is_uncommon = current_hour_percentage < (avg_hour_percentage * 0.25)
            context["hour_is_uncommon"] = hour_is_uncommon

            # Flag if user typically transacts during business hours but this is off-hours
            deviates_from_business_hours_pattern = (
                context["historical_business_hours_ratio"] > 0.75 and
                current_window != "business_hours"
            )
            context["deviates_from_business_hours_pattern"] = deviates_from_business_hours_pattern

            # Flag if user rarely transacts on weekends but this is weekend
            deviates_from_weekday_pattern = (
                is_weekend and
                context["historical_weekend_ratio"] < 0.15
            )
            context["deviates_from_weekday_pattern"] = deviates_from_weekday_pattern

            # Detect sudden change in timing patterns (possible account takeover)
            # Look at last 7 days vs prior 83 days
            recent_cutoff = (now - datetime.timedelta(days=7)).isoformat()
            recent_txs = [tx for tx in historical_txs
                         if tx.timestamp > recent_cutoff]
            older_txs = [tx for tx in historical_txs
                        if tx.timestamp <= recent_cutoff]

            if len(recent_txs) >= 3 and len(older_txs) >= 5:
                # Compare timing patterns
                recent_business_hours = sum(1 for tx in recent_txs
                                          if 9 <= datetime.datetime.fromisoformat(tx.timestamp).hour < 17)
                older_business_hours = sum(1 for tx in older_txs
                                         if 9 <= datetime.datetime.fromisoformat(tx.timestamp).hour < 17)

                recent_bh_ratio = recent_business_hours / len(recent_txs)
                older_bh_ratio = older_business_hours / len(older_txs)

                # Significant shift in timing pattern (>40% change)
                timing_pattern_shift = abs(recent_bh_ratio - older_bh_ratio)
                context["timing_pattern_shift"] = timing_pattern_shift
                context["sudden_timing_change"] = timing_pattern_shift > 0.4

                # If recently shifted to odd hours, high risk
                context["shifted_to_odd_hours"] = (
                    recent_bh_ratio < 0.3 and older_bh_ratio > 0.7
                )
            else:
                context["timing_pattern_shift"] = 0
                context["sudden_timing_change"] = False
                context["shifted_to_odd_hours"] = False
        else:
            # Insufficient historical data
            context["insufficient_timing_history"] = True
            context["hour_is_uncommon"] = False
            context["deviates_from_business_hours_pattern"] = False
            context["deviates_from_weekday_pattern"] = False
            context["timing_pattern_shift"] = 0
            context["sudden_timing_change"] = False
            context["shifted_to_odd_hours"] = False

        # Analyze recent velocity at unusual times
        recent_7d_cutoff = (now - datetime.timedelta(days=7)).isoformat()
        recent_7d_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > recent_7d_cutoff
        ).all()

        recent_deep_night_txs = []
        recent_weekend_txs = []
        recent_holiday_txs = []

        for tx in recent_7d_txs:
            tx_time = datetime.datetime.fromisoformat(tx.timestamp)
            tx_h = tx_time.hour
            tx_wd = tx_time.weekday()

            if 0 <= tx_h < 5:
                recent_deep_night_txs.append(tx)

            if tx_wd >= 5:
                recent_weekend_txs.append(tx)

            if is_holiday(tx_time)[0]:
                recent_holiday_txs.append(tx)

        context["recent_deep_night_transaction_count"] = len(recent_deep_night_txs)
        context["recent_weekend_transaction_count"] = len(recent_weekend_txs)
        context["recent_holiday_transaction_count"] = len(recent_holiday_txs)

        # Calculate total amounts for unusual times
        if recent_deep_night_txs:
            context["recent_deep_night_total_amount"] = sum(abs(tx.amount) for tx in recent_deep_night_txs)

        if recent_weekend_txs:
            context["recent_weekend_total_amount"] = sum(abs(tx.amount) for tx in recent_weekend_txs)

        if recent_holiday_txs:
            context["recent_holiday_total_amount"] = sum(abs(tx.amount) for tx in recent_holiday_txs)

        # Check for timezone anomalies (rapid location changes)
        # Look for transactions from different time zones in short period
        recent_24h_cutoff = (now - datetime.timedelta(hours=24)).isoformat()
        recent_24h_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > recent_24h_cutoff
        ).order_by(Transaction.timestamp).all()

        if len(recent_24h_txs) >= 2:
            # Check if transactions show rapid timezone changes
            # (This is simplified - in production, you'd use actual location data)
            transaction_hours = [datetime.datetime.fromisoformat(tx.timestamp).hour
                               for tx in recent_24h_txs]

            # Look for unusual hour jumping (possible VPN/location spoofing)
            hour_jumps = []
            for i in range(1, len(transaction_hours)):
                jump = abs(transaction_hours[i] - transaction_hours[i-1])
                # Normalize for 24-hour wrap
                if jump > 12:
                    jump = 24 - jump
                hour_jumps.append(jump)

            if hour_jumps:
                max_hour_jump = max(hour_jumps)
                context["max_hour_jump_24h"] = max_hour_jump

                # Significant timezone jump (>6 hours) in 24h period
                context["rapid_timezone_change"] = max_hour_jump > 6
            else:
                context["max_hour_jump_24h"] = 0
                context["rapid_timezone_change"] = False
        else:
            context["max_hour_jump_24h"] = 0
            context["rapid_timezone_change"] = False

        # Generate risk flags
        risk_flags = []

        # Deep night transaction (12 AM - 5 AM)
        if current_window == "deep_night":
            risk_flags.append("deep_night_transaction")

            # Extra flag for midnight hour (12 AM - 1 AM)
            if tx_hour == 0:
                risk_flags.append("midnight_transaction")

        # Late night transaction (10 PM - 12 AM)
        if current_window == "late_night":
            risk_flags.append("late_night_transaction")

        # Weekend large transaction
        if is_weekend and tx_amount > 5000:
            risk_flags.append("weekend_large_transaction")

        # Weekend unusual (if user rarely transacts on weekends)
        if is_weekend and context.get("deviates_from_weekday_pattern", False):
            risk_flags.append("weekend_unusual_for_user")

        # Holiday transaction
        if is_holiday_flag:
            risk_flags.append("holiday_transaction")

            if tx_amount > 5000:
                risk_flags.append("holiday_large_transaction")

        # Outside business hours high value
        if current_window != "business_hours" and tx_amount > 10000:
            risk_flags.append("outside_business_hours_high_value")

        # Unusual time for user
        if context.get("hour_is_uncommon", False):
            risk_flags.append("unusual_time_for_user")

        # Deviates from business hours pattern
        if context.get("deviates_from_business_hours_pattern", False):
            risk_flags.append("deviates_from_typical_hours")

        # Rapid timezone change
        if context.get("rapid_timezone_change", False):
            risk_flags.append("rapid_timezone_change")

        # Sudden timing pattern change (possible account takeover)
        if context.get("sudden_timing_change", False):
            risk_flags.append("sudden_timing_pattern_change")

        # Shifted to odd hours recently
        if context.get("shifted_to_odd_hours", False):
            risk_flags.append("recently_shifted_to_odd_hours")

        # Consistent deep night activity (multiple in recent period)
        if context.get("recent_deep_night_transaction_count", 0) >= 3:
            risk_flags.append("consistent_deep_night_activity")

        # Multiple weekend transactions recently
        if context.get("recent_weekend_transaction_count", 0) >= 3:
            if context.get("historical_weekend_ratio", 1) < 0.2:
                risk_flags.append("unusual_weekend_activity_spike")

        # Early morning high value (5 AM - 7 AM with large amount)
        if current_window == "early_morning" and tx_amount > 7500:
            risk_flags.append("early_morning_high_value")

        context["high_risk_time_flags"] = risk_flags
        context["high_risk_time_flag_count"] = len(risk_flags)

        # Calculate comprehensive time-based risk score (0-100)
        risk_score = base_time_risk

        # Adjust for amount
        if tx_amount > 10000:
            risk_score += 15
        elif tx_amount > 5000:
            risk_score += 10
        elif tx_amount > 2000:
            risk_score += 5

        # Adjust for weekend
        if is_weekend:
            risk_score += 10

        # Adjust for holiday
        if is_holiday_flag:
            risk_score += 15

        # Adjust for pattern deviation
        if context.get("deviates_from_business_hours_pattern", False):
            risk_score += 20

        if context.get("hour_is_uncommon", False):
            risk_score += 15

        # Adjust for timezone anomaly
        if context.get("rapid_timezone_change", False):
            risk_score += 25

        # Adjust for sudden timing change (major red flag)
        if context.get("sudden_timing_change", False):
            risk_score += 30

        # Adjust for shifted to odd hours
        if context.get("shifted_to_odd_hours", False):
            risk_score += 35

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["high_risk_time_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            time_risk_level = "critical"
        elif risk_score >= 60:
            time_risk_level = "high"
        elif risk_score >= 40:
            time_risk_level = "medium"
        elif risk_score >= 20:
            time_risk_level = "low"
        else:
            time_risk_level = "minimal"

        context["time_risk_level"] = time_risk_level

    def _add_past_fraud_flags_context(self, context: Dict[str, Any],
                                       account_id: str,
                                       transaction: Dict[str, Any]) -> None:
        """
        Add past fraudulent behavior flags detection for fraud analysis.

        Checks if the user or recipient has been flagged for prior fraudulent
        activity, including fraud type, severity, recency, and patterns of
        repeat offenses.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        now = datetime.datetime.utcnow()
        tx_amount = abs(float(transaction.get("amount", 0)))

        # Get beneficiary/recipient ID
        beneficiary_id = transaction.get("beneficiary_id") or transaction.get("recipient_id")

        # Initialize fraud history containers
        context["account_has_fraud_history"] = False
        context["beneficiary_has_fraud_history"] = False
        context["combined_fraud_risk_score"] = 0

        # Query account fraud flags
        account_fraud_flags = self.db.query(FraudFlag).filter(
            FraudFlag.entity_type == "account",
            FraudFlag.entity_id == account_id
        ).order_by(FraudFlag.incident_date.desc()).all()

        if account_fraud_flags:
            context["account_has_fraud_history"] = True
            context["account_total_fraud_flags"] = len(account_fraud_flags)

            # Categorize by status
            active_flags = [f for f in account_fraud_flags if f.status == "active"]
            confirmed_flags = [f for f in account_fraud_flags if f.disposition == "confirmed_fraud"]
            resolved_flags = [f for f in account_fraud_flags if f.status == "resolved"]
            disputed_flags = [f for f in account_fraud_flags if f.status == "disputed"]

            context["account_active_fraud_flags"] = len(active_flags)
            context["account_confirmed_fraud_flags"] = len(confirmed_flags)
            context["account_resolved_fraud_flags"] = len(resolved_flags)
            context["account_disputed_fraud_flags"] = len(disputed_flags)

            # Categorize by severity
            critical_flags = [f for f in account_fraud_flags if f.severity == "critical"]
            high_flags = [f for f in account_fraud_flags if f.severity == "high"]
            medium_flags = [f for f in account_fraud_flags if f.severity == "medium"]
            low_flags = [f for f in account_fraud_flags if f.severity == "low"]

            context["account_critical_fraud_flags"] = len(critical_flags)
            context["account_high_fraud_flags"] = len(high_flags)
            context["account_medium_fraud_flags"] = len(medium_flags)
            context["account_low_fraud_flags"] = len(low_flags)

            # Get fraud types and categories
            fraud_types = list(set([f.fraud_type for f in account_fraud_flags]))
            fraud_categories = list(set([f.fraud_category for f in account_fraud_flags]))

            context["account_fraud_types"] = fraud_types
            context["account_fraud_categories"] = fraud_categories
            context["account_unique_fraud_types"] = len(fraud_types)

            # Analyze recency of most recent fraud
            most_recent_flag = account_fraud_flags[0]  # Already sorted by incident_date desc
            days_since_last_fraud = (now - most_recent_flag.incident_date).days

            context["account_days_since_last_fraud"] = days_since_last_fraud
            context["account_most_recent_fraud_type"] = most_recent_flag.fraud_type
            context["account_most_recent_fraud_severity"] = most_recent_flag.severity
            context["account_most_recent_fraud_status"] = most_recent_flag.status

            # Recency classification
            if days_since_last_fraud <= 7:
                recency_category = "very_recent"
            elif days_since_last_fraud <= 30:
                recency_category = "recent"
            elif days_since_last_fraud <= 90:
                recency_category = "moderately_recent"
            elif days_since_last_fraud <= 180:
                recency_category = "somewhat_recent"
            elif days_since_last_fraud <= 365:
                recency_category = "past_year"
            else:
                recency_category = "historical"

            context["account_fraud_recency_category"] = recency_category

            # Calculate total amount involved in past fraud
            total_fraud_amount = sum(float(f.amount_involved or 0) for f in account_fraud_flags
                                    if f.amount_involved is not None)
            context["account_total_fraud_amount"] = total_fraud_amount

            # Analyze fraud patterns
            # Check for repeat fraud (multiple incidents within time windows)
            fraud_last_30d = [f for f in account_fraud_flags
                             if (now - f.incident_date).days <= 30]
            fraud_last_90d = [f for f in account_fraud_flags
                             if (now - f.incident_date).days <= 90]
            fraud_last_365d = [f for f in account_fraud_flags
                              if (now - f.incident_date).days <= 365]

            context["account_fraud_flags_last_30d"] = len(fraud_last_30d)
            context["account_fraud_flags_last_90d"] = len(fraud_last_90d)
            context["account_fraud_flags_last_365d"] = len(fraud_last_365d)

            # Check for escalating pattern (increasing severity over time)
            severity_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}

            if len(account_fraud_flags) >= 2:
                # Compare recent vs older incidents
                recent_avg_severity = sum(severity_scores.get(f.severity, 0)
                                        for f in fraud_last_90d) / len(fraud_last_90d) if fraud_last_90d else 0

                older_flags = [f for f in account_fraud_flags
                              if (now - f.incident_date).days > 90]
                older_avg_severity = sum(severity_scores.get(f.severity, 0)
                                       for f in older_flags) / len(older_flags) if older_flags else 0

                escalating_pattern = recent_avg_severity > older_avg_severity and recent_avg_severity >= 2.5
                context["account_fraud_escalating_pattern"] = escalating_pattern
            else:
                context["account_fraud_escalating_pattern"] = False

            # Check if account was previously closed for fraud and reopened
            account_closed_flags = [f for f in account_fraud_flags
                                   if f.resolution_action == "account_closed"]
            context["account_previously_closed_for_fraud"] = len(account_closed_flags) > 0

            # Check for specific high-risk fraud types
            high_risk_fraud_types = [
                "identity_theft",
                "account_takeover",
                "money_laundering",
                "terrorist_financing",
                "synthetic_identity",
                "credit_card_fraud"
            ]

            has_high_risk_type = any(f.fraud_type.lower() in [t.lower() for t in high_risk_fraud_types]
                                    for f in account_fraud_flags)
            context["account_has_high_risk_fraud_type"] = has_high_risk_type

        else:
            # No fraud history for account
            context["account_total_fraud_flags"] = 0
            context["account_active_fraud_flags"] = 0
            context["account_confirmed_fraud_flags"] = 0
            context["account_fraud_types"] = []
            context["account_fraud_categories"] = []
            context["account_days_since_last_fraud"] = None
            context["account_fraud_recency_category"] = "none"
            context["account_total_fraud_amount"] = 0
            context["account_fraud_flags_last_30d"] = 0
            context["account_fraud_flags_last_90d"] = 0
            context["account_fraud_flags_last_365d"] = 0
            context["account_fraud_escalating_pattern"] = False
            context["account_previously_closed_for_fraud"] = False
            context["account_has_high_risk_fraud_type"] = False

        # Query beneficiary/recipient fraud flags
        if beneficiary_id:
            beneficiary_fraud_flags = self.db.query(FraudFlag).filter(
                FraudFlag.entity_type == "beneficiary",
                FraudFlag.entity_id == beneficiary_id
            ).order_by(FraudFlag.incident_date.desc()).all()

            if beneficiary_fraud_flags:
                context["beneficiary_has_fraud_history"] = True
                context["beneficiary_total_fraud_flags"] = len(beneficiary_fraud_flags)

                # Categorize by status
                ben_active_flags = [f for f in beneficiary_fraud_flags if f.status == "active"]
                ben_confirmed_flags = [f for f in beneficiary_fraud_flags if f.disposition == "confirmed_fraud"]

                context["beneficiary_active_fraud_flags"] = len(ben_active_flags)
                context["beneficiary_confirmed_fraud_flags"] = len(ben_confirmed_flags)

                # Categorize by severity
                ben_critical_flags = [f for f in beneficiary_fraud_flags if f.severity == "critical"]
                ben_high_flags = [f for f in beneficiary_fraud_flags if f.severity == "high"]

                context["beneficiary_critical_fraud_flags"] = len(ben_critical_flags)
                context["beneficiary_high_fraud_flags"] = len(ben_high_flags)

                # Get fraud types
                ben_fraud_types = list(set([f.fraud_type for f in beneficiary_fraud_flags]))
                context["beneficiary_fraud_types"] = ben_fraud_types

                # Recency of most recent fraud
                ben_most_recent = beneficiary_fraud_flags[0]
                ben_days_since_last = (now - ben_most_recent.incident_date).days

                context["beneficiary_days_since_last_fraud"] = ben_days_since_last
                context["beneficiary_most_recent_fraud_type"] = ben_most_recent.fraud_type
                context["beneficiary_most_recent_fraud_severity"] = ben_most_recent.severity

                # Recency classification
                if ben_days_since_last <= 30:
                    ben_recency = "recent"
                elif ben_days_since_last <= 90:
                    ben_recency = "moderately_recent"
                elif ben_days_since_last <= 365:
                    ben_recency = "past_year"
                else:
                    ben_recency = "historical"

                context["beneficiary_fraud_recency_category"] = ben_recency

                # Total amount
                ben_total_amount = sum(float(f.amount_involved or 0) for f in beneficiary_fraud_flags
                                      if f.amount_involved is not None)
                context["beneficiary_total_fraud_amount"] = ben_total_amount

                # Recent activity
                ben_fraud_last_90d = [f for f in beneficiary_fraud_flags
                                     if (now - f.incident_date).days <= 90]
                context["beneficiary_fraud_flags_last_90d"] = len(ben_fraud_last_90d)

            else:
                # No fraud history for beneficiary
                context["beneficiary_total_fraud_flags"] = 0
                context["beneficiary_active_fraud_flags"] = 0
                context["beneficiary_confirmed_fraud_flags"] = 0
                context["beneficiary_fraud_types"] = []
                context["beneficiary_days_since_last_fraud"] = None
                context["beneficiary_fraud_recency_category"] = "none"
                context["beneficiary_total_fraud_amount"] = 0
                context["beneficiary_fraud_flags_last_90d"] = 0
        else:
            # No beneficiary in transaction
            context["beneficiary_total_fraud_flags"] = 0
            context["beneficiary_active_fraud_flags"] = 0
            context["beneficiary_fraud_types"] = []
            context["beneficiary_fraud_recency_category"] = "none"

        # Generate combined risk flags
        risk_flags = []

        # Account has active fraud flags
        if context.get("account_active_fraud_flags", 0) > 0:
            risk_flags.append("account_has_active_fraud_flags")

        # Account has confirmed fraud
        if context.get("account_confirmed_fraud_flags", 0) > 0:
            risk_flags.append("account_has_confirmed_fraud")

        # Account has critical severity fraud
        if context.get("account_critical_fraud_flags", 0) > 0:
            risk_flags.append("account_has_critical_fraud")

        # Recent fraud (within 30 days)
        if context.get("account_fraud_recency_category") in ["very_recent", "recent"]:
            risk_flags.append("account_very_recent_fraud")

        # Multiple fraud incidents
        if context.get("account_total_fraud_flags", 0) >= 3:
            risk_flags.append("account_repeat_fraud_offender")

        # Escalating fraud pattern
        if context.get("account_fraud_escalating_pattern", False):
            risk_flags.append("account_fraud_escalating")

        # Previously closed for fraud
        if context.get("account_previously_closed_for_fraud", False):
            risk_flags.append("account_previously_closed_for_fraud")

        # High-risk fraud type
        if context.get("account_has_high_risk_fraud_type", False):
            risk_flags.append("account_high_risk_fraud_type")

        # Recent fraud activity (multiple in 90 days)
        if context.get("account_fraud_flags_last_90d", 0) >= 2:
            risk_flags.append("account_recent_fraud_activity")

        # Beneficiary has active fraud
        if context.get("beneficiary_active_fraud_flags", 0) > 0:
            risk_flags.append("beneficiary_has_active_fraud_flags")

        # Beneficiary has confirmed fraud
        if context.get("beneficiary_confirmed_fraud_flags", 0) > 0:
            risk_flags.append("beneficiary_has_confirmed_fraud")

        # Beneficiary has critical fraud
        if context.get("beneficiary_critical_fraud_flags", 0) > 0:
            risk_flags.append("beneficiary_has_critical_fraud")

        # Beneficiary recent fraud
        if context.get("beneficiary_fraud_recency_category") in ["recent", "moderately_recent"]:
            risk_flags.append("beneficiary_recent_fraud")

        # Both parties have fraud history
        if context.get("account_has_fraud_history") and context.get("beneficiary_has_fraud_history"):
            risk_flags.append("both_parties_have_fraud_history")

        # Large transaction from account with fraud history
        if context.get("account_has_fraud_history") and tx_amount > 5000:
            risk_flags.append("large_transaction_fraud_history_account")

        # Transaction to beneficiary with fraud history
        if context.get("beneficiary_has_fraud_history") and tx_amount > 2000:
            risk_flags.append("transaction_to_fraud_history_beneficiary")

        # Fraud history with similar transaction patterns
        if context.get("account_has_fraud_history"):
            # Check if past fraud involved similar amounts
            account_fraud_amounts = [float(f.amount_involved) for f in account_fraud_flags
                                    if f.amount_involved is not None and float(f.amount_involved) > 0]

            if account_fraud_amounts:
                import statistics
                avg_fraud_amount = statistics.mean(account_fraud_amounts)

                # If current transaction is within 20% of average fraud amount
                if avg_fraud_amount > 0:
                    similarity_ratio = abs(tx_amount - avg_fraud_amount) / avg_fraud_amount
                    if similarity_ratio < 0.2:
                        risk_flags.append("transaction_similar_to_past_fraud_amount")

        context["past_fraud_risk_flags"] = risk_flags
        context["past_fraud_risk_flag_count"] = len(risk_flags)

        # Calculate comprehensive past fraud risk score (0-100)
        risk_score = 0

        # Account fraud history scoring
        if context.get("account_has_fraud_history"):
            # Base score for having fraud history
            risk_score += 20

            # Add for active flags
            risk_score += min(context.get("account_active_fraud_flags", 0) * 15, 45)

            # Add for confirmed fraud
            risk_score += min(context.get("account_confirmed_fraud_flags", 0) * 10, 30)

            # Add for severity
            risk_score += context.get("account_critical_fraud_flags", 0) * 20
            risk_score += context.get("account_high_fraud_flags", 0) * 10

            # Add for recency
            recency_scores = {
                "very_recent": 35,
                "recent": 25,
                "moderately_recent": 15,
                "somewhat_recent": 10,
                "past_year": 5,
                "historical": 2
            }
            risk_score += recency_scores.get(context.get("account_fraud_recency_category"), 0)

            # Add for repeat offenses
            if context.get("account_total_fraud_flags", 0) >= 5:
                risk_score += 25
            elif context.get("account_total_fraud_flags", 0) >= 3:
                risk_score += 15

            # Add for escalating pattern
            if context.get("account_fraud_escalating_pattern", False):
                risk_score += 20

            # Add for previously closed
            if context.get("account_previously_closed_for_fraud", False):
                risk_score += 30

            # Add for high-risk type
            if context.get("account_has_high_risk_fraud_type", False):
                risk_score += 25

        # Beneficiary fraud history scoring
        if context.get("beneficiary_has_fraud_history"):
            # Base score for beneficiary fraud
            risk_score += 15

            # Add for active/confirmed
            risk_score += min(context.get("beneficiary_active_fraud_flags", 0) * 10, 30)
            risk_score += min(context.get("beneficiary_confirmed_fraud_flags", 0) * 8, 24)

            # Add for severity
            risk_score += context.get("beneficiary_critical_fraud_flags", 0) * 15

            # Add for recency
            ben_recency_scores = {
                "recent": 20,
                "moderately_recent": 12,
                "past_year": 6,
                "historical": 2
            }
            risk_score += ben_recency_scores.get(context.get("beneficiary_fraud_recency_category"), 0)

        # Add for both parties having fraud history
        if context.get("account_has_fraud_history") and context.get("beneficiary_has_fraud_history"):
            risk_score += 30

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["past_fraud_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 80:
            fraud_risk_level = "critical"
        elif risk_score >= 60:
            fraud_risk_level = "high"
        elif risk_score >= 40:
            fraud_risk_level = "medium"
        elif risk_score >= 20:
            fraud_risk_level = "low"
        else:
            fraud_risk_level = "minimal"

        context["past_fraud_risk_level"] = fraud_risk_level

    def _add_location_inconsistent_transactions_context(self, context: Dict[str, Any],
                                                         account_id: str,
                                                         transaction: Dict[str, Any]) -> None:
        """
        Add location-inconsistent transactions detection for fraud analysis.

        Detects transactions initiated from multiple geolocations in short time
        frames, indicating compromised credentials or account takeover. Focuses
        on location hopping patterns and rapid geographic changes.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        import math
        from collections import Counter

        now = datetime.datetime.utcnow()

        # Extract current transaction location
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            tx_metadata = {}

        current_country = tx_metadata.get("country") or tx_metadata.get("country_code")
        current_city = tx_metadata.get("city")
        current_region = tx_metadata.get("region") or tx_metadata.get("state")
        current_ip = tx_metadata.get("ip_address")
        current_lat = tx_metadata.get("latitude")
        current_lon = tx_metadata.get("longitude")
        tx_timestamp_str = transaction.get("timestamp", now.isoformat())
        tx_timestamp = datetime.datetime.fromisoformat(tx_timestamp_str)

        if not current_country:
            context["location_inconsistency_check_available"] = False
            return

        context["location_inconsistency_check_available"] = True

        # Helper function: Calculate distance using Haversine formula
        def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
            """Calculate distance between two coordinates in kilometers"""
            R = 6371  # Earth radius in km

            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            delta_lat = math.radians(lat2 - lat1)
            delta_lon = math.radians(lon2 - lon1)

            a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return R * c

        # Get recent transactions/sessions with location data for velocity analysis
        # Check multiple time windows for pattern detection
        time_windows = {
            "1_hour": 1,
            "3_hours": 3,
            "6_hours": 6,
            "12_hours": 12,
            "24_hours": 24,
            "48_hours": 48,
            "7_days": 168,
            "30_days": 720
        }

        # Query recent device sessions with location data
        thirty_days_ago = (now - datetime.timedelta(days=30)).isoformat()
        recent_sessions = self.db.query(DeviceSession).filter(
            DeviceSession.account_id == account_id,
            DeviceSession.timestamp > thirty_days_ago,
            DeviceSession.ip_country.isnot(None)
        ).order_by(DeviceSession.timestamp.desc()).all()

        # Also query recent transactions with location data
        recent_transactions = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > thirty_days_ago
        ).order_by(Transaction.timestamp.desc()).all()

        # Combine location data from both sources
        location_events = []

        # Add session locations
        for session in recent_sessions:
            try:
                session_time = datetime.datetime.fromisoformat(session.timestamp)
                session_metadata = json.loads(session.user_agent) if session.user_agent else {}

                location_events.append({
                    "timestamp": session_time,
                    "country": session.ip_country,
                    "city": session.ip_city,
                    "latitude": session_metadata.get("latitude"),
                    "longitude": session_metadata.get("longitude"),
                    "ip": session.ip_address,
                    "source": "session"
                })
            except (json.JSONDecodeError, ValueError):
                continue

        # Add transaction locations
        for tx in recent_transactions:
            try:
                tx_time = datetime.datetime.fromisoformat(tx.timestamp)
                tx_meta = json.loads(tx.tx_metadata) if tx.tx_metadata else {}

                tx_country = tx_meta.get("country") or tx_meta.get("country_code")
                if tx_country:
                    location_events.append({
                        "timestamp": tx_time,
                        "country": tx_country,
                        "city": tx_meta.get("city"),
                        "latitude": tx_meta.get("latitude"),
                        "longitude": tx_meta.get("longitude"),
                        "ip": tx_meta.get("ip_address"),
                        "source": "transaction"
                    })
            except (json.JSONDecodeError, ValueError):
                continue

        # Sort by timestamp descending
        location_events.sort(key=lambda x: x["timestamp"], reverse=True)

        # Analyze location inconsistencies across time windows
        for window_name, hours in time_windows.items():
            window_start = now - datetime.timedelta(hours=hours)

            window_events = [e for e in location_events
                           if e["timestamp"] >= window_start]

            if not window_events:
                continue

            # Count unique countries
            window_countries = [e["country"] for e in window_events if e["country"]]
            unique_countries = list(set(window_countries))
            context[f"unique_countries_{window_name}"] = len(unique_countries)
            context[f"countries_list_{window_name}"] = unique_countries

            # Count unique cities
            window_cities = [e["city"] for e in window_events if e["city"]]
            unique_cities = list(set(window_cities))
            context[f"unique_cities_{window_name}"] = len(unique_cities)

            # Count unique IPs
            window_ips = [e["ip"] for e in window_events if e["ip"]]
            unique_ips = list(set(window_ips))
            context[f"unique_ips_{window_name}"] = len(unique_ips)

            # Count total location events
            context[f"location_events_{window_name}"] = len(window_events)

        # Detailed analysis for critical short time windows
        critical_windows = {
            "1h": (1, "1_hour"),
            "3h": (3, "3_hours"),
            "6h": (6, "6_hours"),
            "24h": (24, "24_hours")
        }

        location_hopping_detected = False
        max_location_changes = 0

        for label, (hours, window_key) in critical_windows.items():
            unique_countries = context.get(f"unique_countries_{window_key}", 0)

            # Flag location hopping: 3+ countries in 24 hours, 2+ in 6 hours, etc.
            if (hours <= 6 and unique_countries >= 2) or \
               (hours <= 24 and unique_countries >= 3):
                location_hopping_detected = True
                max_location_changes = max(max_location_changes, unique_countries)

        context["location_hopping_detected"] = location_hopping_detected
        context["max_location_changes_24h"] = context.get("unique_countries_24_hours", 0)

        # Calculate location velocity (location changes per day)
        window_7d_events = [e for e in location_events
                           if e["timestamp"] >= (now - datetime.timedelta(days=7))]

        if len(window_7d_events) >= 2:
            # Count location changes (country or city changes)
            location_changes = 0
            for i in range(1, len(window_7d_events)):
                prev = window_7d_events[i-1]
                curr = window_7d_events[i]

                if prev["country"] != curr["country"] or \
                   (prev["city"] and curr["city"] and prev["city"] != curr["city"]):
                    location_changes += 1

            # Calculate velocity (changes per day)
            days_span = 7
            location_velocity = location_changes / days_span
            context["location_velocity_changes_per_day"] = location_velocity
            context["total_location_changes_7d"] = location_changes
        else:
            context["location_velocity_changes_per_day"] = 0
            context["total_location_changes_7d"] = 0

        # Analyze travel patterns - detect impossible/suspicious travel
        impossible_travel_count = 0
        suspicious_travel_count = 0
        rapid_travel_events = []

        if current_lat and current_lon:
            # Check against recent events with coordinates
            for event in location_events[:10]:  # Check last 10 events
                if not event["latitude"] or not event["longitude"]:
                    continue

                # Calculate distance and time
                distance_km = calculate_distance_km(
                    float(event["latitude"]), float(event["longitude"]),
                    float(current_lat), float(current_lon)
                )

                time_diff_hours = (tx_timestamp - event["timestamp"]).total_seconds() / 3600

                if time_diff_hours > 0 and distance_km > 50:  # Only check if meaningful distance
                    required_speed = distance_km / time_diff_hours

                    # Impossible travel: >900 km/h (faster than commercial flight)
                    if required_speed > 900:
                        impossible_travel_count += 1
                        rapid_travel_events.append({
                            "from": f"{event['city']}, {event['country']}",
                            "distance_km": round(distance_km, 2),
                            "time_hours": round(time_diff_hours, 2),
                            "speed_kmh": round(required_speed, 2),
                            "category": "impossible"
                        })
                    # Suspicious travel: >500 km/h (very fast, unusual)
                    elif required_speed > 500:
                        suspicious_travel_count += 1
                        rapid_travel_events.append({
                            "from": f"{event['city']}, {event['country']}",
                            "distance_km": round(distance_km, 2),
                            "time_hours": round(time_diff_hours, 2),
                            "speed_kmh": round(required_speed, 2),
                            "category": "suspicious"
                        })

        context["impossible_travel_count_recent"] = impossible_travel_count
        context["suspicious_travel_count_recent"] = suspicious_travel_count
        context["rapid_travel_events"] = rapid_travel_events[:5]  # Top 5 most suspicious

        # Analyze historical location stability
        # Check if user typically operates from consistent locations
        historical_90d_events = [e for e in location_events
                                if e["timestamp"] >= (now - datetime.timedelta(days=90))]

        if len(historical_90d_events) >= 10:
            # Calculate location entropy/consistency
            country_counts = Counter([e["country"] for e in historical_90d_events if e["country"]])

            if country_counts:
                total_events = sum(country_counts.values())
                # Primary country percentage
                most_common_country, primary_count = country_counts.most_common(1)[0]
                primary_country_pct = (primary_count / total_events) * 100

                context["historical_primary_country"] = most_common_country
                context["historical_primary_country_percentage"] = primary_country_pct

                # Location consistency score (0-100, higher = more consistent)
                # Based on how concentrated locations are
                location_consistency = primary_country_pct
                context["location_consistency_score"] = location_consistency

                # Flag if currently deviating from typical location
                if current_country.upper()[:2] != most_common_country.upper()[:2]:
                    context["current_location_deviates_from_primary"] = True

                    # Extra flag if user is highly location-consistent (>90%)
                    if primary_country_pct >= 90:
                        context["high_consistency_user_in_unusual_location"] = True
                    else:
                        context["high_consistency_user_in_unusual_location"] = False
                else:
                    context["current_location_deviates_from_primary"] = False
                    context["high_consistency_user_in_unusual_location"] = False
        else:
            context["location_consistency_score"] = None
            context["current_location_deviates_from_primary"] = False
            context["high_consistency_user_in_unusual_location"] = False

        # Detect simultaneous or near-simultaneous access from different locations
        # Check for transactions/sessions within 15 minutes from different countries
        simultaneous_access_events = []

        for i in range(len(location_events) - 1):
            event1 = location_events[i]
            event2 = location_events[i + 1]

            time_diff_minutes = (event1["timestamp"] - event2["timestamp"]).total_seconds() / 60

            # If within 15 minutes and different countries
            if time_diff_minutes <= 15 and \
               event1["country"] and event2["country"] and \
               event1["country"].upper()[:2] != event2["country"].upper()[:2]:
                simultaneous_access_events.append({
                    "country1": event1["country"],
                    "country2": event2["country"],
                    "city1": event1["city"],
                    "city2": event2["city"],
                    "time_diff_minutes": round(time_diff_minutes, 2),
                    "timestamp1": event1["timestamp"].isoformat(),
                    "timestamp2": event2["timestamp"].isoformat()
                })

        context["simultaneous_access_count"] = len(simultaneous_access_events)
        context["simultaneous_access_events"] = simultaneous_access_events[:3]  # Top 3

        # Detect geographic clustering vs dispersion
        # Check if recent locations are geographically dispersed
        if len(historical_90d_events) >= 5:
            unique_countries_90d = list(set([e["country"] for e in historical_90d_events if e["country"]]))

            # Geographic dispersion score (higher = more dispersed)
            dispersion_score = len(unique_countries_90d)
            context["geographic_dispersion_score"] = dispersion_score

            # Flag high dispersion (5+ countries in 90 days)
            context["high_geographic_dispersion"] = dispersion_score >= 5
        else:
            context["geographic_dispersion_score"] = 0
            context["high_geographic_dispersion"] = False

        # Analyze continent changes
        continent_mapping = {
            "US": "North America", "CA": "North America", "MX": "North America",
            "GB": "Europe", "DE": "Europe", "FR": "Europe", "IT": "Europe", "ES": "Europe",
            "CN": "Asia", "JP": "Asia", "IN": "Asia", "KR": "Asia", "SG": "Asia",
            "BR": "South America", "AR": "South America", "CL": "South America",
            "AU": "Oceania", "NZ": "Oceania",
            "ZA": "Africa", "EG": "Africa", "NG": "Africa"
        }

        def get_continent(country_code):
            if not country_code:
                return "Unknown"
            return continent_mapping.get(country_code.upper()[:2], "Other")

        # Count unique continents in 24 hours
        window_24h_events = [e for e in location_events
                            if e["timestamp"] >= (now - datetime.timedelta(hours=24))]

        continents_24h = [get_continent(e["country"]) for e in window_24h_events if e["country"]]
        unique_continents_24h = list(set(continents_24h))
        context["unique_continents_24h"] = len(unique_continents_24h)
        context["continents_list_24h"] = unique_continents_24h

        # Generate risk flags
        risk_flags = []

        # Multiple countries in short time
        if context.get("unique_countries_1_hour", 0) >= 2:
            risk_flags.append("multiple_countries_1_hour")

        if context.get("unique_countries_3_hours", 0) >= 2:
            risk_flags.append("multiple_countries_3_hours")

        if context.get("unique_countries_6_hours", 0) >= 3:
            risk_flags.append("multiple_countries_6_hours")

        if context.get("unique_countries_24_hours", 0) >= 4:
            risk_flags.append("multiple_countries_24_hours")

        # Location hopping pattern
        if location_hopping_detected:
            risk_flags.append("location_hopping_pattern_detected")

        # High location velocity
        if context.get("location_velocity_changes_per_day", 0) >= 2:
            risk_flags.append("high_location_velocity")

        # Impossible travel
        if impossible_travel_count > 0:
            risk_flags.append("impossible_travel_detected")

        # Suspicious travel
        if suspicious_travel_count >= 2:
            risk_flags.append("multiple_suspicious_travel_events")

        # Simultaneous access
        if context.get("simultaneous_access_count", 0) > 0:
            risk_flags.append("simultaneous_access_different_locations")

        # High geographic dispersion
        if context.get("high_geographic_dispersion", False):
            risk_flags.append("high_geographic_dispersion")

        # Multiple continents in 24h
        if context.get("unique_continents_24h", 0) >= 3:
            risk_flags.append("multiple_continents_24h")

        # Deviation from primary location for consistent user
        if context.get("high_consistency_user_in_unusual_location", False):
            risk_flags.append("consistent_user_unusual_location")

        # Current location deviates from primary
        if context.get("current_location_deviates_from_primary", False):
            risk_flags.append("location_deviates_from_primary")

        # High number of unique IPs in short time
        if context.get("unique_ips_24_hours", 0) >= 5:
            risk_flags.append("multiple_ips_24_hours")

        # Many location events in short time (high activity)
        if context.get("location_events_6_hours", 0) >= 10:
            risk_flags.append("high_location_activity_6_hours")

        context["location_inconsistency_flags"] = risk_flags
        context["location_inconsistency_flag_count"] = len(risk_flags)

        # Calculate comprehensive location inconsistency risk score (0-100)
        risk_score = 0

        # Multiple countries scoring
        risk_score += context.get("unique_countries_1_hour", 0) * 30  # Very high risk
        risk_score += context.get("unique_countries_3_hours", 0) * 20
        risk_score += context.get("unique_countries_6_hours", 0) * 10
        risk_score += min(context.get("unique_countries_24_hours", 0) * 5, 30)

        # Location velocity scoring
        velocity = context.get("location_velocity_changes_per_day", 0)
        if velocity >= 3:
            risk_score += 35
        elif velocity >= 2:
            risk_score += 25
        elif velocity >= 1:
            risk_score += 15

        # Impossible/suspicious travel
        risk_score += impossible_travel_count * 40
        risk_score += suspicious_travel_count * 20

        # Simultaneous access
        risk_score += context.get("simultaneous_access_count", 0) * 35

        # Geographic dispersion
        dispersion = context.get("geographic_dispersion_score", 0)
        if dispersion >= 10:
            risk_score += 30
        elif dispersion >= 5:
            risk_score += 20
        elif dispersion >= 3:
            risk_score += 10

        # Deviation from primary location
        if context.get("high_consistency_user_in_unusual_location", False):
            risk_score += 30
        elif context.get("current_location_deviates_from_primary", False):
            risk_score += 15

        # Multiple continents
        continents = context.get("unique_continents_24h", 0)
        if continents >= 3:
            risk_score += 25
        elif continents >= 2:
            risk_score += 10

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["location_inconsistency_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            inconsistency_risk_level = "critical"
        elif risk_score >= 60:
            inconsistency_risk_level = "high"
        elif risk_score >= 40:
            inconsistency_risk_level = "medium"
        elif risk_score >= 20:
            inconsistency_risk_level = "low"
        else:
            inconsistency_risk_level = "minimal"

        context["location_inconsistency_risk_level"] = inconsistency_risk_level

    def _add_normalized_transaction_amount_context(self, context: Dict[str, Any],
                                                     account_id: str,
                                                     transaction: Dict[str, Any]) -> None:
        """
        Add normalized transaction amount analysis for fraud detection.

        Compares transaction amount against normalized values for similar users
        and demographic profiles. Detects anomalies when transactions deviate
        significantly from peer group norms.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        import statistics
        from collections import defaultdict

        now = datetime.datetime.utcnow()
        tx_amount = abs(float(transaction.get("amount", 0)))

        if tx_amount == 0:
            context["normalized_amount_check_available"] = False
            return

        context["normalized_amount_check_available"] = True
        context["current_transaction_amount"] = tx_amount

        # Get account information for demographic segmentation
        account = self.db.query(Account).filter(Account.account_id == account_id).first()

        if not account:
            context["account_not_found"] = True
            return

        # Extract demographic attributes
        account_creation = datetime.datetime.fromisoformat(account.creation_date)
        account_age_days = (now - account_creation).days
        risk_tier = account.risk_tier if hasattr(account, 'risk_tier') else "medium"

        # Parse account metadata for additional demographics
        # Note: Account model doesn't have metadata field, using empty dict
        account_metadata = {}

        # Extract demographic factors (with defaults since Account doesn't have metadata)
        user_location = account_metadata.get("country") or account_metadata.get("location")
        user_occupation = account_metadata.get("occupation") or account_metadata.get("industry")
        account_balance = float(account_metadata.get("balance", 0))
        income_level = account_metadata.get("income_level")

        # Transaction metadata
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            tx_metadata = {}

        tx_type = transaction.get("transaction_type") or tx_metadata.get("type", "transfer")
        tx_category = tx_metadata.get("category", "general")

        # Store demographic profile
        context["user_demographic_profile"] = {
            "account_age_days": account_age_days,
            "risk_tier": risk_tier,
            "location": user_location,
            "occupation": user_occupation,
            "income_level": income_level,
            "account_balance": account_balance
        }

        # Define cohort segments for comparison
        # Account Age Cohorts
        if account_age_days < 7:
            age_cohort = "brand_new"
        elif account_age_days < 30:
            age_cohort = "very_new"
        elif account_age_days < 90:
            age_cohort = "new"
        elif account_age_days < 180:
            age_cohort = "young"
        elif account_age_days < 365:
            age_cohort = "maturing"
        else:
            age_cohort = "established"

        context["account_age_cohort"] = age_cohort

        # Get historical transactions for normalization
        # 1. Get user's own historical baseline
        ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()
        user_historical_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > ninety_days_ago
        ).all()

        user_amounts = [abs(float(tx.amount)) for tx in user_historical_txs if tx.amount]

        if user_amounts:
            context["user_historical_transaction_count"] = len(user_amounts)
            context["user_avg_transaction_amount"] = statistics.mean(user_amounts)
            context["user_median_transaction_amount"] = statistics.median(user_amounts)

            if len(user_amounts) >= 2:
                context["user_stddev_transaction_amount"] = statistics.stdev(user_amounts)

                # Calculate percentiles
                user_amounts_sorted = sorted(user_amounts)
                context["user_p25_amount"] = user_amounts_sorted[len(user_amounts_sorted) // 4]
                context["user_p50_amount"] = statistics.median(user_amounts)
                context["user_p75_amount"] = user_amounts_sorted[3 * len(user_amounts_sorted) // 4]
                context["user_p90_amount"] = user_amounts_sorted[int(len(user_amounts_sorted) * 0.9)]
                context["user_p95_amount"] = user_amounts_sorted[int(len(user_amounts_sorted) * 0.95)]
                context["user_max_amount"] = max(user_amounts)
                context["user_min_amount"] = min(user_amounts)

                # Calculate current transaction percentile in user's own history
                rank = sum(1 for amt in user_amounts if amt <= tx_amount)
                user_percentile = (rank / len(user_amounts)) * 100
                context["current_tx_user_percentile"] = user_percentile

                # Z-score for user's own distribution
                if context["user_stddev_transaction_amount"] > 0:
                    user_z_score = (tx_amount - context["user_avg_transaction_amount"]) / context["user_stddev_transaction_amount"]
                    context["current_tx_user_z_score"] = user_z_score
                else:
                    context["current_tx_user_z_score"] = 0
            else:
                context["user_stddev_transaction_amount"] = 0
                context["current_tx_user_percentile"] = 50
                context["current_tx_user_z_score"] = 0
        else:
            context["user_historical_transaction_count"] = 0
            context["insufficient_user_history"] = True

        # 2. Get peer group statistics (same cohort)
        # Query transactions from similar accounts
        peer_accounts_query = self.db.query(Account).filter(
            Account.account_id != account_id
        )

        # Filter by risk tier
        peer_accounts_query = peer_accounts_query.filter(Account.risk_tier == risk_tier)

        # Get peer accounts (limit to 1000 for performance)
        peer_accounts = peer_accounts_query.limit(1000).all()

        # Filter peer accounts by age cohort
        peer_account_ids = []
        for peer in peer_accounts:
            try:
                peer_creation = datetime.datetime.fromisoformat(peer.creation_date)
                peer_age_days = (now - peer_creation).days

                # Check if same cohort
                if account_age_days < 7 and peer_age_days < 7:
                    peer_account_ids.append(peer.account_id)
                elif 7 <= account_age_days < 30 and 7 <= peer_age_days < 30:
                    peer_account_ids.append(peer.account_id)
                elif 30 <= account_age_days < 90 and 30 <= peer_age_days < 90:
                    peer_account_ids.append(peer.account_id)
                elif 90 <= account_age_days < 180 and 90 <= peer_age_days < 180:
                    peer_account_ids.append(peer.account_id)
                elif 180 <= account_age_days < 365 and 180 <= peer_age_days < 365:
                    peer_account_ids.append(peer.account_id)
                elif account_age_days >= 365 and peer_age_days >= 365:
                    peer_account_ids.append(peer.account_id)
            except (ValueError, AttributeError):
                continue

        context["peer_group_size"] = len(peer_account_ids)

        if peer_account_ids:
            # Get peer transactions
            peer_txs = self.db.query(Transaction).filter(
                Transaction.account_id.in_(peer_account_ids),
                Transaction.timestamp > ninety_days_ago
            ).limit(10000).all()  # Limit for performance

            peer_amounts = [abs(float(tx.amount)) for tx in peer_txs if tx.amount]

            if peer_amounts and len(peer_amounts) >= 10:
                context["peer_transaction_count"] = len(peer_amounts)
                context["peer_avg_transaction_amount"] = statistics.mean(peer_amounts)
                context["peer_median_transaction_amount"] = statistics.median(peer_amounts)

                if len(peer_amounts) >= 2:
                    context["peer_stddev_transaction_amount"] = statistics.stdev(peer_amounts)

                    # Peer percentiles
                    peer_amounts_sorted = sorted(peer_amounts)
                    context["peer_p25_amount"] = peer_amounts_sorted[len(peer_amounts_sorted) // 4]
                    context["peer_p50_amount"] = statistics.median(peer_amounts)
                    context["peer_p75_amount"] = peer_amounts_sorted[3 * len(peer_amounts_sorted) // 4]
                    context["peer_p90_amount"] = peer_amounts_sorted[int(len(peer_amounts_sorted) * 0.9)]
                    context["peer_p95_amount"] = peer_amounts_sorted[int(len(peer_amounts_sorted) * 0.95)]
                    context["peer_p99_amount"] = peer_amounts_sorted[int(len(peer_amounts_sorted) * 0.99)]
                    context["peer_max_amount"] = max(peer_amounts)

                    # Calculate current transaction percentile in peer group
                    peer_rank = sum(1 for amt in peer_amounts if amt <= tx_amount)
                    peer_percentile = (peer_rank / len(peer_amounts)) * 100
                    context["current_tx_peer_percentile"] = peer_percentile

                    # Z-score for peer distribution
                    if context["peer_stddev_transaction_amount"] > 0:
                        peer_z_score = (tx_amount - context["peer_avg_transaction_amount"]) / context["peer_stddev_transaction_amount"]
                        context["current_tx_peer_z_score"] = peer_z_score
                    else:
                        context["current_tx_peer_z_score"] = 0
                else:
                    context["peer_stddev_transaction_amount"] = 0
                    context["current_tx_peer_percentile"] = 50
                    context["current_tx_peer_z_score"] = 0
            else:
                context["insufficient_peer_data"] = True
                context["peer_transaction_count"] = len(peer_amounts) if peer_amounts else 0
        else:
            context["no_peer_group_found"] = True

        # 3. Calculate normalized scores and deviations
        normalization_results = {}

        # User baseline comparison
        if context.get("user_avg_transaction_amount"):
            user_avg = context["user_avg_transaction_amount"]
            deviation_from_user_avg = ((tx_amount - user_avg) / user_avg) * 100 if user_avg > 0 else 0
            normalization_results["deviation_from_user_avg_pct"] = deviation_from_user_avg

            # Flag significant deviations
            if abs(deviation_from_user_avg) > 200:
                normalization_results["extreme_deviation_from_user_baseline"] = True
            elif abs(deviation_from_user_avg) > 100:
                normalization_results["high_deviation_from_user_baseline"] = True
            else:
                normalization_results["extreme_deviation_from_user_baseline"] = False
                normalization_results["high_deviation_from_user_baseline"] = False

        # Peer group comparison
        if context.get("peer_avg_transaction_amount"):
            peer_avg = context["peer_avg_transaction_amount"]
            deviation_from_peer_avg = ((tx_amount - peer_avg) / peer_avg) * 100 if peer_avg > 0 else 0
            normalization_results["deviation_from_peer_avg_pct"] = deviation_from_peer_avg

            # Flag deviations
            if abs(deviation_from_peer_avg) > 300:
                normalization_results["extreme_deviation_from_peer_baseline"] = True
            elif abs(deviation_from_peer_avg) > 150:
                normalization_results["high_deviation_from_peer_baseline"] = True
            else:
                normalization_results["extreme_deviation_from_peer_baseline"] = False
                normalization_results["high_deviation_from_peer_baseline"] = False

        # Account balance ratio
        if account_balance > 0:
            balance_ratio = (tx_amount / account_balance) * 100
            context["transaction_to_balance_ratio_pct"] = balance_ratio

            if balance_ratio > 80:
                normalization_results["high_balance_ratio"] = True
            else:
                normalization_results["high_balance_ratio"] = False
        else:
            context["transaction_to_balance_ratio_pct"] = 0
            normalization_results["high_balance_ratio"] = False

        context["normalization_results"] = normalization_results

        # 4. Generate risk flags
        risk_flags = []

        # Exceeds user's historical maximum
        if context.get("user_max_amount") and tx_amount > context["user_max_amount"]:
            risk_flags.append("exceeds_user_historical_max")

        # High percentile in user's distribution
        if context.get("current_tx_user_percentile"):
            if context["current_tx_user_percentile"] >= 99:
                risk_flags.append("top_1_percent_user_history")
            elif context["current_tx_user_percentile"] >= 95:
                risk_flags.append("top_5_percent_user_history")
            elif context["current_tx_user_percentile"] >= 90:
                risk_flags.append("top_10_percent_user_history")

        # High percentile in peer distribution
        if context.get("current_tx_peer_percentile"):
            if context["current_tx_peer_percentile"] >= 99:
                risk_flags.append("top_1_percent_peer_group")
            elif context["current_tx_peer_percentile"] >= 95:
                risk_flags.append("top_5_percent_peer_group")
            elif context["current_tx_peer_percentile"] >= 90:
                risk_flags.append("top_10_percent_peer_group")

        # High Z-scores
        user_z = context.get("current_tx_user_z_score", 0)
        peer_z = context.get("current_tx_peer_z_score", 0)

        if abs(user_z) >= 3:
            risk_flags.append("user_z_score_extreme")  # 3+ std deviations
        elif abs(user_z) >= 2:
            risk_flags.append("user_z_score_high")  # 2+ std deviations

        if abs(peer_z) >= 3:
            risk_flags.append("peer_z_score_extreme")
        elif abs(peer_z) >= 2:
            risk_flags.append("peer_z_score_high")

        # Deviation flags
        if normalization_results.get("extreme_deviation_from_user_baseline"):
            risk_flags.append("extreme_deviation_from_user_baseline")
        elif normalization_results.get("high_deviation_from_user_baseline"):
            risk_flags.append("high_deviation_from_user_baseline")

        if normalization_results.get("extreme_deviation_from_peer_baseline"):
            risk_flags.append("extreme_deviation_from_peer_baseline")
        elif normalization_results.get("high_deviation_from_peer_baseline"):
            risk_flags.append("high_deviation_from_peer_baseline")

        # High balance ratio
        if normalization_results.get("high_balance_ratio"):
            risk_flags.append("high_transaction_to_balance_ratio")

        # New user with large transaction
        if age_cohort in ["brand_new", "very_new"] and tx_amount > 5000:
            risk_flags.append("new_user_large_transaction")

        # Peer outlier (significantly higher than peer group)
        if context.get("peer_avg_transaction_amount"):
            if tx_amount > context.get("peer_p95_amount", float('inf')):
                risk_flags.append("exceeds_peer_95th_percentile")

        # Inconsistent with user's typical behavior
        if context.get("user_avg_transaction_amount"):
            if tx_amount > context.get("user_p90_amount", float('inf')):
                risk_flags.append("exceeds_user_90th_percentile")

        # First large transaction
        if context.get("user_historical_transaction_count", 0) <= 3 and tx_amount > 2000:
            risk_flags.append("first_few_transactions_large_amount")

        context["normalized_amount_risk_flags"] = risk_flags
        context["normalized_amount_risk_flag_count"] = len(risk_flags)

        # 5. Calculate comprehensive normalized amount risk score (0-100)
        risk_score = 0

        # Base score for amount size
        if tx_amount > 50000:
            risk_score += 20
        elif tx_amount > 25000:
            risk_score += 15
        elif tx_amount > 10000:
            risk_score += 10
        elif tx_amount > 5000:
            risk_score += 5

        # User percentile scoring
        user_pct = context.get("current_tx_user_percentile", 50)
        if user_pct >= 99:
            risk_score += 30
        elif user_pct >= 95:
            risk_score += 20
        elif user_pct >= 90:
            risk_score += 12
        elif user_pct >= 75:
            risk_score += 5

        # Peer percentile scoring
        peer_pct = context.get("current_tx_peer_percentile", 50)
        if peer_pct >= 99:
            risk_score += 25
        elif peer_pct >= 18:
            risk_score += 18
        elif peer_pct >= 90:
            risk_score += 10

        # Z-score scoring
        if abs(user_z) >= 3:
            risk_score += 25
        elif abs(user_z) >= 2:
            risk_score += 15

        if abs(peer_z) >= 3:
            risk_score += 20
        elif abs(peer_z) >= 2:
            risk_score += 12

        # Deviation scoring
        user_dev = abs(normalization_results.get("deviation_from_user_avg_pct", 0))
        if user_dev > 300:
            risk_score += 25
        elif user_dev > 200:
            risk_score += 18
        elif user_dev > 100:
            risk_score += 10

        peer_dev = abs(normalization_results.get("deviation_from_peer_avg_pct", 0))
        if peer_dev > 400:
            risk_score += 20
        elif peer_dev > 300:
            risk_score += 15
        elif peer_dev > 150:
            risk_score += 8

        # Balance ratio scoring
        balance_ratio = context.get("transaction_to_balance_ratio_pct", 0)
        if balance_ratio > 90:
            risk_score += 30
        elif balance_ratio > 80:
            risk_score += 20
        elif balance_ratio > 50:
            risk_score += 10

        # New account large transaction
        if age_cohort in ["brand_new", "very_new"]:
            if tx_amount > 10000:
                risk_score += 30
            elif tx_amount > 5000:
                risk_score += 20

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["normalized_amount_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            amount_risk_level = "critical"
        elif risk_score >= 60:
            amount_risk_level = "high"
        elif risk_score >= 40:
            amount_risk_level = "medium"
        elif risk_score >= 20:
            amount_risk_level = "low"
        else:
            amount_risk_level = "minimal"

        context["normalized_amount_risk_level"] = amount_risk_level

    def _add_transaction_context_anomalies_context(self, context: Dict[str, Any],
                                                     account_id: str,
                                                     transaction: Dict[str, Any]) -> None:
        """
        Add transaction context anomalies detection for fraud analysis.

        Flags transactions that don't align with user's typical spending habits
        including sudden large purchases, unusual categories, merchant type changes,
        velocity anomalies, and behavioral pattern breaks.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        import statistics
        from collections import Counter, defaultdict

        now = datetime.datetime.utcnow()
        tx_amount = abs(float(transaction.get("amount", 0)))

        if tx_amount == 0:
            context["context_anomaly_check_available"] = False
            return

        context["context_anomaly_check_available"] = True

        # Extract transaction context
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            tx_metadata = {}

        # Current transaction attributes
        current_category = tx_metadata.get("category", "unknown")
        current_merchant = tx_metadata.get("merchant_name") or tx_metadata.get("merchant")
        current_merchant_type = tx_metadata.get("merchant_type") or tx_metadata.get("mcc_category")
        current_description = transaction.get("description", "")
        tx_type = transaction.get("transaction_type", "transfer")
        tx_timestamp = datetime.datetime.fromisoformat(transaction.get("timestamp", now.isoformat()))
        tx_day_of_week = tx_timestamp.weekday()  # 0=Monday, 6=Sunday
        tx_hour = tx_timestamp.hour

        context["current_transaction_category"] = current_category
        context["current_transaction_merchant_type"] = current_merchant_type
        context["current_transaction_type"] = tx_type

        # Get historical transactions for pattern analysis
        ninety_days_ago = (now - datetime.timedelta(days=90)).isoformat()
        thirty_days_ago = (now - datetime.timedelta(days=30)).isoformat()
        seven_days_ago = (now - datetime.timedelta(days=7)).isoformat()

        # Query historical transactions
        historical_txs_90d = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp > ninety_days_ago
        ).order_by(Transaction.timestamp.desc()).all()

        if len(historical_txs_90d) < 5:
            context["insufficient_context_history"] = True
            context["historical_transaction_count_for_context"] = len(historical_txs_90d)
            return

        context["historical_transaction_count_for_context"] = len(historical_txs_90d)

        # 1. CATEGORY ANALYSIS
        category_counts = Counter()
        category_amounts = defaultdict(list)
        merchant_type_counts = Counter()
        merchant_counts = Counter()

        for tx in historical_txs_90d:
            tx_meta = {}
            if tx.tx_metadata:
                try:
                    tx_meta = json.loads(tx.tx_metadata) if isinstance(tx.tx_metadata, str) else tx.tx_metadata
                except (json.JSONDecodeError, TypeError):
                    tx_meta = {}

            cat = tx_meta.get("category", "unknown")
            merchant_type = tx_meta.get("merchant_type") or tx_meta.get("mcc_category")
            merchant = tx_meta.get("merchant_name") or tx_meta.get("merchant")

            category_counts[cat] += 1
            category_amounts[cat].append(abs(float(tx.amount)))

            if merchant_type:
                merchant_type_counts[merchant_type] += 1

            if merchant:
                merchant_counts[merchant] += 1

        # Analyze current category
        total_txs = len(historical_txs_90d)
        current_category_count = category_counts.get(current_category, 0)
        current_category_frequency = (current_category_count / total_txs) * 100 if total_txs > 0 else 0

        context["current_category_historical_frequency_pct"] = current_category_frequency
        context["current_category_historical_count"] = current_category_count

        # Is this a new/rare category?
        is_new_category = current_category_count == 0
        is_rare_category = 0 < current_category_frequency < 5  # Less than 5% of transactions

        context["is_new_spending_category"] = is_new_category
        context["is_rare_spending_category"] = is_rare_category

        # Top categories
        top_categories = category_counts.most_common(5)
        context["top_spending_categories"] = [{"category": cat, "count": count, "percentage": (count/total_txs)*100}
                                             for cat, count in top_categories]

        # Category diversity (how many different categories)
        unique_categories = len(category_counts)
        context["unique_spending_categories"] = unique_categories

        # Check if amount is unusual for this category
        if current_category in category_amounts and category_amounts[current_category]:
            cat_amounts = category_amounts[current_category]
            cat_avg = statistics.mean(cat_amounts)
            cat_max = max(cat_amounts)

            context["category_avg_amount"] = cat_avg
            context["category_max_amount"] = cat_max

            # Is current amount unusually high for this category?
            if tx_amount > cat_max:
                context["exceeds_category_historical_max"] = True
            else:
                context["exceeds_category_historical_max"] = False

            if len(cat_amounts) >= 2:
                cat_stddev = statistics.stdev(cat_amounts)
                if cat_stddev > 0:
                    category_z_score = (tx_amount - cat_avg) / cat_stddev
                    context["category_amount_z_score"] = category_z_score

                    if abs(category_z_score) >= 2:
                        context["category_amount_anomaly"] = True
                    else:
                        context["category_amount_anomaly"] = False
                else:
                    context["category_amount_anomaly"] = False
            else:
                context["category_amount_anomaly"] = False

        # 2. MERCHANT TYPE ANALYSIS
        if current_merchant_type:
            merchant_type_count = merchant_type_counts.get(current_merchant_type, 0)
            merchant_type_frequency = (merchant_type_count / total_txs) * 100 if total_txs > 0 else 0

            context["current_merchant_type_frequency_pct"] = merchant_type_frequency
            context["is_new_merchant_type"] = merchant_type_count == 0
            context["is_rare_merchant_type"] = 0 < merchant_type_frequency < 5

            # Top merchant types
            top_merchant_types = merchant_type_counts.most_common(5)
            context["top_merchant_types"] = [{"type": mt, "count": count, "percentage": (count/total_txs)*100}
                                            for mt, count in top_merchant_types]
        else:
            context["merchant_type_analysis_unavailable"] = True

        # 3. MERCHANT ANALYSIS
        if current_merchant:
            merchant_count = merchant_counts.get(current_merchant, 0)
            context["current_merchant_historical_count"] = merchant_count
            context["is_new_merchant"] = merchant_count == 0

            # Top merchants
            top_merchants = merchant_counts.most_common(10)
            context["top_merchants"] = [{"merchant": merch, "count": count}
                                       for merch, count in top_merchants]
        else:
            context["merchant_analysis_unavailable"] = True

        # 4. TRANSACTION TYPE ANALYSIS
        type_counts = Counter()
        for tx in historical_txs_90d:
            type_counts[tx.transaction_type] += 1

        current_type_count = type_counts.get(tx_type, 0)
        current_type_frequency = (current_type_count / total_txs) * 100 if total_txs > 0 else 0

        context["current_type_frequency_pct"] = current_type_frequency
        context["is_new_transaction_type"] = current_type_count == 0
        context["transaction_type_distribution"] = dict(type_counts)

        # 5. VELOCITY ANALYSIS (sudden changes in transaction frequency)
        # Compare recent 7 days vs previous weeks
        recent_7d_txs = [tx for tx in historical_txs_90d
                        if datetime.datetime.fromisoformat(tx.timestamp) >= datetime.datetime.fromisoformat(seven_days_ago)]
        week_2_txs = [tx for tx in historical_txs_90d
                     if datetime.datetime.fromisoformat(seven_days_ago) > datetime.datetime.fromisoformat(tx.timestamp) >=
                        datetime.datetime.fromisoformat((now - datetime.timedelta(days=14)).isoformat())]
        week_3_txs = [tx for tx in historical_txs_90d
                     if datetime.datetime.fromisoformat((now - datetime.timedelta(days=14)).isoformat()) >
                        datetime.datetime.fromisoformat(tx.timestamp) >=
                        datetime.datetime.fromisoformat((now - datetime.timedelta(days=21)).isoformat())]

        context["transactions_last_7_days"] = len(recent_7d_txs)
        context["transactions_week_2"] = len(week_2_txs)
        context["transactions_week_3"] = len(week_3_txs)

        # Calculate velocity change
        if len(week_2_txs) > 0 and len(week_3_txs) > 0:
            avg_weekly_baseline = (len(week_2_txs) + len(week_3_txs)) / 2
            velocity_change = ((len(recent_7d_txs) - avg_weekly_baseline) / avg_weekly_baseline) * 100 if avg_weekly_baseline > 0 else 0
            context["velocity_change_pct"] = velocity_change

            # Flag sudden velocity increase
            if velocity_change > 100:
                context["velocity_spike_detected"] = True
            else:
                context["velocity_spike_detected"] = False
        else:
            context["velocity_spike_detected"] = False

        # 6. AMOUNT PATTERN ANALYSIS
        amounts_90d = [abs(float(tx.amount)) for tx in historical_txs_90d]

        # Sudden large purchase detection
        if amounts_90d:
            avg_amount = statistics.mean(amounts_90d)
            max_amount = max(amounts_90d)

            context["avg_transaction_amount_90d"] = avg_amount
            context["max_transaction_amount_90d"] = max_amount

            # Sudden large purchase: amount significantly higher than typical
            amount_multiplier = tx_amount / avg_amount if avg_amount > 0 else 0
            context["current_to_avg_multiplier"] = amount_multiplier

            if amount_multiplier >= 5:
                context["sudden_large_purchase_detected"] = True
            else:
                context["sudden_large_purchase_detected"] = False

            # Amount consistency analysis
            if len(amounts_90d) >= 10:
                amount_stddev = statistics.stdev(amounts_90d)
                coefficient_of_variation = (amount_stddev / avg_amount) * 100 if avg_amount > 0 else 0
                context["amount_coefficient_of_variation"] = coefficient_of_variation

                # High CV indicates inconsistent spending; low CV indicates consistent
                if coefficient_of_variation < 30:
                    spending_consistency = "very_consistent"
                elif coefficient_of_variation < 60:
                    spending_consistency = "consistent"
                elif coefficient_of_variation < 100:
                    spending_consistency = "moderate"
                else:
                    spending_consistency = "inconsistent"

                context["spending_consistency_classification"] = spending_consistency

                # For consistent spenders, unusual amounts are more suspicious
                if spending_consistency in ["very_consistent", "consistent"] and amount_multiplier >= 3:
                    context["consistent_spender_unusual_amount"] = True
                else:
                    context["consistent_spender_unusual_amount"] = False

        # 7. DAY-OF-WEEK PATTERN ANALYSIS
        day_of_week_counts = Counter()
        for tx in historical_txs_90d:
            tx_time = datetime.datetime.fromisoformat(tx.timestamp)
            day_of_week_counts[tx_time.weekday()] += 1

        current_day_count = day_of_week_counts.get(tx_day_of_week, 0)
        context["current_day_historical_count"] = current_day_count

        # Typical transaction days
        avg_day_count = total_txs / 7 if total_txs > 0 else 0
        context["avg_transactions_per_day_of_week"] = avg_day_count

        if current_day_count < avg_day_count * 0.5:
            context["unusual_day_of_week"] = True
        else:
            context["unusual_day_of_week"] = False

        # 8. TIME-OF-DAY PATTERN ANALYSIS
        hour_of_day_counts = Counter()
        for tx in historical_txs_90d:
            tx_time = datetime.datetime.fromisoformat(tx.timestamp)
            hour_of_day_counts[tx_time.hour] += 1

        current_hour_count = hour_of_day_counts.get(tx_hour, 0)
        context["current_hour_historical_count"] = current_hour_count

        avg_hour_count = total_txs / 24 if total_txs > 0 else 0
        if current_hour_count < avg_hour_count * 0.25:
            context["unusual_hour_of_day"] = True
        else:
            context["unusual_hour_of_day"] = False

        # 9. RECURRING TRANSACTION DETECTION
        # Check if user has recurring patterns (similar amounts at regular intervals)
        recurring_patterns = []
        amount_tolerance = 0.05  # 5% tolerance

        # Group similar amounts
        amount_groups = defaultdict(list)
        for tx in historical_txs_90d:
            amount = abs(float(tx.amount))
            # Find group with similar amount
            found_group = False
            for key_amount in amount_groups.keys():
                if abs(amount - key_amount) / key_amount < amount_tolerance:
                    amount_groups[key_amount].append(tx)
                    found_group = True
                    break
            if not found_group:
                amount_groups[amount].append(tx)

        # Check for recurring patterns (3+ occurrences of similar amount)
        for amount, txs in amount_groups.items():
            if len(txs) >= 3:
                # Check time intervals
                timestamps = sorted([datetime.datetime.fromisoformat(tx.timestamp) for tx in txs])
                intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]

                if intervals:
                    avg_interval = statistics.mean(intervals)
                    if 25 <= avg_interval <= 35:  # Monthly-ish
                        recurring_patterns.append({
                            "amount": amount,
                            "frequency": "monthly",
                            "count": len(txs),
                            "avg_interval_days": avg_interval
                        })
                    elif 5 <= avg_interval <= 9:  # Weekly-ish
                        recurring_patterns.append({
                            "amount": amount,
                            "frequency": "weekly",
                            "count": len(txs),
                            "avg_interval_days": avg_interval
                        })

        context["recurring_payment_patterns"] = recurring_patterns
        context["recurring_pattern_count"] = len(recurring_patterns)

        # Check if current transaction breaks recurring pattern
        breaks_recurring_pattern = False
        for pattern in recurring_patterns:
            if abs(tx_amount - pattern["amount"]) / pattern["amount"] > amount_tolerance:
                # Similar timing but different amount
                breaks_recurring_pattern = True

        context["breaks_recurring_pattern"] = breaks_recurring_pattern

        # 10. BENEFICIARY CONSISTENCY ANALYSIS
        beneficiary_id = transaction.get("beneficiary_id")
        if beneficiary_id:
            beneficiary_txs = [tx for tx in historical_txs_90d
                              if json.loads(tx.tx_metadata if tx.tx_metadata else '{}').get("beneficiary_id") == beneficiary_id
                              or (hasattr(tx, 'beneficiary_id') and tx.beneficiary_id == beneficiary_id)]

            context["transactions_to_current_beneficiary"] = len(beneficiary_txs)

            if beneficiary_txs:
                # Typical amounts to this beneficiary
                beneficiary_amounts = [abs(float(tx.amount)) for tx in beneficiary_txs]
                beneficiary_avg = statistics.mean(beneficiary_amounts)
                beneficiary_max = max(beneficiary_amounts)

                context["beneficiary_avg_amount"] = beneficiary_avg
                context["beneficiary_max_amount"] = beneficiary_max

                if tx_amount > beneficiary_max:
                    context["exceeds_beneficiary_historical_max"] = True
                else:
                    context["exceeds_beneficiary_historical_max"] = False

                # Check if amount is unusual for this beneficiary
                if len(beneficiary_amounts) >= 2:
                    beneficiary_stddev = statistics.stdev(beneficiary_amounts)
                    if beneficiary_stddev > 0:
                        beneficiary_z = (tx_amount - beneficiary_avg) / beneficiary_stddev
                        if abs(beneficiary_z) >= 2:
                            context["unusual_amount_for_beneficiary"] = True
                        else:
                            context["unusual_amount_for_beneficiary"] = False
                    else:
                        context["unusual_amount_for_beneficiary"] = False
            else:
                context["first_transaction_to_beneficiary"] = True

        # 11. GENERATE CONTEXT ANOMALY RISK FLAGS
        risk_flags = []

        # Category anomalies
        if is_new_category:
            risk_flags.append("new_spending_category")

        if is_rare_category:
            risk_flags.append("rare_spending_category")

        if context.get("exceeds_category_historical_max"):
            risk_flags.append("exceeds_category_max")

        if context.get("category_amount_anomaly"):
            risk_flags.append("category_amount_anomaly")

        # Merchant anomalies
        if context.get("is_new_merchant_type"):
            risk_flags.append("new_merchant_type")

        if context.get("is_new_merchant"):
            risk_flags.append("new_merchant")

        # Transaction type anomaly
        if context.get("is_new_transaction_type"):
            risk_flags.append("new_transaction_type")

        # Velocity anomalies
        if context.get("velocity_spike_detected"):
            risk_flags.append("velocity_spike")

        # Amount anomalies
        if context.get("sudden_large_purchase_detected"):
            risk_flags.append("sudden_large_purchase")

        if context.get("consistent_spender_unusual_amount"):
            risk_flags.append("consistent_spender_unusual_amount")

        # Timing anomalies
        if context.get("unusual_day_of_week"):
            risk_flags.append("unusual_day_of_week")

        if context.get("unusual_hour_of_day"):
            risk_flags.append("unusual_hour_of_day")

        # Pattern breaks
        if context.get("breaks_recurring_pattern"):
            risk_flags.append("breaks_recurring_pattern")

        # Beneficiary anomalies
        if context.get("exceeds_beneficiary_historical_max"):
            risk_flags.append("exceeds_beneficiary_max")

        if context.get("unusual_amount_for_beneficiary"):
            risk_flags.append("unusual_amount_for_beneficiary")

        if context.get("first_transaction_to_beneficiary"):
            risk_flags.append("first_transaction_to_beneficiary")

        # Multiple simultaneous anomalies (compound risk)
        if len(risk_flags) >= 4:
            risk_flags.append("multiple_context_anomalies")

        context["context_anomaly_flags"] = risk_flags
        context["context_anomaly_flag_count"] = len(risk_flags)

        # 12. CALCULATE CONTEXT ANOMALY RISK SCORE (0-100)
        risk_score = 0

        # Category anomalies
        if is_new_category:
            risk_score += 20
        elif is_rare_category:
            risk_score += 10

        if context.get("exceeds_category_historical_max"):
            risk_score += 15

        if context.get("category_amount_anomaly"):
            risk_score += 12

        # Merchant anomalies
        if context.get("is_new_merchant_type"):
            risk_score += 15

        if context.get("is_new_merchant"):
            risk_score += 8

        # Transaction type
        if context.get("is_new_transaction_type"):
            risk_score += 18

        # Velocity
        velocity_change = context.get("velocity_change_pct", 0)
        if velocity_change > 200:
            risk_score += 25
        elif velocity_change > 100:
            risk_score += 15

        # Amount anomalies
        if context.get("sudden_large_purchase_detected"):
            risk_score += 20

        multiplier = context.get("current_to_avg_multiplier", 1)
        if multiplier >= 10:
            risk_score += 25
        elif multiplier >= 5:
            risk_score += 15
        elif multiplier >= 3:
            risk_score += 8

        if context.get("consistent_spender_unusual_amount"):
            risk_score += 15

        # Timing
        if context.get("unusual_day_of_week"):
            risk_score += 8

        if context.get("unusual_hour_of_day"):
            risk_score += 8

        # Pattern breaks
        if context.get("breaks_recurring_pattern"):
            risk_score += 15

        # Beneficiary
        if context.get("first_transaction_to_beneficiary") and tx_amount > 1000:
            risk_score += 12

        if context.get("unusual_amount_for_beneficiary"):
            risk_score += 10

        # Multiple anomalies multiplier
        if len(risk_flags) >= 5:
            risk_score += 20
        elif len(risk_flags) >= 3:
            risk_score += 10

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["context_anomaly_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            context_risk_level = "critical"
        elif risk_score >= 60:
            context_risk_level = "high"
        elif risk_score >= 40:
            context_risk_level = "medium"
        elif risk_score >= 20:
            context_risk_level = "low"
        else:
            context_risk_level = "minimal"

        context["context_anomaly_risk_level"] = context_risk_level

    def _add_fraud_complaints_count_context(self, context: Dict[str, Any],
                                             account_id: str,
                                             transaction: Dict[str, Any]) -> None:
        """
        Add fraud complaints count analysis for fraud detection.

        Analyzes frequency of fraud complaints linked to UPI ID, device, or account
        including complaint types, severity, recency, and resolution patterns.

        Args:
            context: Context dictionary to update
            account_id: Account identifier
            transaction: Current transaction data
        """
        from collections import Counter

        now = datetime.datetime.utcnow()

        # Extract transaction identifiers
        tx_metadata = transaction.get("tx_metadata") or transaction.get("metadata")
        if tx_metadata and isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except json.JSONDecodeError:
                tx_metadata = {}

        if not tx_metadata:
            tx_metadata = {}

        # Extract UPI ID and device ID from transaction
        upi_id = tx_metadata.get("upi_id") or transaction.get("upi_id")
        device_id = tx_metadata.get("device_id") or transaction.get("device_id")
        beneficiary_id = transaction.get("beneficiary_id")

        context["fraud_complaints_check_available"] = True

        # 1. ACCOUNT-LEVEL COMPLAINTS
        account_complaints = self.db.query(FraudComplaint).filter(
            FraudComplaint.entity_type == "account",
            FraudComplaint.entity_id == account_id
        ).order_by(FraudComplaint.complaint_date.desc()).all()

        context["account_complaint_count"] = len(account_complaints)

        if account_complaints:
            # Categorize by status
            status_counts = Counter([c.status for c in account_complaints])
            context["account_complaints_by_status"] = dict(status_counts)

            # Active complaints (not resolved/closed)
            active_complaints = [c for c in account_complaints
                               if c.status not in ["resolved", "closed", "rejected"]]
            context["account_active_complaint_count"] = len(active_complaints)

            # Confirmed fraud complaints
            confirmed_complaints = [c for c in account_complaints
                                  if c.resolution == "confirmed_fraud"]
            context["account_confirmed_fraud_complaint_count"] = len(confirmed_complaints)

            # Categorize by severity
            severity_counts = Counter([c.severity for c in account_complaints])
            context["account_complaints_by_severity"] = dict(severity_counts)

            critical_complaints = [c for c in account_complaints if c.severity == "critical"]
            high_complaints = [c for c in account_complaints if c.severity == "high"]

            context["account_critical_complaint_count"] = len(critical_complaints)
            context["account_high_complaint_count"] = len(high_complaints)

            # Categorize by type
            type_counts = Counter([c.complaint_type for c in account_complaints])
            context["account_complaints_by_type"] = dict(type_counts)

            # Most recent complaint
            most_recent = account_complaints[0]
            days_since_last_complaint = (now - most_recent.complaint_date).days

            context["account_days_since_last_complaint"] = days_since_last_complaint
            context["account_most_recent_complaint_type"] = most_recent.complaint_type
            context["account_most_recent_complaint_severity"] = most_recent.severity
            context["account_most_recent_complaint_status"] = most_recent.status

            # Recency analysis
            complaints_last_30d = [c for c in account_complaints
                                  if (now - c.complaint_date).days <= 30]
            complaints_last_90d = [c for c in account_complaints
                                  if (now - c.complaint_date).days <= 90]
            complaints_last_365d = [c for c in account_complaints
                                   if (now - c.complaint_date).days <= 365]

            context["account_complaints_last_30d"] = len(complaints_last_30d)
            context["account_complaints_last_90d"] = len(complaints_last_90d)
            context["account_complaints_last_365d"] = len(complaints_last_365d)

            # Total amount involved in complaints
            total_complaint_amount = sum(float(c.amount_involved or 0)
                                       for c in account_complaints if c.amount_involved)
            context["account_total_complaint_amount"] = total_complaint_amount

            # Fraud ring detection
            in_fraud_ring = any(c.is_part_of_fraud_ring for c in account_complaints)
            context["account_in_fraud_ring"] = in_fraud_ring

            if in_fraud_ring:
                fraud_ring_ids = list(set([c.fraud_ring_id for c in account_complaints
                                         if c.fraud_ring_id]))
                context["account_fraud_ring_ids"] = fraud_ring_ids
        else:
            context["account_active_complaint_count"] = 0
            context["account_confirmed_fraud_complaint_count"] = 0
            context["account_critical_complaint_count"] = 0
            context["account_in_fraud_ring"] = False

        # 2. UPI ID-LEVEL COMPLAINTS
        if upi_id:
            upi_complaints = self.db.query(FraudComplaint).filter(
                FraudComplaint.upi_id == upi_id
            ).order_by(FraudComplaint.complaint_date.desc()).all()

            context["upi_complaint_count"] = len(upi_complaints)

            if upi_complaints:
                # Active and confirmed
                upi_active = [c for c in upi_complaints
                            if c.status not in ["resolved", "closed", "rejected"]]
                upi_confirmed = [c for c in upi_complaints
                               if c.resolution == "confirmed_fraud"]

                context["upi_active_complaint_count"] = len(upi_active)
                context["upi_confirmed_fraud_complaint_count"] = len(upi_confirmed)

                # Severity
                upi_critical = [c for c in upi_complaints if c.severity == "critical"]
                upi_high = [c for c in upi_complaints if c.severity == "high"]

                context["upi_critical_complaint_count"] = len(upi_critical)
                context["upi_high_complaint_count"] = len(upi_high)

                # Recency
                upi_recent = upi_complaints[0]
                upi_days_since = (now - upi_recent.complaint_date).days

                context["upi_days_since_last_complaint"] = upi_days_since
                context["upi_most_recent_complaint_type"] = upi_recent.complaint_type

                # Recent complaints
                upi_last_90d = [c for c in upi_complaints
                              if (now - c.complaint_date).days <= 90]
                context["upi_complaints_last_90d"] = len(upi_last_90d)

                # Fraud ring
                upi_in_fraud_ring = any(c.is_part_of_fraud_ring for c in upi_complaints)
                context["upi_in_fraud_ring"] = upi_in_fraud_ring
            else:
                context["upi_active_complaint_count"] = 0
                context["upi_confirmed_fraud_complaint_count"] = 0
                context["upi_in_fraud_ring"] = False
        else:
            context["upi_id_not_available"] = True
            context["upi_complaint_count"] = 0

        # 3. DEVICE-LEVEL COMPLAINTS
        if device_id:
            device_complaints = self.db.query(FraudComplaint).filter(
                FraudComplaint.device_id == device_id
            ).order_by(FraudComplaint.complaint_date.desc()).all()

            context["device_complaint_count"] = len(device_complaints)

            if device_complaints:
                # Active and confirmed
                device_active = [c for c in device_complaints
                               if c.status not in ["resolved", "closed", "rejected"]]
                device_confirmed = [c for c in device_complaints
                                  if c.resolution == "confirmed_fraud"]

                context["device_active_complaint_count"] = len(device_active)
                context["device_confirmed_fraud_complaint_count"] = len(device_confirmed)

                # Severity
                device_critical = [c for c in device_complaints if c.severity == "critical"]
                device_high = [c for c in device_complaints if c.severity == "high"]

                context["device_critical_complaint_count"] = len(device_critical)
                context["device_high_complaint_count"] = len(device_high)

                # Recency
                device_recent = device_complaints[0]
                device_days_since = (now - device_recent.complaint_date).days

                context["device_days_since_last_complaint"] = device_days_since
                context["device_most_recent_complaint_type"] = device_recent.complaint_type

                # Recent complaints
                device_last_90d = [c for c in device_complaints
                                 if (now - c.complaint_date).days <= 90]
                context["device_complaints_last_90d"] = len(device_last_90d)

                # Fraud ring
                device_in_fraud_ring = any(c.is_part_of_fraud_ring for c in device_complaints)
                context["device_in_fraud_ring"] = device_in_fraud_ring

                # Check for device reuse across multiple accounts
                device_entity_ids = list(set([c.entity_id for c in device_complaints
                                            if c.entity_type == "account"]))
                context["device_linked_account_count"] = len(device_entity_ids)

                if len(device_entity_ids) >= 3:
                    context["device_used_across_multiple_accounts"] = True
                else:
                    context["device_used_across_multiple_accounts"] = False
            else:
                context["device_active_complaint_count"] = 0
                context["device_confirmed_fraud_complaint_count"] = 0
                context["device_in_fraud_ring"] = False
        else:
            context["device_id_not_available"] = True
            context["device_complaint_count"] = 0

        # 4. BENEFICIARY-LEVEL COMPLAINTS
        if beneficiary_id:
            beneficiary_complaints = self.db.query(FraudComplaint).filter(
                FraudComplaint.entity_type == "beneficiary",
                FraudComplaint.entity_id == beneficiary_id
            ).order_by(FraudComplaint.complaint_date.desc()).all()

            context["beneficiary_complaint_count"] = len(beneficiary_complaints)

            if beneficiary_complaints:
                # Active and confirmed
                ben_active = [c for c in beneficiary_complaints
                            if c.status not in ["resolved", "closed", "rejected"]]
                ben_confirmed = [c for c in beneficiary_complaints
                               if c.resolution == "confirmed_fraud"]

                context["beneficiary_active_complaint_count"] = len(ben_active)
                context["beneficiary_confirmed_fraud_complaint_count"] = len(ben_confirmed)

                # Severity
                ben_critical = [c for c in beneficiary_complaints if c.severity == "critical"]
                context["beneficiary_critical_complaint_count"] = len(ben_critical)

                # Recency
                ben_recent = beneficiary_complaints[0]
                ben_days_since = (now - ben_recent.complaint_date).days

                context["beneficiary_days_since_last_complaint"] = ben_days_since

                # Recent complaints
                ben_last_90d = [c for c in beneficiary_complaints
                              if (now - c.complaint_date).days <= 90]
                context["beneficiary_complaints_last_90d"] = len(ben_last_90d)
            else:
                context["beneficiary_active_complaint_count"] = 0
                context["beneficiary_confirmed_fraud_complaint_count"] = 0
        else:
            context["beneficiary_complaint_count"] = 0

        # 5. COMPLAINT VELOCITY ANALYSIS
        # Check if complaints are increasing over time
        if len(account_complaints) >= 3:
            # Compare recent vs older complaints
            recent_complaints = [c for c in account_complaints
                               if (now - c.complaint_date).days <= 30]
            older_complaints = [c for c in account_complaints
                              if 30 < (now - c.complaint_date).days <= 90]

            if older_complaints:
                recent_rate = len(recent_complaints) / 30  # per day
                older_rate = len(older_complaints) / 60  # per day

                if recent_rate > older_rate * 2:
                    context["complaint_velocity_increasing"] = True
                else:
                    context["complaint_velocity_increasing"] = False
            else:
                context["complaint_velocity_increasing"] = False
        else:
            context["complaint_velocity_increasing"] = False

        # 6. GENERATE FRAUD COMPLAINT RISK FLAGS
        risk_flags = []

        # Account flags
        if context.get("account_active_complaint_count", 0) > 0:
            risk_flags.append("account_has_active_complaints")

        if context.get("account_confirmed_fraud_complaint_count", 0) > 0:
            risk_flags.append("account_has_confirmed_fraud_complaints")

        if context.get("account_critical_complaint_count", 0) > 0:
            risk_flags.append("account_has_critical_complaints")

        if context.get("account_complaints_last_30d", 0) >= 2:
            risk_flags.append("account_multiple_recent_complaints")

        if context.get("account_complaint_count", 0) >= 5:
            risk_flags.append("account_repeat_complaint_target")

        if context.get("account_in_fraud_ring", False):
            risk_flags.append("account_in_known_fraud_ring")

        # UPI flags
        if context.get("upi_active_complaint_count", 0) > 0:
            risk_flags.append("upi_has_active_complaints")

        if context.get("upi_confirmed_fraud_complaint_count", 0) > 0:
            risk_flags.append("upi_has_confirmed_fraud_complaints")

        if context.get("upi_critical_complaint_count", 0) > 0:
            risk_flags.append("upi_has_critical_complaints")

        if context.get("upi_complaint_count", 0) >= 3:
            risk_flags.append("upi_multiple_complaints")

        if context.get("upi_in_fraud_ring", False):
            risk_flags.append("upi_in_known_fraud_ring")

        # Device flags
        if context.get("device_active_complaint_count", 0) > 0:
            risk_flags.append("device_has_active_complaints")

        if context.get("device_confirmed_fraud_complaint_count", 0) > 0:
            risk_flags.append("device_has_confirmed_fraud_complaints")

        if context.get("device_critical_complaint_count", 0) > 0:
            risk_flags.append("device_has_critical_complaints")

        if context.get("device_complaint_count", 0) >= 3:
            risk_flags.append("device_multiple_complaints")

        if context.get("device_used_across_multiple_accounts", False):
            risk_flags.append("device_shared_across_accounts")

        if context.get("device_in_fraud_ring", False):
            risk_flags.append("device_in_known_fraud_ring")

        # Beneficiary flags
        if context.get("beneficiary_active_complaint_count", 0) > 0:
            risk_flags.append("beneficiary_has_active_complaints")

        if context.get("beneficiary_confirmed_fraud_complaint_count", 0) > 0:
            risk_flags.append("beneficiary_has_confirmed_fraud_complaints")

        if context.get("beneficiary_complaint_count", 0) >= 3:
            risk_flags.append("beneficiary_multiple_complaints")

        # Velocity flag
        if context.get("complaint_velocity_increasing", False):
            risk_flags.append("complaint_frequency_increasing")

        # Multiple entity types with complaints
        entities_with_complaints = 0
        if context.get("account_complaint_count", 0) > 0:
            entities_with_complaints += 1
        if context.get("upi_complaint_count", 0) > 0:
            entities_with_complaints += 1
        if context.get("device_complaint_count", 0) > 0:
            entities_with_complaints += 1
        if context.get("beneficiary_complaint_count", 0) > 0:
            entities_with_complaints += 1

        if entities_with_complaints >= 3:
            risk_flags.append("multiple_entities_with_complaints")

        context["fraud_complaint_risk_flags"] = risk_flags
        context["fraud_complaint_risk_flag_count"] = len(risk_flags)

        # 7. CALCULATE FRAUD COMPLAINT RISK SCORE (0-100)
        risk_score = 0

        # Account complaint scoring
        account_count = context.get("account_complaint_count", 0)
        if account_count >= 10:
            risk_score += 40
        elif account_count >= 5:
            risk_score += 30
        elif account_count >= 3:
            risk_score += 20
        elif account_count >= 1:
            risk_score += 10

        # Active complaints
        risk_score += min(context.get("account_active_complaint_count", 0) * 10, 30)

        # Confirmed fraud
        risk_score += min(context.get("account_confirmed_fraud_complaint_count", 0) * 15, 45)

        # Critical/high severity
        risk_score += context.get("account_critical_complaint_count", 0) * 20
        risk_score += context.get("account_high_complaint_count", 0) * 10

        # Recent complaints
        if context.get("account_complaints_last_30d", 0) >= 3:
            risk_score += 30
        elif context.get("account_complaints_last_30d", 0) >= 2:
            risk_score += 20
        elif context.get("account_complaints_last_30d", 0) >= 1:
            risk_score += 10

        # UPI complaint scoring
        upi_count = context.get("upi_complaint_count", 0)
        if upi_count >= 5:
            risk_score += 30
        elif upi_count >= 3:
            risk_score += 20
        elif upi_count >= 1:
            risk_score += 10

        risk_score += min(context.get("upi_confirmed_fraud_complaint_count", 0) * 12, 36)

        # Device complaint scoring
        device_count = context.get("device_complaint_count", 0)
        if device_count >= 5:
            risk_score += 30
        elif device_count >= 3:
            risk_score += 20
        elif device_count >= 1:
            risk_score += 10

        risk_score += min(context.get("device_confirmed_fraud_complaint_count", 0) * 12, 36)

        if context.get("device_used_across_multiple_accounts", False):
            risk_score += 25

        # Beneficiary complaint scoring
        ben_count = context.get("beneficiary_complaint_count", 0)
        if ben_count >= 3:
            risk_score += 20
        elif ben_count >= 1:
            risk_score += 10

        risk_score += min(context.get("beneficiary_confirmed_fraud_complaint_count", 0) * 10, 30)

        # Fraud ring membership
        if context.get("account_in_fraud_ring", False):
            risk_score += 40

        if context.get("upi_in_fraud_ring", False):
            risk_score += 35

        if context.get("device_in_fraud_ring", False):
            risk_score += 35

        # Complaint velocity
        if context.get("complaint_velocity_increasing", False):
            risk_score += 20

        # Multiple entities
        if entities_with_complaints >= 3:
            risk_score += 25

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["fraud_complaint_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            complaint_risk_level = "critical"
        elif risk_score >= 60:
            complaint_risk_level = "high"
        elif risk_score >= 40:
            complaint_risk_level = "medium"
        elif risk_score >= 20:
            complaint_risk_level = "low"
        else:
            complaint_risk_level = "minimal"

        context["fraud_complaint_risk_level"] = complaint_risk_level

    def _add_merchant_category_mismatch_context(self, context: Dict[str, Any],
                                                 account_id: str,
                                                 transaction: Dict[str, Any]) -> None:
        """
        Add merchant category mismatch detection context to transaction.

        Detects:
        - Transactions where merchant processes payments outside their registered category
        - Merchants with frequent category mismatches
        - MCC hopping (frequent category changes)
        - High-risk merchant indicators
        - Category-based fraud patterns

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Get merchant identifier from transaction
        merchant_id = transaction.get("counterparty_id") or transaction.get("merchant_id")

        if not merchant_id:
            context["merchant_category_mismatch_check_possible"] = False
            context["merchant_category_mismatch_reason"] = "no_merchant_id"
            return

        # Get merchant profile
        merchant = self.db.query(MerchantProfile).filter(
            MerchantProfile.merchant_id == merchant_id
        ).first()

        if not merchant:
            context["merchant_category_mismatch_check_possible"] = False
            context["merchant_category_mismatch_reason"] = "merchant_not_found"
            context["merchant_unknown"] = True
            context["merchant_unknown_is_suspicious"] = True  # Unknown merchants are suspicious
            return

        context["merchant_category_mismatch_check_possible"] = True

        # Extract transaction MCC from metadata
        tx_metadata = transaction.get("tx_metadata", "{}")
        if isinstance(tx_metadata, str):
            try:
                tx_metadata = json.loads(tx_metadata)
            except:
                tx_metadata = {}

        tx_mcc = tx_metadata.get("mcc") or tx_metadata.get("merchant_category_code")
        tx_category = tx_metadata.get("category") or tx_metadata.get("transaction_category")

        # Merchant basic information
        context["merchant_id"] = merchant.merchant_id
        context["merchant_name"] = merchant.merchant_name
        context["merchant_registered_mcc"] = merchant.registered_mcc
        context["merchant_registered_category"] = merchant.registered_category
        context["merchant_business_type"] = merchant.business_type
        context["merchant_industry"] = merchant.industry
        context["merchant_status"] = merchant.status
        context["merchant_risk_level"] = merchant.risk_level
        context["merchant_verified"] = merchant.verified

        # Transaction MCC information
        context["transaction_mcc"] = tx_mcc
        context["transaction_category"] = tx_category

        # Calculate merchant age
        if merchant.registration_date:
            merchant_age_days = (datetime.datetime.utcnow() - merchant.registration_date).days
            context["merchant_age_days"] = merchant_age_days
            context["merchant_is_new"] = merchant_age_days < 90  # Less than 3 months
            context["merchant_is_very_new"] = merchant_age_days < 30  # Less than 1 month

        # High-risk merchant indicators
        context["merchant_is_high_risk"] = merchant.is_high_risk_merchant
        context["merchant_is_flagged_for_fraud"] = merchant.is_flagged_for_fraud
        if merchant.is_flagged_for_fraud:
            context["merchant_fraud_flag_reason"] = merchant.fraud_flag_reason

        # Category mismatch detection
        mismatch_detected = False
        mismatch_severity = "none"

        if tx_mcc:
            # Direct MCC comparison
            if tx_mcc != merchant.registered_mcc:
                mismatch_detected = True
                context["mcc_mismatch_detected"] = True
                context["expected_mcc"] = merchant.registered_mcc
                context["actual_mcc"] = tx_mcc

                # Check if MCC is in allowed secondary categories
                allowed_mccs = []
                if merchant.allowed_secondary_mccs:
                    try:
                        allowed_mccs = json.loads(merchant.allowed_secondary_mccs)
                    except:
                        allowed_mccs = []

                if tx_mcc in allowed_mccs:
                    context["mcc_mismatch_is_allowed"] = True
                    context["mismatch_in_allowed_secondary_categories"] = True
                    mismatch_severity = "low"
                else:
                    context["mcc_mismatch_is_allowed"] = False
                    mismatch_severity = "high"

                    # Determine if categories are completely unrelated
                    # MCC groups: 0-999 (Airlines), 1000-1999 (Car Rental), 3000-3999 (Hotels),
                    # 4000-4999 (Transportation), 5000-5999 (Retail), 7000-7999 (Services), etc.
                    try:
                        registered_mcc_int = int(merchant.registered_mcc)
                        tx_mcc_int = int(tx_mcc)

                        # Get MCC group (thousands digit)
                        registered_group = registered_mcc_int // 1000
                        tx_group = tx_mcc_int // 1000

                        if registered_group == tx_group:
                            context["mcc_mismatch_same_group"] = True
                            mismatch_severity = "medium"
                        else:
                            context["mcc_mismatch_different_group"] = True
                            mismatch_severity = "high"
                    except:
                        pass

        elif tx_category and merchant.registered_category:
            # Category-based comparison (less precise)
            if tx_category.lower() != merchant.registered_category.lower():
                mismatch_detected = True
                context["category_mismatch_detected"] = True
                context["expected_category"] = merchant.registered_category
                context["actual_category"] = tx_category
                mismatch_severity = "medium"

        context["merchant_category_mismatch_detected"] = mismatch_detected
        context["mismatch_severity"] = mismatch_severity

        # Historical mismatch patterns
        context["merchant_total_mismatch_count"] = merchant.category_mismatch_count
        context["merchant_mismatch_rate"] = merchant.category_mismatch_rate

        # High mismatch rate flags
        if merchant.category_mismatch_rate > 0.5:  # More than 50% mismatches
            context["merchant_has_high_mismatch_rate"] = True
            context["merchant_mismatch_rate_critical"] = True
        elif merchant.category_mismatch_rate > 0.3:  # More than 30% mismatches
            context["merchant_has_high_mismatch_rate"] = True
            context["merchant_mismatch_rate_high"] = True
        elif merchant.category_mismatch_rate > 0.15:  # More than 15% mismatches
            context["merchant_has_elevated_mismatch_rate"] = True

        # Recent mismatch activity
        if merchant.last_mismatch_date:
            days_since_last_mismatch = (datetime.datetime.utcnow() - merchant.last_mismatch_date).days
            context["merchant_days_since_last_mismatch"] = days_since_last_mismatch
            context["merchant_had_recent_mismatch"] = days_since_last_mismatch < 30
            context["merchant_had_very_recent_mismatch"] = days_since_last_mismatch < 7

        # MCC change history (category hopping)
        context["merchant_mcc_change_count"] = merchant.mcc_change_count

        if merchant.mcc_change_count > 0:
            context["merchant_has_changed_mcc"] = True

            # Parse previous MCCs
            if merchant.previous_mccs:
                try:
                    previous_mccs = json.loads(merchant.previous_mccs)
                    context["merchant_previous_mcc_count"] = len(previous_mccs)

                    if len(previous_mccs) > 3:
                        context["merchant_frequent_mcc_changes"] = True

                    # Check recency of last change
                    if merchant.last_mcc_change_date:
                        days_since_mcc_change = (datetime.datetime.utcnow() - merchant.last_mcc_change_date).days
                        context["merchant_days_since_mcc_change"] = days_since_mcc_change
                        context["merchant_recent_mcc_change"] = days_since_mcc_change < 90
                        context["merchant_very_recent_mcc_change"] = days_since_mcc_change < 30
                except:
                    pass

        # Transaction amount analysis relative to merchant patterns
        tx_amount = transaction.get("amount", 0)

        if merchant.avg_transaction_amount:
            amount_ratio = tx_amount / merchant.avg_transaction_amount if merchant.avg_transaction_amount > 0 else 0
            context["merchant_tx_amount_vs_avg_ratio"] = round(amount_ratio, 2)

            if amount_ratio > 10:
                context["merchant_tx_significantly_above_avg"] = True
                context["merchant_tx_amount_anomaly"] = "critical"
            elif amount_ratio > 5:
                context["merchant_tx_above_avg"] = True
                context["merchant_tx_amount_anomaly"] = "high"
            elif amount_ratio > 3:
                context["merchant_tx_moderately_above_avg"] = True
                context["merchant_tx_amount_anomaly"] = "medium"

        if merchant.max_transaction_amount:
            if tx_amount > merchant.max_transaction_amount:
                context["merchant_tx_exceeds_historical_max"] = True
                context["merchant_new_maximum_transaction"] = True
                exceed_ratio = tx_amount / merchant.max_transaction_amount
                context["merchant_tx_vs_max_ratio"] = round(exceed_ratio, 2)

        # Merchant transaction volume and patterns
        context["merchant_total_transactions"] = merchant.total_transactions
        context["merchant_total_volume"] = merchant.total_volume

        if merchant.total_transactions < 10:
            context["merchant_has_few_transactions"] = True
            context["merchant_insufficient_history"] = True

        # Merchant status checks
        if merchant.status != "active":
            context["merchant_not_active"] = True
            context["merchant_status_suspicious"] = True

            if merchant.status == "suspended":
                context["merchant_is_suspended"] = True
            elif merchant.status == "terminated":
                context["merchant_is_terminated"] = True
            elif merchant.status == "under_review":
                context["merchant_under_review"] = True

        # Verification status
        if not merchant.verified:
            context["merchant_not_verified"] = True
            context["merchant_unverified_is_risk"] = True

        # Geographic risk (if available)
        if merchant.registration_country:
            context["merchant_country"] = merchant.registration_country

            # High-risk countries for merchant fraud (example list)
            high_risk_countries = ["XX", "YY", "ZZ"]  # Placeholder codes
            if merchant.registration_country in high_risk_countries:
                context["merchant_from_high_risk_country"] = True

        # Query recent transactions from this merchant to detect patterns
        recent_merchant_txs = self.db.query(Transaction).filter(
            Transaction.counterparty_id == merchant_id,
            Transaction.timestamp >= (datetime.datetime.utcnow() - datetime.timedelta(days=30)).isoformat()
        ).order_by(Transaction.timestamp.desc()).limit(100).all()

        if recent_merchant_txs:
            context["merchant_recent_transaction_count_30d"] = len(recent_merchant_txs)

            # Analyze velocity
            if len(recent_merchant_txs) > 50:
                context["merchant_high_transaction_velocity"] = True

            # Check for amount patterns
            recent_amounts = [float(tx.amount) for tx in recent_merchant_txs if tx.amount]
            if recent_amounts:
                avg_recent_amount = sum(recent_amounts) / len(recent_amounts)
                max_recent_amount = max(recent_amounts)
                min_recent_amount = min(recent_amounts)

                context["merchant_avg_amount_30d"] = round(avg_recent_amount, 2)
                context["merchant_max_amount_30d"] = round(max_recent_amount, 2)
                context["merchant_min_amount_30d"] = round(min_recent_amount, 2)

                # Check for structured transactions (consistent amounts)
                if len(set([round(amt, 2) for amt in recent_amounts])) < len(recent_amounts) * 0.3:
                    context["merchant_structured_transaction_pattern"] = True

                # Check current transaction against recent patterns
                if tx_amount > avg_recent_amount * 5:
                    context["tx_significantly_above_merchant_recent_avg"] = True

                if tx_amount > max_recent_amount:
                    context["tx_exceeds_merchant_recent_max"] = True

        # Risk score calculation (0-100)
        risk_score = 0

        # Base score from merchant risk level
        if merchant.risk_level == "critical":
            risk_score += 30
        elif merchant.risk_level == "high":
            risk_score += 20
        elif merchant.risk_level == "medium":
            risk_score += 10

        # Category mismatch scoring
        if mismatch_detected:
            if mismatch_severity == "high":
                risk_score += 25
            elif mismatch_severity == "medium":
                risk_score += 15
            elif mismatch_severity == "low":
                risk_score += 5

        # Historical mismatch rate
        if merchant.category_mismatch_rate > 0.5:
            risk_score += 20
        elif merchant.category_mismatch_rate > 0.3:
            risk_score += 15
        elif merchant.category_mismatch_rate > 0.15:
            risk_score += 10

        # MCC change frequency
        if merchant.mcc_change_count > 5:
            risk_score += 15
        elif merchant.mcc_change_count > 3:
            risk_score += 10
        elif merchant.mcc_change_count > 0:
            risk_score += 5

        # Recent MCC change
        if context.get("merchant_very_recent_mcc_change"):
            risk_score += 10
        elif context.get("merchant_recent_mcc_change"):
            risk_score += 5

        # Fraud flags
        if merchant.is_flagged_for_fraud:
            risk_score += 25
        if merchant.is_high_risk_merchant:
            risk_score += 15

        # Merchant status
        if merchant.status == "suspended":
            risk_score += 20
        elif merchant.status == "terminated":
            risk_score += 30
        elif merchant.status == "under_review":
            risk_score += 10

        # Verification
        if not merchant.verified:
            risk_score += 10

        # New merchant
        if context.get("merchant_is_very_new"):
            risk_score += 10
        elif context.get("merchant_is_new"):
            risk_score += 5

        # Insufficient transaction history
        if context.get("merchant_insufficient_history"):
            risk_score += 8

        # Amount anomalies
        if context.get("merchant_tx_amount_anomaly") == "critical":
            risk_score += 15
        elif context.get("merchant_tx_amount_anomaly") == "high":
            risk_score += 10
        elif context.get("merchant_tx_amount_anomaly") == "medium":
            risk_score += 5

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["merchant_category_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            category_risk_level = "critical"
        elif risk_score >= 60:
            category_risk_level = "high"
        elif risk_score >= 40:
            category_risk_level = "medium"
        elif risk_score >= 20:
            category_risk_level = "low"
        else:
            category_risk_level = "minimal"

        context["merchant_category_risk_level"] = category_risk_level

    def _add_user_daily_limit_exceeded_context(self, context: Dict[str, Any],
                                                account_id: str,
                                                transaction: Dict[str, Any]) -> None:
        """
        Add user daily limit exceeded detection context to transaction.

        Detects:
        - Transactions that exceed daily transaction count limits
        - Transactions that exceed daily amount limits
        - Transactions that exceed single transaction limits
        - Transactions that exceed transaction-type specific limits
        - Pattern of limit violations
        - Override and regulatory limit violations

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Get account limits
        account_limits = self.db.query(AccountLimit).filter(
            AccountLimit.account_id == account_id,
            AccountLimit.status == "active"
        ).order_by(AccountLimit.effective_date.desc()).first()

        if not account_limits:
            context["daily_limit_check_possible"] = False
            context["daily_limit_check_reason"] = "no_limits_configured"
            context["no_limits_is_risk"] = True  # Accounts without limits are risky
            return

        context["daily_limit_check_possible"] = True

        # Check if limits are expired
        if account_limits.expiration_date and account_limits.expiration_date < datetime.datetime.utcnow():
            context["limits_expired"] = True
            context["limits_expired_is_risk"] = True
            return

        # Extract transaction details
        tx_amount = transaction.get("amount", 0)
        tx_direction = transaction.get("direction", "debit")
        tx_type = transaction.get("transaction_type", "").lower()
        tx_timestamp = transaction.get("timestamp")
        counterparty_id = transaction.get("counterparty_id")

        # Parse timestamp
        if isinstance(tx_timestamp, str):
            try:
                tx_datetime = datetime.datetime.fromisoformat(tx_timestamp)
            except:
                tx_datetime = datetime.datetime.utcnow()
        else:
            tx_datetime = datetime.datetime.utcnow()

        # Get today's date boundaries
        today_start = datetime.datetime.combine(tx_datetime.date(), datetime.time.min)
        today_end = datetime.datetime.combine(tx_datetime.date(), datetime.time.max)

        # Query today's transactions for the account (excluding current transaction)
        today_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp >= today_start.isoformat(),
            Transaction.timestamp <= today_end.isoformat()
        ).all()

        # Calculate today's usage
        today_tx_count = len(today_txs)
        today_total_amount = sum([float(tx.amount) for tx in today_txs if tx.amount])
        today_debit_amount = sum([float(tx.amount) for tx in today_txs if tx.amount and tx.direction == "debit"])
        today_credit_amount = sum([float(tx.amount) for tx in today_txs if tx.amount and tx.direction == "credit"])

        # Calculate amounts including current transaction
        projected_tx_count = today_tx_count + 1
        projected_total_amount = today_total_amount + tx_amount
        projected_debit_amount = today_debit_amount + (tx_amount if tx_direction == "debit" else 0)
        projected_credit_amount = today_credit_amount + (tx_amount if tx_direction == "credit" else 0)

        # Store current usage context
        context["daily_tx_count_today"] = today_tx_count
        context["daily_total_amount_today"] = round(today_total_amount, 2)
        context["daily_debit_amount_today"] = round(today_debit_amount, 2)
        context["daily_credit_amount_today"] = round(today_credit_amount, 2)

        # Store limit information
        context["has_daily_transaction_count_limit"] = account_limits.daily_transaction_count_limit is not None
        context["has_daily_amount_limit"] = account_limits.daily_transaction_amount_limit is not None
        context["has_single_transaction_limit"] = account_limits.single_transaction_limit is not None
        context["is_custom_limit"] = account_limits.is_custom_limit
        context["regulatory_limit"] = account_limits.regulatory_limit

        # Override status
        context["limit_override_enabled"] = account_limits.override_enabled
        if account_limits.override_enabled:
            context["limit_override_reason"] = account_limits.override_reason
            context["limit_override_approved_by"] = account_limits.override_approved_by

            # Check if override is expired
            if account_limits.override_expiration and account_limits.override_expiration < datetime.datetime.utcnow():
                context["limit_override_expired"] = True

        # Violation tracking
        context["total_limit_violations"] = account_limits.total_violations
        context["consecutive_limit_violations"] = account_limits.consecutive_violations

        if account_limits.last_violation_date:
            days_since_violation = (datetime.datetime.utcnow() - account_limits.last_violation_date).days
            context["days_since_last_violation"] = days_since_violation
            context["recent_violation_history"] = days_since_violation < 30

        # Apply risk-based adjustment
        risk_adjustment = account_limits.risk_based_adjustment
        context["risk_based_adjustment_factor"] = risk_adjustment

        # Initialize violation flags
        violations = []
        violation_severity = "none"

        # 1. Check daily transaction count limit
        if account_limits.daily_transaction_count_limit:
            limit_value = account_limits.daily_transaction_count_limit
            context["daily_transaction_count_limit"] = limit_value
            context["projected_daily_tx_count"] = projected_tx_count

            if projected_tx_count > limit_value:
                violations.append("daily_transaction_count_exceeded")
                context["daily_transaction_count_exceeded"] = True
                context["daily_tx_count_overage"] = projected_tx_count - limit_value
                context["daily_tx_count_overage_pct"] = round(((projected_tx_count - limit_value) / limit_value) * 100, 2)
                violation_severity = max(violation_severity, "high")

        # 2. Check daily transaction amount limit
        if account_limits.daily_transaction_amount_limit:
            adjusted_limit = account_limits.daily_transaction_amount_limit * risk_adjustment
            context["daily_transaction_amount_limit"] = round(adjusted_limit, 2)
            context["projected_daily_amount"] = round(projected_total_amount, 2)

            if projected_total_amount > adjusted_limit:
                violations.append("daily_transaction_amount_exceeded")
                context["daily_transaction_amount_exceeded"] = True
                context["daily_amount_overage"] = round(projected_total_amount - adjusted_limit, 2)
                context["daily_amount_overage_pct"] = round(((projected_total_amount - adjusted_limit) / adjusted_limit) * 100, 2)

                if projected_total_amount > adjusted_limit * 1.5:
                    violation_severity = max(violation_severity, "critical")
                    context["daily_amount_significantly_exceeded"] = True
                elif projected_total_amount > adjusted_limit * 1.2:
                    violation_severity = max(violation_severity, "high")
                    context["daily_amount_moderately_exceeded"] = True
                else:
                    violation_severity = max(violation_severity, "medium")

        # 3. Check daily debit limit
        if account_limits.daily_debit_limit and tx_direction == "debit":
            adjusted_limit = account_limits.daily_debit_limit * risk_adjustment
            context["daily_debit_limit"] = round(adjusted_limit, 2)
            context["projected_daily_debit"] = round(projected_debit_amount, 2)

            if projected_debit_amount > adjusted_limit:
                violations.append("daily_debit_limit_exceeded")
                context["daily_debit_limit_exceeded"] = True
                context["daily_debit_overage"] = round(projected_debit_amount - adjusted_limit, 2)
                context["daily_debit_overage_pct"] = round(((projected_debit_amount - adjusted_limit) / adjusted_limit) * 100, 2)
                violation_severity = max(violation_severity, "high")

        # 4. Check daily credit limit
        if account_limits.daily_credit_limit and tx_direction == "credit":
            adjusted_limit = account_limits.daily_credit_limit * risk_adjustment
            context["daily_credit_limit"] = round(adjusted_limit, 2)
            context["projected_daily_credit"] = round(projected_credit_amount, 2)

            if projected_credit_amount > adjusted_limit:
                violations.append("daily_credit_limit_exceeded")
                context["daily_credit_limit_exceeded"] = True
                context["daily_credit_overage"] = round(projected_credit_amount - adjusted_limit, 2)
                violation_severity = max(violation_severity, "medium")

        # 5. Check single transaction limit
        if account_limits.single_transaction_limit:
            adjusted_limit = account_limits.single_transaction_limit * risk_adjustment
            context["single_transaction_limit"] = round(adjusted_limit, 2)

            if tx_amount > adjusted_limit:
                violations.append("single_transaction_limit_exceeded")
                context["single_transaction_limit_exceeded"] = True
                context["single_tx_overage"] = round(tx_amount - adjusted_limit, 2)
                context["single_tx_overage_pct"] = round(((tx_amount - adjusted_limit) / adjusted_limit) * 100, 2)

                if tx_amount > adjusted_limit * 2:
                    violation_severity = max(violation_severity, "critical")
                    context["single_tx_significantly_exceeded"] = True
                else:
                    violation_severity = max(violation_severity, "high")

        # 6. Check single debit/credit limits
        if tx_direction == "debit" and account_limits.single_debit_limit:
            adjusted_limit = account_limits.single_debit_limit * risk_adjustment
            context["single_debit_limit"] = round(adjusted_limit, 2)

            if tx_amount > adjusted_limit:
                violations.append("single_debit_limit_exceeded")
                context["single_debit_limit_exceeded"] = True
                violation_severity = max(violation_severity, "high")

        if tx_direction == "credit" and account_limits.single_credit_limit:
            adjusted_limit = account_limits.single_credit_limit * risk_adjustment
            context["single_credit_limit"] = round(adjusted_limit, 2)

            if tx_amount > adjusted_limit:
                violations.append("single_credit_limit_exceeded")
                context["single_credit_limit_exceeded"] = True
                violation_severity = max(violation_severity, "medium")

        # 7. Check transaction-type specific limits
        tx_type_limit = None
        if tx_type == "ach" and account_limits.ach_daily_limit:
            tx_type_limit = account_limits.ach_daily_limit
            limit_name = "ach_daily_limit"
        elif tx_type == "wire" and account_limits.wire_daily_limit:
            tx_type_limit = account_limits.wire_daily_limit
            limit_name = "wire_daily_limit"
        elif tx_type == "card" and account_limits.card_daily_limit:
            tx_type_limit = account_limits.card_daily_limit
            limit_name = "card_daily_limit"
        elif tx_type == "upi" and account_limits.upi_daily_limit:
            tx_type_limit = account_limits.upi_daily_limit
            limit_name = "upi_daily_limit"

        if tx_type_limit:
            # Calculate today's amount for this transaction type
            today_type_amount = sum([
                float(tx.amount) for tx in today_txs
                if tx.amount and tx.transaction_type and tx.transaction_type.lower() == tx_type
            ])
            projected_type_amount = today_type_amount + tx_amount

            adjusted_limit = tx_type_limit * risk_adjustment
            context[f"{limit_name}"] = round(adjusted_limit, 2)
            context[f"today_{tx_type}_amount"] = round(today_type_amount, 2)
            context[f"projected_{tx_type}_amount"] = round(projected_type_amount, 2)

            if projected_type_amount > adjusted_limit:
                violations.append(f"{limit_name}_exceeded")
                context[f"{limit_name}_exceeded"] = True
                context[f"{tx_type}_overage"] = round(projected_type_amount - adjusted_limit, 2)
                violation_severity = max(violation_severity, "high")

        # 8. Check per-beneficiary limits
        if counterparty_id and account_limits.per_beneficiary_daily_limit:
            # Calculate today's amount to this beneficiary
            today_beneficiary_txs = [
                tx for tx in today_txs
                if tx.counterparty_id == counterparty_id
            ]
            today_beneficiary_amount = sum([float(tx.amount) for tx in today_beneficiary_txs if tx.amount])
            projected_beneficiary_amount = today_beneficiary_amount + tx_amount

            adjusted_limit = account_limits.per_beneficiary_daily_limit * risk_adjustment
            context["per_beneficiary_daily_limit"] = round(adjusted_limit, 2)
            context["today_beneficiary_amount"] = round(today_beneficiary_amount, 2)
            context["projected_beneficiary_amount"] = round(projected_beneficiary_amount, 2)

            if projected_beneficiary_amount > adjusted_limit:
                violations.append("per_beneficiary_daily_limit_exceeded")
                context["per_beneficiary_daily_limit_exceeded"] = True
                context["beneficiary_overage"] = round(projected_beneficiary_amount - adjusted_limit, 2)
                violation_severity = max(violation_severity, "medium")

        if counterparty_id and account_limits.per_beneficiary_transaction_limit:
            adjusted_limit = account_limits.per_beneficiary_transaction_limit * risk_adjustment
            context["per_beneficiary_transaction_limit"] = round(adjusted_limit, 2)

            if tx_amount > adjusted_limit:
                violations.append("per_beneficiary_transaction_limit_exceeded")
                context["per_beneficiary_transaction_limit_exceeded"] = True
                violation_severity = max(violation_severity, "medium")

        # Aggregate violation information
        context["limit_violations"] = violations
        context["limit_violation_count"] = len(violations)
        context["has_limit_violations"] = len(violations) > 0
        context["limit_violation_severity"] = violation_severity

        # Multiple violations are more serious
        if len(violations) >= 3:
            context["multiple_limit_violations"] = True
            context["multiple_violations_critical"] = True
        elif len(violations) >= 2:
            context["multiple_limit_violations"] = True

        # Regulatory limit violation is critical
        if account_limits.regulatory_limit and len(violations) > 0:
            context["regulatory_limit_violated"] = True
            violation_severity = "critical"

        # Pattern analysis - check for systematic limit testing
        if today_tx_count >= 5:
            # Check if multiple transactions are close to limits
            near_limit_count = 0
            for tx in today_txs:
                tx_amt = float(tx.amount) if tx.amount else 0
                if account_limits.single_transaction_limit:
                    limit_ratio = tx_amt / (account_limits.single_transaction_limit * risk_adjustment)
                    if 0.8 <= limit_ratio <= 1.0:  # 80-100% of limit
                        near_limit_count += 1

            if near_limit_count >= 3:
                context["systematic_limit_testing_detected"] = True
                violation_severity = "critical"

        # Check for limit change recency (potential fraud after limit increase)
        if account_limits.limit_change_count > 0 and account_limits.last_limit_change_date:
            days_since_change = (datetime.datetime.utcnow() - account_limits.last_limit_change_date).days
            context["days_since_limit_change"] = days_since_change

            if days_since_change < 7 and len(violations) > 0:
                context["violation_shortly_after_limit_change"] = True
                violation_severity = max(violation_severity, "high")

        # Consecutive violations pattern
        if account_limits.consecutive_violations >= 3:
            context["pattern_of_violations"] = True
            context["persistent_violator"] = True
            violation_severity = max(violation_severity, "high")

        # Risk score calculation (0-100)
        risk_score = 0

        # Base score from violation severity
        if violation_severity == "critical":
            risk_score += 40
        elif violation_severity == "high":
            risk_score += 30
        elif violation_severity == "medium":
            risk_score += 20

        # Multiple violations
        if len(violations) >= 3:
            risk_score += 25
        elif len(violations) >= 2:
            risk_score += 15
        elif len(violations) == 1:
            risk_score += 10

        # Overage magnitude
        if context.get("daily_amount_significantly_exceeded"):
            risk_score += 20
        elif context.get("daily_amount_moderately_exceeded"):
            risk_score += 10

        if context.get("single_tx_significantly_exceeded"):
            risk_score += 15

        # Regulatory limit violation
        if context.get("regulatory_limit_violated"):
            risk_score += 30

        # Systematic limit testing
        if context.get("systematic_limit_testing_detected"):
            risk_score += 20

        # Pattern of violations
        if context.get("persistent_violator"):
            risk_score += 15
        elif account_limits.consecutive_violations >= 2:
            risk_score += 10

        # Recent limit change
        if context.get("violation_shortly_after_limit_change"):
            risk_score += 10

        # No limits configured (risky)
        if context.get("no_limits_is_risk"):
            risk_score += 25

        # Override expired
        if context.get("limit_override_expired"):
            risk_score += 10

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["daily_limit_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            limit_risk_level = "critical"
        elif risk_score >= 60:
            limit_risk_level = "high"
        elif risk_score >= 40:
            limit_risk_level = "medium"
        elif risk_score >= 20:
            limit_risk_level = "low"
        else:
            limit_risk_level = "minimal"

        context["daily_limit_risk_level"] = limit_risk_level

    def _add_recent_high_value_transaction_flags_context(self, context: Dict[str, Any],
                                                          account_id: str,
                                                          transaction: Dict[str, Any]) -> None:
        """
        Add recent high-value transaction flags detection context.

        Detects:
        - Recent high-value transactions that increase fraud risk
        - Rapid transactions following high-value transactions
        - Multiple high-value transactions in succession
        - Velocity of high-value transactions
        - Abnormal patterns after high-value transactions

        Args:
            context: Context dictionary to update
            account_id: Account ID
            transaction: Transaction data
        """
        # Extract current transaction details
        tx_amount = transaction.get("amount", 0)
        tx_timestamp = transaction.get("timestamp")
        tx_direction = transaction.get("direction", "debit")

        # Parse timestamp
        if isinstance(tx_timestamp, str):
            try:
                tx_datetime = datetime.datetime.fromisoformat(tx_timestamp)
            except:
                tx_datetime = datetime.datetime.utcnow()
        else:
            tx_datetime = datetime.datetime.utcnow()

        # Define time windows for "recent" analysis
        windows = {
            "1h": datetime.timedelta(hours=1),
            "6h": datetime.timedelta(hours=6),
            "24h": datetime.timedelta(hours=24),
            "48h": datetime.timedelta(hours=48),
            "7d": datetime.timedelta(days=7),
            "30d": datetime.timedelta(days=30)
        }

        # Get historical transactions to establish baseline
        historical_cutoff = tx_datetime - datetime.timedelta(days=90)
        historical_txs = self.db.query(Transaction).filter(
            Transaction.account_id == account_id,
            Transaction.timestamp >= historical_cutoff.isoformat(),
            Transaction.timestamp < tx_datetime.isoformat()
        ).order_by(Transaction.timestamp.desc()).all()

        if not historical_txs:
            context["recent_high_value_check_possible"] = False
            context["recent_high_value_check_reason"] = "insufficient_history"
            return

        context["recent_high_value_check_possible"] = True

        # Calculate statistical baselines
        amounts = [float(tx.amount) for tx in historical_txs if tx.amount]
        if not amounts:
            context["recent_high_value_check_possible"] = False
            context["recent_high_value_check_reason"] = "no_amount_data"
            return

        avg_amount = sum(amounts) / len(amounts)
        max_amount = max(amounts)
        min_amount = min(amounts)

        # Calculate percentiles
        sorted_amounts = sorted(amounts)
        p95_idx = int(len(sorted_amounts) * 0.95)
        p90_idx = int(len(sorted_amounts) * 0.90)
        p75_idx = int(len(sorted_amounts) * 0.75)

        p95_amount = sorted_amounts[p95_idx] if p95_idx < len(sorted_amounts) else max_amount
        p90_amount = sorted_amounts[p90_idx] if p90_idx < len(sorted_amounts) else max_amount
        p75_amount = sorted_amounts[p75_idx] if p75_idx < len(sorted_amounts) else max_amount

        # Calculate standard deviation
        variance = sum([(amt - avg_amount) ** 2 for amt in amounts]) / len(amounts)
        std_dev = variance ** 0.5

        # Store baseline context
        context["historical_avg_amount"] = round(avg_amount, 2)
        context["historical_max_amount"] = round(max_amount, 2)
        context["historical_p95_amount"] = round(p95_amount, 2)
        context["historical_p90_amount"] = round(p90_amount, 2)
        context["historical_p75_amount"] = round(p75_amount, 2)
        context["historical_std_dev"] = round(std_dev, 2)

        # Define high-value thresholds (multiple criteria)
        high_value_thresholds = {
            "absolute": 10000,  # Fixed threshold (e.g., $10,000)
            "p95": p95_amount,  # 95th percentile of user's history
            "p90": p90_amount,  # 90th percentile
            "3x_avg": avg_amount * 3,  # 3x average
            "2_std_dev": avg_amount + (2 * std_dev)  # 2 standard deviations above mean
        }

        context["high_value_thresholds"] = {k: round(v, 2) for k, v in high_value_thresholds.items()}

        # Analyze recent high-value transactions for each time window
        recent_high_value_analysis = {}

        for window_name, window_delta in windows.items():
            window_start = tx_datetime - window_delta

            # Get transactions in this window
            window_txs = [
                tx for tx in historical_txs
                if tx.timestamp and datetime.datetime.fromisoformat(tx.timestamp) >= window_start
            ]

            if not window_txs:
                continue

            # Identify high-value transactions in this window
            high_value_txs = []
            for tx in window_txs:
                tx_amt = float(tx.amount) if tx.amount else 0

                # Check against multiple thresholds
                is_high_value = (
                    tx_amt >= high_value_thresholds["absolute"] or
                    tx_amt >= high_value_thresholds["p95"] or
                    tx_amt >= high_value_thresholds["3x_avg"]
                )

                if is_high_value:
                    high_value_txs.append({
                        "transaction_id": tx.transaction_id,
                        "amount": tx_amt,
                        "timestamp": tx.timestamp,
                        "direction": tx.direction,
                        "type": tx.transaction_type,
                        "counterparty": tx.counterparty_id
                    })

            window_analysis = {
                "total_transactions": len(window_txs),
                "high_value_count": len(high_value_txs),
                "has_high_value_transactions": len(high_value_txs) > 0
            }

            if high_value_txs:
                # Calculate metrics for high-value transactions
                hv_amounts = [tx["amount"] for tx in high_value_txs]
                window_analysis["high_value_total_amount"] = round(sum(hv_amounts), 2)
                window_analysis["high_value_max_amount"] = round(max(hv_amounts), 2)
                window_analysis["high_value_avg_amount"] = round(sum(hv_amounts) / len(hv_amounts), 2)

                # Find most recent high-value transaction
                most_recent_hv = high_value_txs[0]  # Already sorted by timestamp desc
                most_recent_time = datetime.datetime.fromisoformat(most_recent_hv["timestamp"])
                minutes_since_hv = (tx_datetime - most_recent_time).total_seconds() / 60

                window_analysis["most_recent_hv_amount"] = round(most_recent_hv["amount"], 2)
                window_analysis["minutes_since_most_recent_hv"] = round(minutes_since_hv, 2)
                window_analysis["most_recent_hv_direction"] = most_recent_hv["direction"]

                # Check for rapid follow-up transactions
                if minutes_since_hv < 60:  # Within 1 hour
                    window_analysis["rapid_followup_after_hv"] = True
                    window_analysis["very_rapid_followup"] = minutes_since_hv < 15

                # Check for multiple high-value transactions
                if len(high_value_txs) >= 3:
                    window_analysis["multiple_high_value_txs"] = True
                    window_analysis["high_value_cluster"] = True

            recent_high_value_analysis[window_name] = window_analysis

        context["recent_high_value_analysis"] = recent_high_value_analysis

        # Aggregate flags across all windows
        context["has_recent_high_value_tx_1h"] = recent_high_value_analysis.get("1h", {}).get("has_high_value_transactions", False)
        context["has_recent_high_value_tx_6h"] = recent_high_value_analysis.get("6h", {}).get("has_high_value_transactions", False)
        context["has_recent_high_value_tx_24h"] = recent_high_value_analysis.get("24h", {}).get("has_high_value_transactions", False)
        context["has_recent_high_value_tx_48h"] = recent_high_value_analysis.get("48h", {}).get("has_high_value_transactions", False)
        context["has_recent_high_value_tx_7d"] = recent_high_value_analysis.get("7d", {}).get("has_high_value_transactions", False)

        # Critical flags
        context["rapid_followup_after_high_value"] = recent_high_value_analysis.get("1h", {}).get("rapid_followup_after_hv", False)
        context["very_rapid_followup_after_high_value"] = recent_high_value_analysis.get("1h", {}).get("very_rapid_followup", False)

        # Multiple high-value transaction clusters
        context["high_value_cluster_24h"] = recent_high_value_analysis.get("24h", {}).get("high_value_cluster", False)
        context["high_value_cluster_48h"] = recent_high_value_analysis.get("48h", {}).get("high_value_cluster", False)

        # Current transaction is also high-value
        current_is_high_value = (
            tx_amount >= high_value_thresholds["absolute"] or
            tx_amount >= high_value_thresholds["p95"] or
            tx_amount >= high_value_thresholds["3x_avg"]
        )
        context["current_transaction_is_high_value"] = current_is_high_value

        if current_is_high_value:
            context["current_tx_vs_avg_ratio"] = round(tx_amount / avg_amount, 2) if avg_amount > 0 else 0
            context["current_tx_exceeds_p95"] = tx_amount >= p95_amount
            context["current_tx_exceeds_p90"] = tx_amount >= p90_amount

        # Consecutive high-value transactions pattern
        if current_is_high_value and context.get("has_recent_high_value_tx_24h"):
            context["consecutive_high_value_transactions"] = True

            # Calculate total high-value amount in 24h including current
            hv_24h_total = recent_high_value_analysis.get("24h", {}).get("high_value_total_amount", 0)
            context["high_value_total_24h_including_current"] = round(hv_24h_total + tx_amount, 2)

            # Check if this exceeds normal daily volume
            daily_avg = avg_amount * (len(amounts) / 90)  # Approximate daily average
            if hv_24h_total + tx_amount > daily_avg * 3:
                context["high_value_volume_anomaly"] = True

        # Velocity analysis - transactions after most recent high-value
        if context.get("has_recent_high_value_tx_24h"):
            most_recent_hv_time_str = recent_high_value_analysis.get("24h", {}).get("most_recent_hv_amount")
            if most_recent_hv_time_str:
                # Count transactions between most recent high-value and now
                minutes_since = recent_high_value_analysis.get("24h", {}).get("minutes_since_most_recent_hv", 0)

                if minutes_since > 0:
                    # Count recent transactions after the high-value transaction
                    recent_tx_count = len([
                        tx for tx in historical_txs
                        if tx.timestamp and
                        (tx_datetime - datetime.datetime.fromisoformat(tx.timestamp)).total_seconds() / 60 <= minutes_since
                    ]) + 1  # Include current transaction

                    context["transactions_since_high_value"] = recent_tx_count

                    if minutes_since < 60 and recent_tx_count >= 5:
                        context["high_velocity_after_high_value"] = True
                    elif minutes_since < 360 and recent_tx_count >= 10:  # 6 hours
                        context["elevated_velocity_after_high_value"] = True

        # Pattern: High-value debit followed by multiple credits (refund fraud)
        if context.get("has_recent_high_value_tx_24h"):
            most_recent_hv_direction = recent_high_value_analysis.get("24h", {}).get("most_recent_hv_direction")

            if most_recent_hv_direction == "debit" and tx_direction == "credit":
                context["credit_after_high_value_debit"] = True

                # Check for multiple credits after high-value debit
                minutes_since = recent_high_value_analysis.get("24h", {}).get("minutes_since_most_recent_hv", 0)
                if minutes_since > 0:
                    recent_credits = len([
                        tx for tx in historical_txs
                        if tx.timestamp and tx.direction == "credit" and
                        (tx_datetime - datetime.datetime.fromisoformat(tx.timestamp)).total_seconds() / 60 <= minutes_since
                    ]) + 1

                    if recent_credits >= 3:
                        context["multiple_credits_after_high_value_debit"] = True
                        context["potential_refund_fraud"] = True

        # Pattern: High-value followed by unusual counterparties
        if context.get("has_recent_high_value_tx_6h"):
            current_counterparty = transaction.get("counterparty_id")

            if current_counterparty:
                # Check if current counterparty is new or unusual
                historical_counterparties = set([tx.counterparty_id for tx in historical_txs if tx.counterparty_id])

                if current_counterparty not in historical_counterparties:
                    context["new_counterparty_after_high_value"] = True

        # Calculate time since last high-value transaction
        if context.get("has_recent_high_value_tx_7d"):
            for window in ["1h", "6h", "24h", "48h", "7d"]:
                if recent_high_value_analysis.get(window, {}).get("minutes_since_most_recent_hv") is not None:
                    context["minutes_since_last_high_value_tx"] = recent_high_value_analysis[window]["minutes_since_most_recent_hv"]
                    break

        # Risk score calculation (0-100)
        risk_score = 0

        # Base score for having recent high-value transactions
        if context.get("has_recent_high_value_tx_1h"):
            risk_score += 20
        elif context.get("has_recent_high_value_tx_6h"):
            risk_score += 15
        elif context.get("has_recent_high_value_tx_24h"):
            risk_score += 10
        elif context.get("has_recent_high_value_tx_48h"):
            risk_score += 5

        # Rapid follow-up
        if context.get("very_rapid_followup_after_high_value"):
            risk_score += 25
        elif context.get("rapid_followup_after_high_value"):
            risk_score += 15

        # Current transaction is also high-value
        if context.get("current_transaction_is_high_value"):
            risk_score += 15

        # Consecutive high-value transactions
        if context.get("consecutive_high_value_transactions"):
            risk_score += 20

        # High-value clusters
        if context.get("high_value_cluster_24h"):
            risk_score += 20
        elif context.get("high_value_cluster_48h"):
            risk_score += 10

        # Volume anomaly
        if context.get("high_value_volume_anomaly"):
            risk_score += 15

        # High velocity after high-value
        if context.get("high_velocity_after_high_value"):
            risk_score += 20
        elif context.get("elevated_velocity_after_high_value"):
            risk_score += 10

        # Refund fraud pattern
        if context.get("potential_refund_fraud"):
            risk_score += 25
        elif context.get("credit_after_high_value_debit"):
            risk_score += 10

        # New counterparty after high-value
        if context.get("new_counterparty_after_high_value"):
            risk_score += 15

        # Multiple credits after high-value debit
        if context.get("multiple_credits_after_high_value_debit"):
            risk_score += 15

        # Cap at 100
        risk_score = min(risk_score, 100)

        context["recent_high_value_risk_score"] = risk_score

        # Risk classification
        if risk_score >= 75:
            hv_risk_level = "critical"
        elif risk_score >= 60:
            hv_risk_level = "high"
        elif risk_score >= 40:
            hv_risk_level = "medium"
        elif risk_score >= 20:
            hv_risk_level = "low"
        else:
            hv_risk_level = "minimal"

        context["recent_high_value_risk_level"] = hv_risk_level
