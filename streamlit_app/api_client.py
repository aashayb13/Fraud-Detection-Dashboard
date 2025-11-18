"""
API Client for Streamlit Dashboard

Handles all communication with the FastAPI backend.
Manages authentication tokens and API requests.
"""

import requests
from typing import Optional, Dict, Any, List
import streamlit as st

class FraudAPIClient:
    """
    Client for interacting with the Transaction Monitoring API.

    Handles authentication, token management, and API requests.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.

        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")
        self.token = None
        self.user_info = {}

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user and obtain JWT token.

        Args:
            username: Username
            password: Password

        Returns:
            User information and token

        Raises:
            requests.HTTPError: If login fails
        """
        response = requests.post(
            f"{self.base_url}/api/v1/auth/login",
            data={"username": username, "password": password}
        )
        response.raise_for_status()

        data = response.json()
        self.token = data["access_token"]
        self.user_info = {
            "user_id": data["user_id"],
            "role": data["role"],
            "username": username
        }

        return self.user_info

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication token"""
        if not self.token:
            raise ValueError("Not authenticated. Please login first.")

        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        return response.json()

    def get_overview_stats(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get overview statistics.

        Args:
            time_window_hours: Time window for stats (default 24)

        Returns:
            Overview statistics
        """
        response = requests.get(
            f"{self.base_url}/api/v1/overview",
            headers=self._get_headers(),
            params={"time_window_hours": time_window_hours}
        )
        response.raise_for_status()
        return response.json()

    def get_live_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get live fraud alerts (manual review queue).

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        response = requests.get(
            f"{self.base_url}/api/v1/alerts/live",
            headers=self._get_headers(),
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_top_triggered_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most frequently triggered rules.

        Args:
            limit: Number of rules to return

        Returns:
            List of rule statistics
        """
        response = requests.get(
            f"{self.base_url}/api/v1/rules/top",
            headers=self._get_headers(),
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_scenario_breakdown(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get fraud activity breakdown by scenario.

        Args:
            time_window_hours: Time window for analysis

        Returns:
            Scenario breakdown data
        """
        response = requests.get(
            f"{self.base_url}/api/v1/scenarios/breakdown",
            headers=self._get_headers(),
            params={"time_window_hours": time_window_hours}
        )
        response.raise_for_status()
        return response.json()

    def get_recent_account_changes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent account changes.

        Args:
            limit: Number of changes to return

        Returns:
            List of account changes
        """
        response = requests.get(
            f"{self.base_url}/api/v1/account-changes/recent",
            headers=self._get_headers(),
            params={"limit": limit}
        )
        response.raise_for_status()
        return response.json()

    def get_transaction_details(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get detailed transaction information.

        Args:
            transaction_id: Transaction ID to lookup

        Returns:
            Transaction details
        """
        response = requests.get(
            f"{self.base_url}/api/v1/transaction/{transaction_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def update_alert_status(
        self,
        assessment_id: str,
        action: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update alert status (approve, reject, escalate).

        Args:
            assessment_id: Risk assessment ID
            action: Action to take (approved, rejected, escalated)
            notes: Optional review notes

        Returns:
            Updated status
        """
        response = requests.post(
            f"{self.base_url}/api/v1/alert/{assessment_id}/action",
            headers=self._get_headers(),
            params={"action": action, "notes": notes}
        )
        response.raise_for_status()
        return response.json()

    def get_time_series_metrics(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get time-series metrics for trend analysis.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Time-series data
        """
        response = requests.get(
            f"{self.base_url}/api/v1/metrics/time-series",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()

    def get_risk_distribution(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get risk score distribution for histogram.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Risk distribution data
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/risk-distribution",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()

    def get_money_saved(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get money saved metrics.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Money saved calculations
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/money-saved",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()

    def get_module_performance(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get fraud detection module performance metrics.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Module performance statistics
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/module-performance",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()


    def search_transactions(
        self,
        transaction_id: Optional[str] = None,
        account_id: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search transactions with filters."""
        params = {"limit": limit}
        if transaction_id:
            params["transaction_id"] = transaction_id
        if account_id:
            params["account_id"] = account_id
        if min_amount is not None:
            params["min_amount"] = min_amount
        if max_amount is not None:
            params["max_amount"] = max_amount
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if risk_level:
            params["risk_level"] = risk_level
        response = requests.get(
            f"{self.base_url}/api/v1/investigation/search-transactions",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_account_investigation(self, account_id: str) -> Dict[str, Any]:
        """Get comprehensive account investigation data."""
        response = requests.get(
            f"{self.base_url}/api/v1/investigation/account/{account_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def get_transaction_module_breakdown(self, transaction_id: str) -> Dict[str, Any]:
        """Get fraud module breakdown for a transaction."""
        response = requests.get(
            f"{self.base_url}/api/v1/investigation/transaction/{transaction_id}/modules",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()

    def get_modules_catalog(self, group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Get complete fraud detection modules catalog.

        Args:
            group_by: Optional grouping (category, severity)

        Returns:
            Catalog of all fraud detection modules
        """
        params = {}
        if group_by:
            params["group_by"] = group_by

        response = requests.get(
            f"{self.base_url}/api/v1/modules/catalog",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_geographic_fraud_data(self, time_range: str = "24h") -> Dict[str, Any]:
        """
        Get geographic fraud data for heatmap visualization.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Geographic distribution of fraud activity
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/geographic-fraud",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()

    def get_high_value_transactions(
        self,
        threshold: float = 10000.0,
        time_range: str = "24h",
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get high-value transactions for monitoring.

        Args:
            threshold: Minimum amount to consider high-value
            time_range: Time range (1h, 24h, 7d, 30d)
            limit: Maximum transactions to return

        Returns:
            High-value transactions with risk analysis
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/high-value-transactions",
            headers=self._get_headers(),
            params={
                "threshold": threshold,
                "time_range": time_range,
                "limit": limit
            }
        )
        response.raise_for_status()
        return response.json()

    def get_limit_violations(
        self,
        time_range: str = "24h",
        severity: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get account limit violations.

        Args:
            time_range: Time range (1h, 24h, 7d, 30d)
            severity: Filter by severity (low, medium, high, critical)
            limit: Maximum violations to return

        Returns:
            List of limit violations with account details
        """
        params = {"time_range": time_range, "limit": limit}
        if severity:
            params["severity"] = severity

        response = requests.get(
            f"{self.base_url}/api/v1/analytics/limit-violations",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()

    def get_account_risk_timeline(
        self,
        account_id: str,
        time_range: str = "7d"
    ) -> Dict[str, Any]:
        """
        Get risk score timeline for a specific account.

        Args:
            account_id: Account ID to analyze
            time_range: Time range (1h, 24h, 7d, 30d)

        Returns:
            Time-series risk score data for the account
        """
        response = requests.get(
            f"{self.base_url}/api/v1/analytics/account-risk-timeline/{account_id}",
            headers=self._get_headers(),
            params={"time_range": time_range}
        )
        response.raise_for_status()
        return response.json()

# ==================== Streamlit Session State Management ====================

def get_api_client() -> FraudAPIClient:
    """
    Get or create API client from Streamlit session state.

    Returns:
        Configured API client instance
    """
    if "api_client" not in st.session_state:
        # Get API URL from environment or use default
        import os
        api_url = os.getenv("API_URL", "http://localhost:8000")
        st.session_state.api_client = FraudAPIClient(api_url)

    return st.session_state.api_client


def is_authenticated() -> bool:
    """
    Check if user is authenticated.

    Returns:
        True if authenticated, False otherwise
    """
    client = get_api_client()
    return client.token is not None


def get_user_info() -> Dict[str, Any]:
    """
    Get authenticated user information.

    Returns:
        User info dictionary
    """
    client = get_api_client()
    return client.user_info


def logout():
    """Logout current user"""
    client = get_api_client()
    client.token = None
    client.user_info = {}
    st.session_state.clear()
