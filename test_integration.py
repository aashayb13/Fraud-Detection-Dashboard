#!/usr/bin/env python3
"""
Integration Test Script

Tests the complete dashboard stack:
1. Database initialization
2. API authentication
3. API endpoints
4. Data flow

Run this before starting the dashboard to ensure everything works.
"""

import requests
import sys
from app.models.database import init_db, get_db, Transaction, RiskAssessment, Account
from dashboard.main import DashboardData
import json

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def test_database():
    """Test database initialization and data"""
    print_info("Testing database...")

    try:
        init_db()
        db = next(get_db())

        # Check if tables exist and have data
        transaction_count = db.query(Transaction).count()
        assessment_count = db.query(RiskAssessment).count()
        account_count = db.query(Account).count()

        if transaction_count == 0:
            print_warning(f"No transactions found. Run 'python run.py --mode demo' to create sample data.")
            return False

        print_success(f"Database OK: {transaction_count} transactions, {assessment_count} assessments, {account_count} accounts")

        # Test DashboardData
        dashboard = DashboardData(db)
        stats = dashboard.get_overview_stats(24)
        print_info(f"  - Total transactions (24h): {stats['total_transactions']}")
        print_info(f"  - Average risk score: {stats['average_risk_score']:.2f}")

        db.close()
        return True

    except Exception as e:
        print_error(f"Database test failed: {str(e)}")
        return False

def test_api_health(base_url="http://localhost:8000"):
    """Test API health check"""
    print_info("Testing API health...")

    try:
        response = requests.get(f"{base_url}/", timeout=5)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "healthy":
            print_success(f"API health check passed")
            return True
        else:
            print_error(f"API unhealthy: {data}")
            return False

    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is it running? Start with: python -m uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print_error(f"API health test failed: {str(e)}")
        return False

def test_api_auth(base_url="http://localhost:8000"):
    """Test API authentication"""
    print_info("Testing API authentication...")

    try:
        # Test login
        response = requests.post(
            f"{base_url}/api/v1/auth/login",
            data={"username": "analyst", "password": "analyst123"}
        )
        response.raise_for_status()
        data = response.json()

        if "access_token" in data:
            print_success(f"Authentication successful (Role: {data['role']})")
            return data["access_token"]
        else:
            print_error("No access token in response")
            return None

    except Exception as e:
        print_error(f"Authentication test failed: {str(e)}")
        return None

def test_api_endpoints(base_url="http://localhost:8000", token=None):
    """Test API endpoints"""
    print_info("Testing API endpoints...")

    if not token:
        print_error("No token provided, skipping endpoint tests")
        return False

    headers = {"Authorization": f"Bearer {token}"}
    endpoints = [
        ("/api/v1/overview", "Overview stats"),
        ("/api/v1/alerts/live", "Live alerts"),
        ("/api/v1/rules/top", "Top rules"),
        ("/api/v1/scenarios/breakdown", "Scenario breakdown"),
        ("/api/v1/modules/catalog", "Modules catalog"),
    ]

    all_passed = True

    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", headers=headers, timeout=5)
            response.raise_for_status()
            data = response.json()

            print_success(f"{description}: OK")
        except Exception as e:
            print_error(f"{description}: FAILED - {str(e)}")
            all_passed = False

    return all_passed

def main():
    """Run all tests"""
    print("")
    print("=" * 70)
    print(" Transaction Monitoring Dashboard - Integration Test")
    print("=" * 70)
    print("")

    all_tests_passed = True

    # Test 1: Database
    print(f"{BLUE}[1/4] Database Test{RESET}")
    print("-" * 70)
    if not test_database():
        all_tests_passed = False
        print_warning("Database test failed. Some tests may not work properly.")
    print("")

    # Test 2: API Health
    print(f"{BLUE}[2/4] API Health Test{RESET}")
    print("-" * 70)
    if not test_api_health():
        all_tests_passed = False
        print_error("API is not running. Cannot proceed with remaining tests.")
        print_info("Start the API with: python -m uvicorn api.main:app --reload")
        print("")
        sys.exit(1)
    print("")

    # Test 3: Authentication
    print(f"{BLUE}[3/4] Authentication Test{RESET}")
    print("-" * 70)
    token = test_api_auth()
    if not token:
        all_tests_passed = False
        print_error("Authentication failed. Cannot test protected endpoints.")
        print("")
        sys.exit(1)
    print("")

    # Test 4: API Endpoints
    print(f"{BLUE}[4/4] API Endpoints Test{RESET}")
    print("-" * 70)
    if not test_api_endpoints(token=token):
        all_tests_passed = False
    print("")

    # Summary
    print("=" * 70)
    if all_tests_passed:
        print_success("All tests passed! ✨")
        print("")
        print_info("You can now start the dashboard with:")
        print(f"{GREEN}  ./start_dashboard.sh{RESET}")
        print("")
        print("Or manually:")
        print("  Terminal 1: python -m uvicorn api.main:app --reload")
        print("  Terminal 2: streamlit run streamlit_app/app.py")
    else:
        print_error("Some tests failed. Please fix the issues before starting the dashboard.")
    print("=" * 70)
    print("")

    sys.exit(0 if all_tests_passed else 1)

if __name__ == "__main__":
    main()
