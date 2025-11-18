# Quick Start Guide - Streamlit Dashboard

Get your fraud detection dashboard running in 3 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
# Install all dependencies
pip install -r requirements-complete.txt
```

If you encounter any issues, install individually:
```bash
pip install sqlalchemy fastapi uvicorn streamlit plotly pandas requests python-jose passlib
```

## Step 2: Create Sample Data (1 minute)

```bash
# Initialize database and create demo transactions
python run.py --mode demo
```

This creates:
- Database with sample accounts, transactions, and fraud scenarios
- Risk assessments for demonstration
- Test data across all 25 fraud detection modules

## Step 3: Start the Dashboard (1 minute)

### Option A: Automated Startup (Recommended)

```bash
# Start both API and dashboard automatically
./start_dashboard.sh
```

### Option B: Manual Startup

Terminal 1 - Start API:
```bash
python -m uvicorn api.main:app --reload --port 8000
```

Terminal 2 - Start Dashboard:
```bash
streamlit run streamlit_app/app.py
```

## Step 4: Login

The dashboard will open at **http://localhost:8501**

Use these test credentials:

| Username      | Password         | Role         |
|---------------|------------------|--------------|
| `analyst`     | `analyst123`     | Analyst      |
| `manager`     | `manager123`     | Manager      |
| `investigator`| `investigator123`| Investigator |
| `admin`       | `admin123`       | Admin        |

## What You'll See

### Real-Time Monitoring Page
- **Overview Stats**: Total transactions, risk scores, review rates
- **Live Alert Queue**: Pending fraud alerts sorted by risk
- **Top Triggered Rules**: Most active fraud detection modules
- **Scenario Breakdown**: Activity by fraud type (payroll, beneficiary, etc.)
- **Quick Actions**: Approve/Reject/Escalate buttons for each alert

### Risk Analytics Page
- **Time-Series Analysis**: Transaction and risk trends over time
- **Risk Distribution**: Histogram of risk scores
- **Money Saved**: Financial impact of fraud prevention
- **Module Performance**: Top performing fraud detection modules

### Investigation Tools Page
- **Transaction Search**: Multi-criteria search with filters
- **Account Deep-Dive**: Comprehensive account investigation
- **Module Breakdown**: Detailed view of triggered modules per transaction

### Module Catalog Page (NEW!)
- **All 25+ Modules**: Complete showcase of fraud detection modules
- **Grouped by Category**: Behavioral, Financial Crime, Location Analysis, etc.
- **Grouped by Severity**: Critical, High, Medium, Low
- **Search & Filter**: Find specific modules quickly
- **Detection Capabilities**: What each module detects and why

## Troubleshooting

### "Cannot connect to API server"
**Solution**: Make sure the API is running:
```bash
python -m uvicorn api.main:app --reload
```

### "No module named 'X'"
**Solution**: Install missing dependencies:
```bash
pip install -r requirements-complete.txt
```

### "No data in dashboard"
**Solution**: Create sample data:
```bash
python run.py --mode demo
```

### "Login failed"
**Solution**: Make sure you're using the correct test credentials (see table above)

## Testing the Integration

Before starting the dashboard, you can run tests:

```bash
# Test database only
python test_integration.py

# This will check:
# ✅ Database initialization
# ✅ API health
# ✅ Authentication
# ✅ All API endpoints
```

## Next Steps

Once you're up and running:

1. **Explore the Dashboard**: Check out the real-time monitoring features
2. **Create More Data**: Run additional demo scenarios
3. **Customize**: Modify the dashboard pages in `streamlit_app/pages/`
4. **Add Features**: Check `README-DASHBOARD.md` for development guide

## API Documentation

While the API is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Common Tasks

### Adding New Transactions
```python
# Use your existing system
from app.scenarios.payroll_reroute_scenario import main
main()
```

### Clearing Data
```bash
# Delete database and start fresh
rm transaction_monitoring.db
python run.py --mode demo
```

### Changing Refresh Interval
In the dashboard, toggle "Auto-refresh" and select interval (30s, 60s, 90s)

## Need Help?

1. Check the full documentation: `README-DASHBOARD.md`
2. View API docs: http://localhost:8000/docs
3. Review logs in terminal
4. Check GitHub issues

## Architecture Overview

```
User Browser
     ↓
Streamlit Dashboard (Port 8501)
     ↓ (HTTP/REST + JWT)
FastAPI Backend (Port 8000)
     ↓
Your Existing Fraud Detection System
     ↓
SQLite Database
```

**You're all set! Happy fraud hunting! 🛡️**
