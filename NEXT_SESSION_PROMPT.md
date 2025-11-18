# Next Session Prompt - Continue Dashboard Development

**Use this prompt to start your next Claude Code session and continue building the fraud detection dashboard.**

---

## 📋 Paste This Prompt at the Start of Your Next Session

```
I'm continuing development of a Streamlit fraud detection dashboard for my transaction monitoring system on GitHub (https://github.com/Texasdada13/transaction-monitoring).

IMPORTANT CONTEXT:
- Branch: claude/session-011CUYvXd6EFkotCyShPBnoT
- Location: /home/user/transaction-monitoring (Windows: C:\Users\dada_\OneDrive\Documents\transaction-monitoring)
- Working directory: transaction-monitoring/

CURRENT STATE:
I have a fraud detection system with 25 fraud detection modules in app/services/context_provider.py and database models in app/models/database.py.

I've already built a Streamlit dashboard with FastAPI backend:
✅ Page 1: Real-Time Monitoring (complete)
✅ Page 2: Risk Analytics (complete)
✅ Page 3: Investigation Tools (complete)
❌ Page 4: System Health (NOT STARTED - this is what I need help with)

The dashboard has:
- FastAPI backend with 17 API endpoints
- JWT authentication with 4 user roles
- 3 fully functional Streamlit pages
- Integration with all 25 fraud detection modules
- SQLite database with 15 tables

SESSION SUMMARY:
Please read SESSION_SUMMARY.md in the repo for full details of what's been built.

WHAT I NEED HELP WITH:
1. Build Page 4: System Health monitoring dashboard
2. Add export functionality (CSV download) to analytics pages
3. Improve any existing pages based on my feedback
4. [ADD YOUR SPECIFIC REQUESTS HERE]

KEY FILES TO REVIEW:
- SESSION_SUMMARY.md - Complete overview of current state
- api/main.py - FastAPI backend (17 endpoints)
- streamlit_app/app.py - Main dashboard app
- streamlit_app/pages/ - Existing dashboard pages
- app/services/context_provider.py - 25 fraud detection modules
- app/models/database.py - Database models

IMPORTANT:
- The dashboard is currently working and running on localhost:8501
- API is running on localhost:8000
- Test credentials: analyst/analyst123 (see SESSION_SUMMARY.md for all credentials)
- All code should maintain Windows compatibility
- Use existing patterns from Pages 1-3

Please start by:
1. Reading SESSION_SUMMARY.md to understand what's been built
2. Checking current git status
3. Confirming what I want to build next
4. Then proceed with implementation

Let's continue building!
```

---

## 🎯 Specific Tasks for Next Session

### **Priority 1: Build Page 4 - System Health**

**File to Create**: `streamlit_app/pages/system_health.py`

**Features to Build**:

#### **1. Module Performance Metrics**
- Table showing all 25 fraud detection modules
- Metrics per module:
  - Total triggers (last 24h, 7d, 30d)
  - True positives (confirmed fraud)
  - False positives (approved transactions that were flagged)
  - Precision rate (TP / (TP + FP))
  - Average weight contribution
  - Last triggered timestamp

#### **2. False Positive Rate Trends**
- Line chart showing FP rate over time
- Breakdown by module
- Overall system FP rate
- Comparison to baseline

#### **3. Alert Queue Metrics**
- Current queue size
- Queue size over time (trend chart)
- Average time to resolution
- Pending vs resolved breakdown
- SLA compliance (if applicable)

#### **4. Processing Performance**
- Average transaction processing time
- API response times
- Database query performance
- System load metrics

#### **5. Module Effectiveness Heatmap**
- 2D heatmap showing module performance
- X-axis: Time periods
- Y-axis: Modules
- Color: Effectiveness score

**API Endpoints Needed**:

```python
# Add to api/main.py

@app.get("/api/v1/health/module-metrics")
async def get_module_health_metrics(time_range: str = "24h"):
    """
    Get detailed health metrics for all fraud detection modules.

    Returns:
        - Module trigger counts
        - True positive / false positive rates
        - Precision and recall metrics
        - Last triggered timestamps
    """
    # Implementation here
    pass

@app.get("/api/v1/health/queue-stats")
async def get_queue_statistics(time_range: str = "24h"):
    """
    Get alert queue statistics over time.

    Returns:
        - Queue size trends
        - Average resolution time
        - Pending vs resolved counts
        - SLA metrics
    """
    # Implementation here
    pass

@app.get("/api/v1/health/performance")
async def get_system_performance():
    """
    Get system performance metrics.

    Returns:
        - API response times
        - Database query times
        - Transaction processing times
    """
    # Implementation here
    pass
```

**API Client Methods**:

Add to `streamlit_app/api_client.py`:
```python
def get_module_health_metrics(self, time_range: str = "24h"):
    """Get module health metrics"""
    pass

def get_queue_statistics(self, time_range: str = "24h"):
    """Get queue statistics"""
    pass

def get_system_performance(self):
    """Get system performance metrics"""
    pass
```

---

### **Priority 2: Add Export Functionality**

**Feature**: Add CSV/Excel export buttons to analytics pages

**Implementation**:
1. Add export buttons to Pages 2 and 3
2. Use `pandas.to_csv()` or `pandas.to_excel()`
3. Streamlit's `st.download_button()` for file download

**Example Code**:
```python
# In risk_analytics.py

import pandas as pd
import io

# After displaying data
if st.button("📥 Export Data to CSV"):
    # Convert data to CSV
    csv = df.to_csv(index=False)

    # Download button
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"fraud_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
```

---

### **Priority 3: Batch Actions for Alerts**

**Feature**: Allow analysts to approve/reject multiple alerts at once

**Implementation**:
1. Add checkboxes to alert table in Page 1
2. "Select All" checkbox
3. Bulk action buttons (Approve Selected, Reject Selected)
4. API endpoint for bulk updates

**API Endpoint**:
```python
@app.post("/api/v1/alerts/bulk-action")
async def bulk_update_alerts(
    assessment_ids: List[str],
    action: str,
    notes: Optional[str] = None
):
    """
    Update multiple alerts at once.

    Args:
        assessment_ids: List of assessment IDs
        action: Action to take (approved, rejected)
        notes: Optional notes
    """
    # Implementation here
    pass
```

---

### **Priority 4: Email Notifications**

**Feature**: Send email alerts for high-risk transactions

**Implementation**:
1. Add SMTP configuration to `config/settings.py`
2. Create email template
3. Add notification endpoint
4. Trigger on high-risk alerts (risk_score >= 0.8)

**Libraries Needed**:
```bash
pip install python-multipart
# Add to requirements-complete.txt
```

**Example Code**:
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_fraud_alert_email(transaction_id, risk_score, amount):
    """Send email notification for high-risk transaction"""

    msg = MIMEMultipart()
    msg['From'] = "fraud-alerts@company.com"
    msg['To'] = "security-team@company.com"
    msg['Subject'] = f"🚨 High-Risk Transaction Alert: {transaction_id}"

    body = f"""
    High-risk transaction detected:

    Transaction ID: {transaction_id}
    Risk Score: {risk_score:.2f}
    Amount: ${amount:,.2f}

    Please review immediately in the fraud detection dashboard.
    """

    msg.attach(MIMEText(body, 'plain'))

    # Send email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("your-email@gmail.com", "your-password")
    server.send_message(msg)
    server.quit()
```

---

## 🗂️ File Locations Reference

```
transaction-monitoring/
├── api/
│   ├── main.py              ← Add new health endpoints here
│   └── auth.py
├── streamlit_app/
│   ├── app.py               ← Update page navigation here
│   ├── api_client.py        ← Add new API client methods here
│   └── pages/
│       ├── real_time_monitoring.py     ← Add export + batch actions
│       ├── risk_analytics.py           ← Add export functionality
│       ├── investigation_tools.py
│       └── system_health.py  ← CREATE THIS FILE
├── app/
│   ├── models/
│   │   └── database.py      ← Database models (15 tables)
│   └── services/
│       └── context_provider.py  ← 25 fraud detection modules
├── config/
│   └── settings.py          ← Add email config here
├── SESSION_SUMMARY.md       ← READ THIS FIRST
└── NEXT_SESSION_PROMPT.md   ← This file
```

---

## 🎨 Design Patterns to Follow

### **1. API Endpoint Pattern**
```python
@app.get("/api/v1/category/endpoint-name")
async def function_name(
    param1: str,
    param2: int = 10,
    db: Session = Depends(get_db),
    user: dict = Depends(verify_token)
):
    """
    Clear docstring explaining what this does.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value
    """
    # Query database
    # Process data
    # Return JSON
    return {"data": result}
```

### **2. API Client Method Pattern**
```python
def method_name(self, param1: str, param2: int = 10) -> Dict[str, Any]:
    """Clear docstring"""
    response = requests.get(
        f"{self.base_url}/api/v1/category/endpoint-name",
        headers=self._get_headers(),
        params={"param1": param1, "param2": param2}
    )
    response.raise_for_status()
    return response.json()
```

### **3. Streamlit Page Pattern**
```python
def render():
    """Main render function for the page"""

    # Header
    st.markdown("# Page Title")
    st.markdown("Description")

    # Controls
    time_range = st.selectbox("Time Range", ["24h", "7d", "30d"])

    if st.button("Refresh"):
        st.rerun()

    st.divider()

    # Fetch data
    client = get_api_client()
    try:
        with st.spinner("Loading..."):
            data = client.get_some_data(time_range)

        # Render visualizations
        render_chart(data)
        render_table(data)

    except Exception as e:
        st.error(f"Error: {str(e)}")
```

---

## 🧪 Testing Checklist

After building Page 4, test:

- [ ] Page loads without errors
- [ ] Time range selector works
- [ ] All charts render correctly
- [ ] Metrics display accurate data
- [ ] Refresh button works
- [ ] Export functionality works (if added)
- [ ] API endpoints return valid JSON
- [ ] No authentication errors
- [ ] Windows compatible
- [ ] Responsive layout

---

## 📊 Success Criteria

Page 4 is complete when:

✅ All 25 fraud modules shown in health metrics
✅ False positive rate calculations working
✅ Queue size trend chart displaying
✅ Performance metrics accurate
✅ Export to CSV working
✅ No errors in console
✅ Matches design patterns from Pages 1-3
✅ Committed to git with clear message
✅ Pushed to branch: claude/session-011CUYvXd6EFkotCyShPBnoT

---

## 🎯 Quick Start Commands for Next Session

```powershell
# Navigate to project
cd C:\Users\dada_\OneDrive\Documents\transaction-monitoring

# Pull latest code (from this session)
git pull origin claude/session-011CUYvXd6EFkotCyShPBnoT

# Check status
git status

# Start API
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python -m uvicorn api.main:app --reload --port 8000"

# Wait and start dashboard
Start-Sleep -Seconds 5
streamlit run streamlit_app/app.py

# Login: analyst / analyst123
```

---

## 💡 Pro Tips for Next Session

1. **Start by reviewing SESSION_SUMMARY.md** - It has everything you need to know
2. **Read existing page files** - Copy patterns from Pages 1-3
3. **Test frequently** - Run the dashboard after each feature
4. **Commit often** - Small, focused commits
5. **Check Windows compatibility** - Test on Windows environment
6. **Use existing helpers** - `format_currency()`, `format_timestamp()`, etc.
7. **Leverage Plotly** - Keep charts interactive
8. **Session state** - Use `st.session_state` for navigation

---

## 📚 Key Documentation Links

- **Streamlit Docs**: https://docs.streamlit.io
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Plotly Docs**: https://plotly.com/python/
- **SQLAlchemy Docs**: https://docs.sqlalchemy.org

---

## ⚠️ Important Notes

1. **Don't break existing pages** - Pages 1-3 are working perfectly
2. **Maintain authentication** - All endpoints need `Depends(verify_token)`
3. **Use existing database** - Don't modify schema
4. **Keep Windows compatible** - No bash-specific commands
5. **Follow commit message format** - See existing commits for examples

---

**Ready for your next session!** 🚀

This prompt gives Claude everything needed to continue seamlessly from where we left off.
