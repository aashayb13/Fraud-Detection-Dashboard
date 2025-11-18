# Transaction Monitoring Dashboard

A real-time fraud detection dashboard built with FastAPI and Streamlit.

## Features

- **Real-Time Monitoring**: Live fraud alert queue with auto-refresh
- **JWT Authentication**: Secure role-based access control
- **Interactive Dashboard**: Rich visualizations with Plotly
- **RESTful API**: FastAPI backend with comprehensive endpoints

## Architecture

```
┌─────────────────┐      HTTP/REST      ┌──────────────┐
│   Streamlit     │ ◄─────────────────► │   FastAPI    │
│   Dashboard     │   JSON + JWT Auth   │   Backend    │
└─────────────────┘                     └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │   SQLite/    │
                                        │  PostgreSQL  │
                                        └──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-dashboard.txt
```

### 2. Initialize Database

```bash
# Run the existing setup to create database and sample data
python run.py --mode demo
```

### 3. Start FastAPI Backend

```bash
# Terminal 1: Start API server
python -m uvicorn api.main:app --reload --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 4. Start Streamlit Dashboard

```bash
# Terminal 2: Start Streamlit app
streamlit run streamlit_app/app.py
```

The dashboard will open in your browser at http://localhost:8501

## Test Credentials

| Role         | Username      | Password         | Permissions                           |
|--------------|---------------|------------------|---------------------------------------|
| Analyst      | `analyst`     | `analyst123`     | View alerts, update status            |
| Manager      | `manager`     | `manager123`     | View analytics, export reports        |
| Investigator | `investigator`| `investigator123`| Full investigation access             |
| Admin        | `admin`       | `admin123`       | All permissions                       |

## API Endpoints

### Authentication
- `POST /api/v1/auth/login` - Login and get JWT token

### Dashboard Data
- `GET /api/v1/overview` - Overview statistics
- `GET /api/v1/alerts/live` - Live fraud alerts
- `GET /api/v1/rules/top` - Top triggered rules
- `GET /api/v1/scenarios/breakdown` - Fraud scenario breakdown
- `GET /api/v1/transaction/{id}` - Transaction details

### Alert Management
- `POST /api/v1/alert/{id}/action` - Update alert status (approve/reject/escalate)

### Analytics
- `GET /api/v1/metrics/time-series` - Time-series metrics for charts

## Dashboard Pages

### 1. Real-Time Monitoring (MVP)
- Live alert queue with auto-refresh
- Overview statistics (transactions, risk scores, review rates)
- Top triggered fraud detection rules
- Fraud scenario breakdown
- Quick action buttons (approve/reject/escalate)

### 2. Risk Analytics (Coming Soon)
- KPI trends over time
- Risk score distributions
- Geographic fraud heatmaps
- Fraud type analysis

### 3. Investigation Tools (Coming Soon)
- Transaction search and deep-dive
- Account history viewer
- Module-level feature breakdown
- Related transaction graph

### 4. System Health (Coming Soon)
- Detection module performance
- False positive rates
- Queue size monitoring
- System metrics

## Configuration

### Environment Variables

Create a `.env` file:

```bash
# API Configuration
API_URL=http://localhost:8000
DATABASE_URL=sqlite:///./transaction_monitoring.db

# JWT Secret (change in production!)
SECRET_KEY=your-secret-key-change-in-production

# Token expiration (minutes)
ACCESS_TOKEN_EXPIRE_MINUTES=480
```

### Dashboard Settings

Edit `streamlit_app/app.py` for:
- Auto-refresh intervals
- Default time windows
- Page configurations

## Leveraging Your Existing System

The dashboard integrates with your existing codebase:

✅ **Uses existing database models** from `app/models/database.py`
✅ **Leverages DashboardData class** from `dashboard/main.py`
✅ **Connects to your 25 fraud detection modules** via `ContextProvider`
✅ **Queries your TransactionMonitor** for risk assessments

No changes needed to your core fraud detection logic!

## Development

### Project Structure

```
transaction-monitoring/
├── api/
│   ├── main.py              # FastAPI application
│   ├── auth.py              # JWT authentication
│   └── __init__.py
├── streamlit_app/
│   ├── app.py               # Main Streamlit app
│   ├── api_client.py        # API client wrapper
│   ├── pages/
│   │   ├── real_time_monitoring.py
│   │   └── __init__.py
│   └── __init__.py
├── app/                     # Your existing fraud detection system
├── dashboard/               # Your existing dashboard logic
└── requirements-dashboard.txt
```

### Adding New Pages

1. Create a new file in `streamlit_app/pages/`
2. Import and call in `streamlit_app/app.py`
3. Add necessary API endpoints in `api/main.py`

### Testing

```bash
# Test API health
curl http://localhost:8000/

# Test login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=analyst&password=analyst123"

# Test protected endpoint (replace TOKEN with JWT from login)
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/api/v1/overview
```

## Production Deployment

### Security Checklist
- [ ] Change SECRET_KEY to a secure random value
- [ ] Use environment variables for sensitive config
- [ ] Enable HTTPS/TLS
- [ ] Restrict CORS to your domain
- [ ] Use PostgreSQL instead of SQLite
- [ ] Implement rate limiting
- [ ] Add request logging
- [ ] Set up monitoring/alerting

### Deployment Options

**Option 1: Docker**
```bash
# Coming soon: Dockerfile and docker-compose.yml
```

**Option 2: Cloud Platforms**
- FastAPI: Deploy to AWS Lambda, Google Cloud Run, or Azure Functions
- Streamlit: Deploy to Streamlit Cloud, Heroku, or AWS EC2

## Troubleshooting

### API Connection Issues
```
Error: Cannot connect to API server
```
**Solution**: Ensure FastAPI is running on port 8000:
```bash
python -m uvicorn api.main:app --reload
```

### Authentication Errors
```
Error: Invalid authentication token
```
**Solution**: Check that you're using the correct test credentials or re-login

### Database Errors
```
Error: No such table: transactions
```
**Solution**: Initialize the database:
```bash
python run.py --mode demo
```

## Next Steps

1. ✅ **MVP Complete**: Real-Time Monitoring page
2. 🚧 **In Progress**: Risk Analytics page
3. 📋 **Planned**: Investigation Tools page
4. 📋 **Planned**: System Health page
5. 📋 **Planned**: Export/reporting functionality

## Support

For issues or questions:
1. Check the API docs at http://localhost:8000/docs
2. Review the logs in terminal
3. Open an issue on GitHub

## License

Same as your main project license.
