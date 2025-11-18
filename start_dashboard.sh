#!/bin/bash
# Start the Transaction Monitoring Dashboard
# This script starts both the FastAPI backend and Streamlit frontend

echo "🚀 Starting Transaction Monitoring Dashboard..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements-dashboard.txt
else
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    fi
fi

# Check if database exists
if [ ! -f "transaction_monitoring.db" ]; then
    echo "📊 Database not found. Initializing..."
    python run.py --mode demo
fi

echo ""
echo "✅ Prerequisites ready!"
echo ""
echo "Starting services..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start FastAPI in background
echo "🔧 Starting FastAPI backend on http://localhost:8000"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start
sleep 3

# Check if API started successfully
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ FastAPI backend is running (PID: $API_PID)"
else
    echo "❌ Failed to start FastAPI backend"
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎨 Starting Streamlit dashboard on http://localhost:8501"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Dashboard will open in your browser automatically"
echo ""
echo "🔐 Test Credentials:"
echo "   - Analyst:      Username: analyst      Password: analyst123"
echo "   - Manager:      Username: manager      Password: manager123"
echo "   - Investigator: Username: investigator Password: investigator123"
echo "   - Admin:        Username: admin        Password: admin123"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start Streamlit (this will block)
streamlit run streamlit_app/app.py

# When Streamlit exits, kill the API server
echo ""
echo "🛑 Stopping services..."
kill $API_PID 2>/dev/null
echo "✅ All services stopped"
