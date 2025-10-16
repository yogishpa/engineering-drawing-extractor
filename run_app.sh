#!/bin/bash

# Engineering Drawing Extractor - Startup Script 
echo "🚀 Starting Engineering Drawing Extractor..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create secrets file if it doesn't exist
if [ ! -f ".streamlit/secrets.toml" ]; then
    echo "⚙️ Creating secrets.toml from example..."
    cp .streamlit/secrets.toml.example .streamlit/secrets.toml
    echo "⚠️  Please edit .streamlit/secrets.toml with your AWS credentials"
fi

# Start the application
echo "🎯 Starting Streamlit application..."
echo "📱 Access the app at: http://localhost:8501"
streamlit run app.py

echo "👋 Application stopped."
