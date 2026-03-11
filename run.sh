#!/bin/bash
echo "Starting mongod service..."
sudo systemctl start mongod

echo "Activating virtual environment..."
source venv/bin/activate

echo "Starting Flask server..."
python src/app.py
