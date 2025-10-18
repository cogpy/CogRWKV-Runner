#!/bin/bash
# Startup script for OpenCog-RWKV AGI Backend

echo "Starting OpenCog-RWKV AGI Backend..."
echo "=================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend-opencog"

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "
import sys
missing_deps = []
try:
    import fastapi
except ImportError:
    missing_deps.append('fastapi')
try:
    import uvicorn
except ImportError:
    missing_deps.append('uvicorn')
try:
    import networkx
except ImportError:
    missing_deps.append('networkx')
try:
    import numpy
except ImportError:
    missing_deps.append('numpy')
try:
    import pydantic
except ImportError:
    missing_deps.append('pydantic')

if missing_deps:
    print(f'Missing dependencies: {missing_deps}')
    print('Installing...')
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_deps + ['aiohttp'])
else:
    print('All dependencies available')
"

# Start the OpenCog backend server
echo "Starting OpenCog AGI Backend Server..."
cd backend-opencog

# Default parameters - can be overridden
HOST=${OPENCOG_HOST:-"127.0.0.1"}
PORT=${OPENCOG_PORT:-8001}
DEBUG=${OPENCOG_DEBUG:-false}
DEMO_MODE=${OPENCOG_DEMO:-true}

# Build command line arguments
ARGS="--host $HOST --port $PORT"
if [ "$DEBUG" = "true" ]; then
    ARGS="$ARGS --debug --reload"
fi
if [ "$DEMO_MODE" = "true" ]; then
    ARGS="$ARGS --demo-mode"
fi

echo "Starting server at http://$HOST:$PORT"
echo "Demo mode: $DEMO_MODE"
echo "Debug mode: $DEBUG"
echo ""

# Start the server
python3 main.py $ARGS