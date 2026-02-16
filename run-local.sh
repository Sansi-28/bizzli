#!/bin/bash

# ==============================================================================
# PowerGuard Local Deployment Script (without Docker)
# ==============================================================================

set -e

echo "🚀 Manipur PowerGuard - Local Deployment"
echo "========================================="

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python: $(python3 --version)${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js: $(node --version)${NC}"

# Function to install dependencies
install_deps() {
    echo -e "\n${YELLOW}📦 Installing dependencies...${NC}\n"
    
    # Backend
    echo -e "${BLUE}Installing Python dependencies...${NC}"
    cd "$PROJECT_DIR/backend"
    pip3 install -r requirements.txt
    pip3 install gunicorn
    
    # Frontend
    echo -e "${BLUE}Installing Node.js dependencies...${NC}"
    cd "$PROJECT_DIR/frontend"
    npm install
    
    echo -e "${GREEN}✓ Dependencies installed${NC}"
}

# Function to build frontend
build_frontend() {
    echo -e "\n${YELLOW}🔨 Building frontend for production...${NC}\n"
    cd "$PROJECT_DIR/frontend"
    
    # Set API URL for production build
    export REACT_APP_API_URL="http://localhost:5000/api"
    npm run build
    
    echo -e "${GREEN}✓ Frontend built successfully${NC}"
}

# Function to start backend
start_backend() {
    echo -e "\n${YELLOW}🔧 Starting backend server...${NC}\n"
    cd "$PROJECT_DIR/backend"
    
    # Kill any existing process on port 5000
    fuser -k 5000/tcp 2>/dev/null || true
    
    # Start with gunicorn
    gunicorn --bind 0.0.0.0:5000 --workers 4 --daemon app:app
    
    echo -e "${GREEN}✓ Backend running on http://localhost:5000${NC}"
}

# Function to start development servers
dev() {
    echo -e "\n${YELLOW}🔧 Starting development servers...${NC}\n"
    
    # Start backend in background
    cd "$PROJECT_DIR/backend"
    python3 app.py &
    BACKEND_PID=$!
    echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
    
    # Start frontend
    cd "$PROJECT_DIR/frontend"
    echo -e "${BLUE}Starting frontend development server...${NC}"
    npm start
}

# Function to serve production build
serve_prod() {
    echo -e "\n${YELLOW}🚀 Starting production servers...${NC}\n"
    
    # Build frontend if not exists
    if [ ! -d "$PROJECT_DIR/frontend/build" ]; then
        build_frontend
    fi
    
    start_backend
    
    # Serve frontend with serve (install if needed)
    if ! command -v serve &> /dev/null; then
        echo -e "${BLUE}Installing 'serve' for frontend...${NC}"
        npm install -g serve
    fi
    
    cd "$PROJECT_DIR/frontend"
    serve -s build -l 3000 &
    
    echo -e "\n${GREEN}✓ Production deployment complete!${NC}"
    echo ""
    echo "🌐 Frontend: http://localhost:3000"
    echo "🔧 Backend API: http://localhost:5000/api/health"
}

# Function to stop all servers
stop() {
    echo -e "\n${YELLOW}🛑 Stopping servers...${NC}\n"
    
    # Kill processes on ports
    fuser -k 5000/tcp 2>/dev/null || true
    fuser -k 3000/tcp 2>/dev/null || true
    
    # Kill gunicorn
    pkill -f gunicorn 2>/dev/null || true
    
    echo -e "${GREEN}✓ Servers stopped${NC}"
}

# Parse arguments
case "${1:-dev}" in
    install)
        install_deps
        ;;
    build)
        build_frontend
        ;;
    dev)
        dev
        ;;
    prod|production)
        serve_prod
        ;;
    stop)
        stop
        ;;
    *)
        echo "Usage: $0 {install|build|dev|prod|stop}"
        echo ""
        echo "Commands:"
        echo "  install  - Install all dependencies"
        echo "  build    - Build frontend for production"
        echo "  dev      - Start development servers"
        echo "  prod     - Start production servers"
        echo "  stop     - Stop all servers"
        exit 1
        ;;
esac
