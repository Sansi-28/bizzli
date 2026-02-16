#!/bin/bash

# ==============================================================================
# PowerGuard Deployment Script
# ==============================================================================

set -e

echo "🚀 Manipur PowerGuard Deployment Script"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"

# Function to deploy
deploy() {
    echo -e "\n${YELLOW}📦 Building and deploying containers...${NC}\n"
    
    # Use docker compose (v2) or docker-compose (v1)
    if docker compose version &> /dev/null; then
        docker compose up --build -d
    else
        docker-compose up --build -d
    fi
    
    echo -e "\n${GREEN}✓ Deployment complete!${NC}"
    echo ""
    echo "🌐 Frontend: http://localhost"
    echo "🔧 Backend API: http://localhost:5000/api/health"
    echo ""
    echo "To view logs: docker compose logs -f"
    echo "To stop: docker compose down"
}

# Function to stop
stop() {
    echo -e "\n${YELLOW}🛑 Stopping containers...${NC}\n"
    
    if docker compose version &> /dev/null; then
        docker compose down
    else
        docker-compose down
    fi
    
    echo -e "${GREEN}✓ Containers stopped${NC}"
}

# Function to view logs
logs() {
    if docker compose version &> /dev/null; then
        docker compose logs -f
    else
        docker-compose logs -f
    fi
}

# Function to show status
status() {
    echo -e "\n${YELLOW}📊 Container Status:${NC}\n"
    docker ps --filter "name=powerguard"
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy|start|up)
        deploy
        ;;
    stop|down)
        stop
        ;;
    restart)
        stop
        deploy
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status}"
        exit 1
        ;;
esac
