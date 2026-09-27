#!/bin/bash
#
# Validate Prometheus port-forwards and restart if needed
# This script checks connectivity to Prometheus on fog (9091) and edge (9090)
# and attempts to restart port-forwards if they're not accessible
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Prometheus Port-Forward Validator${NC}"
echo -e "${BLUE}================================${NC}\n"

# Function to check port connectivity
check_port() {
    local host=$1
    local port=$2
    local name=$3
    
    echo -n "Checking ${name} (${host}:${port})... "
    
    if timeout 2 bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
        echo -e "${GREEN}✓ Accessible${NC}"
        return 0
    else
        echo -e "${RED}✗ NOT accessible${NC}"
        return 1
    fi
}

# Function to kill and restart port-forward
restart_port_forward() {
    local port=$1
    local kubeconfig=$2
    local name=$3
    
    echo -e "${YELLOW}Restarting port-forward for ${name}...${NC}"
    
    # Kill existing port-forward
    echo "  Killing existing port-forward on port ${port}..."
    pkill -f "port-forward.*${port}" || true
    sleep 1
    
    # Start new port-forward
    echo "  Starting new port-forward on port ${port}..."
    export KUBECONFIG="${kubeconfig}"
    kubectl port-forward -n cattle-monitoring-system svc/rancher-monitoring-prometheus ${port}:9090 > /tmp/pf_${port}.log 2>&1 &
    PF_PID=$!
    echo "  Port-forward started (PID: ${PF_PID})"
    
    # Wait for port to be accessible
    echo -n "  Waiting for port to become accessible... "
    for i in {1..10}; do
        sleep 1
        if timeout 2 bash -c "echo >/dev/tcp/localhost/${port}" 2>/dev/null; then
            echo -e "${GREEN}✓ Success${NC}"
            return 0
        fi
    done
    
    echo -e "${RED}✗ Timeout${NC}"
    return 1
}

# Check fog cluster port-forward (9091)
FOG_KUBECONFIG=~/.kube/config.lab.fog
FOG_OK=1
if ! check_port "localhost" "9091" "Fog Prometheus"; then
    FOG_OK=0
    if [ ! -f "$FOG_KUBECONFIG" ]; then
        echo -e "${RED}  Error: Fog kubeconfig not found at $FOG_KUBECONFIG${NC}"
    else
        restart_port_forward "9091" "$FOG_KUBECONFIG" "Fog"
        FOG_OK=$?
    fi
fi

echo ""

# Check edge cluster port-forward (9090)
EDGE_KUBECONFIG=~/.kube/config.lab.edge
EDGE_OK=1
if ! check_port "localhost" "9090" "Edge Prometheus"; then
    EDGE_OK=0
    if [ ! -f "$EDGE_KUBECONFIG" ]; then
        echo -e "${RED}  Error: Edge kubeconfig not found at $EDGE_KUBECONFIG${NC}"
    else
        restart_port_forward "9090" "$EDGE_KUBECONFIG" "Edge"
        EDGE_OK=$?
    fi
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}================================${NC}"

if [ $FOG_OK -eq 0 ]; then
    echo -e "${RED}✗ Fog Prometheus (9091): NOT accessible${NC}"
else
    echo -e "${GREEN}✓ Fog Prometheus (9091): Accessible${NC}"
fi

if [ $EDGE_OK -eq 0 ]; then
    echo -e "${RED}✗ Edge Prometheus (9090): NOT accessible${NC}"
else
    echo -e "${GREEN}✓ Edge Prometheus (9090): Accessible${NC}"
fi

# Test actual Prometheus queries
echo ""
echo -e "${BLUE}Testing Prometheus Queries${NC}"
echo -e "${BLUE}================================${NC}"

if [ $FOG_OK -eq 1 ]; then
    echo -n "Fog - Query status: "
    if curl -s 'http://localhost:9091/api/v1/query?query=up' | grep -q '"status":"success"'; then
        echo -e "${GREEN}✓ Responding${NC}"
    else
        echo -e "${RED}✗ Not responding${NC}"
    fi
fi

if [ $EDGE_OK -eq 1 ]; then
    echo -n "Edge - Query status: "
    if curl -s 'http://localhost:9090/api/v1/query?query=up' | grep -q '"status":"success"'; then
        echo -e "${GREEN}✓ Responding${NC}"
    else
        echo -e "${RED}✗ Not responding${NC}"
    fi
fi

echo ""

# Final status
if [ $FOG_OK -eq 1 ] && [ $EDGE_OK -eq 1 ]; then
    echo -e "${GREEN}✓ All port-forwards are operational!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some port-forwards are still not accessible.${NC}"
    echo -e "${YELLOW}Check logs in /tmp/pf_*.log for details${NC}"
    exit 1
fi
