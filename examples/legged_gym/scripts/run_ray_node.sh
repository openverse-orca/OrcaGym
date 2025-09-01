#!/bin/bash

# Script to start Ray head node using orcagym_addresses from config file
# This script implements two main functions:
# 1. For head node: Use orcagym_addresses IP from rllib_appo_config.yaml to start head node
# 2. Workers can connect to this IP to communicate with the head

set -e  # Exit on error

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../configs/rllib_appo_config.yaml"

# Check if config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Function to parse YAML and extract head node IP
get_head_node_ip() {
    local config_file="$1"
    # Use python to parse YAML and extract the head node IP
    python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    head_node_ip = config.get('orcagym_addresses', {}).get('head_node', '127.0.0.1')
    port = config.get('orcagym_addresses', {}).get('port', 10001)
    
    print(f'{head_node_ip}:{port}')
except Exception as e:
    print('127.0.0.1:10001', file=sys.stderr)
    sys.exit(1)
"
}

# Function to start Ray head node
start_head_node() {
    local head_address="$1"
    local head_ip=$(echo "$head_address" | cut -d':' -f1)
    local port=$(echo "$head_address" | cut -d':' -f2)
    
    echo "Starting Ray head node..."
    echo "Head IP: $head_ip"
    echo "Port: $port"
    
    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        echo "Activating conda environment 'orca'..."
        eval "$(conda shell.bash hook)"
        conda activate orca
    fi
    
    # Start Ray head node with specified address
    ray start --head \
        --node-ip-address="$head_ip" \
        --port="$port" \
        --dashboard-host="0.0.0.0" \
        --dashboard-port=8265 \
        --object-manager-port=8076 \
        --ray-client-server-port=10001 \
        --verbose
    
    echo "Ray head node started successfully!"
    echo "Workers can connect to: $head_address"
    echo "Dashboard available at: http://$head_ip:8265"
}

# Function to start Ray worker node
start_worker_node() {
    local head_address="$1"
    
    echo "Starting Ray worker node..."
    echo "Connecting to head node: $head_address"
    
    # Activate conda environment if available
    if command -v conda &> /dev/null; then
        echo "Activating conda environment 'orca'..."
        eval "$(conda shell.bash hook)"
        conda activate orca
    fi
    
    # Start Ray worker node
    ray start --address="$head_address" --verbose
    
    echo "Ray worker node connected successfully!"
}

# Function to stop Ray
stop_ray() {
    echo "Stopping Ray..."
    ray stop
    echo "Ray stopped successfully!"
}

# Function to show Ray cluster status
show_status() {
    echo "Ray cluster status:"
    ray status
}

# Main function
main() {
    local command="${1:-head}"  # Default to head node
    local head_address
    
    case "$command" in
        "head")
            echo "=== Starting Ray Head Node ==="
            head_address=$(get_head_node_ip "$CONFIG_FILE")
            if [[ $? -ne 0 ]]; then
                echo "Error: Failed to parse configuration file"
                exit 1
            fi
            start_head_node "$head_address"
            ;;
        "worker")
            echo "=== Starting Ray Worker Node ==="
            head_address=$(get_head_node_ip "$CONFIG_FILE")
            if [[ $? -ne 0 ]]; then
                echo "Error: Failed to parse configuration file"
                exit 1
            fi
            start_worker_node "$head_address"
            ;;
        "stop")
            echo "=== Stopping Ray ==="
            stop_ray
            ;;
        "status")
            echo "=== Ray Status ==="
            show_status
            ;;
        *)
            echo "Usage: $0 [head|worker|stop|status]"
            echo ""
            echo "Commands:"
            echo "  head     - Start Ray head node (default)"
            echo "  worker   - Start Ray worker node"
            echo "  stop     - Stop Ray cluster"
            echo "  status   - Show Ray cluster status"
            echo ""
            echo "Configuration file: $CONFIG_FILE"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"