#!/bin/bash -ex

# Define network ranges
KAFFE_NET="192.168.1.0/24"
CORP_NET="10.0.1.0/24"
INTERNET_NET="192.0.2.0/24"

# Container names
GW_CONTAINER_NAME="test-gw"
CORP_HTTP_CONTAINER_NAME="test-corp-http"
CLIENT_CONTAINER_NAME="test-client"

# Container image
CONTAINER_IMAGE="quay.io/cathay4t/test-env"
PODMAN_OPTS="--privileged -v /tmp:/mnt"

# Helper functions
function podman_start {
    name=$1
    shift
    podman run $PODMAN_OPTS -d --name $name --hostname $name "$@" $CONTAINER_IMAGE
}

function podman_exec {
    podman exec -ti $1 /bin/bash -c "$2"
}

# Cleanup function
function cleanup {
    echo "Cleaning up resources..."
    podman rm -f $GW_CONTAINER_NAME $CORP_HTTP_CONTAINER_NAME $CLIENT_CONTAINER_NAME || true
    podman network rm -f internet kaffe corp || true
}

# Gateway setup
function setup_gateway {
    podman_exec $GW_CONTAINER_NAME "
        # Configure IPv4 addresses
        ip addr add 192.168.1.1/24 dev eth1;
        ip addr add 10.0.1.1/24 dev eth2;
        ip addr add 192.0.2.1/24 dev eth0;

        # Enable global IPv4 forwarding
        sysctl -w net.ipv4.ip_forward=0;

        # Enable per-device IPv4 forwarding
        sysctl -w net.ipv4.conf.eth0.forwarding=1;
        sysctl -w net.ipv4.conf.eth1.forwarding=1;
        sysctl -w net.ipv4.conf.eth2.forwarding=1;


        # Install iptables if not available
        command -v iptables || yum install -y iptables iptables-services

        # Set default FORWARD policy to ACCEPT
        iptables -P FORWARD ACCEPT

        # Enable NAT for IPv4
        iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE;

        # Configure routing
        ip route add $KAFFE_NET dev eth1;
        ip route add $CORP_NET dev eth2;

        # Allow forwarding between KAFFE and CORP
        iptables -A FORWARD -i eth1 -o eth2 -j ACCEPT;
        iptables -A FORWARD -i eth2 -o eth1 -j ACCEPT;

        # Enable logging for dropped packets (optional)
        iptables -A FORWARD -j LOG --log-prefix 'FORWARD DROP: ' --log-level 4
    "
}

# Configure CLIENT and CORP routes
function configure_routes {
    # Add routes for KAFFE client
    podman_exec $CLIENT_CONTAINER_NAME "
        ip addr add 192.168.1.2/24 dev eth0;
        ip route add $CORP_NET via 192.168.1.1 dev eth0;
        ip route add default via 192.168.1.1;
    "

    # Add routes for CORP HTTP server
    podman_exec $CORP_HTTP_CONTAINER_NAME "
        ip addr add 10.0.1.2/24 dev eth0;
        ip route add $KAFFE_NET via 10.0.1.1 dev eth0;
        ip route add default via 10.0.1.1;
    "
}

# Perform IPv4 tests
function perform_tests {
    echo "Performing IPv4 forwarding tests..."

    # Check forwarding status
    echo "Checking IPv4 forwarding status for devices:"
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv4/conf/eth0/forwarding"
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv4/conf/eth1/forwarding"
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv4/conf/eth2/forwarding"

    # Test IPv4 forwarding
    echo "Testing IPv4 forwarding from KAFFE to CORP..."
    podman_exec $CLIENT_CONTAINER_NAME "ping -c 3 10.0.1.2" || echo "IPv4 forwarding failed from KAFFE to CORP."

    echo "Testing IPv4 forwarding from CORP to KAFFE..."
    podman_exec $CORP_HTTP_CONTAINER_NAME "ping -c 3 192.168.1.2" || echo "IPv4 forwarding failed from CORP to KAFFE."
}

# Main function
function main {
    cleanup

    # Create networks
    podman network create --subnet 192.0.2.0/24 --gateway 192.0.2.1 internet || true
    podman network create --internal kaffe -o no_default_route=1 || true
    podman network create --internal corp -o no_default_route=1 || true

    # Pull container image
    podman pull $CONTAINER_IMAGE

    # Start containers
    podman_start $GW_CONTAINER_NAME --network internet --network kaffe --network corp
    podman_start $CORP_HTTP_CONTAINER_NAME --network corp
    podman_start $CLIENT_CONTAINER_NAME --network kaffe

    # Setup gateway and routes
    setup_gateway
    configure_routes

    # Perform tests
    perform_tests

    echo "Tests completed."
}

main

