#!/bin/bash -ex

# Define network ranges
KAFFE_NET="2001:db8:1::/64"
CORP_NET="2001:db8:2::/64"
INTERNET_NET="2001:db8:3::/64"
IPV4_NET="192.168.122.0/24"

# Container names
GW_CONTAINER_NAME="test-gw"
CORP_HTTP_CONTAINER_NAME="test-corp-http"
CLIENT_CONTAINER_NAME="test-client"

# Container image
CONTAINER_IMAGE="quay.io/cathay4t/test-env"
PODMAN_OPTS="--privileged -v /tmp:/mnt"

# Helper functions
function podman_start {
    local name=$1
    shift
    podman run $PODMAN_OPTS -d --name "$name" --hostname "$name" "$@" $CONTAINER_IMAGE
}

function podman_exec {
    podman exec -ti "$1" /bin/bash -c "$2"
}

# Cleanup function
function cleanup {
    echo "Cleaning up resources..."
    podman rm -f $GW_CONTAINER_NAME $CORP_HTTP_CONTAINER_NAME $CLIENT_CONTAINER_NAME || true
    podman network rm -f internet kaffe corp || true
}

# Gateway setup
function setup_gateway {
    # Copy the local RPM into the container
    podman cp ./iptables-nft-1.8.10-5.el9.x86_64.rpm $GW_CONTAINER_NAME:/tmp/

    podman_exec $GW_CONTAINER_NAME "
        # Configure IPv6 addresses
        ip -6 addr add 2001:db8:1::1/64 dev eth1;
        ip -6 addr add 2001:db8:2::1/64 dev eth2;
        ip -6 addr add 2001:db8:3::1/64 dev eth0;

        # Configure IPv4 fallback for package installation
        ip addr add 192.168.122.1/24 dev eth0;

        # Enable global IPv6 forwarding
        sysctl -w net.ipv6.conf.all.forwarding=1;

        # Enable IPv6 forwarding on individual interfaces
        sysctl -w net.ipv6.conf.eth0.forwarding=1;
        sysctl -w net.ipv6.conf.eth1.forwarding=1;
        sysctl -w net.ipv6.conf.eth2.forwarding=1;

        # Ensure iptables-nft is installed using the copied RPM
        if ! command -v ip6tables &>/dev/null; then
            rpm -ivh /tmp/iptables-nft-1.8.10-5.el9.x86_64.rpm || { echo 'Failed to install iptables-nft'; exit 1; }
        fi

        # Set default FORWARD policy to ACCEPT
        ip6tables -P FORWARD ACCEPT;

        # Enable NAT for IPv6
        ip6tables -t nat -A POSTROUTING -o eth0 -j MASQUERADE;

        # Configure routing
        ip -6 route add $KAFFE_NET dev eth1;
        ip -6 route add $CORP_NET dev eth2;

        # Allow forwarding between KAFFE and CORP
        ip6tables -A FORWARD -i eth1 -o eth2 -j ACCEPT;
        ip6tables -A FORWARD -i eth2 -o eth1 -j ACCEPT;
    "
}


# Configure CLIENT and CORP routes
function configure_routes {
    podman_exec $CLIENT_CONTAINER_NAME "
        ip -6 addr add 2001:db8:1::2/64 dev eth0;
        ip -6 route add $CORP_NET via 2001:db8:1::1 dev eth0;
        ip -6 route add default via 2001:db8:1::1;
    "
    podman_exec $CORP_HTTP_CONTAINER_NAME "
        ip -6 addr add 2001:db8:2::2/64 dev eth0;
        ip -6 route add $KAFFE_NET via 2001:db8:2::1 dev eth0;
        ip -6 route add default via 2001:db8:2::1;
    "
}

# Perform IPv6 tests
function perform_tests {
    echo "Performing IPv6 forwarding tests..."
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv6/conf/eth0/forwarding"
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv6/conf/eth1/forwarding"
    podman_exec $GW_CONTAINER_NAME "cat /proc/sys/net/ipv6/conf/eth2/forwarding"

    echo "Testing IPv6 forwarding from KAFFE to CORP..."
    podman_exec $CLIENT_CONTAINER_NAME "ping6 -c 3 2001:db8:2::2" || echo "IPv6 forwarding failed from KAFFE to CORP."

    echo "Testing IPv6 forwarding from CORP to KAFFE..."
    podman_exec $CORP_HTTP_CONTAINER_NAME "ping6 -c 3 2001:db8:1::2" || echo "IPv6 forwarding failed from CORP to KAFFE."
}

# Main function
function main {
    cleanup
    podman network create --subnet 2001:db8:3::/64 --gateway 2001:db8:3::1 internet || true
    podman network create --internal kaffe -o no_default_route=1 || true
    podman network create --internal corp -o no_default_route=1 || true

    podman pull $CONTAINER_IMAGE

    podman_start $GW_CONTAINER_NAME --network internet --network kaffe --network corp
    podman_start $CORP_HTTP_CONTAINER_NAME --network corp
    podman_start $CLIENT_CONTAINER_NAME --network kaffe

    # Ensure RPM file is copied into the gateway container
    podman cp ./iptables-nft-1.8.10-2.el9.x86_64.rpm $GW_CONTAINER_NAME:/tmp/

    setup_gateway
    configure_routes
    perform_tests

    echo "Tests completed."
}

main

