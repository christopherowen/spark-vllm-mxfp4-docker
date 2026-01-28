#!/bin/bash
# NCCL RDMA/RoCE Configuration for DGX Spark Two-Node Setup
#
# Based on NVIDIA DGX Spark playbook and forum recommendations.
# This enables proper RDMA transport instead of TCP socket fallback.
#
# IMPORTANT: Set these BEFORE starting Ray and vLLM on BOTH nodes!

# =============================================================================
# INTERFACE CONFIGURATION
# =============================================================================

# The network interface for the 10.10.10.x subnet (DAC link)
export IFACE=enp1s0f1np1

# Socket interface for NCCL control messages
export NCCL_SOCKET_IFNAME=$IFACE
export GLOO_SOCKET_IFNAME=$IFACE

# UCX network device (for Ray)
export UCX_NET_DEVICES=$IFACE

# =============================================================================
# RDMA/RoCE CONFIGURATION (THE CRITICAL PART)
# =============================================================================

# Enable InfiniBand/RDMA path (don't disable IB)
export NCCL_IB_DISABLE=0

# Specify which RoCE devices to use (both ACTIVE devices)
# Check your active devices with: cat /sys/class/infiniband/*/ports/*/state
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1

# GDR (GPUDirect RDMA) - enable for lowest latency
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_GDR_LEVEL=5

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Low-latency protocol for small messages (decode)
export NCCL_PROTO=LL128

# Algorithm selection (Ring is optimal for 2 nodes)
export NCCL_ALGO=Ring

# Fewer channels = less overhead for small messages
export NCCL_MIN_NCHANNELS=1
export NCCL_MAX_NCHANNELS=2

# Smaller buffer for lower latency
export NCCL_BUFFSIZE=1048576

# =============================================================================
# DEBUG (uncomment to verify RDMA is being used)
# =============================================================================

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=NET

# =============================================================================
# VERIFICATION
# =============================================================================

echo "NCCL RDMA Configuration:"
echo "  NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo "  NCCL_IB_HCA=$NCCL_IB_HCA"
echo "  NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL"
echo "  NCCL_PROTO=$NCCL_PROTO"
echo ""
echo "Active RoCE devices:"
for dev in /sys/class/infiniband/*/ports/*/state; do
    state=$(cat "$dev" 2>/dev/null)
    if [[ "$state" == *"ACTIVE"* ]]; then
        echo "  ✓ $(dirname $(dirname $dev) | xargs basename): $state"
    fi
done
echo ""
echo "Now start Ray and vLLM with these environment variables set."
