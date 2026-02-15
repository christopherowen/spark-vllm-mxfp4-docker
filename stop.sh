#!/bin/bash
# =============================================================================
# Stop vLLM and related processes, show GPU state
# =============================================================================

echo "=============================================="
echo "Stopping vLLM and related processes"
echo "=============================================="

# Check if container is running
if ! docker ps --filter name=vllm-dev --format "{{.Names}}" | grep -q vllm-dev; then
    echo "Container vllm-dev is not running"
else
    # Kill processes inside the container
    docker exec vllm-dev bash -c '
    echo "Killing processes..."
    
    # Kill by specific patterns (order matters - children first)
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null && echo "  VLLM::EngineCore killed" || true
    pkill -9 -f "vllm" 2>/dev/null && echo "  vllm killed" || true
    pkill -9 -f "resource_tracker" 2>/dev/null && echo "  resource_tracker killed" || true
    pkill -9 -f "multiprocessing" 2>/dev/null && echo "  multiprocessing killed" || true
    
    # nsys processes
    pkill -9 -f "nsys-tee" 2>/dev/null && echo "  nsys-tee killed" || true
    pkill -9 -f "nsys-launcher" 2>/dev/null && echo "  nsys-launcher killed" || true
    pkill -9 -f "NsightSystems" 2>/dev/null && echo "  NsightSystems killed" || true
    nsys shutdown 2>/dev/null || true
    
    # Compilation processes
    pkill -9 -f "nvcc" 2>/dev/null || true
    pkill -9 -f "cicc" 2>/dev/null || true
    pkill -9 -f "ptxas" 2>/dev/null || true

    # Kill any remaining GPU processes
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        echo "  Killing GPU process $pid"
        kill -9 $pid 2>/dev/null || true
    done

    sleep 2
    
    # Show remaining (excluding ps itself and bash)
    echo ""
    echo "Remaining processes:"
    ps aux | grep -v -E "^root.*ps aux|^root.*bash$|^root.*/bin/bash$" | grep -E "vllm|nsys|VLLM|python|EngineCore" | head -10 || echo "  None"
    
    true
    '
fi

echo ""
echo "=============================================="
echo "GPU State"
echo "=============================================="

# Try container first, then host
if docker ps --filter name=vllm-dev --format "{{.Names}}" | grep -q vllm-dev; then
    docker exec vllm-dev nvidia-smi
elif command -v nvidia-smi &>/dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not available"
fi

echo ""
echo "Done."
