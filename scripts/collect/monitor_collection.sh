#!/bin/bash
# Monitor trace collection progress

echo "======================================================================"
echo "TRACE COLLECTION MONITOR"
echo "======================================================================"
echo ""

# Count traces in each directory
echo "📊 Current Progress:"
echo ""

if [ -d "data/clean_traces" ]; then
    echo "Clean Traces:"
    for dir in data/clean_traces/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
            dirname=$(basename "$dir")
            echo "  • $dirname: $count traces"
        fi
    done
    total_clean=$(find data/clean_traces -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total clean: $total_clean/1000"
else
    echo "Clean Traces: 0/1000 (not started)"
fi

echo ""

if [ -d "data/backdoor_traces" ]; then
    echo "Backdoor Traces:"
    for dir in data/backdoor_traces/*/; do
        if [ -d "$dir" ]; then
            count=$(find "$dir" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
            dirname=$(basename "$dir")
            echo "  • $dirname: $count traces"
        fi
    done
    total_backdoor=$(find data/backdoor_traces -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  Total backdoor: $total_backdoor/500"
else
    echo "Backdoor Traces: 0/500 (not started)"
fi

echo ""
echo "======================================================================"
echo "Total Progress: $((total_clean + total_backdoor))/1500 traces"
echo "======================================================================"
echo ""

# Show recent trace files
echo "Most recent traces:"
find data -name "*.json" -type f 2>/dev/null | sort -r | head -3
echo ""

# Check if collection is running
if pgrep -f "collect_traces.py" > /dev/null; then
    echo "✅ Collection process is running"
else
    echo "⚠️  No collection process detected"
fi

echo ""
