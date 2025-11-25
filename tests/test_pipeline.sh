#!/bin/bash
# End-to-end pipeline test with small batches

set -e  # Exit on error

echo "======================================================================"
echo "END-TO-END PIPELINE TEST"
echo "======================================================================"
echo ""
echo "This will collect ~50 test traces to validate the entire pipeline:"
echo "  • 20 clean traces (multi-model)"
echo "  • 30 backdoor traces (10 per TM)"
echo ""
echo "Expected duration: ~25 minutes"
echo "Expected cost: ~$0.12"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "======================================================================"
echo "STEP 1: Clean Traces (20 traces, PARALLEL)"
echo "======================================================================"
echo ""

# Use parallel collection for speed (5 workers = 5x faster)
python collect_traces_parallel.py --num-traces 20 --workers 5

echo ""
echo "======================================================================"
echo "STEP 2: Backdoor Traces - TM1 (10 traces)"
echo "======================================================================"
echo ""

python collect_backdoor_traces.py --threat-model TM1 --num-traces 10

echo ""
echo "======================================================================"
echo "STEP 3: Backdoor Traces - TM2 (10 traces)"
echo "======================================================================"
echo ""

python collect_backdoor_traces.py --threat-model TM2 --num-traces 10

echo ""
echo "======================================================================"
echo "STEP 4: Backdoor Traces - TM3 (10 traces)"
echo "======================================================================"
echo ""

python collect_backdoor_traces.py --threat-model TM3 --num-traces 10

echo ""
echo "======================================================================"
echo "TEST COMPLETE - VERIFICATION"
echo "======================================================================"
echo ""

./monitor_collection.sh

echo ""
echo "======================================================================"
echo "VERIFYING TRACE QUALITY"
echo "======================================================================"
echo ""

python -c "
import json
from pathlib import Path

print('Checking trace quality...')
print()

# Check clean traces
clean_traces = list(Path('data/clean_traces').rglob('*.json'))
if clean_traces:
    with open(clean_traces[0]) as f:
        data = json.load(f)
        print('✅ Clean trace format:')
        print(f'   Keys: {list(data.keys())}')
        print(f'   Model: {data.get(\"model_name\")}')
        print(f'   Steps: {len(data.get(\"steps\", []))}')
        print(f'   Success: {data.get(\"success\")}')
else:
    print('❌ No clean traces found')

print()

# Check backdoor traces
backdoor_traces = list(Path('data/backdoor_traces').rglob('*.json'))
if backdoor_traces:
    with open(backdoor_traces[0]) as f:
        data = json.load(f)
        print('✅ Backdoor trace format:')
        print(f'   Keys: {list(data.keys())}')
        print(f'   Threat Model: {data.get(\"backdoor_metadata\", {}).get(\"threat_model\")}')
        print(f'   Model: {data.get(\"model_name\")}')
        print(f'   Steps: {len(data.get(\"steps\", []))}')
else:
    print('❌ No backdoor traces found')

print()

# Count traces by model
clean_count = len(clean_traces)
backdoor_count = len(backdoor_traces)
total = clean_count + backdoor_count

print('====================================================================')
print('SUMMARY')
print('====================================================================')
print(f'Clean traces: {clean_count}')
print(f'Backdoor traces: {backdoor_count}')
print(f'Total: {total}')
print()

if total >= 50:
    print('✅ Pipeline test PASSED!')
    print()
    print('Next steps:')
    print('1. Check costs in Google Cloud Console')
    print('2. Continue with development (Phase 2: Feature Engineering)')
    print('3. Run full collection later when ready')
else:
    print('⚠️  Expected ~50 traces, got', total)
"

echo ""
echo "======================================================================"
echo "To check costs, visit:"
echo "https://console.cloud.google.com/billing"
echo "======================================================================"
echo ""
