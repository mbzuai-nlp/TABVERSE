#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mominaahsan/Desktop/VisualTableBench"
TASK_DIR="$ROOT/data_full/2-task"

echo "Checking for duplicate table-query-label triplets in 2-task files..."
echo "================================================================"

for json_file in "$TASK_DIR"/*.json; do
    if [[ -f "$json_file" ]]; then
        filename=$(basename "$json_file")
        
        python3 - "$json_file" <<'PYTHON'
import json
import sys
from collections import Counter

filepath = sys.argv[1]
filename = filepath.split('/')[-1]

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract table-query-label triplets
triplets = []
for sample in data:
    table = sample.get('table', {})
    # Serialize table to make it hashable
    table_str = json.dumps(table, sort_keys=True)
    query = sample.get('query', '')
    label = tuple(sample.get('label', []))  # Convert list to tuple for hashing
    triplets.append((table_str, query, label))

# Count occurrences
triplet_counts = Counter(triplets)

# Find duplicates
total_samples = len(data)
unique_triplets = len(triplet_counts)
duplicates = {triplet: count for triplet, count in triplet_counts.items() if count > 1}
num_duplicate_triplets = len(duplicates)
total_duplicate_instances = sum(count - 1 for count in duplicates.values())

print(f"\n📊 {filename}")
print(f"   Total samples: {total_samples}")
print(f"   Unique table-query-label triplets: {unique_triplets}")
print(f"   Duplicate triplets: {num_duplicate_triplets}")
print(f"   Total duplicate instances: {total_duplicate_instances}")

if duplicates:
    print(f"\n   Top 5 most repeated table-query-label triplets:")
    for i, (triplet, count) in enumerate(sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5], 1):
        table_str, query, label = triplet
        label_str = str(list(label))
        query_preview = query[:60] + "..." if len(query) > 60 else query
        print(f"      {i}. ({count}x) Query: \"{query_preview}\" | Label: {label_str}")

PYTHON
    fi
done

echo ""
echo "================================================================"
echo "✓ Check complete"
