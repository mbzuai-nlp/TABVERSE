"""
Filter 2-task_full to create 2-task with specific sampling criteria:
- feverous: keep all 794 samples
- tabfact: keep unique image_ids only (1695 unique tables)
- hybridqa: keep unique image_ids only (1608 unique tables)
- sqa: 1000 samples, ensuring all 185 unique image_ids
- wikitq: 1000 samples, ensuring all 421 unique image_ids
"""

import json
import random
from pathlib import Path
from collections import defaultdict

INPUT_DIR = Path("data_full/2-task_full")
OUTPUT_DIR = Path("data_full/2-task")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

random.seed(42)  # For reproducibility


def copy_dataset_fully(dataset_name):
    """Copy dataset completely without filtering."""
    input_path = INPUT_DIR / f"{dataset_name}.json"
    output_path = OUTPUT_DIR / f"{dataset_name}.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    unique_images = len(set(item['image_id'] for item in data))
    print(f"✓ {dataset_name}: copied all {len(data)} samples ({unique_images} unique tables)")


def deduplicate_to_unique_tables(dataset_name, expected_unique_images):
    """
    Keep only one sample per unique image_id (deduplicate to unique tables).
    Similar to 3-suc structure but in 2-task format.
    """
    input_path = INPUT_DIR / f"{dataset_name}.json"
    output_path = OUTPUT_DIR / f"{dataset_name}.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by image_id
    image_groups = defaultdict(list)
    for item in data:
        image_groups[item['image_id']].append(item)
    
    unique_images = len(image_groups)
    print(f"  {dataset_name}: {len(data)} total samples, {unique_images} unique tables")
    
    # Verify expected unique images
    if unique_images != expected_unique_images:
        print(f"  ⚠️  Warning: Expected {expected_unique_images} unique image_ids, found {unique_images}")
    
    # Take one sample per image_id (pick first occurrence)
    deduplicated_samples = []
    for image_id, samples in sorted(image_groups.items()):
        # Pick the first sample for consistency
        deduplicated_samples.append(samples[0])
    
    # Reassign sequential IDs
    for i, item in enumerate(deduplicated_samples):
        item['id'] = i
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(deduplicated_samples, f, ensure_ascii=False, indent=4)
    
    print(f"✓ {dataset_name}: deduplicated to {len(deduplicated_samples)} unique tables → saved to {output_path.name}")


def sample_dataset_with_all_images(dataset_name, target_samples, expected_unique_images):
    """
    Sample dataset to target_samples while ensuring ALL unique image_ids are included.
    
    Strategy:
    1. Group samples by image_id
    2. Take one sample from each unique image_id
    3. Fill remaining slots with random samples from entire dataset
    4. Reassign sequential IDs
    """
    input_path = INPUT_DIR / f"{dataset_name}.json"
    output_path = OUTPUT_DIR / f"{dataset_name}.json"
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group by image_id
    image_groups = defaultdict(list)
    for item in data:
        image_groups[item['image_id']].append(item)
    
    unique_images = len(image_groups)
    print(f"  {dataset_name}: {len(data)} total samples, {unique_images} unique tables")
    
    # Verify expected unique images
    if unique_images != expected_unique_images:
        print(f"  ⚠️  Warning: Expected {expected_unique_images} unique image_ids, found {unique_images}")
    
    # Check if target is achievable
    if target_samples < unique_images:
        print(f"  ⚠️  Error: Cannot sample {target_samples} while including all {unique_images} unique image_ids")
        print(f"  → Copying all samples instead")
        copy_dataset_fully(dataset_name)
        return
    
    # Step 1: Take one sample from each image_id
    selected_samples = []
    for image_id, samples in image_groups.items():
        selected_samples.append(random.choice(samples))
    
    print(f"  → Selected 1 sample per image_id: {len(selected_samples)} samples")
    
    # Step 2: Fill remaining slots with random samples
    remaining_needed = target_samples - len(selected_samples)
    
    if remaining_needed > 0:
        # Pool of all samples (excluding already selected ones to avoid exact duplicates)
        selected_ids = set(id(s) for s in selected_samples)
        remaining_pool = [s for s in data if id(s) not in selected_ids]
        
        # If pool is smaller than needed, we'll allow duplicates
        if remaining_needed <= len(remaining_pool):
            additional_samples = random.sample(remaining_pool, remaining_needed)
        else:
            # Sample with replacement
            additional_samples = random.choices(data, k=remaining_needed)
        
        selected_samples.extend(additional_samples)
        print(f"  → Added {len(additional_samples)} random samples: total {len(selected_samples)}")
    
    # Step 3: Shuffle to mix the guaranteed + random samples
    random.shuffle(selected_samples)
    
    # Step 4: Reassign sequential IDs
    for i, item in enumerate(selected_samples):
        item['id'] = i
    
    # Verify all image_ids present
    final_image_ids = set(item['image_id'] for item in selected_samples)
    all_present = len(final_image_ids) == unique_images
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=4)
    
    status = "✓" if all_present else "⚠️"
    print(f"{status} {dataset_name}: sampled {len(selected_samples)} samples "
          f"({len(final_image_ids)} unique tables) → saved to {output_path.name}")
    
    if not all_present:
        print(f"  ⚠️  Warning: Only {len(final_image_ids)}/{unique_images} unique image_ids in output!")


def main():
    print(f"Filtering datasets from {INPUT_DIR} → {OUTPUT_DIR}\n")
    
    # Check input directory exists
    if not INPUT_DIR.exists():
        print(f"❌ Error: {INPUT_DIR} does not exist!")
        print(f"   Please rename data_full/2-task to data_full/2-task_full first")
        return
    
    # 1. Feverous: Keep all samples
    print("1. Feverous - keeping all samples:")
    feverous_file = INPUT_DIR / "feverous.json"
    if feverous_file.exists():
        copy_dataset_fully('feverous')
    else:
        print(f"⚠️  feverous.json not found in {INPUT_DIR}")
    
    # 2. TabFact: Deduplicate to unique tables
    print("\n2. TabFact - deduplicating to unique tables:")
    tabfact_file = INPUT_DIR / "tabfact.json"
    if tabfact_file.exists():
        deduplicate_to_unique_tables('tabfact', expected_unique_images=1695)
    else:
        print(f"⚠️  tabfact.json not found in {INPUT_DIR}")
    
    # 3. HybridQA: Deduplicate to unique tables
    print("\n3. HybridQA - deduplicating to unique tables:")
    hybridqa_file = INPUT_DIR / "hybridqa.json"
    if hybridqa_file.exists():
        deduplicate_to_unique_tables('hybridqa', expected_unique_images=1608)
    else:
        print(f"⚠️  hybridqa.json not found in {INPUT_DIR}")
    
    # 4. SQA: 1000 samples, all 185 unique image_ids
    print("\n4. SQA - sampling 1000 with all unique tables:")
    sqa_file = INPUT_DIR / "sqa.json"
    if sqa_file.exists():
        sample_dataset_with_all_images('sqa', target_samples=1000, expected_unique_images=185)
    else:
        print(f"⚠️  sqa.json not found in {INPUT_DIR}")
    
    # 5. WikiTQ: 1000 samples, all 421 unique image_ids
    print("\n5. WikiTQ - sampling 1000 with all unique tables:")
    wikitq_file = INPUT_DIR / "wikitq.json"
    if wikitq_file.exists():
        sample_dataset_with_all_images('wikitq', target_samples=1000, expected_unique_images=421)
    else:
        print(f"⚠️  wikitq.json not found in {INPUT_DIR}")
    
    print(f"\n{'='*70}")
    print(f"✓ Filtering complete! Output saved to {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
