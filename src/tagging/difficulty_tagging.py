import os
import json
from typing import Dict, Optional

# Configuration
GPT5_RESULTS_DIR = "results/vlmpipeline/gpt-5.2/task"
GEMINI_RESULTS_DIR = "results/vlmpipeline/gemini-3-flash-preview/task"
SOURCE_DIR = "results/question_complexity/gemini-3-flash-preview"  # Source files (read-only)
OUTPUT_DIR = "results/question_difficulty_complexity"  # New output directory
DATASETS = ["feverous", "hybridqa", "sqa", "tabfact", "wikitq"]
FORMATS = ["html", "markdown", "latex"]


def normalize_text(text: str) -> str:
    """Normalize text by standardizing look-alike characters."""
    if not text:
        return ""
    text = str(text)
    
    # Replace semicolons with commas
    text = text.replace(";", ",")
    
    # Normalize dashes
    text = text.replace("–", "-")  # en-dash
    text = text.replace("—", "-")  # em-dash
    text = text.replace("−", "-")  # minus sign
    
    # Normalize quotes
    text = text.replace(""", '"')  # left double quote
    text = text.replace(""", '"')  # right double quote
    text = text.replace("'", "'")  # left single quote
    text = text.replace("'", "'")  # right single quote
    
    # Normalize apostrophes
    text = text.replace("'", "'")  # right single quote as apostrophe
    
    # Normalize spaces
    text = text.replace("\u00A0", " ")  # non-breaking space
    text = text.replace("\u2009", " ")  # thin space
    text = text.replace("\u202F", " ")  # narrow no-break space
    
    return text


def split_prediction(pred: str) -> set:
    """
    Split a multi-value prediction string into a set of individual items.
    Handles predictions separated by newlines and/or commas (semicolons are
    already normalised to commas by normalize_text before this is called).
    """
    import re
    parts = re.split(r'[\n,]+', pred)
    return {p.strip() for p in parts if p.strip()}


def calculate_difficulty(gpt5_sample: Dict, gemini_sample: Dict, label) -> tuple:
    """
    Calculate question difficulty based on model predictions.
    
    Compares task_prediction values with ground truth label (exact match after normalization).
    Counts exact matches across 6 predictions (3 from GPT-5, 3 from Gemini).
    Score 0-3: Hard
    Score 4-6: Easy
    
    Args:
        gpt5_sample: GPT-5 predictions for all formats
        gemini_sample: Gemini predictions for all formats
        label: Ground truth label (can be string or list)
    
    Returns:
        tuple: (score, difficulty) where score is 0-6 and difficulty is "Hard" or "Easy"
    """
    # Handle label as list or single value and normalize
    if isinstance(label, list):
        labels = [normalize_text(l) for l in label]
    else:
        labels = [normalize_text(label)]

    is_multi_label = len(labels) > 1
    labels_set = set(labels)

    def is_correct(pred: str) -> bool:
        pred = normalize_text(pred)
        if is_multi_label:
            # Split prediction into individual items and compare as sets
            return split_prediction(pred) == labels_set
        else:
            # Single label: keep original exact-match behaviour
            return pred in labels

    score = 0

    # Count GPT-5 predictions
    for fmt in FORMATS:
        if fmt in gpt5_sample:
            pred = gpt5_sample[fmt].get("task_prediction", "")
            if is_correct(pred):
                score += 1

    # Count Gemini predictions
    for fmt in FORMATS:
        if fmt in gemini_sample:
            pred = gemini_sample[fmt].get("task_prediction", "")
            if is_correct(pred):
                score += 1
    
    # Determine difficulty
    if score <= 3:
        difficulty = "Hard"
    else:
        difficulty = "Easy"
    
    return score, difficulty


def process_dataset(dataset_name: str, max_samples: Optional[int] = None):
    """
    Process a single dataset and append question_difficulty.
    """
    print(f"\n{'='*70}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Load GPT-5 results
    gpt5_path = os.path.join(GPT5_RESULTS_DIR, f"{dataset_name}.json")
    print(f"Loading GPT-5 results from: {gpt5_path}")
    with open(gpt5_path, "r", encoding="utf-8") as f:
        gpt5_data = json.load(f)
    gpt5_map = {sample["id"]: sample for sample in gpt5_data}
    
    # Load Gemini results
    gemini_path = os.path.join(GEMINI_RESULTS_DIR, f"{dataset_name}.json")
    print(f"Loading Gemini results from: {gemini_path}")
    with open(gemini_path, "r", encoding="utf-8") as f:
        gemini_data = json.load(f)
    gemini_map = {sample["id"]: sample for sample in gemini_data}
    
    # Load target file (output from complexity tagging) - READ ONLY
    source_path = os.path.join(SOURCE_DIR, f"{dataset_name}.json")
    print(f"Loading source file from: {source_path}")
    with open(source_path, "r", encoding="utf-8") as f:
        target_data = json.load(f)
    
    # Limit samples if specified (for testing)
    if max_samples and max_samples > 0:
        target_data = target_data[:max_samples]
        print(f"Processing first {max_samples} sample(s) only")
    
    # Process each sample
    easy_count = 0
    hard_count = 0
    missing_count = 0
    
    for sample in target_data:
        sample_id = sample["id"]
        
        # Get corresponding samples from GPT-5 and Gemini
        gpt5_sample = gpt5_map.get(sample_id)
        gemini_sample = gemini_map.get(sample_id)
        
        if gpt5_sample is None or gemini_sample is None:
            print(f"  ⚠ Warning: Sample {sample_id} missing in model results, skipping")
            missing_count += 1
            continue
        
        # Get ground truth label
        label = sample.get("label", [])
        
        # Calculate difficulty
        score, difficulty = calculate_difficulty(gpt5_sample, gemini_sample, label)
        sample["score"] = score
        sample["question_difficulty"] = difficulty
        
        if difficulty == "Easy":
            easy_count += 1
        else:
            hard_count += 1
    
    # Save to new output directory
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(target_data, f, ensure_ascii=False, indent=4)
    
    print(f"\n✓ Completed {dataset_name}")
    print(f"  Total samples: {len(target_data)}")
    print(f"  Easy: {easy_count} ({easy_count/len(target_data)*100:.1f}%)")
    print(f"  Hard: {hard_count} ({hard_count/len(target_data)*100:.1f}%)")
    if missing_count > 0:
        print(f"  Missing: {missing_count}")
    print(f"  Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Question Difficulty Tagging Pipeline"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples per dataset (for testing)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("Question Difficulty Tagging Pipeline")
    print("="*70)
    print(f"GPT-5 Results   : {GPT5_RESULTS_DIR}")
    print(f"Gemini Results  : {GEMINI_RESULTS_DIR}")
    print(f"Source Directory: {SOURCE_DIR} (read-only)")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Datasets        : {', '.join(DATASETS)}")
    print(f"Max Samples     : {args.max_samples or 'All'}")
    print(f"\nDifficulty Criteria:")
    print(f"  Easy: 4-6 correct predictions (out of 6)")
    print(f"  Hard: 0-3 correct predictions (out of 6)")
    print("="*70)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Overall statistics
    total_easy = 0
    total_hard = 0
    total_samples = 0
    
    for dataset_name in DATASETS:
        try:
            process_dataset(dataset_name, args.max_samples)
            
            # Count after processing
            output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}.json")
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            dataset_size = len(data)
            easy = sum(1 for s in data if s.get("question_difficulty") == "Easy")
            hard = sum(1 for s in data if s.get("question_difficulty") == "Hard")
            
            total_easy += easy
            total_hard += hard
            total_samples += dataset_size
            
        except FileNotFoundError as e:
            print(f"✗ Error: File not found - {e}")
        except Exception as e:
            print(f"✗ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    print(f"Total Samples: {total_samples}")
    print(f"Easy: {total_easy} ({total_easy/total_samples*100:.1f}%)")
    print(f"Hard: {total_hard} ({total_hard/total_samples*100:.1f}%)")
    print("="*70)
    print("\n✓ Question Difficulty Tagging Completed!")
    print(f"Results saved in: {OUTPUT_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
