from pathlib import Path
from render_html import render_html_to_image
from render_markdown import render_markdown_to_image
from render_latex import render_latex_to_image
from utils import get_all_unrendered_files

def run_renderer(name, folder, ext, func, needs_output_path=False, limit=None):
    print(f"--- Processing {name} ---")
    files = get_all_unrendered_files(folder, ext)
    if limit:
        files = files[:limit]
    for f in files:
        try:
            if needs_output_path:
                output = f.with_suffix(".png")
                func(f, output)
            else:
                func(f)
        except Exception as e:
            print(f"Error rendering {f.name}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render structured representations to images")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of files to process")
    args = parser.parse_args()
    
    # List of datasets to process sequentially
    datasets = ["sqa", "tabfact", "wikitq"]
    base_path = Path("/Users/mominaahsan/Desktop/VisualTableBench/structured_representations")
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset.upper()}")
        print(f"{'='*60}\n")
        
        base = base_path / dataset
        run_renderer("HTML", base / "html", ".html", render_html_to_image, needs_output_path=True, limit=args.limit)
        run_renderer("Markdown", base / "markdown", ".md", render_markdown_to_image, limit=args.limit)
        run_renderer("LaTeX", base / "latex", ".tex", render_latex_to_image, limit=args.limit)
        
        print(f"\n✓ Completed dataset: {dataset.upper()}\n")
