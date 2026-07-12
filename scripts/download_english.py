import os
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library: pip install datasets")
    exit(1)

OUTPUT_DIR = Path("data/raw/english_grammar")

def download_english_grammar():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading English grammar dataset from HuggingFace (Wikitext)...")
    try:
        # Wikitext-103 is the gold standard for teaching AI models basic English grammar
        # We will download a large chunk of the train split.
        dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
        
        # We don't need the entire 100M+ word dataset (that would drown out our finance data)
        # We just need enough to teach it grammar, so we'll grab the first 50,000 paragraphs.
        subset = dataset.select(range(50000))
        print(f"Found {len(subset)} grammar items.")
        
        for i, item in enumerate(tqdm(subset, desc="Saving data")):
            text = item.get("text", "").strip()
            
            # Skip empty lines
            if len(text) > 50:
                filename = f"english_text_{i}.txt"
                out_path = OUTPUT_DIR / filename
                out_path.write_text(text, encoding="utf8")
        
        print(f"Successfully downloaded English grammar data to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error downloading English data: {e}")

if __name__ == "__main__":
    download_english_grammar()
