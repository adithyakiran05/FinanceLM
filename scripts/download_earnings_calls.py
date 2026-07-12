import os
from pathlib import Path
from tqdm import tqdm

# We will use the HuggingFace datasets library which has excellent open-source conversational finance datasets
try:
    from datasets import load_dataset
except ImportError:
    print("Please install the 'datasets' library: pip install datasets")
    exit(1)

OUTPUT_DIR = Path("data/raw/earnings")

def download_earnings_calls():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Downloading earnings call transcripts from HuggingFace (financial_phrasebank / earnings datasets)...")
    try:
        # We MUST use 'FinGPT/fingpt-sentiment-train' which is a standard pure Parquet dataset 
        # because the 'datasets' library completely removed support for python loading scripts like 'jlh-ibm'.
        dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        
        print(f"Found {len(dataset)} items.")
        
        for i, item in enumerate(tqdm(dataset, desc="Saving data")):
            # FinGPT schema contains instructions, inputs, and outputs of conversational finance
            text = item.get("instruction", "") + "\n" + item.get("input", "") + "\n" + item.get("output", "")
            
            if len(text) > 10:
                filename = f"conversational_{i}.txt"
                out_path = OUTPUT_DIR / filename
                out_path.write_text(text, encoding="utf8")
        
        print(f"Successfully downloaded data to {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error downloading earnings calls: {e}")
        print("Note: You may need to authenticate with HuggingFace, or try an alternative open dataset like 'financial_phrasebank'.")

if __name__ == "__main__":
    download_earnings_calls()
