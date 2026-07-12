import os
import numpy as np
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

def build_corpus():
    data_dir = Path("data/cleaned")
    tokenizer_dir = Path("models/finance-tokenizer")
    output_file = Path("data/processed/corpus.bin")
    
    if not tokenizer_dir.exists():
        print("Error: Run train_tokenizer.py first!")
        return

    print("Loading custom tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))

    files = list(data_dir.rglob("*.txt"))
    if not files:
        print("No text files found in data/cleaned/")
        return
        
    print(f"Found {len(files)} files. Building massive binary corpus...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # We write raw 16-bit integer tokens to a binary file to avoid RAM limits
    with open(output_file, 'wb') as f:
        for filepath in tqdm(files, desc="Tokenizing and Writing"):
            try:
                text = filepath.read_text(encoding="utf8", errors="ignore")
                if not text.strip():
                    continue
                # Encode text into token IDs
                tokens = tokenizer.encode(text)
                
                # Convert to 16-bit integers (vocab size is 32000, so it fits easily in uint16)
                token_array = np.array(tokens, dtype=np.uint16)
                
                # Stream the binary data directly to the SSD
                f.write(token_array.tobytes())
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")

    # Verify the size
    file_size_bytes = os.path.getsize(output_file)
    total_tokens = file_size_bytes // 2  # 2 bytes per uint16 token
    print(f"Corpus built successfully! Total Tokens: {total_tokens:,}")
    print(f"File saved to {output_file} ({file_size_bytes / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    build_corpus()
