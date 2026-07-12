from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def count_file(f):
    try:
        text = f.read_text(encoding="utf8", errors="ignore")
        return len(text), len(text.split())
    except:
        return 0, 0

if __name__ == "__main__":
    clean_dir = Path("data/cleaned")
    if not clean_dir.exists():
        print(f"Directory {clean_dir} does not exist. Run the pipeline first.")
    else:
        files = list(clean_dir.rglob("*.txt"))
        
        if not files:
            print("No files found.")
            exit(0)

        chars = 0
        words = 0

        print(f"Counting words across {len(files)} files on all CPU cores...")
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(count_file, files, chunksize=100), total=len(files), desc="Calculating Stats"))

        for c, w in results:
            chars += c
            words += w

        print("Documents:", len(files))
        print("Words:", words)
        print("Characters:", chars)