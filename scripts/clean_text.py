import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

DIRS_TO_CLEAN = [
    (Path("data/processed/sec"), Path("data/cleaned/sec")),
    (Path("data/raw/news"), Path("data/cleaned/news")),
    (Path("data/raw/definitions"), Path("data/cleaned/definitions")),
    (Path("data/raw/fed"), Path("data/cleaned/fed")),
    (Path("data/raw/global_finance"), Path("data/cleaned/global_finance")),
    (Path("data/raw/earnings"), Path("data/cleaned/earnings")),
    (Path("data/raw/arxiv"), Path("data/cleaned/arxiv")),
    (Path("data/raw/gutenberg"), Path("data/cleaned/gutenberg")),
    (Path("data/raw/textbooks"), Path("data/cleaned/textbooks")),
    (Path("data/raw/english_grammar"), Path("data/cleaned/english_grammar")),
    (Path("data/raw/movie_scripts_repo"), Path("data/cleaned/movie_scripts")),
]

def clean_file(args):
    file, out_dir = args
    try:
        text = file.read_text(encoding="utf8", errors="ignore")

        # Remove excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove repeated spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Remove page numbers
        text = re.sub(r"Page \d+", "", text)

        out = out_dir / file.name
        out.write_text(text, encoding="utf8")
    except Exception as e:
        print(f"Failed to clean {file.name}: {e}")

if __name__ == "__main__":
    tasks = []
    for in_dir, out_dir in DIRS_TO_CLEAN:
        out_dir.mkdir(parents=True, exist_ok=True)
        if in_dir.exists():
            for f in in_dir.rglob("*.txt"):
                tasks.append((f, out_dir))
                
    print(f"Starting multi-processing for {len(tasks)} files across all data sources...")
    
    if tasks:
        with ProcessPoolExecutor() as executor:
            list(tqdm(executor.map(clean_file, tasks, chunksize=100), total=len(tasks), desc="Cleaning"))
            
    print("Cleaning complete!")