from pathlib import Path
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

RAW_DIR = Path("data/raw/sec")
OUTPUT_DIR = Path("data/processed/sec")

def process_file(file):
    try:
        html = file.read_text(errors="ignore")
        # SEC filings have no ads or sidebars, so trafilatura is overkill.
        # BeautifulSoup's lxml parser is 10x to 100x faster for simple text extraction.
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")

        if text:
            # Create a unique filename: TICKER_ACCESSION_FILENAME.txt
            ticker = file.parent.parent.parent.name
            accession = file.parent.name
            unique_name = f"{ticker}_{accession}_{file.stem}.txt"
            
            output_file = OUTPUT_DIR / unique_name
            output_file.write_text(text, encoding="utf8")
    except Exception as e:
        print(f"Failed to parse {file.name}: {e}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = list(RAW_DIR.rglob("*.html"))
    print(f"Starting multi-processing for {len(files)} files...")
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_file, files, chunksize=100), total=len(files), desc="Parsing"))
    
    print("Parsing complete!")