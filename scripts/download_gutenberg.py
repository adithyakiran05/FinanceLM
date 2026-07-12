import requests
from pathlib import Path
from tqdm import tqdm
import time

OUTPUT_DIR = Path("data/raw/gutenberg")

# Map of classic English literature and economic texts on Project Gutenberg
BOOKS = {
    "Adam_Smith_Wealth_of_Nations": "3300",
    "Jonathan_Swift_Modest_Proposal": "1080",
    "Jane_Austen_Pride_and_Prejudice": "1342",
    "Charles_Dickens_Tale_of_Two_Cities": "98",
    "Mary_Shelley_Frankenstein": "84",
    "Arthur_Conan_Doyle_Sherlock_Holmes": "1661",
    "Herman_Melville_Moby_Dick": "2701",
    "Lewis_Carroll_Alice_in_Wonderland": "11"
}

def download_gutenberg_books():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    print("Downloading foundational economic textbooks from Project Gutenberg...")
    
    for book_name, book_id in tqdm(BOOKS.items(), desc="Downloading texts"):
        # The standard format for plain text files on the Gutenberg cache
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        
        try:
            res = requests.get(url, headers=headers)
            if res.status_code == 200:
                # Gutenberg plain text files are typically utf-8
                text = res.text
                
                out_path = OUTPUT_DIR / f"{book_name}.txt"
                out_path.write_text(text, encoding="utf8", errors="ignore")
            else:
                print(f"Failed to fetch {book_name} - Status code: {res.status_code}")
                
            time.sleep(1) # Be respectful to Gutenberg servers
            
        except Exception as e:
            print(f"Error downloading {book_name}: {e}")
            
    print(f"Successfully saved texts to {OUTPUT_DIR}")

if __name__ == "__main__":
    download_gutenberg_books()
