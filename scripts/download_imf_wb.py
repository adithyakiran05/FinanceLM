import requests
from pathlib import Path
from tqdm import tqdm
import json

OUTPUT_DIR = Path("data/raw/global_finance")

def download_world_bank_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # World Bank Documents API
    # We query for English documents, e.g., Policy Research Working Papers
    url = "https://search.worldbank.org/api/v2/wds?format=json&strtitle=finance&lang=English&rows=50"
    print(f"Querying World Bank API for financial reports...")
    
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()
        
        documents = data.get("documents", {})
        
        for doc_id, doc_info in tqdm(documents.items(), desc="Saving World Bank Reports"):
            if isinstance(doc_info, dict):
                title = doc_info.get("display_title", "Untitled")
                abstract = doc_info.get("abstracts", {}).get("cdata", "")
                author = doc_info.get("author", "Unknown")
                
                if abstract:
                    content = f"Title: {title}\nAuthor: {author}\n\nAbstract/Summary:\n{abstract}"
                    
                    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                    filename = f"WB_{doc_id}_{safe_title[:30]}.txt"
                    
                    out_path = OUTPUT_DIR / filename
                    out_path.write_text(content, encoding="utf8", errors="ignore")
                    
        print("Successfully collected World Bank abstracts/reports.")
                
    except Exception as e:
        print(f"Error querying World Bank API: {e}")

if __name__ == "__main__":
    download_world_bank_data()
