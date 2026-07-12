import fitz  # PyMuPDF
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = Path("data/raw/textbooks_repo")
OUTPUT_DIR = Path("data/raw/textbooks")

def parse_pdfs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Recursively find all PDFs
    pdf_files = list(INPUT_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF textbooks to parse.")
    
    for pdf_path in tqdm(pdf_files, desc="Parsing Textbooks"):
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for page in doc:
                full_text.append(page.get_text())
                
            text = "\n".join(full_text)
            
            # Save extracted text
            safe_name = pdf_path.stem.replace(" ", "_")
            out_path = OUTPUT_DIR / f"{safe_name}.txt"
            out_path.write_text(text, encoding="utf8", errors="ignore")
            
            doc.close()
            
        except Exception as e:
            print(f"Failed to parse {pdf_path.name}: {e}")
            
if __name__ == "__main__":
    parse_pdfs()
