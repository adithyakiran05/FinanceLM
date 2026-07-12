import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return filepath, hasher.hexdigest()

def deduplicate():
    clean_dir = Path("data/cleaned")
    if not clean_dir.exists():
        print(f"Directory {clean_dir} does not exist.")
        return

    files = list(clean_dir.rglob("*.txt"))
    total_files = len(files)
    if total_files == 0:
        return
        
    seen_hashes = set()
    files_removed = 0
    
    print(f"Hashing {total_files} files across all cores...")
    
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(get_file_hash, files, chunksize=100), total=total_files, desc="Hashing"))
        
    print("Deleting duplicates...")
    for filepath, file_hash in tqdm(results, desc="Deduplicating"):
        if file_hash in seen_hashes:
            filepath.unlink()
            files_removed += 1
        else:
            seen_hashes.add(file_hash)
            
    print(f"Deduplication complete.")
    print(f"Total files checked: {total_files}")
    print(f"Total duplicates removed: {files_removed}")
    print(f"Remaining files: {total_files - files_removed}")

if __name__ == "__main__":
    deduplicate()
