
import os
import json
import requests
from tqdm import tqdm

HF_TOKEN = os.getenv("HF_TOKEN")  # Make sure to export HF_TOKEN in your shell
DATASET = "ai4bharat/IndicVoices"
CONFIG = "nepali"
SPLIT = "valid"
PAGE_LIMIT = 100
SAVE_DIR = os.path.join("downloads", CONFIG, SPLIT)
METADATA_PATH = os.path.join(SAVE_DIR, "metadata.json")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

API_BASE = "https://datasets-server.huggingface.co/rows"

def fetch_page(offset: int, limit: int = PAGE_LIMIT):
    params = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "offset": offset,
        "length": limit
    }
    response = requests.get(API_BASE, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def download_file(url, save_path):
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_rows = []
    offset = 0
    total_rows = None

    print("📥 Downloading all rows from Hugging Face...")

    # Step 1: Paginated fetch
    while True:
        print(f"🔄 Fetching rows {offset} to {offset + PAGE_LIMIT - 1}")
        data = fetch_page(offset, PAGE_LIMIT)

        if total_rows is None:
            total_rows = data.get("num_rows_total", 0)
            print(f"🔢 Total rows: {total_rows}")

        rows = data.get("rows", [])
        if not rows:
            break

        all_rows.extend(rows)
        offset += PAGE_LIMIT

        if offset >= total_rows:
            break

    # Step 2: Save metadata
    print(f"📝 Saving metadata for {len(all_rows)} rows...")
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # Step 3: Download audio files
    print("⬇️ Downloading audio files...")
    for row in tqdm(all_rows):
        row_idx = row.get("row_idx")
        row_data = row.get("row", {})
        audio_files = row_data.get("audio_filepath", [])

        for i, audio in enumerate(audio_files):
            url = audio.get("src")
            ext = audio.get("type", "audio/wav").split("/")[-1]
            filename = f"{row_idx}_{i}.{ext}"
            filepath = os.path.join(SAVE_DIR, filename)
            download_file(url, filepath)

    print("✅ All data downloaded successfully.")

if __name__ == "__main__":
    main()
