import os
import json
import soundfile as sf
from tqdm import tqdm

def process_directories(directories, dataset_name):
    """
    Process a list of directories and generate manifest entries.
    
    Args:
        directories: List of directory paths to process
        dataset_name: Name of the dataset (for logging purposes)
    
    Returns:
        tuple: (manifest_entries, statistics_dict)
    """
    manifest = []
    n_audio = 0
    n_missing_trans = 0
    n_missing_utt = 0
    
    # First pass: collect all audio files to get total count for progress bar
    print(f"\nScanning {dataset_name} directories for audio files...")
    all_audio_files = []
    
    for libri_dir in directories:
        if not os.path.exists(libri_dir):
            print(f"[WARNING] Directory does not exist: {libri_dir}")
            continue
            
        for root, _, files in os.walk(libri_dir):
            for file in files:
                if file.endswith(".flac") or file.endswith(".wav"):
                    audio_path = os.path.join(root, file)
                    all_audio_files.append(audio_path)
    
    print(f"Found {len(all_audio_files)} audio files in {dataset_name} dataset")
    
    # Second pass: process each audio file with progress bar
    print(f"\nProcessing {dataset_name} audio files...")
    
    with tqdm(total=len(all_audio_files), desc=f"Processing {dataset_name}", unit="files") as pbar:
        for audio_path in all_audio_files:
            n_audio += 1
            
            root = os.path.dirname(audio_path)
            file = os.path.basename(audio_path)
            
            chapter_id = os.path.basename(root)                  # e.g. "496"
            speaker_id = os.path.basename(os.path.dirname(root)) # e.g. "26"
            transcript_file = os.path.join(root, f"{speaker_id}-{chapter_id}.trans.txt")

            if not os.path.exists(transcript_file):
                n_missing_trans += 1
                pbar.set_postfix({"Skipped": "No transcript", "File": os.path.basename(audio_path)})
                pbar.update(1)
                continue

            # Build dict utt‑id → text
            with open(transcript_file, "r") as tf:
                trans_dict = {
                    line.split(" ")[0]: " ".join(line.strip().split(" ")[1:])
                    for line in tf
                }

            utt_id = os.path.splitext(file)[0]
            text = trans_dict.get(utt_id)

            if text is None:
                n_missing_utt += 1
                pbar.set_postfix({"Skipped": "No utterance", "File": os.path.basename(audio_path)})
                pbar.update(1)
                continue

            duration = sf.info(audio_path).duration

            manifest.append(
                {
                    "audio_filepath": os.path.abspath(audio_path),
                    "duration": duration,
                    "text": text,
                }
            )
            
            # Update progress bar with current file info
            pbar.set_postfix({"Processed": len(manifest), "File": os.path.basename(audio_path)})
            pbar.update(1)
    
    stats = {
        "n_audio": n_audio,
        "n_missing_trans": n_missing_trans,
        "n_missing_utt": n_missing_utt,
        "n_manifest_entries": len(manifest)
    }
    
    return manifest, stats

def save_manifest(manifest, manifest_file):
    """Save manifest entries to a JSON file."""
    os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
    with open(manifest_file, "w") as fout:
        for entry in manifest:
            fout.write(json.dumps(entry) + "\n")
    print(f"Manifest saved to: {manifest_file}")

# Define directories for each dataset
train_dirs = [
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/train-other-500",
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/train-clean-100",
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/train-clean-360"
]

test_clean_dirs = [
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/test-clean"
]

test_other_dirs = [
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/test-other"
]

val_dirs = [
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/dev-clean",
    "/mnt/diyo_data/datasets/librispeechData/LibriSpeech/dev-other"
]

# Define manifest file paths
train_manifest_file = "manifest/train_manifest.json"
test_clean_manifest_file = "manifest/test_clean_manifest.json"
test_other_manifest_file = "manifest/test_other_manifest.json"
val_manifest_file = "manifest/val_manifest.json"

print("=" * 60)
print("LIBRISPEECH MANIFEST GENERATION")
print("=" * 60)

# Process train dataset 
print("\n" + "=" * 20 + " TRAIN DATASET " + "=" * 20)
train_manifest, train_stats = process_directories(train_dirs, "train")
save_manifest(train_manifest, train_manifest_file)

print("\n=== TRAIN SUMMARY ===")
print(f"Audio files inspected : {train_stats['n_audio']}")
print(f"Missing transcripts   : {train_stats['n_missing_trans']}")
print(f"Missing utt in trans  : {train_stats['n_missing_utt']}")
print(f"Added to manifest     : {train_stats['n_manifest_entries']}")

# Process test-clean dataset
print("\n" + "=" * 18 + " TEST-CLEAN DATASET " + "=" * 18)
test_clean_manifest, test_clean_stats = process_directories(test_clean_dirs, "test-clean")
save_manifest(test_clean_manifest, test_clean_manifest_file)

print("\n=== TEST-CLEAN SUMMARY ===")
print(f"Audio files inspected : {test_clean_stats['n_audio']}")
print(f"Missing transcripts   : {test_clean_stats['n_missing_trans']}")
print(f"Missing utt in trans  : {test_clean_stats['n_missing_utt']}")
print(f"Added to manifest     : {test_clean_stats['n_manifest_entries']}")

# Process test-other dataset
print("\n" + "=" * 18 + " TEST-OTHER DATASET " + "=" * 18)
test_other_manifest, test_other_stats = process_directories(test_other_dirs, "test-other")
save_manifest(test_other_manifest, test_other_manifest_file)

print("\n=== TEST-OTHER SUMMARY ===")
print(f"Audio files inspected : {test_other_stats['n_audio']}")
print(f"Missing transcripts   : {test_other_stats['n_missing_trans']}")
print(f"Missing utt in trans  : {test_other_stats['n_missing_utt']}")
print(f"Added to manifest     : {test_other_stats['n_manifest_entries']}")

# Process validation dataset
print("\n" + "=" * 20 + " VALIDATION DATASET " + "=" * 20)
val_manifest, val_stats = process_directories(val_dirs, "validation")
save_manifest(val_manifest, val_manifest_file)

print("\n=== VALIDATION SUMMARY ===")
print(f"Audio files inspected : {val_stats['n_audio']}")
print(f"Missing transcripts   : {val_stats['n_missing_trans']}")
print(f"Missing utt in trans  : {val_stats['n_missing_utt']}")
print(f"Added to manifest     : {val_stats['n_manifest_entries']}")

