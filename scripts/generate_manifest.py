import os
import json
import soundfile as sf

LIBRI_DIRS = [
    "/diyoData/experiments/knowledgedistill/data/train-clean-100",
    "/diyoData/experiments/knowledgedistill/data/LibriSpeech/train-clean-360",
    "/diyoData/experiments/knowledgedistill/data/LibriSpeech/train-other-500"

]
MANIFEST_FILE = "manifest/train-clean-100-360-500_manifest.json"

os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)

manifest = []
n_audio          = 0
n_missing_trans  = 0
n_missing_utt    = 0

for libri_dir in LIBRI_DIRS:
    print(f"\nProcessing directory: {libri_dir}")
    
    if not os.path.exists(libri_dir):
        print(f"[WARNING] Directory does not exist: {libri_dir}")
        continue
        
    for root, _, files in os.walk(libri_dir):
        for file in files:
            if not (file.endswith(".flac") or file.endswith(".wav")):
                continue

            n_audio += 1
            audio_path = os.path.join(root, file)

            chapter_id = os.path.basename(root)                  # e.g. "496"
            speaker_id = os.path.basename(os.path.dirname(root)) # e.g. "26"
            transcript_file = os.path.join(root, f"{speaker_id}-{chapter_id}.trans.txt")

            if not os.path.exists(transcript_file):
                n_missing_trans += 1
                print(f"[SKIP] transcript file not found → {transcript_file}")
                continue

            # Build dict utt‑id → text
            with open(transcript_file, "r") as tf:
                trans_dict = {
                    line.split(" ")[0]: " ".join(line.strip().split(" ")[1:])
                    for line in tf
                }

            utt_id = os.path.splitext(file)[0]
            text   = trans_dict.get(utt_id)

            if text is None:
                n_missing_utt += 1
                print(f"[SKIP] utterance id {utt_id} not in {os.path.basename(transcript_file)}")
                continue

            duration = sf.info(audio_path).duration

            manifest.append(
                {
                    "audio_filepath": os.path.abspath(audio_path),
                    "duration": duration,
                    "text": text,
                }
            )

print("\n=== SUMMARY ===")
print(f"Audio files inspected : {n_audio}")
print(f"Missing transcripts   : {n_missing_trans}")
print(f"Missing utt in trans  : {n_missing_utt}")
print(f"Added to manifest     : {len(manifest)}")
# 
with open(MANIFEST_FILE, "w") as fout:
    for entry in manifest:
        fout.write(json.dumps(entry) + "\n")

print(f"\nManifest saved to: {MANIFEST_FILE}")
