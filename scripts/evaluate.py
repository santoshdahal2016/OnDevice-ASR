import nemo.collections.asr as nemo_asr
import json
import os

MODEL_PATH = "/diyoData/experiments/knowledgedistill/experiments/full_train/2025-08-08_15-33-14/checkpoints/full_train.nemo"
TEST_MANIFEST = "/diyoData/experiments/knowledgedistill/manifest/test-clean_manifest.json"
TRANSCRIPTIONS_FILE = "transcriptions.jsonl"

# Load manifest and collect audio file paths and ground truth
audio_files = []
ground_truth = []
with open(TEST_MANIFEST, "r") as f:
    for line in f:
        item = json.loads(line)
        audio_files.append(item["audio_filepath"])
        ground_truth.append(item["text"])

# Check if transcriptions already exist
if os.path.exists(TRANSCRIPTIONS_FILE):
    print(f"Loading existing transcriptions from {TRANSCRIPTIONS_FILE}")
    pred_texts = []
    with open(TRANSCRIPTIONS_FILE, "r") as f:
        for line in f:
            item = json.loads(line)
            pred_texts.append(item["pred_text"])
else:
    print("Transcribing audio files...")
    # Load model            self.teacher = EncDecRNNTBPEModel.restore_from(teacher_path, map_location='cpu')

    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=MODEL_PATH)
    # Transcribe
    pred_hyps = asr_model.transcribe(audio_files, batch_size=16)
    pred_texts = [h.text for h in pred_hyps]
    # Save transcriptions
    with open(TRANSCRIPTIONS_FILE, "w") as f:
        for audio_fp, gt, pred in zip(audio_files, ground_truth, pred_texts):
            f.write(json.dumps({
                "audio_filepath": audio_fp,
                "text": gt,
                "pred_text": pred
            }) + "\n")
    print(f"Transcriptions saved to {TRANSCRIPTIONS_FILE}")

# Lowercase for case-insensitive evaluation
ground_truth = [t.lower() for t in ground_truth]
pred_texts = [t.lower() for t in pred_texts]

# Compute WER and CER
from nemo.collections.asr.metrics.wer import word_error_rate

wer = word_error_rate(hypotheses=pred_texts, references=ground_truth, use_cer=False)
cer = word_error_rate(hypotheses=pred_texts, references=ground_truth, use_cer=True)

print(f"WER: {wer:.4f}")
print(f"CER: {cer:.4f}")
