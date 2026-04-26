import os
import csv
import torch
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

# --- CONFIGURATION (Edit these before running) ---
AUDIO_NAME = "bdopb"

# 1. Pyannote reads the file and returns a dictionary 
# (Key = filename, Value = Pyannote Annotation Object)
rttm_dict = load_rttm(f"E:/Downloads/classifai_datasets/archive/labels/dev/{AUDIO_NAME}.rttm")

# Grab the annotation timeline for this specific audio file
ground_truth = rttm_dict[AUDIO_NAME]

# 2. Run your app.py prediction on "cmfyw.wav" here
audio_path = f"E:/Downloads/classifai_datasets/archive/voxconverse_dev_wav/audio/{AUDIO_NAME}.wav"

print("Loading Pyannote pipeline...")
token = os.environ.get("HF_TOKEN")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token
)
if torch.cuda.is_available():
    pipeline.to(torch.device("cuda"))

print("Running diarization... (this may take a moment)")
prediction = pipeline(audio_path)

# 3. Grade it instantly
der_metric = DiarizationErrorRate()
score = der_metric(ground_truth, prediction)

print(f"Diarization Error Rate (DER): {score * 100:.2f}%")

# 4. Log results to CSV
log_dir = "dev_debug_logs/diarization_debug_logs"
os.makedirs(log_dir, exist_ok=True)
csv_path = os.path.join(log_dir, "diarization_results.csv")
file_exists = os.path.isfile(csv_path)
wav_filename = os.path.basename(audio_path)

with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(["Audio Filename", "DER (%)"])
    writer.writerow([wav_filename, round(score * 100, 2)])