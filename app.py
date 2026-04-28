import streamlit as st
import tempfile
import os
import time
import csv
import gc

import whisper
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
from pyannote.audio import Pipeline
import torch


# --- Audio Processing ---

def format_audio(input_file_path, output_file_path):
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file_path, format="wav")

def denoise_audio(input_file_path: str, output_file_path: str) -> None:
    try:
        rate, data = wavfile.read(input_file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        wavfile.write(output_file_path, rate, reduced_noise)
    except FileNotFoundError:
        st.error(f"Audio file not found at {input_file_path}")
    except Exception as e:
        st.error(f"An error occurred during denoising: {e}")


# --- Transcription ---

def transcribe_audio(path, model_name):
    model = whisper.load_model(model_name)
    start = time.time()
    result = model.transcribe(path)
    runtime = time.time() - start
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, runtime

def save_benchmark(model_name, audio_filename, runtime):
    csv_filename = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Audio Filename", "Runtime (Seconds)"])
        writer.writerow([model_name, audio_filename, round(runtime, 2)])


# --- Diarization ---

def run_diarization(path, token):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    start = time.time()
    diarization = pipeline(path)
    runtime = time.time() - start
    return diarization, runtime


# --- Analysis ---

def is_question(text):
    text_clean = text.strip().lower()
    score = 0
    QUESTION_WORDS = (
        "who", "what", "when", "where", "why", "how",
        "is", "are", "do", "does", "did",
        "can", "could", "would", "should",
        "will", "have", "has"
    )
    if text_clean.endswith("?") or ("?" in text_clean):
        score += 1
    if any(text_clean.startswith(q) for q in QUESTION_WORDS):
        score += 1
    return score == 2

def merge_transcript_and_diarization(whisper_segments, diarization):
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })

    if not speaker_turns:
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": "SPEAKER_00",
                "text": seg["text"].strip()
            }
            for seg in whisper_segments
        ]

    merged_segments = []
    for segment in whisper_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_mid = (seg_start + seg_end) / 2.0
        seg_text = segment["text"].strip()

        best_turn = None
        best_overlap = -1
        best_distance = float("inf")

        for turn in speaker_turns:
            turn_start = turn["start"]
            turn_end = turn["end"]
            overlap = max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))
            turn_mid = (turn_start + turn_end) / 2.0
            distance = abs(seg_mid - turn_mid)

            if (
                overlap > best_overlap
                or (overlap == best_overlap and distance < best_distance)
                or (
                    overlap == best_overlap
                    and distance == best_distance
                    and best_turn is not None
                    and (turn_end - turn_start) > (best_turn["end"] - best_turn["start"])
                )
            ):
                best_turn = turn
                best_overlap = overlap
                best_distance = distance

        merged_segments.append({
            "start": seg_start,
            "end": seg_end,
            "speaker": best_turn["speaker"],
            "text": seg_text
        })

    return merged_segments

def classify_segments(merged_segments):
    for item in merged_segments:
        item["question"] = is_question(item["text"])
    return merged_segments

def format_transcript(merged_segments):
    merged_text = ""
    for item in merged_segments:
        label = "❓" if item["question"] else ""
        merged_text += (
            f"[{item['start']:.2f}s - {item['end']:.2f}s] "
            f"{item['speaker']}: {item['text']} {label}\n"
        )
    return merged_text

def cleanup_temp_files(*paths):
    for path in paths:
        try:
            os.remove(path)
        except Exception:
            pass


# --- Streamlit UI ---

st.title("ClassifAI 🎓")

selected_model = st.selectbox(
    "Select Whisper Model for Benchmarking:",
    ("tiny.en", "base.en", "small.en", "medium.en"),
    index=2
)

uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if st.button("Analyze Audio", type="primary"):
    if uploaded_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".m4a") as raw_file:
            raw_file.write(uploaded_file.getvalue())
            raw_path = raw_file.name

        standardized_path = "temp_standardized.wav"
        denoised_path = "clean_ready_for_ai.wav"

        st.write("Processing audio pipeline...")

        with st.spinner("Standardizing and Denoising Audio..."):
            format_audio(raw_path, standardized_path)
            denoise_audio(standardized_path, denoised_path)

        with st.spinner(f"Transcribing with {selected_model}..."):
            result, transcription_runtime = transcribe_audio(standardized_path, selected_model)
            save_benchmark(selected_model, uploaded_file.name, transcription_runtime)

        with st.spinner("Diarizing with Pyannote..."):
            token = st.secrets["HF_TOKEN"]
            diarization, diarization_runtime = run_diarization(denoised_path, token)

        merged_segments = merge_transcript_and_diarization(result["segments"], diarization)

        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_segments = classify_segments(merged_segments)
        merged_text = format_transcript(merged_segments)

        total_runtime = transcription_runtime + diarization_runtime
        st.success(f"Analysis Complete! Total GPU Runtime: {total_runtime:.2f} seconds")

        st.header("Merged Transcript with Speaker Labels")
        st.text_area("Speaker-Labeled Transcript", merged_text, height=500, disabled=True)

        st.download_button(
            label="Download Transcript",
            data=merged_text,
            file_name="speaker_labeled_transcript.txt",
            mime="text/plain"
        )

        cleanup_temp_files(raw_path, standardized_path, denoised_path)