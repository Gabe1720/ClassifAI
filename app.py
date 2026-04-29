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


# Function to format the audio into a .wav file.
# Allow whisper to properly read the audio by configuring
# the file to these settings:
# -> Convert the audio to a single channel.
# -> Set the framerate to 16,000Hz
# -> Output a .wav of the reconfigured file.
def format_audio(input_file_path, output_file_path):
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file_path, format="wav")


# Function to reduce any background noise in the audio file before
# transcription. This will provide a clean method to improve the quality of
# both transcription and diarization. 
def denoise_audio(input_file_path: str, output_file_path: str) -> None:
    try:
        rate, data = wavfile.read(input_file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        wavfile.write(output_file_path, rate, reduced_noise)
    except FileNotFoundError:
        st.error(f"Audio file not found at {input_file_path}")
    except Exception as e:
        st.error(f"An error occurred during denoising: {e}")


# Function to handle the transcription pipeline using
# the "whisper" library from OpenAI. The model's timestamps
# are recorded along with the transcribed text.
def transcribe_audio(path, model_name):
    model = whisper.load_model(model_name)
    start = time.time()
    result = model.transcribe(path)
    runtime = time.time() - start
    del model
    gc.collect()

    # Allow the transcription to run without a GPU 
    # (Not recommended for the large models):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, runtime


# Function for benchmarking model results into a csv file.
# -> Records the model name, audio file, and system runtime.
def save_benchmark(model_name, audio_filename, runtime):
    csv_filename = "benchmark_results.csv"
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Audio Filename", "Runtime (Seconds)"])
        writer.writerow([model_name, audio_filename, round(runtime, 2)])


# Function for handling the speaker diarization 
# pipeline using the Pyannote library. The 
# goal of this pipeline is to provide the most accurate 
# possible speaker labels. Speaker timestamps are
# also recorded.
def run_diarization(path, token):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )

    # Allow the diarization to run without a GPU 
    # (Not recommended for the large models):
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    # Record speaker timestamps:
    start = time.time()
    diarization = pipeline(path)
    runtime = time.time() - start
    return diarization, runtime


# Function to handle question classification.
# A score function will determine the identification
# of a meaningful question within the transcribed
# text. The current algorithm handles the following:
# -> Identify starting question words.
# -> Identify transcribed question marks.
# A text segment is classified as a question if it either 
# contains a question mark or starts with a known question word.
def is_question(text):
    text_clean = text.strip().lower()

    # 1. Direct punctuation check is highly reliable from Whisper
    if "?" in text_clean:
        return True

    # 2. Fallback: Check if the segment starts with a question word
    # Split text into words to prevent false substring matches (e.g., "issue" starting with "is")
    words = text_clean.split()
    if not words:
        return False
        
    first_word = words[0].strip(".,!:'\"-")

    QUESTION_WORDS = {
        "who", "what", "when", "where", "why", "how",
        "is", "are", "do", "does", "did",
        "can", "could", "would", "should",
        "will", "have", "has", "which", "whom", "whose", "may", "might"
    }

    # If punctuation is missing, starting with an interrogative word is a strong indicator
    return first_word in QUESTION_WORDS


# Function to handle the merging of transcipt and diarization outputs.
# By using an algorithm to estimate relevant timestamps, transcripts
# are aligned with the closest possible speaker.
# The implemented algorithm performs the following:
# -> Convert the diarization output to a list of timestamps and labeled speakers.
# -> If pyannote diarization failed at any timeframe, label the timeframe to "SPEAKER_00" 
#    (The first person to be labeled in the file).
# -> Each transcribed segment is matched with a speaker by calculating overlap and
#    midpoint distance. This checks if both timestamp segments share certain time frames.
# -> Output a unified list that matches transcript text with the best possible
#    speaker, while keeping the transcript timestamps.
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


# Helper function to perform question classification on each
# line within the merged results. If the text is identified
# as a question, it will be labeled for formatting.
def classify_segments(merged_segments):
    # for item in merged_segments:
    #     item["question"] = is_question(item["text"])
    current_sentence_segments = []
    current_sentence_text = ""

    for i, item in enumerate(merged_segments):
        text = item["text"].strip()
        current_sentence_segments.append(item)
        current_sentence_text += " " + text

        # If segment ends with punctuation or is the last segment, evaluate the combined sentence
        if text.endswith(('.', '?', '!')) or i == len(merged_segments) - 1:
            is_q = is_question(current_sentence_text)
            for seg in current_sentence_segments:
                seg["question"] = is_q

            # Reset the buffer for the next sentence
            current_sentence_segments = []
            current_sentence_text = ""

    return merged_segments


# Helper function to format the results. Each line including the speaker,
# text line, and timestamp data is appended to an output string. If the line
# is identified as a question, it will be labeled with a question mark to
# allow the user to categorize meaningful questions.
def format_transcript(merged_segments):
    merged_text = ""
    for item in merged_segments:
        label = "❓" if item["question"] else ""
        merged_text += (
            f"[{item['start']:.2f}s - {item['end']:.2f}s] "
            f"{item['speaker']}: {item['text']} {label}\n"
        )
    return merged_text


# Function to clean up any temporary files created by the entire process.
# Main use is to clean the system data after analysis is completed.
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