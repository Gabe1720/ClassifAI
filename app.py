import streamlit as st
import tempfile
import os
import time
import csv
import gc

# --- AI and Audio Libraries ---
import whisper
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
from pyannote.audio import Pipeline


# --- Helper Functions ---
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

# Function to merge whisper transcript to its estimated speaker:
def merge_transcript_and_diarization(whisper_segments, diarization):

    # Convert pyannote output into a list we can reuse
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker
        })

    # Fallback only if diarization completely failed
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

            # overlap length
            overlap = max(
                0.0,
                min(seg_end, turn_end) - max(seg_start, turn_start)
            )

            # midpoint distance for fallback / tie-breaking
            turn_mid = (turn_start + turn_end) / 2.0
            distance = abs(seg_mid - turn_mid)

            # choose this speaker if:
            # - larger overlap
            # - same overlap but closer in time
            # - same overlap and distance but longer turn
            if (
                overlap > best_overlap
                or (
                    overlap == best_overlap
                    and distance < best_distance
                )
                or (
                    overlap == best_overlap
                    and distance == best_distance
                    and best_turn is not None
                    and (turn_end - turn_start)
                    > (best_turn["end"] - best_turn["start"])
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

        st.write("Processing audio pipeline...")

        standardized_path = "temp_standardized.wav"
        denoised_path = "clean_ready_for_ai.wav"

        with st.spinner("Standardizing and Denoising Audio..."):
            format_audio(raw_path, standardized_path)
            denoise_audio(standardized_path, denoised_path)

        with st.spinner(f"Transcribing with {selected_model}..."):
            transcription_start = time.time()

            model = whisper.load_model(selected_model)
            result = model.transcribe(standardized_path)

            transcription_end = time.time()
            transcription_runtime = transcription_end - transcription_start

            csv_filename = "benchmark_results.csv"
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Model", "Audio Filename", "Runtime (Seconds)"])
                writer.writerow([
                    selected_model,
                    uploaded_file.name,
                    round(transcription_runtime, 2)
                ])

        del model
        gc.collect()

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with st.spinner("Diarizing with Pyannote..."):
            token = st.secrets["HF_TOKEN"]

            diarization_start = time.time()

            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            )

            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))

            diarization = pipeline(denoised_path)

            diarization_end = time.time()
            diarization_runtime = diarization_end - diarization_start

        merged_segments = merge_transcript_and_diarization(
            result["segments"],
            diarization
        )

        del result
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        merged_text = ""
        for item in merged_segments:
            merged_text += (
                f"[{item['start']:.2f}s - {item['end']:.2f}s] "
                f"{item['speaker']}: {item['text']}\n"
            )

        total_runtime = transcription_runtime + diarization_runtime

        st.success(
            f"Analysis Complete! Total GPU Runtime: {total_runtime:.2f} seconds"
        )

        st.header("Merged Transcript with Speaker Labels")

        st.text_area(
            "Speaker-Labeled Transcript",
            merged_text,
            height=500,
            disabled=True
        )

        st.download_button(
            label="Download Transcript",
            data=merged_text,
            file_name="speaker_labeled_transcript.txt",
            mime="text/plain"
        )

        try:
            os.remove(raw_path)
            os.remove(standardized_path)
            os.remove(denoised_path)
        except Exception:
            pass
