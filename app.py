import streamlit as st
import tempfile
import os
import whisper
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
from pyannote.audio import Pipeline
import time
import csv # Added for logging results

# --- Helper Functions ---
def format_audio(input_file_path, output_file_path):
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file_path, format="wav")

def denoise_audio(input_file_path, output_file_path):
    rate, data = wavfile.read(input_file_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
    wavfile.write(output_file_path, rate, reduced_noise)

# --- Streamlit UI ---
st.title("ClassifAI 🎓")

# --- NEW: Model Selection Dropdown ---
selected_model = st.selectbox(
    "Select Whisper Model for Benchmarking:",
    ("tiny.en", "base.en", "small.en", "medium.en"),
    index=2 # Defaults to small.en
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
            # --- NEW: Timing and Model Loading ---
            transcription_start = time.time()
            
            # Load the user-selected model
            model = whisper.load_model(selected_model)
            result = model.transcribe(standardized_path)
            
            transcription_end = time.time()
            runtime = transcription_end - transcription_start
            
            st.success(f"Transcription complete in {runtime:.2f} seconds!")
            
            # --- NEW: Save to Local CSV ---
            csv_filename = "benchmark_results.csv"
            file_exists = os.path.isfile(csv_filename)
            
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write the header if the file is brand new
                if not file_exists:
                    writer.writerow(["Model", "Audio Filename", "Runtime (Seconds)"])
                
                # Write the benchmark data
                writer.writerow([selected_model, uploaded_file.name, round(runtime, 2)])
            
            st.info(f"Result saved to `{csv_filename}` in your project folder.")

        with st.spinner("Diarizing with Pyannote..."):
            token = st.secrets["HF_TOKEN"]
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
            diarization = pipeline(denoised_path)

            '''
            Code for iterating over diarization results:

            annotation = diarization.speaker_diarization
            # Print out the timestamp and the assigned speaker
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

            '''
            st.success("Diarization complete!")

        # Cleanup
        os.remove(raw_path)
        os.remove(standardized_path)
        os.remove(denoised_path)
        
        # (Optional) Clear GPU memory after the run to be extra safe
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()