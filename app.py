import streamlit as st
import tempfile
import os
import time
import csv # Added for logging results
import gc # Free up GPU VRAM before Diarization

# --- The Torchaudio Cloud Patch ---
import torchaudio
if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None
if not hasattr(torchaudio, 'get_audio_backend'):
    torchaudio.get_audio_backend = lambda: "soundfile"
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox_io"]

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
            transcription_start = time.time()
            
            # Load model and transcribe
            model = whisper.load_model(selected_model)
            result = model.transcribe(standardized_path)
            
            transcription_end = time.time()
            transcription_runtime = transcription_end - transcription_start
            
            # --- NEW: Format Whisper Output ---
            whisper_text_output = ""
            for segment in result["segments"]:
                whisper_text_output += f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}\n"
            
            # Log Benchmark Data
            csv_filename = "benchmark_results.csv"
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Model", "Audio Filename", "Runtime (Seconds)"])
                writer.writerow([selected_model, uploaded_file.name, round(transcription_runtime, 2)])

            # Delete the Whisper model and result from Python's memory
            del model 
            del result
            
            # Force Python to run garbage collection
            gc.collect()

            # Empty the PyTorch GPU cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        with st.spinner("Diarizing with Pyannote..."):
            token = st.secrets["HF_TOKEN"]
            diarization_start = time.time()
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
            # --- NEW: Force Pyannote onto the GPU ---
            import torch
            if torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
                
            diarization = pipeline(denoised_path)
            diarization_end = time.time()
            diarization_runtime = diarization_end - diarization_start

            # Log Benchmark Data
            csv_filename = "benchmark_results_diarization.csv"
            file_exists = os.path.isfile(csv_filename)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["Audio Filename", "Runtime (Seconds)"])
                writer.writerow([uploaded_file.name, round(diarization_runtime, 2)])
            
            # --- NEW: Format Pyannote Output ---
            pyannote_text_output = ""
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                pyannote_text_output += f"[{turn.start:.2f}s - {turn.end:.2f}s]: {speaker}\n"
                
        st.success(f"Analysis Complete! Total GPU Runtime: {transcription_runtime + diarization_runtime:.2f} seconds")
        st.divider()
        
        # --- NEW: Side-by-Side Display Dashboard ---
        st.header("Raw Model Outputs")
        st.write("These raw outputs will be merged in the next phase of development.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Whisper Transcription")
            # Using st.text_area to create a nice scrollable box for the text
            st.text_area("Timestamps & Text", whisper_text_output, height=400, disabled=True)
            
        with col2:
            st.subheader("🗣️ Pyannote Diarization")
            st.text_area("Timestamps & Speakers", pyannote_text_output, height=400, disabled=True)

        # Cleanup
        os.remove(raw_path)
        os.remove(standardized_path)
        os.remove(denoised_path)
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()