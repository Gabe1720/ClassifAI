from pydub import AudioSegment
from pyannote.audio import Pipeline
import time

# --- 1. PREPROCESSING: Format Standardization ---
def format_audio(input_file_path, output_file_path):
    print("Standardizing audio format...")
    # Load whatever file the user uploaded
    audio = AudioSegment.from_file(input_file_path)
    
    # Force it into the exact format Pyannote and Whisper love:
    # 1 Channel (Mono) and 16000Hz Frame Rate
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    
    # Export it as a clean .wav
    audio.export(output_file_path, format="wav")
    print("Audio standardized!")

# --- 2. DIARIZATION ---
def run_diarization(processed_audio_path, hf_token):
    print("Loading Pyannote model... (This downloads the model the first time)")
    # Instantiate the pipeline using your token
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token
    )
    
    print("Running diarization... this may take a few minutes depending on audio length.")
    start_time = time.time()
    
    # Run the model on the audio file
    diarization = pipeline(processed_audio_path)
    
    print(f"Finished in {time.time() - start_time:.2f} seconds.")
    print("\n--- Diarization Results ---")
    
    annotation = diarization.speaker_diarization
    # Print out the timestamp and the assigned speaker
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")


# --- RUN THE CODE ---
if __name__ == "__main__":
    # Replace these with  actual files and token
    YOUR_TEST_FILE = "Lecture_snippet.m4a"
    STANDARDIZED_FILE = "clean_ready_for_ai.wav"
    YOUR_HUGGINGFACE_TOKEN = "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" 

    # 1. Format the audio
    format_audio(YOUR_TEST_FILE, STANDARDIZED_FILE)
    
    # 2. Run the diarization
    run_diarization(STANDARDIZED_FILE, YOUR_HUGGINGFACE_TOKEN)