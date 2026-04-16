import os
import time
import csv
import gc
import torch
import whisper
from pydub import AudioSegment
from scipy.io import wavfile
import noisereduce as nr
from bert_score import score
import string
import pandas as pd
import re
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# --- CONFIGURATION (Edit these before running) ---
AUDIO_FILE_PATH = "E:/Downloads/9.14-spring-2014/mit9_14s14_lec01_360p_16_9.wav" # Path to your test audio
GROUND_TRUTH_PATH = "E:/Downloads/9.14-spring-2014/dev_utterance_metadata.csv" # Path to your ground truth CSV
WHISPER_MODEL = "medium.en"
GROUND_TRUTH_ROWS = 526 # Only evaluate the first 526 rows (Lecture 1)
# -----------------------------------------------

# --- Helper Functions ---
def format_audio(input_file_path, output_file_path):
    print("Standardizing audio format...")
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file_path, format="wav")

def denoise_audio(input_file_path: str, output_file_path: str) -> None:
    """
    Reads a standardized WAV file, applies noise reduction, and writes the output.
    """
    try:
        print("Denoising audio...")
        rate, data = wavfile.read(input_file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        wavfile.write(output_file_path, rate, reduced_noise)
    except Exception as e:
        print(f"Error denoising audio: {e}")
        raise

def extract_dense_jargon(file_path):
    print("Using KeyBERT to extract semantic keywords from transcript...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # 1. Scrub the weird numbers/punctuation
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)

    # 2. Define the "Academic Fluff" we want to ignore
    academic_fluff = [
        'course', 'courses', 'ebook', 'ebooks', 'supplementary', 'ocwmitedu',
        'table', 'tables', 'slide', 'slides', 'bring', 'just', 'redistribute',
        'integrate', 'mit', 'lecture', 'reading', 'okay', 'right', 'know', 'like',
        'going', 'want', 'think', 'people', 'time', 'thing', 'things', 'terms', 'look'
    ]

    # Combine standard English stopwords with our custom fluff
    custom_stopwords = list(ENGLISH_STOP_WORDS) + academic_fluff

    # 3. Create a custom Vectorizer using our new massive stopword list
    vectorizer = CountVectorizer(stop_words=custom_stopwords)

    kw_model = KeyBERT()
    
    # 4. Pass the vectorizer into KeyBERT
    # Note: I slightly lowered diversity to 0.6 so it allows more related biological terms
    keywords = kw_model.extract_keywords(
        clean_text, 
        vectorizer=vectorizer,      # <-- Our new filter
        keyphrase_ngram_range=(1, 1), 
        use_mmr=True, 
        diversity=0.6, 
        top_n=20
    )
    
    clean_words = [kw[0] for kw in keywords]
    return ", ".join(clean_words)

def align_by_word(whisper_result, csv_path, max_rows=None):
    print("\nAligning timelines using Word-Level Precision...")
    if max_rows is not None:
        df_gt = pd.read_csv(csv_path, nrows=max_rows)
    else:
        df_gt = pd.read_csv(csv_path)

    # 1. Extract every single word and its timestamp from Whisper into a flat list
    all_whisper_words = []
    for segment in whisper_result.get('segments', []):
        for word_info in segment.get('words', []):
            all_whisper_words.append(word_info)

    aligned_candidates = []
    aligned_references = []

    # 2. Iterate through your exact CSV rows
    for index, row in df_gt.iterrows():
        gt_start = float(row['audio_start_sec'])
        gt_end = gt_start + float(row['duration'])
        gt_text = str(row['text'])

        matched_words = []

        # 3. Drop Whisper words directly into the CSV row if they happened in that timeframe
        for w in all_whisper_words:
            w_mid = (w['start'] + w['end']) / 2.0
            
            if gt_start <= w_mid <= gt_end:
                matched_words.append(w['word'].strip())

        # Combine the matched words into a single sentence
        matched_whisper_text = " ".join(matched_words)

        # 4. Text Normalization (Critical for BERTScore with Baseline Rescaling)
        # Convert both to lowercase and strip all punctuation
        candidate = matched_whisper_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        reference = gt_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        # We only evaluate rows that contain at least 2 words.
        # This prevents the "Empty Candidate" warning from stray commas or silence gaps.
        if len(candidate.split()) > 1 and len(reference.split()) > 1:
            aligned_candidates.append(candidate)
            aligned_references.append(reference)

    print(f"Successfully aligned {len(aligned_references)} exact CSV rows!")
    return aligned_candidates, aligned_references

def main():
    print("========== ClassifAI Benchmarking Tool ==========")
    print(f"Target Audio:  {AUDIO_FILE_PATH}")
    print(f"Ground Truth:  {GROUND_TRUTH_PATH}")
    print(f"Whisper Model: {WHISPER_MODEL}")
    print("=================================================\n")

    # Define temporary processing paths
    standardized_path = "temp_standardized.wav"
    denoised_path = "clean_ready_for_ai.wav"

    # 1. Pre-process Audio
    format_audio(AUDIO_FILE_PATH, standardized_path)
    denoise_audio(standardized_path, denoised_path)

    # --- Read and Format Lecture Notes ---
    NOTES_FILE_PATH = "E:/Downloads/9.14-spring-2014/dev/lecture_01/transcript.txt"
    dense_keywords = extract_dense_jargon(NOTES_FILE_PATH)
    
    print("Extracting context from lecture notes...")
        
    # Wrap the notes in a conversational prompt to set the grammatical style
    custom_prompt = f"The following lecture covers specialized topics including: {dense_keywords}. Let us begin."
    print(f"Custom Prompt:\n{custom_prompt}")

    # 2. Transcription
    print(f"\nLoading {WHISPER_MODEL} model...")
    model = whisper.load_model(WHISPER_MODEL)
    
    print("Transcribing... (This may take a while for large files)")
    transcription_start = time.time()
    result = model.transcribe(denoised_path,
                              condition_on_previous_text=False,
                              initial_prompt=custom_prompt,
                              word_timestamps=True)
    transcription_end = time.time()
    
    transcription_runtime = transcription_end - transcription_start
    print(f"Transcription complete in {transcription_runtime:.2f} seconds.")

    # 3. Format Output
    transcript_text = ""
    for segment in result["segments"]:
        transcript_text += segment['text'] + " "

    # 4. Cleanup Memory before BERTScore
    del model 
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. BERTScore Evaluation
    print("\nLoading ground truth and calculating BERTScore...")

    # Run the new Word-Level Alignment Algorithm
    candidates, references = align_by_word(result, GROUND_TRUTH_PATH, max_rows=GROUND_TRUTH_ROWS)
    del result # Finish memory cleanup from step 4

    start_time = time.time()
    P, R, F1 = score(candidates, references, lang="en", verbose=True, rescale_with_baseline=True)
    end_time = time.time()

    print("\n========== LOWEST 10 F1 SCORES ==========")
    f1_list = F1.tolist()
    indexed_scores = [(score_val, idx) for idx, score_val in enumerate(f1_list)]
    indexed_scores.sort(key=lambda x: x[0])
    
    for rank, (score_val, idx) in enumerate(indexed_scores[:10], start=1):
        print(f"\n--- Rank {rank} Worst Match (F1: {score_val:.4f}) ---")
        print(f"WHISPER: {candidates[idx]}")
        print(f"CSV:     {references[idx]}")
    print("=========================================\n")

    # 6. Display Results
    print(f"\n--- Evaluation Results (Calculated in {end_time - start_time:.2f}s) ---")
    print(f"Precision: {P.mean().item():.4f}")
    print(f"Recall:    {R.mean().item():.4f}")
    print(f"F1 Score:  {F1.mean().item():.4f}")

    # 7. Log to CSV
    csv_filename = "dev_debug_logs/benchmark_results.csv"
    file_exists = os.path.isfile(csv_filename)
    audio_filename = os.path.basename(AUDIO_FILE_PATH)
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model", "Audio Filename", "Runtime (s)", "Precision", "Recall", "F1"])
        writer.writerow([WHISPER_MODEL, audio_filename, round(transcription_runtime, 2), 
                         round(P.mean().item(), 4), round(R.mean().item(), 4), round(F1.mean().item(), 4)])
    
    print(f"\nResults appended to {csv_filename}")

    # --- NEW: Generate Detailed Text Report ---
    report_filename = f"./dev_debug_logs/transcript_comparison_{WHISPER_MODEL}.txt"
    
    with open(report_filename, "w", encoding="utf-8") as report_file:
        report_file.write(f"========== BENCHMARK REPORT: {WHISPER_MODEL} ==========\n")
        report_file.write(f"Audio File: {AUDIO_FILE_PATH}\n")
        report_file.write(f"F1 Score:   {F1.mean().item():.4f}\n")
        report_file.write("========================================================\n\n")
        
        report_file.write("--- PREDICTED TRANSCRIPT (Whisper) ---\n")
        # Format it nicely with line breaks every ~100 characters for readability
        import textwrap
        report_file.write(textwrap.fill(transcript_text, width=100) + "\n\n")
        
        report_file.write("--- GROUND TRUTH TRANSCRIPT ---\n")
        full_reference_text = " ".join(references)
        report_file.write(textwrap.fill(full_reference_text, width=100) + "\n")

    print(f"Detailed transcript comparison saved to {report_filename}")

    # 8. File Cleanup
    if os.path.exists(standardized_path):
        os.remove(standardized_path)
    if os.path.exists(denoised_path):
        os.remove(denoised_path)

if __name__ == "__main__":
    main()