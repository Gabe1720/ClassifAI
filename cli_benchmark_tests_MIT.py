import os
import time
import csv
import gc
import torch
import whisper
from pydub import AudioSegment
from pydub.effects import normalize as pydub_normalize
from scipy.io import wavfile
import noisereduce as nr
from bert_score import score
from jiwer import wer
import string
import pandas as pd
import re
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# --- CONFIGURATION (Edit these before running) ---
AUDIO_FILE_PATH = "E:/Downloads/classifai_datasets/9.14-spring-2014/mit9_14s14_lec01_360p_16_9.wav"
GROUND_TRUTH_PATH = "E:/Downloads/classifai_datasets/9.14-spring-2014/dev_utterance_metadata.csv"
GT_TRANSCRIPT_FILE_PATH = "E:/Downloads/classifai_datasets/9.14-spring-2014/dev/lecture_01/transcript.txt"
GROUND_TRUTH_ROWS = 526 
RESULTS_CSV = "dev_debug_logs/MIT_debug_logs/benchmark_MIT.csv"
EVAL_METRIC = "BOTH" # Options: "WER", "BERTSCORE", "BOTH"

# --- TEST SUITE CONFIGURATIONS ---
# We define a list of configurations to test. Each dictionary represents one full pipeline run.
# This grid tests the most impactful parameters identified: audio normalization, noise reduction strength, 
# n-gram keyword extraction, Whisper beam size, decoding temperature, and alignment tolerance.
model = "small.en"  # Default model for tests, can be overridden in individual configs
TEST_CONFIGS = [
    {
        "name": f"Baseline ({model}, No NR, No Prompt)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Add Audio Normalization",
        "model": model,
        "normalize_audio": True,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Heavy Noise Reduction (0.8)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": True,
        "nr_prop_decrease": 0.8,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Light Noise Reduction (0.3)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": True,
        "nr_prop_decrease": 0.3,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "With Prompt (1-gram, top 20)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": True,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "With Prompt (2-gram, top 30) & CondPrevText=False",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": True,
        "keyword_ngram": (1, 2),
        "keyword_top_n": 30,
        "condition_on_previous_text": False,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Increased Beam Size (10)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 10,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Alignment Tolerance (0.2s padding)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.2
    },
    {
        "name": "Combined Best-Guess 1 (Norm + Light NR + 2-gram + Tol 0.2)",
        "model": model,
        "normalize_audio": True,
        "use_noise_reduction": True,
        "nr_prop_decrease": 0.3,
        "use_prompt": True,
        "keyword_ngram": (1, 2),
        "keyword_top_n": 30,
        "condition_on_previous_text": False,
        "beam_size": 5,
        "temperature": 0.0,
        "alignment_tolerance": 0.2
    },
    {
        "name": "Combined Best-Guess 2 (Norm + Light NR + Beam Size 10)",
        "model": model,
        "normalize_audio": True,
        "use_noise_reduction": True,
        "nr_prop_decrease": 0.3,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 10,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    },
    {
        "name": "Combined Best-Guess 3 (Norm + Beam Size 10)",
        "model": model,
        "normalize_audio": True,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 10,
        "temperature": 0.0,
        "alignment_tolerance": 0.0
    }
]

# --- Helper Functions ---
def format_audio(input_file_path, output_file_path, normalize=False):
    print(f"Standardizing audio format (Normalize={normalize})...")
    audio = AudioSegment.from_file(input_file_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    if normalize:
        audio = pydub_normalize(audio)
    audio.export(output_file_path, format="wav")

def denoise_audio(input_file_path: str, output_file_path: str, prop_decrease: float) -> None:
    try:
        print(f"Denoising audio (prop_decrease={prop_decrease})...")
        rate, data = wavfile.read(input_file_path)
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=prop_decrease)
        wavfile.write(output_file_path, rate, reduced_noise)
    except Exception as e:
        print(f"Error denoising audio: {e}")
        raise

def extract_dense_jargon(file_path, ngram_range=(1, 1), top_n=20):
    print(f"Using KeyBERT (ngram={ngram_range}, top_n={top_n})...")
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)

    academic_fluff = [
        'course', 'courses', 'ebook', 'ebooks', 'supplementary', 'ocwmitedu',
        'table', 'tables', 'slide', 'slides', 'bring', 'just', 'redistribute',
        'integrate', 'mit', 'lecture', 'reading', 'okay', 'right', 'know', 'like',
        'going', 'want', 'think', 'people', 'time', 'thing', 'things', 'terms', 'look',
        'resources', 'opencourseware', 'quizzes', 'concept', 'talk', 'sorrythe',
        'educational', 'help', 'creative', 'commons', 'content', 'book', 'purposes', 
        'class', 'make', 'change', 'future', 'ask', 'sounds', 'term', 'bother', 
        'means', 'following', 'interesting', 'special', 'talking', 'differences', 'don'
    ]

    custom_stopwords = list(ENGLISH_STOP_WORDS) + academic_fluff
    
    # We pass the ngram_range to the vectorizer to catch multi-word jargon
    vectorizer = CountVectorizer(stop_words=custom_stopwords, ngram_range=ngram_range)
    kw_model = KeyBERT()
    
    keywords = kw_model.extract_keywords(
        clean_text, 
        vectorizer=vectorizer,
        keyphrase_ngram_range=ngram_range, 
        use_mmr=True, 
        diversity=0.6, 
        top_n=top_n
    )
    
    clean_words = [kw[0] for kw in keywords]
    print(f"Extracted Keywords: {clean_words}")
    return ", ".join(clean_words)

def align_by_word(whisper_result, csv_path, max_rows=None, tolerance=0.0):
    print(f"Aligning timelines (Tolerance={tolerance}s)...")
    if max_rows is not None:
        df_gt = pd.read_csv(csv_path, nrows=max_rows)
    else:
        df_gt = pd.read_csv(csv_path)

    all_whisper_words = []
    for segment in whisper_result.get('segments', []):
        for word_info in segment.get('words', []):
            all_whisper_words.append(word_info)

    aligned_candidates = []
    aligned_references = []

    for index, row in df_gt.iterrows():
        # Apply alignment tolerance padding to the ground truth window
        gt_start = float(row['audio_start_sec']) - tolerance
        gt_end = float(row['audio_start_sec']) + float(row['duration']) + tolerance
        gt_text = str(row['text'])

        matched_words = []
        for w in all_whisper_words:
            w_mid = (w['start'] + w['end']) / 2.0
            if gt_start <= w_mid <= gt_end:
                matched_words.append(w['word'].strip())

        matched_whisper_text = " ".join(matched_words)

        candidate = matched_whisper_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        reference = gt_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        if len(candidate.split()) > 1 and len(reference.split()) > 1:
            aligned_candidates.append(candidate)
            aligned_references.append(reference)

    return aligned_candidates, aligned_references

def main():
    print("========== ClassifAI Comprehensive Benchmarking Tool ==========")
    print(f"Target Audio:  {AUDIO_FILE_PATH}")
    print(f"Ground Truth:  {GROUND_TRUTH_PATH}")
    print("=================================================================\n")

    os.makedirs("dev_debug_logs/MIT_debug_logs", exist_ok=True)
    
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Test Name", "Model", "Audio Normalization", "Noise Reduction", "NR Prop Decrease", 
                "Use Prompt", "Keyword N-Gram", "Keyword Top N", "Condition on Prev", 
                "Beam Size", "Temperature", "Align Tolerance", 
                "Runtime (s)", "WER", "Precision", "Recall", "F1"
            ])
            
    for i, config in enumerate(TEST_CONFIGS):
        # if i != 0:
        #     continue  # TEMP: Run only the a particular test(s) for quick debugging. Remove this line to run all tests.
        
        print(f"\n\n{'='*60}")
        print(f"RUNNING TEST: {config['name']}")
        print(f"{'='*60}")

        standardized_path = "temp_standardized.wav"
        denoised_path = "clean_ready_for_ai.wav"

        # 1. Pre-process Audio
        format_audio(AUDIO_FILE_PATH, standardized_path, normalize=config['normalize_audio'])
        
        target_audio_path = standardized_path
        if config['use_noise_reduction']:
            denoise_audio(standardized_path, denoised_path, prop_decrease=config['nr_prop_decrease'])
            target_audio_path = denoised_path

        # 2. Prompt Generation
        custom_prompt = None
        if config['use_prompt']:
            dense_keywords = extract_dense_jargon(
                GT_TRANSCRIPT_FILE_PATH, 
                ngram_range=config['keyword_ngram'], 
                top_n=config['keyword_top_n']
            )
            custom_prompt = f"The following lecture covers specialized topics including: {dense_keywords}. Let us begin."
            print(f"Generated Prompt: {custom_prompt[:100]}...")

        # 3. Transcription
        print(f"\nLoading {config['model']} model...")
        model = whisper.load_model(config['model'])
        
        print("Transcribing... (This may take a while)")
        transcription_start = time.time()
        
        transcribe_kwargs = {
            "audio": target_audio_path,
            "condition_on_previous_text": config['condition_on_previous_text'],
            "word_timestamps": True,
            "beam_size": config['beam_size'],
            "temperature": config['temperature'],
            "language": "en" # Enforce English to prevent translation mode bugs
        }
        if custom_prompt:
            transcribe_kwargs["initial_prompt"] = custom_prompt

        result = model.transcribe(**transcribe_kwargs)
        transcription_end = time.time()
        
        transcription_runtime = transcription_end - transcription_start
        print(f"Transcription complete in {transcription_runtime:.2f} seconds.")

        # 3.5 Save Predicted Transcript for Sanity Check
        import textwrap
        safe_config_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', config['name'])
        transcript_filename = f"dev_debug_logs/MIT_debug_logs/transcript_prediction_{safe_config_name}.txt"
        
        transcript_text = "".join([segment.get('text', '') + " " for segment in result.get('segments', [])])
        
        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"--- PREDICTED TRANSCRIPT: {config['name']} ---\n\n")
            f.write(textwrap.fill(transcript_text.strip(), width=100) + "\n")
        print(f"Saved predicted transcript to {transcript_filename}")

        # 4. Cleanup Memory
        del model 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 5. BERTScore Evaluation
        candidates, references = align_by_word(
            result, GROUND_TRUTH_PATH, 
            max_rows=GROUND_TRUTH_ROWS, 
            tolerance=config['alignment_tolerance']
        )
        del result
        gc.collect()

        print(f"\n--- Evaluation Results ---")

        wer_score = None
        p_mean, r_mean, f1_mean = None, None, None

        if EVAL_METRIC in ["WER", "BOTH"]:
            print("Calculating Word Error Rate (WER)...")
            wer_score = wer(references, candidates)
            print(f"Word Error Rate (WER): {wer_score:.4f}")

        if EVAL_METRIC in ["BERTSCORE", "BOTH"]:
            print("Calculating BERTScore...")
            start_time = time.time()
            # Set verbose=False to keep terminal clean during loop
            P, R, F1 = score(candidates, references, lang="en", verbose=False, rescale_with_baseline=True)
            end_time = time.time()

            p_mean = P.mean().item()
            r_mean = R.mean().item()
            f1_mean = F1.mean().item()

            print("\n========== LOWEST 10 F1 SCORES ==========")
            f1_list = F1.tolist()
            indexed_scores = [(score_val, idx) for idx, score_val in enumerate(f1_list)]
            indexed_scores.sort(key=lambda x: x[0])
            
            for rank, (score_val, idx) in enumerate(indexed_scores[:10], start=1):
                print(f"\n--- Rank {rank} Worst Match (F1: {score_val:.4f}) ---")
                print(f"WHISPER: {candidates[idx]}")
                print(f"CSV:     {references[idx]}")
            print("=========================================\n")
            
            print(f"BERTScore Precision: {p_mean:.4f}")
            print(f"BERTScore Recall:    {r_mean:.4f}")
            print(f"BERTScore F1 Score:  {f1_mean:.4f}")

        # 6. Log to CSV
        with open(RESULTS_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                config['name'], config['model'], config['normalize_audio'], config['use_noise_reduction'],
                config['nr_prop_decrease'], config['use_prompt'], str(config['keyword_ngram']), 
                config['keyword_top_n'], config['condition_on_previous_text'], config['beam_size'], 
                config['temperature'], config['alignment_tolerance'],
                round(transcription_runtime, 2), 
                round(wer_score, 4) if wer_score is not None else "N/A",
                round(p_mean, 4) if p_mean is not None else "N/A", 
                round(r_mean, 4) if r_mean is not None else "N/A", 
                round(f1_mean, 4) if f1_mean is not None else "N/A"
            ])
        
        print(f"Results logged to {RESULTS_CSV}")

        # 7. File Cleanup
        if os.path.exists(standardized_path):
            os.remove(standardized_path)
        if os.path.exists(denoised_path):
            os.remove(denoised_path)

if __name__ == "__main__":
    main()
