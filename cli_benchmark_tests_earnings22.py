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
import re
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from datasets import load_dataset
import soundfile as sf
import textwrap

# --- CONFIGURATION (Edit these before running) ---
RESULTS_CSV = "dev_debug_logs/earnings22_debug_logs/earnings22_benchmark_results.csv"

# --- TEST SUITE CONFIGURATIONS ---
# We define a list of configurations to test. Each dictionary represents one full pipeline run.
model = "medium.en"  # Default model for tests, can be overridden in individual configs
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
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
    },
    {
        "name": "Combined Best-Guess 1 (Norm + Light NR + 2-gram)",
        "model": model,
        "normalize_audio": True,
        "use_noise_reduction": True,
        "nr_prop_decrease": 0.3,
        "use_prompt": True,
        "keyword_ngram": (1, 2),
        "keyword_top_n": 30,
        "condition_on_previous_text": False,
        "beam_size": 5,
        "temperature": 0.0
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
        "temperature": 0.0
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
        "temperature": 0.0
    },
    {
        "name": f"Increased Temperature (0.5)",
        "model": model,
        "normalize_audio": False,
        "use_noise_reduction": False,
        "nr_prop_decrease": 0.0,
        "use_prompt": False,
        "keyword_ngram": (1, 1),
        "keyword_top_n": 20,
        "condition_on_previous_text": True,
        "beam_size": 5,
        "temperature": 0.5
    }
]

def load_earnings_dataset():
    print("Fetching Earnings22 Dataset from Hugging Face...")
    dataset = load_dataset("distil-whisper/earnings22", name="full", split="test")
    print(f"Successfully loaded {len(dataset)} full-length corporate meetings!")
    
    meeting = dataset[0]
    
    audio_array = meeting["audio"]["array"]
    sample_rate = meeting["audio"]["sampling_rate"]
    
    temp_audio_path = "current_benchmark_audio.wav"
    sf.write(temp_audio_path, audio_array, sample_rate)
    print(f"Saved audio to {temp_audio_path} for Whisper to process.")
    
    ground_truth_text = meeting["transcription"]
    return temp_audio_path, ground_truth_text

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

def extract_dense_jargon(raw_text, ngram_range=(1, 1), top_n=20):
    print(f"Using KeyBERT (ngram={ngram_range}, top_n={top_n})...")
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', raw_text)

    corporate_fluff = [
        'company', 'quarter', 'earnings', 'call', 'welcome', 'everyone',
        'forward', 'looking', 'statements', 'results', 'financial', 'year',
        'business', 'growth', 'thank', 'questions', 'operator', 'revenue',
        'operations', 'management', 'investors', 'performance', 'slide',
        'slides', 'presentation', 'turn', 'call', 'today', 'first', 'second',
        'third', 'fourth', 'time', 'like', 'think', 'going', 'know', 'want'
    ]

    custom_stopwords = list(ENGLISH_STOP_WORDS) + corporate_fluff
    
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

def main():
    print("========== ClassifAI Comprehensive Benchmarking Tool (Earnings22) ==========")
    
    os.makedirs("dev_debug_logs/earnings22_debug_logs", exist_ok=True)
    
    # 1. Load Dataset Once For All Tests
    audio_path, ground_truth_text = load_earnings_dataset()
    print("=================================================================\n")
    
    file_exists = os.path.isfile(RESULTS_CSV)
    with open(RESULTS_CSV, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Test Name", "Model", "Audio Normalization", "Noise Reduction", "NR Prop Decrease", 
                "Use Prompt", "Keyword N-Gram", "Keyword Top N", "Condition on Prev", 
                "Beam Size", "Temperature", 
                "Runtime (s)", "WER", "BERT_Precision", "BERT_Recall", "BERT_F1"
            ])
            
    for i, config in enumerate(TEST_CONFIGS):
        if i != 10:  # TEMP: Run only particular test(s) for quick debugging. Remove this condition to run all tests.
            continue  # TEMP: Run only a particular test(s) for quick debugging. Remove this line to run all tests.
        
        print(f"\n\n{'='*60}")
        print(f"RUNNING TEST: {config['name']}")
        print(f"{'='*60}")

        standardized_path = "temp_standardized.wav"
        denoised_path = "clean_ready_for_ai.wav"

        # 2. Pre-process Audio
        format_audio(audio_path, standardized_path, normalize=config['normalize_audio'])
        
        target_audio_path = standardized_path
        if config['use_noise_reduction']:
            denoise_audio(standardized_path, denoised_path, prop_decrease=config['nr_prop_decrease'])
            target_audio_path = denoised_path

        # 3. Prompt Generation
        custom_prompt = None
        if config['use_prompt']:
            dense_keywords = extract_dense_jargon(
                ground_truth_text, 
                ngram_range=config['keyword_ngram'], 
                top_n=config['keyword_top_n']
            )
            custom_prompt = f"The following earnings call covers topics including: {dense_keywords}. Let us begin."
            print(f"Generated Prompt: {custom_prompt[:100]}...")

        # 4. Transcription
        print(f"\nLoading {config['model']} model...")
        model = whisper.load_model(config['model'])
        
        print("Transcribing... (This may take a while for a full 1-hour meeting)")
        transcription_start = time.time()
        
        transcribe_kwargs = {
            "audio": target_audio_path,
            "condition_on_previous_text": config['condition_on_previous_text'],
            "word_timestamps": False, # Word timestamps not needed since we do full text eval
            "beam_size": config['beam_size'],
            "temperature": config['temperature'],
            "language": "en" 
        }
        if custom_prompt:
            transcribe_kwargs["initial_prompt"] = custom_prompt

        result = model.transcribe(**transcribe_kwargs)
        transcription_end = time.time()
        
        transcription_runtime = transcription_end - transcription_start
        print(f"Transcription complete in {transcription_runtime:.2f} seconds.")

        # 5. Save Predicted Transcript
        safe_config_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', config['name'])
        transcript_filename = f"dev_debug_logs/earnings22_debug_logs/transcript_prediction_{safe_config_name}.txt"
        
        transcript_text = "".join([segment.get('text', '') + " " for segment in result.get('segments', [])])
        
        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"--- PREDICTED TRANSCRIPT: {config['name']} ---\n\n")
            f.write(textwrap.fill(transcript_text.strip(), width=100) + "\n")
        print(f"Saved predicted transcript to {transcript_filename}")


        GT_filename = f"dev_debug_logs/earnings22_debug_logs/transcript_GT.txt"
        with open(GT_filename, "w", encoding="utf-8") as f:
            f.write(f"--- GROUND TRUTH TRANSCRIPT ---\n\n")
            f.write(textwrap.fill(ground_truth_text.strip(), width=100) + "\n")
        print(f"Saved ground truth transcript to {GT_filename}")

        # 6. Cleanup Memory
        del model 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 7. Evaluation (WER & BERTScore)
        print("Evaluating Results...")
        candidate_clean = transcript_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        reference_clean = ground_truth_text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

        print("Calculating Word Error Rate (WER)...")
        wer_score = wer(reference_clean, candidate_clean)

        print("Calculating BERTScore (using first 500 words to avoid token limit)...")
        # Earnings22 calls are 1 hour long and exceed BERT's 512 token limit. 
        # We chunk out the first 500 words to represent a slice of semantic similarity.
        cand_short = " ".join(candidate_clean.split()[:500])
        ref_short = " ".join(reference_clean.split()[:500])

        try:
            P, R, F1 = score([cand_short], [ref_short], lang="en", verbose=False, rescale_with_baseline=True)
            p_mean = P.mean().item()
            r_mean = R.mean().item()
            f1_mean = F1.mean().item()
        except Exception as e:
            print(f"BERTScore failed: {e}")
            p_mean, r_mean, f1_mean = 0.0, 0.0, 0.0

        print(f"\n--- Evaluation Results ---")
        print(f"Word Error Rate (WER): {wer_score:.4f}")
        print(f"BERTScore Precision:   {p_mean:.4f}")
        print(f"BERTScore Recall:      {r_mean:.4f}")
        print(f"BERTScore F1:          {f1_mean:.4f}")

        # 8. Log to CSV
        with open(RESULTS_CSV, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                config['name'], config['model'], config['normalize_audio'], config['use_noise_reduction'],
                config['nr_prop_decrease'], config['use_prompt'], str(config['keyword_ngram']), 
                config['keyword_top_n'], config['condition_on_previous_text'], config['beam_size'], 
                config['temperature'],
                round(transcription_runtime, 2), round(wer_score, 4), round(p_mean, 4), round(r_mean, 4), round(f1_mean, 4)
            ])
        
        print(f"Results logged to {RESULTS_CSV}")

        # 9. File Cleanup
        if os.path.exists(standardized_path):
            os.remove(standardized_path)
        if os.path.exists(denoised_path):
            os.remove(denoised_path)

if __name__ == "__main__":
    main()