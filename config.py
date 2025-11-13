# Dateipfade
from re import S
AUDIO_PATH =        "1_can YOU handle this TRUTH_ (Documentary)_HD_(Vocals).wav"
VIDEO_PATH =        "can YOU handle this TRUTH_ (Documentary)_HD.mp4"
FINAL_VIDEO_PATH =  "can YOU handle this TRUTH_ (Documentary)_HD_deutsch.mp4"
ORIGINAL_AUDIO_PATH = "00_original_audio.wav"
PROCESSED_AUDIO_PATH = "processed_audio.wav"
PROCESSED_AUDIO_PATH_SPEED = "processed_audio_speed.wav"
#SAMPLE_PATH_1 = "ich_sample-01.wav"
#SAMPLE_PATH_2 = "ich_sample-02.wav"
#SAMPLE_PATH_3 = "ich_sample-03.wav"
SAMPLE_PATH_1 = "servant_sample-01-2.wav"
SAMPLE_PATH_2 = "servant_sample-02-2.wav"
SAMPLE_PATH_3 = "servant_sample-03-2.wav"
#SAMPLE_PATH_4 = "ich_sample-04.wav"
#SAMPLE_PATH_5 = "papa_sample-05.wav"
SPEECH_TIMESTAMPS = "speech_timestamps.json"
DOWNSAMPLED_AUDIO_PATH = "downsampled_audio.wav"
ONLY_SPEECH = "only_speech.wav"
SAMPLING_RATE_VAD = 16000   
#Transkription
TRANSCRIPTION_FILE = "01_transcription.csv"
REFINED_TRANSCRIPTION_FILE = "01a_refined_transcription.csv"
TRANSCRIPTION_CLEANED = "02_transcription_cleaned.csv"
PUNCTED_TRANSCRIPTION_FILE = "03_puncted_transcription.csv"
CORRECTED_TRANSCRIPTION_FILE = "corrected_transcription.csv"
MERGED_TRANSCRIPTION_FILE = "05_merged_transcription.csv"
PUNCTED_TRANSCRIPTION_FILE_2 = "02_puncted_transcription.csv"
CHAR_LIMIT_TRANSCRIPTION = 512
FORMATTED_TRANSKRIPTION_FILE = "03_formatted_transcription.csv"
# Zusammenführung Transkription
MIN_DUR = 3.0 # Minimale Segmentdauer in Sekunden 
MAX_DUR = 15 # Maximale Segmentdauer in Sekunden
MAX_GAP = 0.5 # Maximaler akzeptierter Zeitabstand zwischen Segmenten
MAX_CHARS = 150 # Maximale Anzahl an Zeichen pro Segment
MIN_WORDS = 10 # Minimale Anzahl an Wörtern pro Segment
ITERATIONS = 3 # Durchläufe
#Translation
MADLAD400_MODEL_DIR = "madlad400-3b-mt-int8_bfloat16"
MARIANMT_MODEL_DIR = "opus-mt-en-de-ct2"
MARIANMT_HYPOTHESES_CSV = "02_hypotheses_marianmt.csv"
HYPOTHESES_CSV = "02_translation_hypotheses_detailed.csv"
TRANSLATION_FILE = "03_translation.csv"
SEMANTIC_BEST_TRANSLATION_FILE = "03b_translation_semantic_best.csv"
REFINED_TRANSLATION_FILE = "05_refined_translation.csv"
CORRECTED_TRANSLATION_FILE = "03c_translation_corrected.csv"
MERGED_TRANSLATION_FILE = "06_merged_translation.csv"
REPAIRED_TRANSLATION_FILE = "repaired_translation_file.csv"
CLEAN_TRANSLATION_FILE = "clean_translation.csv"
PUNCTED_TRANSLATION_FILE = "puncted_translation.csv"
PROVISIONAL_TRANSLATION_CSV = "provisional_translation.csv"
CLEANED_SOURCE_CSV = "cleaned_source.csv"
POLISHED_TRANSLATION_CSV = "04b_translation_polished.csv"
CHAR_LIMIT_TRANSLATION = 210
TTS_FORMATTED_TRANSLATION_FILE = "06_tts_formatted_translation.csv"
# Zusammenführung Übersetzung
MIN_DUR_TRANSLATION = 3.0# Minimale Segmentdauer in Sekunden
MAX_DUR_TRANSLATION = 15 # Maximale Segmentdauer in Sekunden
MAX_GAP_TRANSLATION = 0.5 # Maximaler akzeptierter Zeitabstand zwischen Segmenten
MAX_CHARS_TRANSLATION = 200 # Maximale Anzahl an Zeichen pro Segment
MIN_WORDS_TRANSLATION = 7 # Minimale Anzahl an Wörtern pro Segment
ITERATIONS_TRANSLATION = 3 # Durchläufe
#Quality_Report
TRANSLATION_QUALITY_REPORT = "04_translation_quality_report.csv"
TRANSLATION_QUALITY_SUMMARY = "04_translation_quality_summary.txt"
CLEANED_SOURCE_FOR_QUALITY_CHECK = "04a_cleaned_source_for_quality_check.csv"
EMBEDDINGS_FILE_NPZ = "08_german_text_embeddings.npz"
TRANSLATION_WITH_EMBEDDINGS_CSV = "08_german_text_embeddings.csv"
#TTS
TTS_TEMP_CHUNKS_DIR = "tts_temp_chunks"
TTS_PROGRESS_MANIFEST = "tts_progress_manifest.csv"
TRANSLATED_AUDIO_WITH_PAUSES = "06_translated_audio_with_pauses.wav"
RESAMPLED_AUDIO_FOR_MIXDOWN = "07_resampled_audio_44100.wav"
ADJUSTED_VIDEO_PATH = "08_adjusted_video.mp4"
USE_PIP = True
USE_ONNX_VAD = True
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
# ---------------------------------
# NLLB-200 Einstellungen
# ---------------------------------
NLLB_MODEL_DIR = "nllb-200-1.3B-bfloat16"   # Pfad zum konvertierten CT2-Modell
NLLB_BATCH_MAX_TOKENS = 2048               # konservativ für RTX-40-Laptop-8 GB

# Vocoderpfade für XTTS
vocoder_pth = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\bigvgan_generator.pt"
vocoder_cfg = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\config.json"

# Sentence Transformer Modell-Konfiguration
SENTENCE_TRANSFORMER_MODELS = {
    "quality": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "quality_LaBSE": "sentence-transformers/LaBSE",
    "quality_neu": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "quality_big": "intfloat/multilingual-e5-large",
    "embedding": "sentence-transformers/distiluse-base-multilingual-cased-v2", 
    "embedding_big": "intfloat/multilingual-e5-large-instruct",
    "mini": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "multi_speed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "latest": "Alibaba-NLP/gte-large-en-v1.5"
}
CORRECTION_LLM_MODELS = {
    "gemma": "gemma2:9b",
    "qwen2.5": "qwen2.5:7b",
    "qwen3": "qwen3:8b"
}
SIMILARITY_THRESHOLD_EVAL = 0.78
SIMILARITY_THRESHOLD_POLISHING = 0.9
# Standard-Modelle für verschiedene Aufgaben
ST_QUALITY_MODEL = SENTENCE_TRANSFORMER_MODELS["quality_LaBSE"]
ST_POLISH_MODEL_DE = SENTENCE_TRANSFORMER_MODELS["quality"]
ST_MINI_MODEL = SENTENCE_TRANSFORMER_MODELS["mini"]
GEMMA= CORRECTION_LLM_MODELS["gemma"]
QWEN2_5= CORRECTION_LLM_MODELS["qwen2.5"]
QWEN3= CORRECTION_LLM_MODELS["qwen3"]

# Globale 4-Wort-Grenze gegen Ein-Wort-Segmente
MIN_WORDS_GLOBAL = 4

# TTS Text-Validierung
MIN_TTS_TEXT_LENGTH = 5
MAX_TTS_TEXT_LENGTH = 200

# Deutsche Abkürzungen für robuste Satzaufteilung
GERMAN_ABBREVIATIONS = {
    "Dr.", "Prof.", "z.B.", "u.a.", "Nr.", "Abs.", "etc.", "usw.", "ca.", "i.d.R.",
    "Mr.", "Mrs.", "Ms.", "bzw.", "Co.", "GmbH", "AG", "e.V.", "i.d.R.", "u.a.", "z.T."
}