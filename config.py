# Dateipfade
AUDIO_PATH = "2_the SERPENT god of FORBIDDEN knowledge (PART 2)(DOCUMENTARY)_HD_(Vocals).wav"
VIDEO_PATH = "the SERPENT god of FORBIDDEN knowledge (PART 2)(DOCUMENTARY)_HD.mp4"
FINAL_VIDEO_PATH = "the SERPENT god of FORBIDDEN knowledge (PART 2)(DOCUMENTARY)_HD_deutsch.mp4"
ORIGINAL_AUDIO_PATH = "00_original_audio.wav"
PROCESSED_AUDIO_PATH = "processed_audio.wav"
PROCESSED_AUDIO_PATH_SPEED = "processed_audio_speed.wav"
SAMPLE_PATH_1 = "asleep_sample-01.wav"
SAMPLE_PATH_2 = "asleep_sample-02.wav"
SAMPLE_PATH_3 = "asleep_sample-03.wav"
#SAMPLE_PATH_4 = "gibson_sample-04.wav"
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
REFINED_TRANSLATION_FILE = "03a_refined_translation.csv"
MERGED_TRANSLATION_FILE = "06_merged_translation.csv"
REPAIRED_TRANSLATION_FILE = "repaired_translation_file.csv"
CLEAN_TRANSLATION_FILE = "clean_translation.csv"
PUNCTED_TRANSLATION_FILE = "puncted_translation.csv"
CHAR_LIMIT_TRANSLATION = 210
TTS_FORMATTED_TRANSLATION_FILE = "05_tts_formatted_translation.csv"
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
# Schwellenwert für die Kosinus-Ähnlichkeit (experimentell bestimmen, z.B. 0.6 - 0.8)
SIMILARITY_THRESHOLD = 0.88
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
# ---------- OpenVoice Settings ----------
OV_BASE_SPEAKER_CFG = "openvoice/checkpoints/base_speakers/EN/config.json"
OV_BASE_SPEAKER_CKPT = "openvoice/checkpoints/base_speakers/EN/checkpoint.pth"
OV_CONVERTER_CFG     = "openvoice/checkpoints/converter/config.json"
OV_CONVERTER_CKPT    = "openvoice/checkpoints/converter/checkpoint.pth"
# Liste zulässiger Emotionen laut OpenVoice-Demo[19]
OV_ALLOWED_EMOTIONS = [
    "default", "whispering", "shouting", "excited",
    "cheerful", "terrified", "angry", "sad", "friendly"
]

# Vocoderpfade für XTTS
vocoder_pth = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\bigvgan_generator.pt"
vocoder_cfg = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\config.json"

# Sentence Transformer Modell-Konfiguration
SENTENCE_TRANSFORMER_MODELS = {
    "quality": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "quality_neu": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "quality_big": "intfloat/multilingual-e5-large",
    "embedding": "sentence-transformers/distiluse-base-multilingual-cased-v2", 
    "embedding_big": "intfloat/multilingual-e5-large-instruct",
    "speed": "sentence-transformers/all-MiniLM-L6-v2",
    "multi_speed": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "latest": "Alibaba-NLP/gte-large-en-v1.5"
}

# Globale 4-Wort-Grenze gegen Ein-Wort-Segmente
MIN_WORDS_GLOBAL = 4

# Standard-Modelle für verschiedene Aufgaben
ST_QUALITY_MODEL = SENTENCE_TRANSFORMER_MODELS["quality"]
ST_EMBEDDING_MODEL_DE = SENTENCE_TRANSFORMER_MODELS["embedding"]
ST_SPEED_MODEL = SENTENCE_TRANSFORMER_MODELS["multi_speed"]

# TTS Text-Validierung
MIN_TTS_TEXT_LENGTH = 5
MAX_TTS_TEXT_LENGTH = 200

# Deutsche Abkürzungen für robuste Satzaufteilung
GERMAN_ABBREVIATIONS = {
    "Dr.", "Prof.", "z.B.", "u.a.", "Nr.", "Abs.", "etc.", "usw.", "ca.", "i.d.R.",
    "Mr.", "Mrs.", "Ms.", "bzw.", "Co.", "GmbH", "AG", "e.V.", "i.d.R.", "u.a.", "z.T."
}