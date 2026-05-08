import torch

# Dateipfade
AUDIO_PATH =        "1_BABYLON the DECAPITATOR (DOCUMENTARY) 'RUN...NOAH, HIDE' (They have taken th_Full-HD_(Vocals).wav"
VIDEO_PATH =        "BABYLON the DECAPITATOR (DOCUMENTARY) 'RUN...NOAH, HIDE' (They have taken th_Full-HD.mp4"
FINAL_VIDEO_PATH =  "BABYLON the DECAPITATOR (DOCUMENTARY) 'RUN...NOAH, HIDE' (They have taken th_Full-HD_deutsch.mp4"
ORIGINAL_AUDIO_PATH = "00_original_audio.wav"
PROCESSED_AUDIO_PATH = "processed_audio.wav"
PROCESSED_AUDIO_PATH_SPEED = "processed_audio_speed.wav"
#SAMPLE_PATH_1 = "ich_sample-01.wav"
#SAMPLE_PATH_2 = "ich_sample-02.wav"
#SAMPLE_PATH_3 = "ich_sample-03.wav"
#SAMPLE_PATH_4 = "ich_sample-04.wav"
#SAMPLE_PATH_5 = "ich_sample-05.wav"
#SAMPLE_PATH_6 = "ich_sample-06.wav"
SAMPLE_PATH_1 = "servant_sample-01.wav"
SAMPLE_PATH_2 = "servant_sample-02.wav"
SAMPLE_PATH_3 = "servant_sample-03.wav"
#SAMPLE_PATH_4 = "servant_sample-04.wav"

# HuggingFace-ID des OmniVoice-Modells (lokal cachen nach erstem Download)
OMNIVOICE_MODEL_ID: str = "k2-fsa/OmniVoice"

# BF16 halbiert den VRAM-Bedarf vs. FP32; auf RTX 4050 (Ampere) nativ unterstützt.
# VRAM ~3.3 GB (Modellgewichte) + ~0.5–1 GB (Aktivierungen) = ~4 GB gesamt.
OMNIVOICE_DTYPE = torch.bfloat16

# Diffusion-Schritte: 16 = schnell+gut; 32 = beste Qualität (langsamer)
OMNIVOICE_NUM_STEPS: int = 50

# Sprechgeschwindigkeit – identisch zum bisherigen XTTS-Wert speed=1.05
OMNIVOICE_SPEED: float = None

# Zielsprache der Synthese
OMNIVOICE_LANGUAGE: str = "de"

# Sampling-Rate – identisch zu XTTS (24 kHz)
OMNIVOICE_SAMPLE_RATE: int = 24_000

# ─── Qwen3-TTS Konfiguration ──────────────────────────────────────────────────

# Modell-ID für Voice Cloning (3-Sek.-Referenz-Cloning)
# VRAM-Schätzung: ~3.5 GB in bfloat16 — sicher für RTX 4050 Mobile (6 GB VRAM)
QWEN3_TTS_CLONE_MODEL_ID: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

# Fallback: kleineres Modell (~1.5 GB VRAM), geringere Qualität
QWEN3_TTS_MODEL_ID: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Gerät für TTS-Synthese (z.B. "cuda:0" für GPU oder "cpu" für CPU)
QWEN3_TTS_DEVICE: str = "cuda:0"

# Zielsprache für TTS-Synthese (Qwen3-TTS-konformer Bezeichner)
QWEN3_TTS_TARGET_LANGUAGE: str = "German"

# Ausgabe-Samplerate von Qwen3-TTS (fest 24 kHz)
QWEN3_TTS_SAMPLE_RATE: int = 24_000

# Maximale Anzahl an Tokens pro Syntheseaufruf
QWEN3_TTS_MAX_NEW_TOKENS: int = 2048

# Gewichtung zwischen STS-Similarity (1-ALPHA) und normiertem Beam-Score (ALPHA)
# 0.0 = nur STS, 1.0 = nur Beam-Score
# 0.35 = leichte Bevorzugung von STS-Treue über Modell-Konfidenz
HYPOTHESIS_SCORE_ALPHA: float = 0.40  # ÄNDERUNG: 0.35 → 0.40; nach Beam-Verbesserung Gewicht leicht anheben

# Minimale Token-Länge einer Hypothese (Schutz vor leeren Ausgaben)
MIN_HYPOTHESIS_TOKENS: int = 1

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
MADLAD400_MODEL_DIR = "madlad400-3b-mt-bfloat16"
MARIANMT_MODEL_DIR = "opus-mt-en-de-ct2"
MARIANMT_HYPOTHESES_CSV = "02_hypotheses_marianmt.csv"
HYPOTHESES_CSV = "02a_translation_hypotheses_detailed.csv"
TRANSLATION_FILE = "02_translation.csv"
SEMANTIC_BEST_TRANSLATION_FILE = "02b_translation_semantic_best.csv"
REFINED_TRANSLATION_FILE = "05_refined_translation.csv"
CORRECTED_TRANSLATION_FILE = "03_translation_corrected.csv"
MERGED_TRANSLATION_FILE = "06_merged_translation.csv"
REPAIRED_TRANSLATION_FILE = "repaired_translation_file.csv"
CLEAN_TRANSLATION_FILE = "clean_translation.csv"
PUNCTED_TRANSLATION_FILE = "puncted_translation.csv"
PROVISIONAL_TRANSLATION_CSV = "provisional_translation.csv"
CLEANED_SOURCE_CSV = "cleaned_source.csv"
POLISHED_TRANSLATION_CSV = "04_translation_polished.csv"
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
TRANSLATION_QUALITY_REPORT = "03a_translation_quality_report.csv"
TRANSLATION_QUALITY_SUMMARY = "03a_translation_quality_summary.txt"
POLISHED_TRANSLATION_SUMMARY = "04a_polished_translation_summary.txt"
CLEANED_SOURCE_FOR_QUALITY_CHECK = "01b_cleaned_source_for_quality_check.csv"
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
vocoder_pth = r"C:\Users\regme\Desktop\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\bigvgan_generator.pt"
vocoder_cfg = r"C:\Users\regme\Desktop\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\config.json"

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
    "qwen3_8": "qwen3:8b",
    "qwen3": "qwen3:14b",
    "translate": "translategemma:4b"
}
SIMILARITY_THRESHOLD_EVAL = 0.95
SIMILARITY_THRESHOLD_POLISHING = 0.9
# Standard-Modelle für verschiedene Aufgaben
ST_QUALITY_MODEL = SENTENCE_TRANSFORMER_MODELS["quality"]
ST_BETTER_MODEL = SENTENCE_TRANSFORMER_MODELS["quality_LaBSE"]
ST_MINI_MODEL = SENTENCE_TRANSFORMER_MODELS["mini"]
GEMMA= CORRECTION_LLM_MODELS["gemma"]
QWEN3_8= CORRECTION_LLM_MODELS["qwen3_8"]
QWEN3= CORRECTION_LLM_MODELS["qwen3"]
TRANSLATE: str= CORRECTION_LLM_MODELS["translate"]

# Globale 4-Wort-Grenze gegen Ein-Wort-Segmente
MIN_WORDS_GLOBAL = 4

# TTS Text-Validierung
MIN_TTS_TEXT_LENGTH = 5
MAX_TTS_TEXT_LENGTH = 200

# Deutsche Abkürzungen für robuste Satzaufteilung
GERMAN_ABBREVIATIONS = {
    "Dipl.-Ing.", "Dr.-Ing.", "m.M.n.", "Oberst", "Forts.",
    "M.B.A.", "n.Chr.", "v.Chr.", "i.d.R.", "habil.",
    "Dipl.", "verh.", "Hptm.", "exkl.", "gest.",
    "gGmbH", "B.Sc.", "Prof.", "Ph.D.", "Oblt.",
    "verw.", "inkl.", "M.Sc.", "LL.B.", "LL.M.",
    "z.Zt.", "evtl.", "Sept.", "u.Ä.", "Nov.",
    "Anl.", "i.O.", "s.u.", "Str.", "NATO",
    "Feb.", "bzw.", "min.", "Jan.", "Abs.",
    "ggf.", "B.A.", "Maj.", "Apr.", "u.a.",
    "Jul.", "u.g.", "o.B.", "i.V.", "StGB",
    "Aug.", "o.g.", "Anm.", "km/h", "pkt.",
    "Mrz.", "Tel.", "m.E.", "Okt.", "Gen.",
    "RAin", "Inc.", "Herr", "Dez.", "Jun.",
    "Doz.", "GmbH", "Jhd.", "Kap.", "led.",
    "etc.", "vgl.", "e.K.", "usw.", "geb.",
    "M.A.", "u.U.", "z.T.", "Abb.", "Ltd.",
    "s.o.", "e.V.", "Ass.", "p.a.", "Sep.",
    "Tab.", "max.", "o.ä.", "Frau", "i.A.",
    "d.h.", "Mag.", "Fig.", "z.B.", "ca.",
    "Di.", "Hr.", "ff.", "So.", "St.",
    "Fax", "Dr.", "Mi.", "MdB", "StR",
    "mfG", "DDR", "BRD", "vs.", "BGB",
    "WHO", "Co.", "Bl.", "Nr.", "m/s",
    "mbH", "HGB", "Pl.", "Mo.", "Jh.",
    "Fr.", "Lt.", "Mai", "Bv.", "OHG",
    "MBA", "Av.", "Sa.", "Do.", "em.",
    "MdL", "Bd.", "MfG", "°C", "UN",
    "AG", "qm", "EU", "S.", "s.",
    "UG", "ha", "cm", "m²", "mm",
    "RA", "kg", "KG", "m³", "mg",
    "km", "€", "§", "%",
}

# Englische Abkürzungen für robuste Satzaufteilung
ENGLISH_ABBREVIATIONS = {
    "approx.", "Assoc.", "D.D.S.", "Corp.", "Prof.", 
    "Ph.D.", "Bros.", "Ed.D.", "Blvd.", "Capt.", 
    "Dept.", "LL.B.", "Terr.", "LL.M.", "Sept.", 
    "Nov.", "Wed.", "Mon.", "e.g.", "Feb.", 
    "vol.", "Tue.", "Thu.", "M.S.", "Sgt.", 
    "Fri.", "Jan.", "B.A.", "Sen.", "Maj.", 
    "Apr.", "M.D.", "Jul.", "U.N.", "a.m.", 
    "Aug.", "Mrs.", "viz.", "E.U.", "p.m.", 
    "Dec.", "P.M.", "Rev.", "Gen.", "Sat.", 
    "Mar.", "Inc.", "Ave.", "Jun.", "Est.", 
    "gal.", "U.K.", "Sun.", "pvt.", "B.S.", 
    "U.S.", "etc.", "i.e.", "Hwy.", "fig.", 
    "M.A.", "Ltd.", "Col.", "Gov.", "mfg.", 
    "A.M.", "Rep.", "Sep.", "Oct.", "Pty.", 
    "Hon.", "Fig.", "mi.", "ca.", "pp.", 
    "Mr.", "lb.", "ed.", "No.", "sq.", 
    "Ct.", "St.", "LLC", "Dr.", "Ms.", 
    "vs.", "Ln.", "Sr.", "yd.", "qt.", 
    "Mt.", "Co.", "cu.", "pt.", "Pl.", 
    "cm.", "ft.", "in.", "Jr.", "Lt.", 
    "cf.", "Sq.", "mm.", "oz.", "km.", 
    "Rd.", "Ft.", "p."
}