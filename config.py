# Dateipfade
VIDEO_PATH = "The WAR between TWO PILLARS (OCCULT pillars VS The pillars of YHWH)_Full-HD.mp4"
FINAL_VIDEO_PATH = "The WAR between TWO PILLARS (OCCULT pillars VS The pillars of YHWH)_Full-HD_deutsch.mp4"
ORIGINAL_AUDIO_PATH = "0_original_audio.wav"
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
TRANSCRIPTION_FILE = "1_transcription.csv"
TRANSCRIPTION_CLEANED = "2_transcription_cleaned.csv"
PUNCTED_TRANSCRIPTION_FILE = "3_puncted_transcription.csv"
CORRECTED_TRANSCRIPTION_FILE = "4_corrected_transcription.csv"
MERGED_TRANSCRIPTION_FILE = "5_merged_transcription.csv"
PUNCTED_TRANSCRIPTION_FILE_2 = "6_puncted_transcription.csv"
CHAR_LIMIT = 150
FORMATTED_TRANSKRIPTION_FILE = "7_formatted_transcription.csv"
# Zusammenführung Transkription
MIN_DUR = 3.0 # Minimale Segmentdauer in Sekunden 
MAX_DUR = 15 # Maximale Segmentdauer in Sekunden
MAX_GAP = 0.5 # Maximaler akzeptierter Zeitabstand zwischen Segmenten
MAX_CHARS = 150 # Maximale Anzahl an Zeichen pro Segment
MIN_WORDS = 10 # Minimale Anzahl an Wörtern pro Segment
ITERATIONS = 3 # Durchläufe
#Translation
TRANSLATION_FILE = "8_translation.csv"
MERGED_TRANSLATION_FILE = "10_merged_translation.csv"
REPAIRED_TRANSLATION_FILE = "repaired_translation_file.csv"
CLEAN_TRANSLATION_FILE = "clean_translation.csv"
PUNCTED_TRANSLATION_FILE = "puncted_translation.csv"
TTS_FORMATTED_TRANSLATION_FILE = "12_tts_formatted_translation.csv"
# Zusammenführung Übersetzung
MIN_DUR_TRANSLATION = 3.0# Minimale Segmentdauer in Sekunden
MAX_DUR_TRANSLATION = 15 # Maximale Segmentdauer in Sekunden
MAX_GAP_TRANSLATION = 0.5 # Maximaler akzeptierter Zeitabstand zwischen Segmenten
MAX_CHARS_TRANSLATION = 200 # Maximale Anzahl an Zeichen pro Segment
MIN_WORDS_TRANSLATION = 7 # Minimale Anzahl an Wörtern pro Segment
ITERATIONS_TRANSLATION = 3 # Durchläufe
#Quality_Report
ST_QUALITY_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TRANSLATION_QUALITY_REPORT = "9_translation_quality_report.csv"
CLEANED_SOURCE_FOR_QUALITY_CHECK = "cleaned_source_for_quality_check.csv"
ST_EMBEDDING_MODEL_DE = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
EMBEDDINGS_FILE_NPZ = "11_german_text_embeddings.npz"
TRANSLATION_WITH_EMBEDDINGS_CSV = "11_german_text_embeddings.csv"
# Schwellenwert für die Kosinus-Ähnlichkeit (experimentell bestimmen, z.B. 0.6 - 0.8)
SIMILARITY_THRESHOLD = 0.7
#TTS
TRANSLATED_AUDIO_WITH_PAUSES = "13_translated_audio_with_pauses.wav"
RESAMPLED_AUDIO_FOR_MIXDOWN = "14_resampled_audio_44100.wav"
ADJUSTED_VIDEO_PATH = "15_adjusted_video.mp4"
USE_PIP = True
USE_ONNX_VAD = True
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
vocoder_pth = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\bigvgan_generator.pt"
vocoder_cfg = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\config.json"
