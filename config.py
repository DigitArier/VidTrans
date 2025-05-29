# Dateipfade
VIDEO_PATH = "The Most Twisted DXM Stories In Existence_HD.mp4"
FINAL_VIDEO_PATH = "The Most Twisted DXM Stories In Existence_HD_deutsch.mp4"
ORIGINAL_AUDIO_PATH = "original_audio.wav"
PROCESSED_AUDIO_PATH = "processed_audio.wav"
PROCESSED_AUDIO_PATH_SPEED = "processed_audio_speed.wav"
SAMPLE_PATH_1 = "JRE_sample-01.wav"
SAMPLE_PATH_2 = "JRE_sample-02.wav"
SAMPLE_PATH_3 = "JRE_sample-03.wav"
#SAMPLE_PATH_4 = "papa_sample-04.wav"
#SAMPLE_PATH_5 = "papa_sample-05.wav"
SPEECH_TIMESTAMPS = "speech_timestamps.json"
DOWNSAMPLED_AUDIO_PATH = "downsampled_audio.wav"
ONLY_SPEECH = "only_speech.wav"
SAMPLING_RATE_VAD = 16000   
#Transkription
TRANSCRIPTION_FILE = "transcription.csv"
MERGED_TRANSCRIPTION_FILE = "merged_transcription.csv"
PUNCTED_TRANSCRIPTION_FILE = "puncted_transcription.csv"
CORRECTED_TRANSCRIPTION_FILE = "corrected_transcription.csv"
#Translation
TRANSLATION_FILE = "translation.csv"
MERGED_TRANSLATION_FILE = "merged_translation.csv"
MERGED_TRANSLATION_FILE_2 = "merged_translation_2.csv"
REPAIRED_TRANSLATION_FILE = "repaired_translation_file.csv"
CLEAN_TRANSLATION_FILE = "clean_translation.csv"
PUNCTED_TRANSLATION_FILE = "puncted_translation.csv"
TTS_FORMATTED_TRANSLATION_FILE = "tts_formatted_translation.csv"
# Zusammenführung
MIN_DUR = 1.5 # Minimale Segmentdauer in Sekunden 
MAX_DUR = 15 # Maximale Segmentdauer in Sekunden
MAX_GAP = 0.5 # Maximaler akzeptierter Zeitabstand zwischen Segmenten
MAX_CHARS = 300 # Maximale Anzahl an Zeichen pro Segment
MIN_WORDS = 7 # Minimale Anzahl an Wörtern pro Segment
ITERATIONS = 2 # Durchläufe
#Quality_Report
ST_QUALITY_MODEL = "paraphrase-multilingual-mpnet-base-v2"
TRANSLATION_QUALITY_REPORT = "translation_quality_report.csv"
ST_EMBEDDING_MODEL_DE = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
EMBEDDINGS_FILE_NPZ = "german_text_embeddings.npz"
TRANSLATION_WITH_EMBEDDINGS_CSV = "german_text_embeddings.csv"
# Schwellenwert für die Kosinus-Ähnlichkeit (experimentell bestimmen, z.B. 0.6 - 0.8)
SIMILARITY_THRESHOLD = 0.75
#TTS
TRANSLATED_AUDIO_WITH_PAUSES = "translated_audio_with_pauses.wav"
RESAMPLED_AUDIO_FOR_MIXDOWN = "resampled_audio_44100.wav"
ADJUSTED_VIDEO_PATH = "adjusted_video.mp4"
USE_PIP = True
USE_ONNX_VAD = True
BOS_TOKEN_ID = 0
EOS_TOKEN_ID = 1
PAD_TOKEN_ID = 2
vocoder_pth = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\bigvgan_generator.pt"
vocoder_cfg = r"D:\Modelle\Vocoder\bigvgan_v2_24khz_100band_256x\config.json"
