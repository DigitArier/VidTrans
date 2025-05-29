from ast import Constant
import os
import logging

# ==============================
# Globale Konfigurationen und Logging
# ==============================
# Logging so früh wie möglich konfigurieren
logging.basicConfig(
    filename='video_translation_final.log',
    format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s", # Detaillierteres Format
    level=logging.INFO,
    filemode='a',  # 'w' zum Überschreiben bei jedem Start, 'a' zum Anhängen (Standard)
    force=True     # Wichtig für Python 3.8+: Stellt sicher, dass diese Konfiguration greift
)
logger = logging.getLogger(__name__) # Logger für dieses Modul holen

# Test-Log-Nachricht direkt nach der Initialisierung
logger.info("Logging wurde erfolgreich initialisiert.")
"""
import coverage
cov = coverage.Coverage(branch=True)
cov.start()
"""
import re
from pathlib import Path
import subprocess
import ffmpeg
from langcodes import Language
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import pandas as pd
import json
import scipy.signal as signal
from scipy.signal import resample_poly
from pydub import AudioSegment
from tokenizers import Encoding, Tokenizer
from tokenizers.models import BPE
import packaging
import spacy
import ftfy
from ftfy import fix_encoding
import torch
from torch import autocast
torch.set_num_threads(12)
import tensor_parallel
import deepspeed
from deepspeed import init_inference, DeepSpeedConfig
from accelerate import init_empty_weights, infer_auto_device_map
import shape as sh
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torchaudio
from audiostretchy.stretch import stretch_audio
import pyrubberband
import time
from datetime import datetime, timedelta
import csv
import traceback
import psutil
import language_tool_python
from language_tool_python import LanguageTool
from functools import partial
from multiprocessing import Pool, cpu_count
from functools import partial
from llama_cpp import Llama
from config import *
from tqdm import tqdm
from contextlib import contextmanager
from deepmultilingualpunctuation import PunctuationModel
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    MarianConfig,
    MarianPreTrainedModel,
    MarianMTModel,
    MarianTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoModelForCausalLM,
    GenerationMixin,
    QuantoConfig,
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TorchAoConfig
    )
import transformers
import ctranslate2
from transformers import modeling_utils, modeling_flash_attention_utils
from torchao.quantization import Int8WeightOnlyConfig
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
#import whisper
from faster_whisper import WhisperModel, BatchedInferencePipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from pydub import AudioSegment
from pydub.effects import(
    high_pass_filter,
    low_pass_filter,
    normalize
    )
from silero_vad import(
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    collect_chunks
    )

torch.backends.cuda.matmul.allow_tf32 = True  # Schnellere Matrix-Multiplikationen
torch.backends.cudnn.allow_tf32 = True        # TF32 für cuDNN aktivieren
torch.backends.cudnn.benchmark = True         # Optimale Kernel-Auswahl
torch.backends.cudnn.deterministic = False    # Nicht-deterministische Optimierungen erlauben
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["CT2_FLASH_ATTENTION"] = "1"
os.environ["CT2_VERBOSE"] = "1"

# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.0   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.0   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 1.0     # Geschwindigkeitsfaktor für die Wiedergabe des Videos

# Lautstärkeanpassungen
VOLUME_ADJUSTMENT_44100 = 1.0   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.045   # Lautstärkefaktor für das Video

# ============================== 
# Hilfsfunktionen
# ==============================

start_time = time.time()
step_times = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
source_lang="en"
target_lang="de"
# Konfigurationen für die Verwendung von CUDA
cuda_options = {
    "hwaccel": "cuda",
    "hwaccel_output_format": "cuda"
}

def run_command(command):
    
    print(f"Ausführung des Befehls: {command}")
    subprocess.run(command, shell=True, check=True)

def time_function(func, *args, **kwargs):
        """Misst die Ausführungszeit einer Funktion."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        logger.info(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
        return result, execution_time

def load_whisper_model():
    """
    Lädt das Whisper-Modell in INT8-Quantisierung für schnellere GPU-Inferenz
    und richtet die gebatchte Pipeline ein.
    """
    model_size = "large-v3"
    # compute_type="int8_float16" nutzt INT8-Gewichte + FP16-Aktivierungen für Speed & geringen Speicher
    fw_model = WhisperModel(model_size, device="auto", compute_type="bfloat16", cpu_threads=12, local_files_only=True)
    pipeline = BatchedInferencePipeline(model=fw_model)
    return pipeline

# Batch-Größe für die Übersetzung
BATCH_SIZE = 2

def load_translation_model(model_path=None):
    """
    Lädt das MADLAD400-Übersetzungsmodell für die Verwendung auf der GPU oder CPU.
    
    Args:
        model_path: Pfad zum Modell. Wenn None, wird ein Standardmodell geladen.
    
    Returns:
        Ein Tuple mit Modell und Tokenizer.
    """
    # Standardmodell für Englisch nach Deutsch, falls kein Pfad angegeben
    if model_path is None:
        model_path = "jbochi/madlad400-7b-mt"  # MADLAD400-Modell für maschinelle Übersetzung
    
    logger.info(f"Lade MADLAD400-Modell von {model_path}...")
    
    # Gerät bestimmen (GPU bevorzugt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Verwende Gerät: {device}")
    
    num_threads=10
    
    torch.set_num_threads(num_threads)
    logger.info(f"Setze PyTorch auf {num_threads} CPU-Threads für maximale Leistung.")
    
    # Speicher freigeben, um Platz für das Modell zu schaffen
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        # Tokenizer und Modell laden
        tokenizer = T5TokenizerFast.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Modell auf das entsprechende Gerät verschieben
        model.to(device)
        model.eval()  # Inferenzmodus aktivieren
        model = torch.compile(model, dynamic=True, mode="max-autotune")  # Optional: TorchScript-Optimierung
        
        logger.info(f"MADLAD400-Modell erfolgreich geladen auf {device}.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Fehler beim Laden des MADLAD400-Modells: {e}")
        raise

def load_xtts_v2():
    """
    Lädt Xtts v2 und konfiguriert DeepSpeed-Inferenz.
    """
    # 1) Konfiguration lesen
    config = XttsConfig()
    config.load_json("D:\\alltalk_tts\\models\\xtts\\v203\\config.json")
    # 2) Modell initialisieren
    xtts_model = Xtts.init_from_config(
        config,
        vocoder_path=vocoder_pth,
        vocoder_config_path=vocoder_cfg
    )
    xtts_model.load_checkpoint(
        config,
        checkpoint_dir="D:\\alltalk_tts\\models\\xtts\\v203",  # Pfad anpassen
        use_deepspeed=False
    )
    xtts_model.to(torch.device(0))
    xtts_model.eval()

    return xtts_model

# Context Manager für GPU-Operationen
@contextmanager
def gpu_context():
    try:
        yield
    finally:
        torch.cuda.empty_cache()
        logger.info("GPU-Speicher bereinigt")

def extract_audio_ffmpeg(video_path, audio_output):
    """Extrahiert die Audiospur aus dem Video (Mono, 44.100 Hz)."""
    if os.path.exists(audio_output):
        if not ask_overwrite(audio_output):
            logger.info(f"Verwende vorhandene Datei: {audio_output}")
            return
    try:
        ffmpeg.input(video_path, hwaccel="cuda", hwaccel_output_format="cuda").output(
            audio_output,
            threads=0,      # Verwendet alle verfügbaren Threads
            f="wav",
            acodec="pcm_s16le",
            ac=1,  # Mono-Audio
            ar="16000"  # 44.100 Hz
            ).run()
        logger.info(f"Audio extrahiert: {audio_output}")
    except ffmpeg.Error as e:
        logger.error(f"Fehler bei der Audioextraktion: {e}")

def process_audio(input_file, output_file):
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Datei: {output_file}", exc_info=True)
            return
        
    print("--------------------------------")
    print("<< Starte Audio-Verarbeitung >>|")
    print("--------------------------------")
        
    # Helper function to save and log intermediate steps
    def save_step(audio_segment, filename):
        audio_segment.export(filename, format="wav")
        logger.info(f"Zwischenschritt gespeichert: {filename} - Größe: {os.path.getsize(filename)} Bytes", exc_info=True)
    
    # 1. Load the audio file
    audio = AudioSegment.from_wav(input_file)
    #save_step(audio, "process_original.wav")
    
    # 2. High-Pass Filter für klare Stimme (z.B. 80-100 Hz)
    #    Filtert tieffrequentes Dröhnen/Brummen heraus [1][2].
    audio_hp = high_pass_filter(audio, cutoff=150)
    save_step(audio_hp, "process_high_pass.wav")
    
    # 3. Noise Gate, um Atem und Hintergrundrauschen zu unterdrücken
    #    Threshold je nach Sprechpegel, z.B. -48 dB [2][7].
    def noise_gate(audio_segment, threshold_db=-40):
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        max_amplitude = np.max(np.abs(samples))
        
        # ermittelter Schwellenwert in linearer Amplitude
        gate_threshold = max_amplitude * (10 ** (threshold_db / 20))
        gated_samples = np.where(np.abs(samples) > gate_threshold, samples, 0)
        return audio_segment._spawn(gated_samples.astype(np.int16).tobytes())
    
    audio_ng = noise_gate(audio_hp, threshold_db=-48)
    save_step(audio_ng, "process_noise_gate.wav")

    # 4. Multiband-Kompressor (Soft-Knee), um Pegelschwankungen auszugleichen [2].
    def apply_multiband_compression(audio_segment):
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        rate = audio_segment.frame_rate
        
        # Butterworth-Filter
        b_low, a_low = signal.butter(4, 300 / (rate / 2), btype='low')
        b_mid, a_mid = signal.butter(4, [300 / (rate / 2), 3000 / (rate / 2)], btype='bandpass')
        b_high, a_high = signal.butter(4, 3000 / (rate / 2), btype='high')
        
        low_band = signal.lfilter(b_low, a_low, samples)
        mid_band = signal.lfilter(b_mid, a_mid, samples)
        high_band = signal.lfilter(b_high, a_high, samples)
        
        def compress(signal_band, threshold=0.1, ratio=5.0):
            # Soft-Knee-ähnliche Funktion
            compressed = np.where(
                np.abs(signal_band) > threshold,
                threshold + (signal_band - threshold) / ratio,
                signal_band
            )
            return compressed
        
        low_band_comp = compress(low_band, threshold=0.1, ratio=3.5)
        mid_band_comp = compress(mid_band, threshold=0.1, ratio=4.5)
        high_band_comp = compress(high_band, threshold=0.1, ratio=4.5)
        
        # Bänder normalisieren:
        def normalize_band(band):
            peak = np.max(np.abs(band)) + 1e-8
            return band / peak
        
        low_band_comp = normalize_band(low_band_comp)
        mid_band_comp = normalize_band(mid_band_comp)
        high_band_comp = normalize_band(high_band_comp)
        
        # Zusammenmischen mit Gewichten
        combined = (
            0.5 * low_band_comp + 
            1.0 * mid_band_comp + 
            0.8 * high_band_comp
        )
        
        # Gesamtnormalisierung
        combined /= (np.max(np.abs(combined)) + 1e-8)
        combined *= np.max(np.abs(samples))
        
        return audio_segment._spawn(combined.astype(np.int16).tobytes())
    
    audio_comp = apply_multiband_compression(audio_ng)
    save_step(audio_comp, "process_compressed.wav")
    
    # 5. Equalizer (z.B. zusätzlicher High-Pass + leichte Absenkung um 400-500 Hz),
    #    sowie Anheben der Präsenz bei ~5 kHz für mehr Klarheit [1][2][7].
    #    Hier sehr simpel mit high_pass_filter() & low_pass_filter() kombiniert:
    audio_eq = high_pass_filter(audio_comp, cutoff=150)   # tiefe Frequenzen raus
    audio_eq = low_pass_filter(audio_eq, cutoff=8000)     # Ultra-Höhen kappen
    save_step(audio_eq, "process_equalized.wav")
    
    # 6. De-Esser, um harte S-/Zischlaute abzuschwächen [2].
    #    Hier sehr rudimentär mit einem Low-Pass-Filter bei ca. 7000 Hz.
    def apply_deesser(audio_segment, cutoff=7000):
        return low_pass_filter(audio_segment, cutoff=cutoff)
    
    audio_deessed = apply_deesser(audio_eq, cutoff=7000)
    save_step(audio_deessed, "process_deessed.wav")
    
    # 7. Finales Normalisieren (bzw. Limiter).
    #    Hebt das Gesamtsignal an, ohne zu übersteuern [1][2].
    audio_normalized = normalize(audio_deessed)
    save_step(audio_normalized, output_file)
    
    print("----------------------------------------")
    print("|<< Audio-Verbesserung abgeschlossen >>|")
    print("----------------------------------------")
    
    logger.info("Verarbeitung abgeschlossen.", exc_info=True)
    logger.info(f"Endgültige Datei: {output_file}", exc_info=True)
        # Lautheitsnormalisierung nach EBU R128
        #meter = Meter(44100)  # Annahme: 44.1 kHz Samplerate
        #loudness = meter.integrated_loudness(samples)
        #normalized_audio = normalize.loudness(samples, loudness, -23.0)

def resample_to_16000_mono(input_path, output_path, speed_factor):
    """Resample the audio to 24 kHz (Mono)."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}", exc_info=True)
            return

    try:
        # Load the audio file
        audio, sr = librosa.load(input_path, sr=16000, mono=True, res_type="kaiser_best")  # Load with original sampling rate
        logger.info(f"Original sampling rate: {sr} Hz", exc_info=True)

#        # Adjust playback speed if necessary
        if speed_factor != 1.0:
#            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            audio = pyrubberband.pyrb.time_stretch(audio, sr, speed_factor)
        # Resample to 16.000 Hz if needed
        target_sr = 16000
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best", fix=True, scale=True)
        else:
            audio_resampled = audio

        # Save the resampled audio
        sf.write(output_path, audio_resampled, samplerate=target_sr)
        logger.info(f"Audio successfully resampled to {target_sr} Hz (Mono): {output_path}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during resampling to 16 kHz: {e}")

def detect_speech(audio_path, only_speech_path):
    """Führt eine Sprachaktivitätserkennung (VAD) mit Silero VAD durch."""
    if os.path.exists(only_speech_path):
        if not ask_overwrite(only_speech_path):
            logger.info(f"Verwende vorhandene Datei: {only_speech_path}", exc_info=True)
            return
    try:
        print("------------------")
        print("|<< Starte VAD >>|")
        print("------------------")
        sampling_rate=SAMPLING_RATE_VAD
        logger.info("Lade Silero VAD-Modell...")
        vad_model = load_silero_vad(onnx=USE_ONNX_VAD)
        logger.info("Starte Sprachaktivitätserkennung...")
        wav = read_audio(audio_path, sampling_rate)


        # Initialize output_audio as a 2D tensor [1, samples]
        output_audio = torch.zeros((1, len(wav)), dtype=wav.dtype, device=wav.device)
        
        speech_timestamps = get_speech_timestamps(
            wav,                                # Audio-Daten
            vad_model,                          # Silero VAD-Modell
            sampling_rate=SAMPLING_RATE_VAD,    # 16.000 Hz
            min_speech_duration_ms=200,         # Minimale Sprachdauer
            min_silence_duration_ms=60,         # Minimale Stille-Dauer
            speech_pad_ms=50,                   # Padding für Sprachsegmente
            return_seconds=True,                # Rückgabe in Sekunden
            threshold=0.30                       # Schwellenwert für Sprachaktivität
        )
        # Überprüfe, ob Sprachaktivität gefunden wurde
        if not speech_timestamps:
            logger.warning("Keine Sprachaktivität gefunden. Das Audio enthält möglicherweise nur Stille oder Rauschen.")
            return []
        
        # Setze die sprachaktiven Abschnitte in das leere Audio-Array
        prev_end = 0                                        # Startzeit des vorherigen Segments
        max_silence_samples = int(2.0 * sampling_rate)      # Maximal 2 Sekunden Stille
        prev_end = 0                                        # Startzeit des vorherigen Segments
        max_silence_samples = int(2.0 * sampling_rate)      # Maximal 2 Sekunden Stille

        for segment in speech_timestamps:
            start = int((segment['start'] * sampling_rate) - 0.5)   # Sekunden in Samples umrechnen
            end = int((segment['end'] * sampling_rate) + 1.0)       # Sekunden in Samples umrechnen
    
            # Kürze die Stille zwischen den Segmenten
            silence_duration = start - prev_end             # Dauer der Stille
            if silence_duration > max_silence_samples:      # Wenn die Stille zu lang ist
                start = prev_end + max_silence_samples      # Startzeit anpassen
    
            output_audio[:, start:end] = wav[start:end]     # Kopiere den Abschnitt in das Ausgabe-Audio
            prev_end = end                                  # Speichere das Ende des aktuellen Segments
            
        # Speichere das modifizierte Audio
        torchaudio.save(only_speech_path, output_audio, sampling_rate)
        
        # Schreibe die Ergebnisse in eine JSON-Datei
        with open(SPEECH_TIMESTAMPS, "w", encoding="utf-8") as file:
            json.dump(speech_timestamps, file, ensure_ascii=False, indent=4)

        speech_timestamps_samples = [
            {"start": int(ts["start"] * SAMPLING_RATE_VAD), "end": int(ts["end"] * SAMPLING_RATE_VAD)}
            for ts in speech_timestamps
        ]
        # Speichere nur die Sprachsegmente als separate WAV-Datei
        save_audio(
            'only_speech_silero.wav',
            collect_chunks(speech_timestamps_samples, wav),
                    sampling_rate=SAMPLING_RATE_VAD
                    )

        logger.info("Sprachaktivitätserkennung abgeschlossen!")
        
        print("--------------------------")
        print("<< VAD abgeschlossen!! >>|")
        print("--------------------------")
        
        return speech_timestamps
    except Exception as e:
        logger.error(f"Fehler bei der Sprachaktivitätserkennung: {e}", exc_info=True)
        return []

def create_voice_sample(audio_path, sample_path_1, sample_path_2, sample_path_3):

    """Erstellt ein Voice-Sample aus dem verarbeiteten Audio für Stimmenklonung."""
    if os.path.exists(sample_path_1):
        choice = input("Ein Sample #1 existiert bereits. Möchten Sie ein neues erstellen? (j/n, ENTER zum Überspringen): ").strip().lower()
        if choice == "" or choice in ["n", "nein"]:
            logger.info("Verwende vorhandene sample.wav.", exc_info=True)
            return

    while True:
        start_time = input("Startzeit für das Sample #1 (in Sekunden): ")
        end_time = input("Endzeit für das Sample #1 (in Sekunden): ")
        
        if start_time == "" or end_time == "":
            logger.info("Erstellung der sample.wav übersprungen.", exc_info=True)
            continue
        
        try:
            start_seconds = float(start_time)
            end_seconds = float(end_time)
            duration = end_seconds - start_seconds
            
            if duration <= 0:
                logger.warning("Endzeit muss nach der Startzeit liegen.")
                continue
            
            ffmpeg.input(audio_path, ss=start_seconds, t=duration).output(
                sample_path_1,
                acodec='pcm_s16le',
                ac=1,
                ar=22050
            ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

        except ValueError:
            logger.error("Ungültige Eingabe. Bitte gültige Zahlen eintragen.")
            continue
        
        if os.path.exists(sample_path_2):
            choice = input("Ein Sample #2 existiert bereits. Möchten Sie ein neues erstellen? (j/n, ENTER zum Überspringen): ").strip().lower()
            if choice == "" or choice in ["n", "nein"]:
                logger.info("Verwende vorhandene sample.wav.", exc_info=True)
                continue

        while True:
            start_time = input("Startzeit für das Sample #2 (in Sekunden): ")
            end_time = input("Endzeit für das Sample #2 (in Sekunden): ")
            
            if start_time == "" or end_time == "":
                logger.info("Erstellung der sample.wav übersprungen.", exc_info=True)
                continue
            
            try:
                start_seconds = float(start_time)
                end_seconds = float(end_time)
                duration = end_seconds - start_seconds
                
                if duration <= 0:
                    logger.warning("Endzeit muss nach der Startzeit liegen.")
                    continue

                ffmpeg.input(audio_path, ss=start_seconds, t=duration).output(
                    sample_path_2,
                    acodec='pcm_s16le',
                    ac=1,
                    ar=22050
                ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

            except ValueError:
                logger.error("Ungültige Eingabe. Bitte gültige Zahlen eintragen.")
                continue

            if os.path.exists(sample_path_3):
                choice = input("Ein Sample #3 existiert bereits. Möchten Sie ein neues erstellen? (j/n, ENTER zum Überspringen): ").strip().lower()
                if choice == "" or choice in ["n", "nein"]:
                    logger.info("Verwende vorhandene sample.wav.", exc_info=True)
                    continue
                
            while True:
                start_time = input("Startzeit für das Sample #3 (in Sekunden): ")
                end_time = input("Endzeit für das Sample #3 (in Sekunden): ")
                
                if start_time == "" or end_time == "":
                    logger.info("Erstellung der sample.wav übersprungen.", exc_info=True)
                    continue
                
                try:
                    start_seconds = float(start_time)
                    end_seconds = float(end_time)
                    duration = end_seconds - start_seconds
                    
                    if duration <= 0:
                        logger.warning("Endzeit muss nach der Startzeit liegen.")
                        continue
                    
                    ffmpeg.input(audio_path, ss=start_seconds, t=duration).output(
                        sample_path_3,
                        acodec='pcm_s16le',
                        ac=1,
                        ar=22050
                    ).overwrite_output().run(capture_stdout=True, capture_stderr=True)

                    logger.info(f"Voice sample erstellt: {sample_path_1, sample_path_2, sample_path_3}", exc_info=True)
                    return
                except ValueError:
                    logger.error("Ungültige Eingabe. Bitte gültige Zahlen eintragen.")
                except ffmpeg.Error as e:
                    logger.error(f"Fehler beim Erstellen des Voice Samples: {e}")

# Transkriberen
def transcribe_audio_with_timestamps(audio_file, transcription_file):
    """Führt eine Spracherkennung mit Whisper durch und speichert die Transkription (inkl. Zeitstempel) in einer JSON-Datei."""
    if os.path.exists(transcription_file):
        if not ask_overwrite(transcription_file):
            logger.info(f"Verwende vorhandene Transkription: {transcription_file}", exc_info=True)
            return read_transcripted_csv(transcription_file)
    try:
        logger.info("Lade Whisper-Modell (large-v3)...", exc_info=True)
        logger.info("Starte Transkription...", exc_info=True)
        print("----------------------------")
        print("|<< Starte Transkription >>|")
        print("----------------------------")
        
        transcription_start_time = time.time()
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True  # Auto-Tuning für beste Performance
        torch.backends.cudnn.enabled = True    # cuDNN aktivieren
        with gpu_context():
            pipeline = load_whisper_model()   # Laden des vortrainierten Whisper-Modells
            
            # VAD-Parameter für die Sprachsegmentierung definieren
            vad_params = {
                "threshold": 0,               # Niedriger Schwellwert für empfindlichere Spracherkennung
                "min_speech_duration_ms": 0,  # Minimale Sprachdauer in Millisekunden
                #"max_speech_duration_s": 30,    # Maximale Sprachdauer in Sekunden
                "min_silence_duration_ms": 0, # Minimale Stille-Dauer zwischen Segmenten
                #"speech_pad_ms": 400            # Polsterzeit vor und nach Sprachsegmenten
            }
            
            segments, info = pipeline.transcribe(
                audio_file,
                batch_size=6,
                beam_size=10,
                patience=1.2,
                vad_filter=True,
                vad_parameters=vad_params,
                #chunk_length=15,
                #compression_ratio_threshold=2.8,    # Schwellenwert für Kompressionsrate
                #log_prob_threshold=-0.2,             # Schwellenwert für Log-Probabilität
                #no_speech_threshold=1.0,            # Schwellenwert für Stille
                #temperature=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5),      # Temperatur für Sampling
                temperature=0.5,                  # Temperatur für Sampling
                word_timestamps=True,               # Zeitstempel für Wörter
                hallucination_silence_threshold=0.2,  # Schwellenwert für Halluzinationen
                condition_on_previous_text=True,    # Bedingung an vorherigen Text
                no_repeat_ngram_size=0,
                repetition_penalty=1.0,
                #verbose=True,                       # Ausführliche Ausgabe
                language="en",                       # Englische Sprache
                #task="translate",                    # Übersetzung aktivieren
            )
                #segments = result["segments"]
            segments_list = []
            
            # Simuliere die verbose-Ausgabe
            print("\n[ Transkriptionsdetails ]")
            for segment in segments:
                start = str(timedelta(seconds=segment.start)).split('.')[0]
                end = str(timedelta(seconds=segment.end)).split('.')[0]
                text = segment.text.strip()
                
                # Text bereinigen: Entferne "..." am Ende des Textes
                text = segment.text.strip()
                text = re.sub(r'\.\.\.$', '', text).strip()  # Entferne "..." nur am Ende
                
                # Ähnlich wie verbose=True bei OpenAI Whisper
                print(f"[{start} --> {end}] {text}")
                
                adjusted_segment = {
                    "start": max(segment.start - 0, 0),
                    "end": max(segment.end + 0, 0),
                    "text": segment.text
                }
                segments_list.append(adjusted_segment)

        # CSV-Export
        transcription_file = transcription_file.replace('.json', '.csv')
        with open(transcription_file, mode='w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='|')
            csv_writer.writerow(['Startzeit', 'Endzeit', 'Text'])
            
            for segment in segments_list:
                start = str(timedelta(seconds=segment["start"])).split('.')[0]
                end = str(timedelta(seconds=segment["end"])).split('.')[0]
                csv_writer.writerow([start, end, segment["text"]])
        print("------------------------------------")
        print("|<< Transkription abgeschlossen! >>|")
        print("------------------------------------")
        logger.info("Transkription abgeschlossen!", exc_info=True)

        transcript_end_time = time.time() - transcription_start_time
        logger.info(f"Step execution time: {(transcript_end_time / 60):.0f}:{(transcript_end_time):.3f} minutes")
        print(f"{(transcript_end_time):.2f} Sekunden")
        print(f"{(transcript_end_time / 60 ):.2f} Minuten")
        print(f"{(transcript_end_time / 3600):.2f} Stunden")
        
        return segments_list

    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}", exc_info=True)
        return []

def parse_time(time_str):
    """Strikte Zeitanalyse ohne Millisekunden"""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    
    # Normalisierung: Entferne Leerzeichen und überflüssige Zeichen
    clean_str = re.sub(r"[^\d:.,]", "", time_str.strip())
    
    # Fall 1: HH:MM:SS
    if match := re.match(r"^(\d+):(\d{1,2}):(\d{1,2})$", clean_str):
        h, m, s = map(int, match.groups())
        if m >= 60 or s >= 60:
            raise ValueError(f"Ungültige Zeit: {time_str}")
        return h * 3600 + m * 60 + s
    
    # Fall 2: MM:SS
    if match := re.match(r"^(\d+):(\d{1,2})$", clean_str):
        m, s = map(int, match.groups())
        if m >= 60 or s >= 60:
            raise ValueError(f"Ungültige Zeit: {time_str}")
        return m * 60 + s
    
    # Fall 3: SS
    if clean_str.isdigit():
        return int(clean_str)
    
    return None

def process_chunk(args):
    chunk, lang, lt_config = args
    tool = lt_config(lang)
    
    try:
        # Zeitkonvertierung mit verbessertem Parsing
        chunk['start_sec'] = chunk['startzeit'].apply(
            lambda x: parse_time(x) or 0.0  # Fallback auf 0 bei Fehlern
        )
        chunk['end_sec'] = chunk['endzeit'].apply(
            lambda x: parse_time(x) or 0.0
        )
        
        # Zeitliche Konsistenzprüfung
        invalid_times = (
            (chunk['start_sec'] >= chunk['end_sec']) |
            (chunk['start_sec'] < 0) |
            (chunk['end_sec'] < 0)
        )
        if invalid_times.any():
            logger.warning(f"{invalid_times.sum()} ungültige Zeitpaare gefunden, korrigiere...")
            # Auto-Korrektur: Endzeit = Startzeit + 1s bei ungültigen Werten
            chunk.loc[invalid_times, 'end_sec'] = chunk['start_sec'] + 1.0
        
        # Restliche Verarbeitung unverändert...
        return chunk.explode('sentences')
    
    except Exception as e:
        logger.error(f"Fehler in Prozess-Chunk: {e}")
        return pd.DataFrame()

def format_time(seconds):
    """Konvertiert Sekunden in HH:MM:SS mit Validierung"""
    if seconds is None or seconds < 0:
        logger.warning("Ungültige Sekundenangabe, setze auf 0")
        return "00:00:00"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    # Overflow-Handling
    if minutes >= 60:
        hours += minutes // 60
        minutes %= 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# === SATZSEGMENTIERUNG MIT spaCy ===
nlp = spacy.blank('de')
nlp.add_pipe('sentencizer')

def split_sentences(text):
    """Text in natürliche Sätze segmentieren"""
    return [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

def split_into_sentences(text):
    """
    Teilt einen Text in Sätze auf unter Berücksichtigung gängiger Satzenden
    
    Args:
        text (str): Zu teilender Text
        
    Returns:
        list: Liste der einzelnen Sätze
    """
    # Erweiterte Regex für bessere Satzerkennung
    # Berücksichtigt gängige Satzenden (., !, ?) aber ignoriert Abkürzungen
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+'
    
    sentences = re.split(pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Fallback für Texte ohne erkennbare Satzgrenzen
    if not sentences:
        return [text]
    
    return sentences

def ask_overwrite(file_path):
    """
    Fragt den Benutzer, ob eine vorhandene Datei überschrieben werden soll

    Args:
        file_path (str): Pfad zur Datei
        
    Returns:
        bool: True wenn überschrieben werden soll, sonst False
    """
    while True:
        answer = input(f"Datei '{file_path}' existiert bereits. Überschreiben? (j/n): ").lower()
        if answer in ['j', 'ja']:
            return True
        elif answer == "" or answer in ['n', 'nein']:
            return False
        print("Bitte mit 'j' oder 'n' antworten.")

def read_translated_csv(file_path):
    """
    Liest eine bereits übersetzte CSV-Datei ein
    
    Args:
        file_path (str): Pfad zur CSV-Datei
        
    Returns:
        DataFrame: Die eingelesenen Daten
    """
    return pd.read_csv(file_path, sep='|', dtype=str)

def merge_transcript_chunks(input_file, output_file, min_dur, max_dur, max_gap, max_chars, min_words, iterations):
    """
    Führt Transkript-Segmente zusammen, ohne Text zu verwerfen.

    Args:
        input_file (str): Eingabedatei mit | als Trennzeichen.
        output_file (str): Zieldatei für Ergebnisse.
        min_dur (float): Ziel-Mindestdauer. Segmente darunter werden nach Möglichkeit zusammengeführt.
        max_dur (float): Maximale Segmentdauer in Sekunden (wird beim initialen Mergen beachtet).
        max_gap (float): Maximaler akzeptierter Zeitabstand für initiales Mergen.
        max_chars (int): Maximale Anzahl von Zeichen (wird beim initialen Mergen und Splitten beachtet).
        min_words (int): Ziel-Mindestwortzahl. Segmente darunter werden nach Möglichkeit zusammengeführt.
        iterations (int): Anzahl der Optimierungsdurchläufe (hauptsächlich für Gap/Dauer/Char-Optimierung).
    """
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {output_file}")
            return read_translated_csv(output_file)

    def force_merge_short_segments(data_list, min_dur, min_words):
        """
        Führt Segmente, die min_dur oder min_words nicht erfüllen, mit Nachbarn zusammen.
        Priorität: Kein Textverlust. Modifiziert die Liste direkt.
        Gibt True zurück, wenn Änderungen vorgenommen wurden, sonst False.
        """
        if not data_list: return False # Nichts zu tun bei leerer Liste

        merged_something = False
        i = 0
        while i < len(data_list):
            item = data_list[i]
            start_sec = parse_time(item['startzeit'])
            end_sec = parse_time(item['endzeit'])

            # Wenn Zeiten ungültig sind, kann nicht geprüft werden -> überspringen
            if start_sec is None or end_sec is None or start_sec > end_sec:
                logger.warning(f"Segment {i} hat ungültige Zeiten ({item['startzeit']} / {item['endzeit']}) - wird in force_merge übersprungen.")
                i += 1
                continue

            duration = round(end_sec - start_sec, 3)
            word_count = len(item['text'].split()) if item.get('text') else 0

            needs_merge = (duration < min_dur or word_count < min_words) and len(data_list) > 1 # Nur wenn es >1 Segment gibt

            if needs_merge:
                merged_this_iteration = False
                # Option 1: Mit Vorgänger zusammenführen (bevorzugt)
                if i > 0:
                    prev_item = data_list[i-1]
                    # Kombiniere Text (füge Leerzeichen hinzu, wenn beide Texte nicht leer sind)
                    separator = " " if prev_item.get('text', '').strip() and item.get('text', '').strip() else ""
                    merged_text = (prev_item.get('text', '') + separator + item.get('text', '')).strip()

                    # Aktualisiere Vorgänger: Startzeit bleibt, Endzeit wird vom aktuellen genommen
                    prev_item['endzeit'] = item['endzeit']
                    prev_item['text'] = merged_text
                    # Entferne aktuelles Element
                    del data_list[i]
                    # Index NICHT erhöhen, da das nächste Element jetzt an Position 'i' ist
                    merged_something = True
                    merged_this_iteration = True
                    logger.debug(f"Segment {i+1} (war {item['startzeit']}) mit Vorgänger {i} zusammengeführt.")

                # Option 2: Mit Nachfolger zusammenführen (wenn kein Vorgänger vorhanden)
                elif i < len(data_list) - 1:
                    next_item = data_list[i+1]
                    # Kombiniere Text
                    separator = " " if item.get('text', '').strip() and next_item.get('text', '').strip() else ""
                    merged_text = (item.get('text', '') + separator + next_item.get('text', '')).strip()

                    # Aktualisiere aktuelles Element: Startzeit bleibt, Endzeit wird vom Nachfolger genommen
                    item['endzeit'] = next_item['endzeit']
                    item['text'] = merged_text
                    # Entferne Nachfolger
                    del data_list[i+1]
                    # Index NICHT erhöhen, da das aktuelle Element (jetzt vergrößert) erneut geprüft werden soll
                    merged_something = True
                    merged_this_iteration = True
                    logger.debug(f"Segment {i} ({item['startzeit']}) mit Nachfolger {i+1} zusammengeführt.")

                # Wenn keine Zusammenführung möglich war (z.B. einziges Segment, das Regeln verletzt)
                if not merged_this_iteration:
                    # Dieses Segment kann nicht zusammengeführt werden, gehe zum nächsten
                    i += 1
            else:
                # Segment ist OK oder das einzige, gehe zum nächsten
                i += 1

        return merged_something # Gibt an, ob in diesem Durchlauf etwas zusammengeführt wurde

    try:
        print(f"Starte Verarbeitung von: {input_file}")
        print(f"Parameter: min_dur={min_dur}s (Ziel), max_dur={max_dur}s, max_gap={max_gap}s, max_chars={max_chars}, min_words={min_words} (Ziel), iterations={iterations}")

        df = pd.read_csv(input_file, sep='|', dtype=str)
        original_segment_count = len(df)
        print(f"Eingelesen: {original_segment_count} Segmente aus {input_file}")

        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
        required_columns = {'startzeit', 'endzeit', 'text'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Fehlende Spalten: {', '.join(missing)}")

        df['start_sec'] = df['startzeit'].apply(parse_time)
        df['end_sec'] = df['endzeit'].apply(parse_time)
        df['text'] = df['text'].astype(str).fillna('')

        invalid_mask = df['start_sec'].isna() | df['end_sec'].isna() | (df['start_sec'] > df['end_sec'])
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"Warnung: {invalid_count} Zeilen mit ungültigen/inkonsistenten Zeitangaben werden übersprungen.")
            df = df[~invalid_mask].copy()

        if df.empty:
            print("Keine gültigen Segmente nach initialer Zeitprüfung.")
            # Schreibe leere Datei
            pd.DataFrame(columns=['startzeit', 'endzeit', 'text']).to_csv(output_file, sep='|', index=False)
            return pd.DataFrame(columns=['startzeit', 'endzeit', 'text'])


        df['duration'] = df['end_sec'] - df['start_sec']
        # Keine Filterung nach Dauer hier, da wir nichts verwerfen wollen.

        df = df.sort_values('start_sec').reset_index(drop=True)
        print(f"Nach Zeitvalidierung und Sortierung: {len(df)} gültige Segmente")

        current_df = df.copy()
        current_data_list = []

        for iteration in range(iterations):
            print(f"\n--- Optimierungsdurchlauf {iteration+1}/{iterations} ---")

            if iteration > 0:
                if not current_data_list:
                    print("Keine Segmente mehr für weiteren Durchlauf vorhanden.")
                    break
                temp_df = pd.DataFrame(current_data_list)
                temp_df['start_sec'] = temp_df['startzeit'].apply(parse_time)
                temp_df['end_sec'] = temp_df['endzeit'].apply(parse_time)
                temp_df['text'] = temp_df['text'].astype(str).fillna('')

                invalid_mask = temp_df['start_sec'].isna() | temp_df['end_sec'].isna() | (temp_df['start_sec'] > temp_df['end_sec'])
                if invalid_mask.any():
                    print(f"Warnung (Durchlauf {iteration+1}): {invalid_mask.sum()} Zeilen mit ungültigen Zeiten nach vorherigem Schritt entfernt.")
                    temp_df = temp_df[~invalid_mask].copy()

                if temp_df.empty:
                    print("Keine gültigen Segmente mehr im Durchlauf.")
                    current_data_list = []
                    break

                temp_df['duration'] = temp_df['end_sec'] - temp_df['start_sec']
                current_df = temp_df.sort_values('start_sec').reset_index(drop=True)
                print(f"Für Durchlauf {iteration+1}: {len(current_df)} Segmente")

            merged_data = []
            current_chunk = None

            # Phase 1: Zeitbasiertes Zusammenführen (max_gap, max_dur, max_chars beachten)
            for _, row in current_df.iterrows():
                row_text = str(row['text']) if pd.notna(row['text']) else ''

                if current_chunk is None:
                    current_chunk = {
                        'start': row['start_sec'], 'end': row['end_sec'],
                        'text': [row_text],
                        'original_start': row['startzeit'], 'original_end': row['endzeit']
                    }
                else:
                    gap = row['start_sec'] - current_chunk['end']
                    potential_new_end = row['end_sec']
                    potential_duration = potential_new_end - current_chunk['start']
                    # Füge Leerzeichen nur hinzu, wenn beide Teile Text haben
                    separator = " " if current_chunk['text'] and current_chunk['text'][-1].strip() and row_text.strip() else ""
                    potential_text_list = current_chunk['text'] + ([row_text] if row_text else [])
                    potential_text = separator.join(potential_text_list).strip() # Korrektur: Join mit Leerzeichen?
                    potential_text = (current_chunk['text'][-1] + separator + row_text).strip() # Nur letztes + neues prüfen? Nein, Gesamttext.
                    # Korrekte Berechnung des potenziellen Gesamttextes
                    temp_texts = current_chunk['text'] + [row_text]
                    potential_full_text = ""
                    first = True
                    for txt in temp_texts:
                        clean_txt = txt.strip()
                        if clean_txt:
                            if not first: potential_full_text += " "
                            potential_full_text += clean_txt
                            first = False
                    potential_chars = len(potential_full_text)


                    # Zusammenführen, wenn Kriterien erfüllt sind
                    if (gap >= 0 and gap <= max_gap) and \
                    (potential_duration <= max_dur) and \
                    (potential_chars <= max_chars):
                        current_chunk['end'] = potential_new_end
                        # Füge Text nur hinzu, wenn er nicht leer ist
                        if row_text:
                            current_chunk['text'].append(row_text)
                        current_chunk['original_end'] = row['endzeit']
                    else:
                        # Finalisiere vorherigen Chunk
                        final_text = ""
                        first = True
                        for txt in current_chunk['text']:
                            clean_txt = txt.strip()
                            if clean_txt:
                                if not first: final_text += " "
                                final_text += clean_txt
                                first = False
                        if final_text: # Nur hinzufügen, wenn Text vorhanden
                            merged_data.append({
                                'startzeit': current_chunk['original_start'],
                                'endzeit': current_chunk['original_end'],
                                'text': final_text
                            })
                        # Beginne neuen Chunk
                        current_chunk = {
                            'start': row['start_sec'], 'end': row['end_sec'],
                            'text': [row_text] if row_text else [], # Leeren Text nicht hinzufügen
                            'original_start': row['startzeit'], 'original_end': row['endzeit']
                        }

            # Letzten Chunk hinzufügen
            if current_chunk:
                final_text = ""
                first = True
                for txt in current_chunk['text']:
                    clean_txt = txt.strip()
                    if clean_txt:
                        if not first: final_text += " "
                        final_text += clean_txt
                        first = False
                if final_text:
                    merged_data.append({
                        'startzeit': current_chunk['original_start'],
                        'endzeit': current_chunk['original_end'],
                        'text': final_text
                    })
            print(f"Nach Zeit-Zusammenführung: {len(merged_data)} Segmente")

            # Phase 2: Aufteilung zu langer Segmente (max_chars)
            # Kein Verwerfen nach min_dur oder min_words hier!
            current_data_list = []
            segmente_aufgeteilt_max_chars = 0

            for item in merged_data:
                text = item['text'].strip()
                start_time_sec = parse_time(item['startzeit'])
                end_time_sec = parse_time(item['endzeit'])

                if start_time_sec is None or end_time_sec is None:
                    logger.warning(f"Überspringe Segment wegen ungültiger Zeit in Phase 2: {item}")
                    # WICHTIG: Da wir nichts verwerfen dürfen, müssen wir es trotzdem behalten!
                    # Füge es unverändert hinzu, auch wenn die Zeit ungültig ist.
                    current_data_list.append(item)
                    continue

                duration = end_time_sec - start_time_sec

                # Wenn Text max_chars nicht überschreitet, direkt übernehmen
                if len(text) <= max_chars:
                    current_data_list.append(item)
                else:
                    # Text muss aufgeteilt werden
                    segmente_aufgeteilt_max_chars += 1
                    print(f"Info: Segment wird wg. Länge aufgeteilt: {len(text)} > {max_chars} ({item['startzeit']}-{item['endzeit']})")

                    # Versuche, an Satzgrenzen zu teilen, sonst an Wortgrenzen
                    sentences = split_into_sentences(text)
                    if not sentences: sentences = [text] # Fallback

                    new_segments_text = []
                    current_segment_text = ""

                    for part in sentences: # Teile können Sätze oder Wörter sein (bei sehr langen Sätzen)
                        part = part.strip()
                        if not part: continue

                        words_in_part = part.split()
                        temp_part_segment = "" # Temporärer Speicher für Teile des aktuellen Parts

                        for word in words_in_part:
                            # Prüfen, ob das Wort selbst > max_chars ist
                            if len(word) > max_chars:
                                # Vorheriges Segment abschließen, wenn vorhanden
                                if current_segment_text:
                                    new_segments_text.append(current_segment_text)
                                    current_segment_text = ""
                                # Langen Word-Chunk abschließen, wenn vorhanden
                                if temp_part_segment:
                                    new_segments_text.append(temp_part_segment)
                                    temp_part_segment = ""

                                logger.warning(f"Wort länger als max_chars ({len(word)} > {max_chars}) im Segment {item['startzeit']}: '{word[:30]}...' - wird als eigenes Segment behalten.")
                                new_segments_text.append(word) # Füge das lange Wort als eigenes Segment hinzu
                                continue # Nächstes Wort

                             # Prüfe, ob das Wort in den aktuellen *Teil*-Segment-Chunk passt
                            proposed_temp_part = f"{temp_part_segment} {word}".strip() if temp_part_segment else word
                            if len(proposed_temp_part) <= max_chars:
                                temp_part_segment = proposed_temp_part
                            else:
                                  # Teil-Segment-Chunk passt nicht mehr, prüfe ob er ins *Gesamt*-Segment passt
                                proposed_full_segment = f"{current_segment_text} {temp_part_segment}".strip() if current_segment_text else temp_part_segment

                                if len(proposed_full_segment) <= max_chars:
                                    # Ja, füge den Teil-Chunk zum Gesamtsegment hinzu
                                    current_segment_text = proposed_full_segment
                                    temp_part_segment = word # Beginne neuen Teil-Chunk mit aktuellem Wort
                                else:
                                    # Nein, Gesamtsegment würde zu lang.
                                    # Schließe zuerst das aktuelle Gesamtsegment ab (wenn es Inhalt hat)
                                    if current_segment_text:
                                        new_segments_text.append(current_segment_text)

                                    # Beginne neues Gesamtsegment mit dem Teil-Chunk
                                    # Falls der Teil-Chunk selbst leer ist (kann passieren?), starte mit dem Wort
                                    current_segment_text = temp_part_segment if temp_part_segment else word
                                    if not temp_part_segment and len(word) <= max_chars:
                                        # Sonderfall: Teil-Chunk war leer, Wort passt, starte neues Segment damit
                                        temp_part_segment = "" # Zurücksetzen, da das Wort jetzt in current_segment_text ist
                                    elif temp_part_segment:
                                        temp_part_segment = word # Beginne neuen Teil-Chunk

                        # Nach der Word-Schleife: Prüfe den verbleibenden Teil-Chunk
                        if temp_part_segment:
                            proposed_full_segment = f"{current_segment_text} {temp_part_segment}".strip() if current_segment_text else temp_part_segment
                            if len(proposed_full_segment) <= max_chars:
                                current_segment_text = proposed_full_segment
                            else:
                                # Passt nicht mehr, schließe altes Segment ab und starte neues
                                if current_segment_text:
                                    new_segments_text.append(current_segment_text)
                                current_segment_text = temp_part_segment

                    # Letztes zusammengesetztes Segment hinzufügen
                    if current_segment_text:
                        new_segments_text.append(current_segment_text)

                    print(f"-> Aufgeteilt in {len(new_segments_text)} Segmente")

                    # Zeitverteilung proportional zur Textlänge
                    num_new_segments = len(new_segments_text)
                    if num_new_segments == 0: continue

                    total_chars_in_split = sum(len(s) for s in new_segments_text)
                    if total_chars_in_split == 0: # Vermeide Division durch Null
                        # Verteile Zeit gleichmäßig, wenn keine Zeichen vorhanden sind (sollte selten sein)
                        segment_duration = duration / num_new_segments if num_new_segments > 0 else 0
                        current_start_time = start_time_sec
                        for i, segment_text in enumerate(new_segments_text):
                            segment_end_time = current_start_time + segment_duration
                            if i == num_new_segments - 1: segment_end_time = end_time_sec # Letztes Segment exakt
                            new_item = {
                                'startzeit': format_time(current_start_time),
                                'endzeit': format_time(segment_end_time),
                                'text': segment_text
                            }
                            current_data_list.append(new_item)
                            current_start_time = segment_end_time
                        continue # Gehe zum nächsten Originalsegment


                    current_start_time = start_time_sec
                    cumulative_duration = 0
                    for i, segment_text in enumerate(new_segments_text):
                        segment_chars = len(segment_text)
                        segment_proportion = segment_chars / total_chars_in_split
                        segment_duration = duration * segment_proportion

                        segment_end_time = current_start_time + segment_duration
                        cumulative_duration += segment_duration

                        # Korrektur für das letzte Segment, um exakte Endzeit sicherzustellen
                        # und Rundungsfehler auszugleichen
                        if i == num_new_segments - 1:
                            segment_end_time = end_time_sec
                            # Optional: leichte Anpassung der Dauer des vorletzten, wenn nötig? Eher nicht.

                        new_item = {
                            'startzeit': format_time(current_start_time),
                            'endzeit': format_time(segment_end_time),
                            'text': segment_text
                        }
                        current_data_list.append(new_item)
                        current_start_time = segment_end_time # Nutze die berechnete Endzeit als Startzeit

            print(f"Nach Längen-Aufteilung: {len(current_data_list)} Segmente")
            print(f"  Segmente aufgeteilt wg. max. Zeichen: {segmente_aufgeteilt_max_chars}")

            # Keine weitere Filterung hier, fahre mit nächster Iteration fort oder beende

        # --- Finale Phase: Erzwinge min_dur und min_words durch Zusammenführen ---
        print("\n--- Finale Bereinigung: Erzwinge Mindestdauer & Mindestwörter ---")
        force_merge_iterations = 0
        max_force_merge_iterations = len(current_data_list) # Sicherheitslimit
        while force_merge_iterations < max_force_merge_iterations:
            force_merge_iterations += 1
            print(f"Bereinigungsdurchlauf {force_merge_iterations}...")
            changed = force_merge_short_segments(current_data_list, min_dur, min_words)
            print(f"Segmente nach Durchlauf {force_merge_iterations}: {len(current_data_list)}")
            if not changed:
                print("Keine weiteren Zusammenführungen nötig.")
                break # Beenden, wenn sich nichts mehr geändert hat
        if force_merge_iterations == max_force_merge_iterations:
            print("Warnung: Maximalzahl an Bereinigungsdurchläufen erreicht. Möglicherweise konnten nicht alle Segmente bereinigt werden.")


        # --- Abschluss und Speichern ---
        if not current_data_list:
            print("\n--- Verarbeitung abgeschlossen: KEINE finalen Segmente erzeugt ---")
            final_df = pd.DataFrame(columns=['startzeit', 'endzeit', 'text'])
            final_df.to_csv(output_file, sep='|', index=False)
            print(f"Leere Ergebnisdatei {output_file} gespeichert.")
            return final_df

        result_df = pd.DataFrame(current_data_list)
        result_df = result_df[['startzeit', 'endzeit', 'text']] # Korrekte Spaltenreihenfolge
        result_df.to_csv(output_file, sep='|', index=False)

        # Abschlussbericht
        final_segment_count = len(result_df)
        print("\n--- Verarbeitungsstatistik (Final) ---")
        print(f"Originale Segmente gelesen:    {original_segment_count}")
        print(f"Gültige Segmente nach Init-Parse: {len(df) if 'df' in locals() else 'N/A'}")
        # Die Zählung aufgeteilter Segmente etc. ist komplexer geworden, da nichts verworfen wird.
        print(f"Segmente nach Iterationen:     {len(current_data_list) if current_data_list else 'N/A'}") # Vor finalem Merge
        print(f"Finale Segmente geschrieben:   {final_segment_count}")
        print(f"Ergebnis in {output_file} gespeichert")
        print("------------------------------------\n")

        return result_df

    except FileNotFoundError:
        print(f"Fehler: Eingabedatei nicht gefunden: {input_file}")
        raise
    except ValueError as e:
        print(f"Fehler bei der Datenvalidierung oder Verarbeitung: {str(e)}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {str(e)}")
        traceback.print_exc()
        raise

def restore_punctuation(input_file, output_file):
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {output_file}", exc_info=True)
            return read_translated_csv(output_file)

    """Stellt die Interpunktion mit deepmultilingualpunctuation wieder her."""
    df = pd.read_csv(input_file, sep='|', dtype=str)
    
    # Identifiziere die ursprüngliche Textspalte (ignoriere Groß/Kleinschreibung)
    text_col_original = None
    for col in df.columns:
        if col.strip().lower() == 'text':
            text_col_original = col
            break
    if text_col_original is None:
        raise ValueError("Keine Spalte 'text' (unabhängig von Groß/Kleinschreibung) in der Eingabedatei gefunden.")
    with gpu_context():
        model = PunctuationModel()
        
        # Wende das Modell an und speichere in einer NEUEN Spalte
        df['punctuated_text'] = df[text_col_original].apply(lambda x: model.restore_punctuation(x) if isinstance(x, str) and x.strip() else x)

        # Lösche die ursprüngliche Textspalte
        df = df.drop(columns=[text_col_original])

        # Benenne die neue Spalte in 'text' um (jetzt garantiert kleingeschrieben)
        df = df.rename(columns={'punctuated_text': 'text'})

        # Stelle sicher, dass die Spaltenreihenfolge sinnvoll ist (optional)
        # Z.B. ['startzeit', 'endzeit', 'text']
        cols = df.columns.tolist()
        if 'startzeit' in cols and 'endzeit' in cols and 'text' in cols:
            # Versuche, die Reihenfolge zu erzwingen
            core_cols = ['startzeit', 'endzeit', 'text']
            other_cols = [c for c in cols if c not in core_cols]
            df = df[core_cols + other_cols]
        
    df.to_csv(output_file, sep='|', index=False)
    return output_file

def correct_grammar_transcription(input_file, output_file, lang="en-US"):
    """
    Überprüft und korrigiert die Grammatik der transkribierten Segmente.
    
    Args:
        input_file (str): Pfad zur CSV-Datei mit den transkribierten Segmenten
        output_file (str): Pfad zur Ausgabedatei mit korrigierten Segmenten
        lang (str): Sprachcode für LanguageTool (z.B. "en-US", "de-DE")
    
    Returns:
        str: Pfad zur Ausgabedatei mit korrigierten Segmenten
    """
    # Datei-Existenzprüfung
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Datei: {output_file}")
            return output_file

    try:
        # LanguageTool initialisieren
        tool = language_tool_python.LanguageTool(lang)
        
        logger.info(f"Starte Grammatikkorrektur für Transkription in {lang}...")
        print("--------------------------------------")
        print("|<< Starte Grammatikkorrektur >>|")
        print("--------------------------------------")
        
        # CSV-Datei einlesen
        df = pd.read_csv(input_file, sep='|', dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        
        if 'text' not in df.columns:
            logger.error(f"Keine 'text'-Spalte in {input_file} gefunden.")
            return None
        
        # Grammatikkorrektur für jedes Segment durchführen
        total_segments = len(df)
        corrected_count = 0
        
        for i, row in tqdm(df.iterrows(), total=total_segments, desc="Korrigiere Grammatik"):
            text = row['text']
            if pd.isna(text) or not text.strip():
                continue
                
            original_text = text
            corrected_text = tool.correct(text)
            
            if original_text != corrected_text:
                corrected_count += 1
            
            df.at[i, 'text'] = corrected_text
            
        # Korrigierte Segmente in CSV-Datei speichern
        df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
        
        logger.info(f"Grammatikkorrektur abgeschlossen: {corrected_count} von {total_segments} Segmenten korrigiert.")
        print(f"Grammatikkorrektur abgeschlossen: {corrected_count} von {total_segments} Segmenten korrigiert.")
        print("--------------------------------------")
        
        return output_file
        
    except ImportError:
        logger.error("language_tool_python nicht installiert. Bitte installieren Sie es mit 'pip install language-tool-python'.")
        return input_file
    except Exception as e:
        logger.error(f"Fehler bei der Grammatikkorrektur: {e}", exc_info=True)
        return input_file

def read_transcripted_csv(file_path):
    """Liest die übersetzte CSV-Datei."""
    segments = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        next(csv_reader)
        for row in csv_reader:
            if len(row) == 3:
                start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[0].split(':'))))
                end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[1].split(':'))))
                segments.append({
                    "start": start,
                    "end": end,
                    "text": row[2]
                })
    return segments

def clean_translation(text):
    """Entfernt Bindestriche und Punkte überall im Text, nicht nur am Zeilenende."""
    # Entferne "..." oder "....." überall im Text
    text = re.sub(r'\.{2,}', '', text)  # Mehr als 3 Punkte
    # Entferne "---", "--" und einzelne Bindestriche, außer bei Wortverbindungen
    text = re.sub(r'\s*-\s*', ' ', text)  # Bindestriche mit Leerzeichen drumherum -> entfernen
    text = re.sub(r'(?<!\w)-(?=\w)', '', text)  # Bindestriche zwischen Buchstaben entfernen
    # Optional: Mehrfache Leerzeichen bereinigen
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sanitize_for_csv_and_tts(text):
    """Entfernt oder ersetzt Zeichen, die in CSV problematisch oder für TTS unleserlich sind."""
    replacements = {
        '|': '︱',   # U+FE31: Präsentationsstrich (Pipe-Ersatz)
        '"': '＂',   # U+FF02: Fullwidth Quote
        "'": '＇',   # U+FF07: Fullwidth Apostroph
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def safe_encode(text):  
    fixed = fix_encoding(text)  
    return fixed.encode('utf-8', 'replace').decode('utf-8') 

# Übersetzen
def batch_translate(batch_texts, model, tokenizer, target_lang):
    """
    Übersetzt einen Batch von Texten mit dem MADLAD400-Modell.
    
    Args:
        batch_texts: Liste von Dictionaries mit 'text'-Schlüssel.
        model: MADLAD400-Modell für die Übersetzung.
        tokenizer: Tokenizer für das MADLAD400-Modell.
        target_lang: Zielsprache-Code (z.B. 'de' für Deutsch).
    
    Returns:
        Liste von übersetzten Texten.
    """
    try:
        # Extrahiere die Texte aus dem Batch
        source_texts = []
        for item in batch_texts:
            text = item['text'].strip() if isinstance(item, dict) and 'text' in item else str(item).strip()
            # Füge Sprachpräfix für MADLAD400 hinzu (z.B. "<2de>" für Deutsch)
            text_with_prefix = f"<2{target_lang}> {text}"
            source_texts.append(text_with_prefix)
        
        # Tokenisiere die Eingabetexte
        inputs = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Verschiebe die Eingaben auf das gleiche Gerät wie das Modell
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Führe die Übersetzung durch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                num_beams=5,  # Reduziert von 8, da MADLAD400 mit weniger Beams oft gute Ergebnisse liefert
                early_stopping=True
            )
        
        # Dekodiere die Ausgaben
        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        logger.info(f"Batch mit {len(translations)} Texten erfolgreich übersetzt.")
        return translations
    
    except Exception as e:
        logger.error(f"Fehler bei der Batch-Übersetzung mit MADLAD400: {e}", exc_info=True)
        # Fallback: Leere Strings zurückgeben
        return ["" for _ in range(len(batch_texts))]

def translate_segments(transcription_file, translation_file, source_lang="en", target_lang="de"):
    """
    Übersetzt die bereits transkribierten Segmente mithilfe des MarianMT-Modells.
    
    Args:
        transcription_file: Pfad zur Datei mit den transkribierten Segmenten.
        translation_file: Pfad zur Ausgabedatei für die Übersetzungen.
        source_lang: Quellsprache (z.B. 'en').
        target_lang: Zielsprache (z.B. 'de').
    
    Returns:
        Dictionary mit den Übersetzungen.
    """
    existing_translations = {}  # Zwischenspeicher für bereits gespeicherte Übersetzungen
    end_times = {}  # Speichert Endzeiten für jeden Startpunkt
    
    # Wenn es bereits eine Übersetzungsdatei gibt, Nutzer fragen
    if os.path.exists(translation_file):
        if not ask_overwrite(translation_file):
            logger.info(f"Fortsetzen mit vorhandenen Übersetzungen: {translation_file}")
            # Lade existierende Übersetzungen
            with open(translation_file, "r", encoding="utf-8") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter='|')
                try:
                    next(csv_reader)  # Header überspringen
                    for row in csv_reader:
                        if len(row) == 3:
                            existing_translations[row[0]] = row[2]
                            end_times[row[0]] = row[1]
                except StopIteration:
                    pass  # Datei ist leer oder nur Header
            return existing_translations  # Sofortiger Exit, keine neue Übersetzung
        else:
            logger.info(f"Starte neue Übersetzung, vorhandene Datei wird überschrieben: {translation_file}")
    
    try:
        # CSV-Datei mit Transkriptionen einlesen
        segments = []
        with open(transcription_file, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='|')
            next(csv_reader)  # Header überspringen
            for row in csv_reader:
                if len(row) == 3:
                    start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[0].split(':'))))
                    end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[1].split(':'))))
                    # Speichere Endzeit für alle Segmente
                    end_times[row[0]] = row[1]
                    # Falls bereits übersetzt, überspringen
                    if row[0] in existing_translations:
                        continue
                    segments.append({"start": start, "end": end, "text": row[2], "start_str": row[0]})
        
        if not segments:
            logger.info("Keine neuen Segmente zu übersetzen!")
            return existing_translations  # Konsistente Rückgabe
        
        print(f"--------------------------")
        print(f"|<< Starte Übersetzung >>|")
        print(f"--------------------------")
        
        translate_start_time = time.time()
        
        with gpu_context():
            print(f">>> MADLAD400-Modell wird initialisiert... <<<")
            model, tokenizer = load_translation_model()
            print(f">>> MADLAD400-Modell initialisiert. Übersetzung startet... <<<")
            
            # Segmente in Batches aufteilen
            segment_batches = [segments[i:i+BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
            
            # Batches übersetzen
            for batch_idx, batch in enumerate(segment_batches):
                batch_start_time = time.time()
                print(f"Verarbeite Batch {batch_idx+1}/{len(segment_batches)} ({len(batch)} Segmente)...")

                # Batch übersetzen
                translations = batch_translate(batch, model, tokenizer, target_lang)
                
                # Übersetzungen anzeigen
                print(f"Übersetzte Texte für Batch {batch_idx+1}:")
                for i, seg in enumerate(batch):
                    print(f"Segment {i+1} (Start: {seg['start_str']}): {translations[i]}")
                    
                # Übersetzungen speichern
                for i, seg in enumerate(batch):
                    existing_translations[seg["start_str"]] = translations[i]
                batch_end_time = time.time() - batch_start_time
                print(f"Batch {batch_idx+1}/{len(segment_batches)} in {batch_end_time:.2f} Sekunden verarbeitet.")
            
        # Ergebnisse speichern
        with open(translation_file, mode='w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='|')
            csv_writer.writerow(['Startzeit', 'Endzeit', 'Text'])
            
            # Schreibe alle Übersetzungen (alte und neue)
            for start_str, text in existing_translations.items():
                if start_str in end_times:  # Sicherheitscheck
                    end_str = end_times[start_str]
                    sanitized_text = sanitize_for_csv_and_tts(text) if 'sanitize_for_csv_and_tts' in globals() else text
                    csv_writer.writerow([start_str, end_str, sanitized_text])
        
        logger.info("Übersetzung abgeschlossen!")
        print(f"-----------------------------------")
        print(f"|<< Übersetzung abgeschlossen!! >>|")
        print(f"-----------------------------------")
        
        translate_end_time = time.time() - translate_start_time
        logger.info(f"Step execution time: {(translate_end_time / 60):.0f}:{(translate_end_time):.3f} minutes")
        print(f"{(translate_end_time):.2f} Sekunden")
        print(f"{(translate_end_time / 60 ):.2f} Minuten")
        print(f"{(translate_end_time / 3600):.2f} Stunden")
        
        return existing_translations
    
    except Exception as e:
        logger.error(f"Fehler bei der Übersetzung: {e}", exc_info=True)
        return existing_translations  # Konsistente Rückgabe auch im Fehlerfall

def safe_restore_punctuation(text, model_instance):
    """ Robuste Funktion aus vorheriger Antwort, fängt Fehler ab """
    if not isinstance(text, str):
        # logger.warning(f"Überspringe Eintrag, da kein String: {type(text)} - {repr(text)}")
        return text
    if not text.strip():
        return text
    # Spezifische Behandlung für "Name: ..., dtype: object" Strings
    if 'Name:' in text and 'dtype: object' in text:
        logger.warning(f"Überspringe wahrscheinlichen Series-String: {repr(text)}")
        return "" # Oder den Original-String zurückgeben, wenn das besser ist

    try:
        return model_instance.restore_punctuation(text)
    except IndexError as e:
        logger.error(f"IndexError bei Interpunktion für: {repr(text)}", exc_info=True)
        return text # Originaltext zurückgeben
    except Exception as e:
        logger.error(f"Anderer Fehler bei Interpunktion für: {repr(text)}", exc_info=True)
        return text # Originaltext zurückgeben

def restore_punctuation_de(input_file, output_file):
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {output_file}", exc_info=True)
            return read_translated_csv(output_file)

    """Stellt die Interpunktion mit deepmultilingualpunctuation wieder her."""
    df = pd.read_csv(input_file, sep='|', dtype=str)
    
    # Mögliche Vorbereitung: NaN-Werte in leere Strings umwandeln, falls vorhanden
    df['Text'] = df['Text'].fillna('')
    # Sicherstellen, dass alles String ist (obwohl fillna('') das meist erledigt)
    df['Text'] = df['Text'].astype(str)
    with gpu_context():
        model = PunctuationModel()
        df['Text'] = df['Text'].apply(lambda x: model.restore_punctuation(x) if x.strip() else x)
    df.to_csv(output_file, sep='|', index=False)
    return output_file

def evaluate_translation_quality(source_csv_path, translated_csv_path, report_path, model_name, threshold):
    """
    Prüft die semantische Ähnlichkeit zwischen Quell- und übersetzten Segmenten
    und erstellt einen Bericht.
    """
    # Vorhandene Strukturen: Datei-Existenzprüfung
    if os.path.exists(report_path):
        if not ask_overwrite(report_path):
            logger.info(f"Verwende vorhandenen Qualitätsbericht: {report_path}")
            return report_path # Gibt den Pfad zurück, falls vorhanden und nicht überschrieben

    logger.info(f"Starte Qualitätsprüfung der Übersetzung mit Modell: {model_name}")
    print("------------------------------------------")
    print("|<< Starte Qualitätsprüfung Übersetzung >>|")
    print("------------------------------------------")

    try:
        # Daten laden
        try:
            df_source = pd.read_csv(source_csv_path, sep='|', dtype=str).fillna('')
            df_translated = pd.read_csv(translated_csv_path, sep='|', dtype=str).fillna('')
            # Spaltennamen normalisieren (klein, keine Leerzeichen)
            df_source.columns = df_source.columns.str.strip().str.lower()
            df_translated.columns = df_translated.columns.str.strip().str.lower()
        except FileNotFoundError as e:
            logger.error(f"Fehler beim Laden der CSV-Dateien für Qualitätsprüfung: {e}")
            return None
        except Exception as e:
            logger.error(f"Allgemeiner Fehler beim Lesen der CSVs: {e}")
            return None

        # Überprüfen, ob die 'text'-Spalte existiert
        if 'text' not in df_source.columns or 'text' not in df_translated.columns:
            logger.error(f"Benötigte Spalte 'text' nicht in {source_csv_path} oder {translated_csv_path} gefunden.")
            return None

        # Überprüfen, ob die Anzahl der Zeilen übereinstimmt (wichtig für 1:1 Vergleich)
        if len(df_source) != len(df_translated):
            logger.warning(f"Zeilenanzahl in Quell- ({len(df_source)}) und Übersetzungsdatei ({len(df_translated)}) stimmt nicht überein. Qualitätsprüfung übersprungen.")
            # Optional: Man könnte versuchen, über Start/Endzeiten zu matchen, ist aber komplexer.
            return None # Abbruch, da 1:1 Vergleich nicht sichergestellt

        # Sentence Transformer Modell laden
        with gpu_context():
            logger.info(f"Lade Sentence Transformer Modell: {model_name}")
            st_model = SentenceTransformer(model_name, device=device) # Nutze globale Variable 'device'

            # Embeddings berechnen
            logger.info("Berechne Embeddings für Quell- und Zielsätze...")
            source_texts = df_source['text'].tolist()
            translated_texts = df_translated['text'].tolist()

            # Handle empty lists to avoid errors with encode
            if not source_texts or not translated_texts:
                logger.warning("Eine der Textlisten (Quelle oder Übersetzung) ist leer. Qualitätsprüfung übersprungen.")
                return None

            embeddings_source = st_model.encode(source_texts, convert_to_tensor=True, show_progress_bar=True)
            embeddings_translated = st_model.encode(translated_texts, convert_to_tensor=True, show_progress_bar=True)

            # Ähnlichkeit berechnen
            logger.info("Berechne Kosinus-Ähnlichkeit...")
            similarities = []
            for i in range(len(embeddings_source)):
                # Ensure embeddings are not empty/None before calculating similarity
                if embeddings_source[i] is not None and embeddings_translated[i] is not None:
                    sim = cos_sim(embeddings_source[i], embeddings_translated[i]).item()
                    similarities.append(sim)
                else:
                    similarities.append(np.nan) # Append NaN if embeddings could not be computed

            # Ergebnisse aufbereiten
            results = []
            issues_found = 0
            for i in range(len(df_source)):
                similarity = similarities[i]
                flag = ""
                if pd.isna(similarity):
                    status = "Fehler bei Embedding"
                    flag = "CHECK MANUALLY"
                    issues_found +=1
                elif similarity < threshold:
                    status = f"Niedrig ({similarity:.3f})"
                    flag = "CHECK MANUALLY"
                    issues_found += 1
                else:
                    status = f"OK ({similarity:.3f})"

                results.append({
                    "startzeit": df_source.iloc[i].get('startzeit', 'N/A'), # Verwende get für Robustheit
                    "endzeit": df_source.iloc[i].get('endzeit', 'N/A'),
                    "quelltext": df_source['text'].iloc[i],
                    "uebersetzung": df_translated['text'].iloc[i],
                    "aehnlichkeit": similarity if not pd.isna(similarity) else 'N/A',
                    "status": status,
                    "flag": flag
                })

        # Bericht speichern
        df_report = pd.DataFrame(results)
        df_report.to_csv(report_path, sep='|', index=False, encoding='utf-8')

        logger.info(f"Qualitätsbericht gespeichert unter: {report_path}")
        logger.info(f"Anzahl potenziell problematischer Segmente (Ähnlichkeit < {threshold} oder Fehler): {issues_found}")
        print(f"Qualitätsprüfung abgeschlossen. Bericht: {report_path}")
        print(f"Gefundene Probleme: {issues_found}")
        print("-------------------------------------------")

        return report_path # Rückgabe des Pfads zum Bericht

    except Exception as e:
        logger.error(f"Fehler bei der Qualitätsprüfung der Übersetzung: {e}", exc_info=True)
        traceback.print_exc()
        return None # Gibt None zurück im Fehlerfall
    finally:
        # Speicher freigeben (optional, aber gute Praxis)
        if 'st_model' in locals():
            del st_model
        torch.cuda.empty_cache()
        logger.info("Sentence Transformer Ressourcen freigegeben.")

def read_translated_csv(file_path):
    """Liest die übersetzte CSV-Datei."""
    segments = []
    with open(file_path, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='|')
        next(csv_reader)
        for row in csv_reader:
            if len(row) == 3:
                start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[0].split(':'))))
                end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[1].split(':'))))
                segments.append({
                    "start": start,
                    "end": end,
                    "text": row[2]
                })
    return segments

def apply_denoising(audio, sr):
    """Rauschfilter für Audio-Postprocessing"""
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
    return torchaudio.functional.lowpass_biquad(
            audio_tensor,
            sr,
            cutoff_freq=7000
            ).squeeze().numpy()

def convert_time_to_seconds(time_str):
    """Konvertiert verschiedene Zeitformate in Sekunden."""
    if not time_str:
        return 0
    
    # Normalisierung des Formats (entfernt führende Nullen)
    time_str = time_str.strip()
    
    parts = time_str.split(':')
    if len(parts) == 3:  # Format: h:mm:ss
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # Format: mm:ss
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        try:
            return float(time_str)  # Falls einfach Sekunden
        except ValueError:
            return 0

def setup_gpu_optimization():
    """Konfiguriert GPU-Optimierungen für maximale Leistung."""
    # Fortgeschrittene CUDA-Optimierungen
    torch.backends.cuda.matmul.allow_tf32 = True  # Schnellere Matrix-Multiplikationen
    torch.backends.cudnn.allow_tf32 = True        # TF32 für cuDNN aktivieren
    torch.backends.cudnn.benchmark = True         # Optimale Kernel-Auswahl
    torch.backends.cudnn.deterministic = False    # Nicht-deterministische Optimierungen erlauben
    
    # Speicherzuweisung optimieren
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # Caching für CUDA-Kernels aktivieren
    return torch.cuda.Stream(priority=-1)  # Hochpriorität-Stream

def generate_semantic_embeddings(input_csv_path, output_npz_path, output_csv_path, model_name):
    """
    Generiert semantische Embeddings für Textsegmente aus einer CSV-Datei
    und speichert sie als NPZ-Datei und optional in einer neuen CSV.
    """
    # Vorhandene Strukturen: Datei-Existenzprüfung
    if os.path.exists(output_npz_path):
        if not ask_overwrite(output_npz_path):
            logger.info(f"Verwende vorhandene NPZ-Embedding-Datei: {output_npz_path}")
            # Prüfe auch CSV, wenn NPZ nicht überschrieben wird
            if os.path.exists(output_csv_path):
                if not ask_overwrite(output_csv_path):
                    logger.info(f"Verwende vorhandene CSV-Embedding-Datei: {output_csv_path}")
                    return output_npz_path, output_csv_path # Beide vorhanden und nicht überschreiben
                else:
                    # CSV überschreiben, NPZ aber nicht -> Warnung oder Abbruch? Hier: nur loggen
                    logger.warning(f"NPZ-Datei '{output_npz_path}' wird beibehalten, aber CSV '{output_csv_path}' wird neu erstellt.")
            # Wenn nur NPZ vorhanden war und nicht überschrieben wird
            return output_npz_path, None # Nur NPZ-Pfad zurückgeben, da CSV neu erstellt oder nicht vorhanden
        else:
            # NPZ wird überschrieben, prüfe CSV separat
            if os.path.exists(output_csv_path):
                if not ask_overwrite(output_csv_path):
                    logger.info(f"Behalte vorhandene CSV '{output_csv_path}', aber NPZ '{output_npz_path}' wird neu erstellt.")
                    # Führe Generierung durch, aber speichere nur NPZ
                    output_csv_path = None # Signalisiert, CSV nicht zu speichern
                #else: Beide werden überschrieben (Normalfall)


    logger.info(f"Starte Generierung semantischer Embeddings mit Modell: {model_name}")
    print("-----------------------------------------")
    print("|<< Starte Embedding-Generierung      >>|")
    print("-----------------------------------------")

    try:
        # Daten laden
        try:
            df = pd.read_csv(input_csv_path, sep='|', dtype=str).fillna('')
            df.columns = df.columns.str.strip().str.lower() # Normalisieren
        except FileNotFoundError as e:
            logger.error(f"Fehler beim Laden der CSV-Datei für Embedding-Generierung: {e}")
            return None, None
        except Exception as e:
            logger.error(f"Allgemeiner Fehler beim Lesen der CSV: {e}")
            return None, None


        if 'text' not in df.columns:
            logger.error(f"Benötigte Spalte 'text' nicht in {input_csv_path} gefunden.")
            return None, None

        texts = df['text'].tolist()
        if not texts:
            logger.warning("Keine Texte in der Eingabedatei gefunden. Embedding-Generierung übersprungen.")
            # Leere Dateien erzeugen? Oder None zurückgeben? Hier: None
            return None, None

        # Sentence Transformer Modell laden
        with gpu_context():
            logger.info(f"Lade Sentence Transformer Modell: {model_name}")
            st_model = SentenceTransformer(model_name, device=device)

            # Embeddings berechnen
            logger.info(f"Berechne Embeddings für {len(texts)} Segmente...")
            embeddings = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        # Embeddings als NPZ speichern
        logger.info(f"Speichere Embeddings in NPZ-Datei: {output_npz_path}")
        # Speichere Metadaten zusammen mit den Embeddings
        metadata = {
            'model_name': model_name,
            'source_file': input_csv_path,
            'original_indices': df.index.to_numpy() # Um die Reihenfolge zu bewahren
        }
        # Erstelle ein Dictionary für np.savez
        save_dict = {'embeddings': embeddings, **metadata}
        # Füge optional Originaldaten hinzu (z.B. Start/Endzeiten)
        if 'startzeit' in df.columns: save_dict['startzeit'] = df['startzeit'].to_numpy()
        if 'endzeit' in df.columns: save_dict['endzeit'] = df['endzeit'].to_numpy()

        np.savez_compressed(output_npz_path, **save_dict) # Komprimiert speichern
        logger.info(f"NPZ-Datei gespeichert.")


        # Optional: Embeddings zur CSV hinzufügen und speichern
        if output_csv_path:
            logger.info(f"Füge Embeddings zur CSV hinzu und speichere als: {output_csv_path}")
            # Konvertiere Embeddings in eine speicherbare Form (z.B. String-Liste)
            df['embedding'] = [str(list(emb)) for emb in embeddings]
            try:
                df.to_csv(output_csv_path, sep='|', index=False, encoding='utf-8')
                logger.info(f"CSV mit Embeddings gespeichert.")
            except Exception as e:
                logger.error(f"Fehler beim Speichern der CSV mit Embeddings: {e}")
                output_csv_path = None # Signalisiert, dass CSV-Speichern fehlschlug

        print("-----------------------------------------")
        print("|<< Embedding-Generierung abgeschlossen >>|")
        print("-----------------------------------------")

        return output_npz_path, output_csv_path # Gibt Pfade zurück

    except Exception as e:
        logger.error(f"Fehler bei der Embedding-Generierung: {e}", exc_info=True)
        traceback.print_exc()
        return None, None # Gibt None zurück im Fehlerfall
    finally:
        # Speicher freigeben
        if 'st_model' in locals():
            del st_model
        torch.cuda.empty_cache()
        logger.info("Sentence Transformer Ressourcen freigegeben.")

def format_translation_for_tts(input_file, output_file, lang="de-DE", use_embeddings=False, embeddings_file=None, lt_path="D:\\LanguageTool-6.6"):
    """
    Optimiert übersetzte Segmente für TTS, indem Grammatik korrigiert und der Text in Abschnitte
    mit maximal 175 Zeichen pro Zeitstempel aufgeteilt wird, es sei denn, das neue Segment hätte
    weniger als 3 Wörter. Implementiert eine offlinefähige Namenerkennung mit spaCy.
    
    Args:
        input_file (str): Pfad zur CSV-Datei mit den übersetzten Segmenten.
        output_file (str): Pfad zur Ausgabedatei mit formatierten Segmenten.
        lang (str): Sprachcode für LanguageTool (z.B. "de-DE").
        use_embeddings (bool): Ob semantische Embeddings für die Analyse verwendet werden sollen.
        embeddings_file (str): Pfad zur NPZ-Datei mit Embeddings.
        lt_path (str): Pfad zum lokalen LanguageTool-Verzeichnis (Standard: "D:\\LanguageTool-6.6").
    
    Returns:
        str: Pfad zur Ausgabedatei mit formatierten Segmenten.
    """
    # Datei-Existenzprüfung
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Datei: {output_file}")
            return output_file

    try:
        logger.info(f"Starte TTS-Formatierung für {lang}...")
        print("----------------------------------")
        print("|<< Starte TTS-Formatierung >>|")
        print("----------------------------------")
        
        # LanguageTool-Server starten aus dem angegebenen Pfad
        import subprocess
        import time
        lt_jar_path = os.path.join(lt_path, "languagetool-server.jar")
        if not os.path.exists(lt_jar_path):
            logger.error(f"LanguageTool-Server JAR-Datei nicht gefunden unter: {lt_jar_path}")
            return input_file
        
        port = 8010  # Standardport für LanguageTool
        lt_process = subprocess.Popen(
            ["java", "-cp", lt_jar_path, "org.languagetool.server.HTTPServer", "--port", str(port)],
            cwd=lt_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"LanguageTool-Server gestartet aus {lt_path} auf Port {port}.")
        time.sleep(5)  # Warten, bis der Server hochgefahren ist
        
        # Umgebungsvariablen für LanguageTool setzen
        os.environ["LANGUAGE_TOOL_HOST"] = "localhost"
        os.environ["LANGUAGE_TOOL_PORT"] = str(port)
        
        # LanguageTool initialisieren
        tool = language_tool_python.LanguageTool(lang)
        logger.info("LanguageTool mit lokalem Server verbunden.")
        
        # CSV-Datei einlesen
        df = pd.read_csv(input_file, sep='|', dtype=str)
        df.columns = df.columns.str.strip().str.lower()
        
        # Prüfen, ob erforderliche Spalten vorhanden sind
        required_cols = ['text', 'startzeit', 'endzeit']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Fehlende Spalten in {input_file}: {', '.join(missing_cols)}")
            tool.close()
            lt_process.terminate()
            return None
        
        # Embeddings laden, falls gewünscht (optional für semantische Analyse)
        embeddings = None
        if use_embeddings and embeddings_file and os.path.exists(embeddings_file):
            try:
                logger.info(f"Lade semantische Embeddings für Satzanalyse...")
                embeddings_data = np.load(embeddings_file)
                embeddings = embeddings_data['embeddings']
            except Exception as e:
                logger.error(f"Fehler beim Laden der Embeddings: {e}")
                use_embeddings = False
        
        # SpaCy-Modell für Namenerkennung laden (offline nutzbar)
        try:
            nlp = spacy.load("de_core_news_sm")  # Kleines deutsches Modell
            logger.info("SpaCy-Modell für Namenerkennung geladen.")
        except OSError:
            logger.error("SpaCy-Modell 'de_core_news_sm' nicht gefunden. Bitte installieren Sie es mit 'python -m spacy download de_core_news_sm'.")
            tool.close()
            lt_process.terminate()
            return input_file
        
        # Hilfsfunktion zur Namenerkennung mit spaCy
        def protect_names(text):
            """
            Erkennt Namen im Text mit spaCy NER und schützt sie vor Korrektur durch LanguageTool,
            indem sie temporär maskiert werden.
            """
            protected_names = {}
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PER":  # Nur Personen-Namen (PER) maskieren
                    placeholder = f"__NAME_{ent.start_char}_{ent.end_char}__"
                    protected_names[placeholder] = ent.text
            modified_text = text
            for placeholder, name in protected_names.items():
                modified_text = modified_text.replace(name, placeholder)
            return modified_text, protected_names
        
        def restore_names(text, protected_names):
            """Stellt die geschützten Namen im Text wieder her."""
            for placeholder, name in protected_names.items():
                text = text.replace(placeholder, name)
            return text
        
        # Korrigierte Hilfsfunktion zum Aufteilen des Textes mit 3-Wörter-Regel
        def split_text_by_char_limit(text, char_limit=175):
            """
            Teilt den Text in Abschnitte mit maximal 'char_limit' Zeichen auf.
            Versucht, an Satzgrenzen zu teilen, falls möglich, sonst an Wortgrenzen.
            Ignoriert das Limit, wenn das neue Segment weniger als 3 Wörter hätte.
            """
            if len(text) <= char_limit:
                return [text]
            
            sentences = split_into_sentences(text)
            if len(sentences) == 1 and len(text) > char_limit:
                # Wenn nur ein Satz, aber zu lang, teile an Wortgrenzen
                words = text.split()
                chunks = []
                current_chunk = ""
                i = 0
                
                while i < len(words):
                    word = words[i]
                    # Teste, ob das aktuelle Wort noch in den Chunk passt
                    test_chunk = current_chunk + (" " + word if current_chunk else word)
                    
                    if len(test_chunk) <= char_limit:
                        # Wort passt in den aktuellen Chunk
                        current_chunk = test_chunk
                        i += 1
                    else:
                        # Wort passt nicht mehr, prüfe verbleibende Wörter
                        remaining_words = words[i:]
                        
                        if len(remaining_words) < 3:
                            # Weniger als 3 Wörter übrig, füge sie zum aktuellen Chunk hinzu (Regel ignorieren)
                            current_chunk += " " + " ".join(remaining_words) if current_chunk else " ".join(remaining_words)
                            chunks.append(current_chunk)
                            break  # Alle Wörter verarbeitet
                        else:
                            # Genug Wörter für ein neues Segment, speichere aktuellen Chunk
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = word
                            i += 1
                
                # Letzten Chunk hinzufügen, falls noch Inhalt vorhanden
                if current_chunk and current_chunk not in chunks:
                    chunks.append(current_chunk)
                    
                return chunks
            else:
                # Teile an Satzgrenzen, wenn möglich, und kombiniere Sätze unter char_limit
                chunks = []
                current_chunk = ""
                
                for j, sentence in enumerate(sentences):
                    test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
                    
                    if len(test_chunk) <= char_limit:
                        # Satz passt in den aktuellen Chunk
                        current_chunk = test_chunk
                    else:
                        # Satz passt nicht mehr, prüfe verbleibende Sätze
                        remaining_sentences = sentences[j:]
                        remaining_text = " ".join(remaining_sentences)
                        
                        if len(remaining_text.split()) < 3:
                            # Weniger als 3 Wörter in verbleibenden Sätzen, füge sie hinzu
                            current_chunk = test_chunk
                        else:
                            # Genug Wörter für neue Segmente, speichere aktuellen Chunk
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                
                # Letzten Chunk hinzufügen
                if current_chunk:
                    chunks.append(current_chunk)
                    
                return chunks
        
        # Liste für formatierte Segmente
        formatted_segments = []
        total_segments = len(df)
        split_count = 0
        segments_exceeding_limit = 0  # Zähler für Segmente, die das Limit überschreiten aber nicht geteilt werden
        
        for i, row in tqdm(df.iterrows(), total=total_segments, desc="Formatiere Segmente"):
            # Zeiten parsen
            start_time = parse_time(row['startzeit'])
            end_time = parse_time(row['endzeit'])
            
            if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
                logger.warning(f"Ungültige Zeiten in Segment {i}: {row['startzeit']} - {row['endzeit']}")
                continue
                
            text = row['text'].strip()
            
            if not text:
                logger.debug(f"Leeres Segment {i}, wird übersprungen.")
                continue
                
            # 1. Namen schützen vor Grammatikkorrektur
            protected_text, protected_names = protect_names(text)
            
            # 2. Grammatikkorrektur durchführen
            corrected_text = tool.correct(protected_text)
            
            # 3. Geschützte Namen wiederherstellen
            corrected_text = restore_names(corrected_text, protected_names)
            
            # 4. Text in Abschnitte mit maximal 175 Zeichen aufteilen
            text_chunks = split_text_by_char_limit(corrected_text, char_limit=175)
            
            # Prüfe, ob Segmente das Limit überschreiten (für Statistiken)
            for chunk in text_chunks:
                if len(chunk) > 175:
                    segments_exceeding_limit += 1
            
            if len(text_chunks) == 1:
                # Wenn nur ein Abschnitt, direkt hinzufügen
                formatted_segments.append({
                    'startzeit': row['startzeit'],
                    'endzeit': row['endzeit'],
                    'text': sanitize_for_csv_and_tts(text_chunks[0])
                })
            else:
                # Mehrere Abschnitte, Zeit proportional aufteilen
                split_count += 1
                total_chars = sum(len(chunk) for chunk in text_chunks)
                segment_duration = end_time - start_time
                current_start = start_time
                
                for j, chunk in enumerate(text_chunks):
                    # Proportionale Zeitaufteilung basierend auf Zeichenanzahl
                    char_proportion = len(chunk) / total_chars
                    this_duration = segment_duration * char_proportion
                    
                    # Letztes Segment bekommt exakt die Endzeit
                    segment_end = end_time if j == len(text_chunks) - 1 else current_start + this_duration
                    
                    formatted_segments.append({
                        'startzeit': format_time(current_start),
                        'endzeit': format_time(segment_end),
                        'text': sanitize_for_csv_and_tts(chunk)
                    })
                    
                    current_start = segment_end
        
        # Ergebnisse in DataFrame konvertieren und speichern
        if formatted_segments:
            result_df = pd.DataFrame(formatted_segments)
            result_df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
            
            logger.info(f"TTS-Formatierung abgeschlossen: {len(formatted_segments)} Segmente erzeugt.")
            print(f"Ergebnis: {len(formatted_segments)} formatierte Segmente")
            print(f"  - {split_count} Segmente wurden aufgrund der Zeichenbegrenzung aufgeteilt")
            print(f"  - {segments_exceeding_limit} Segmente überschreiten das 175-Zeichen-Limit (3-Wörter-Regel angewendet)")
            print("----------------------------------")
            
            tool.close()
            lt_process.terminate()
            logger.info("LanguageTool-Server beendet.")
            return output_file
        else:
            logger.warning("Keine formatierten Segmente erstellt.")
            tool.close()
            lt_process.terminate()
            logger.info("LanguageTool-Server beendet.")
            return input_file
            
    except ImportError:
        logger.error("language_tool_python nicht installiert. Bitte installieren Sie es mit 'pip install language-tool-python'.")
        return input_file
    except Exception as e:
        logger.error(f"Fehler bei der TTS-Formatierung: {e}", exc_info=True)
        traceback.print_exc()
        if 'tool' in locals():
            tool.close()
        if 'lt_process' in locals():
            lt_process.terminate()
            logger.info("LanguageTool-Server beendet (nach Fehler).")
        return input_file

# Synthetisieren
def text_to_speech_with_voice_cloning(
    translation_file,
    sample_path_1,
    sample_path_2,
    sample_path_3,
    #sample_path_4,
    #sample_path_5,
    output_path,
    batch_size=24
):
    """
    Optimiert Text-to-Speech mit Voice Cloning und verschiedenen Beschleunigungen.

    Args:
        translation_file: Pfad zur CSV-Datei mit übersetzten Texten.
        sample_path_1, sample_path_2: Pfade zu Sprachbeispielen für die Stimmenklonung.
        output_path: Ausgabepfad für die generierte Audiodatei.
        batch_size: Größe des Batches für parallele Verarbeitung.
    """
    # Speicher freigeben, um Platz für das TTS-Modell zu schaffen
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Überprüfen, ob die Ausgabedatei bereits existiert und ob sie überschrieben werden soll
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"TTS-Audio bereits vorhanden: {output_path}")
            return
    
    try:
        print(f"------------------")
        print(f"|<< Starte TTS >>|")
        print(f"------------------")

        tts_start_time = time.time()

        # GPU-Optimierungen aktivieren für maximale Leistung
        cuda_stream = setup_gpu_optimization()
        
        # TTS-Modell laden und auf GPU verschieben
        with gpu_context():
            print(f"TTS-Modell wird initialisiert...")
            base_model = load_xtts_v2()
            
            # Bedingungen für die Stimme (Latent und Embedding) aus den Sprachbeispielen generieren
            with torch.cuda.stream(cuda_stream), torch.inference_mode():
                sample_paths = [
                    sample_path_1,
                    sample_path_2,
                    sample_path_3,
                    #sample_path_4,
                    #sample_path_5
                ]
                gpt_cond_latent, speaker_embedding = base_model.get_conditioning_latents(
                    sound_norm_refs=False,
                    audio_path=sample_paths,
                    load_sr=22050,
                )

            # DeepSpeed Inference für TTS initialisieren
            print(f"Initialisiere DeepSpeed Inference für TTS...")
            ds_config = {
                "enable_cuda_graph": False,  # CUDA Graphs sind bei variabler Inputlänge schwierig
                "dtype": torch.float32,      # Float32 für Kompatibilität
                "replace_with_kernel_inject": True  # Optimierung für Inferenz
            }
            ds_engine = deepspeed.init_inference(
                model=base_model,
                tensor_parallel={"tp_size": 1},  # Tensor-Parallelismus auf 1 setzen
                dtype=torch.float32,
                replace_with_kernel_inject=True,
            )
            optimized_tts_model = ds_engine.module
            logger.info("DeepSpeed Inference für TTS initialisiert.")
            
            # Basismodell freigeben, um Speicher zu sparen
            del base_model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # CSV-Datei mit übersetzten Texten einlesen
            all_texts = []
            timestamps = []
            with open(translation_file, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f, delimiter="|")
                next(reader)  # Header überspringen
                for row in reader:
                    if len(row) < 3:
                        continue
                    # Start- und Endzeit in Sekunden umwandeln
                    start = convert_time_to_seconds(row[0])
                    end = convert_time_to_seconds(row[1])
                    text = row[2].strip()
                    all_texts.append(text)
                    timestamps.append((start, end))
            
            # Batches für die parallele Verarbeitung erstellen
            batches = [all_texts[i:i+batch_size] for i in range(0, len(all_texts), batch_size)]
            timestamp_batches = [timestamps[i:i+batch_size] for i in range(0, len(timestamps), batch_size)]
            
            # Maximale Audiolänge schätzen für effiziente Vorallokation
            sampling_rate = 24000
            max_length_seconds = timestamps[-1][1] if timestamps else 0
            max_audio_length = int(max_length_seconds * sampling_rate) + 100000  # Sicherheitspuffer
            
            # Audio-Array vorallozieren für effiziente Zusammenführung
            final_audio = np.zeros(max_audio_length, dtype=np.float32)
            current_position_samples = 0
            
            # Batch-weise TTS durchführen
            for batch_idx, (text_batch, time_batch) in enumerate(zip(batches, timestamp_batches)):
                batch_start_time = time.time()
                print(f"Verarbeite Batch {batch_idx+1}/{len(batches)}...")
                
                batch_results = []
                # Inferenz mit optimiertem Modell durchführen
                with torch.cuda.stream(cuda_stream), torch.inference_mode():
                    for text in text_batch:
                        result = optimized_tts_model.inference(
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            text=text,
                            language="de",  # Sprache auf Deutsch setzen, passend zu MarianMT-Übersetzung
                            speed=0.85,  # Geschwindigkeit der Stimme anpassen
                            temperature=0.9,  # Temperatur für Zufälligkeit der Ausgabe
                            repetition_penalty=10.0,  # Strafe für Wiederholungen
                            enable_text_splitting=False,  # Textaufteilung deaktivieren
                            top_k=65,  # Top-K-Sampling für Qualität
                            top_p=0.95  # Top-P-Sampling für Qualität
                        )
                        batch_results.append(result)
            
                # Audioergebnisse in das finale Array einfügen
                for i, (result, (start, end)) in enumerate(zip(batch_results, time_batch)):
                    audio_clip = result.get("wav", np.zeros(1000, dtype=np.float32))
                    audio_clip = np.array(audio_clip, dtype=np.float32).squeeze()
                    
                    # Startposition in Samples berechnen
                    start_pos_samples = int(start * sampling_rate)
                    
                    # Sicherstellen, dass wir nicht rückwärts gehen
                    if start_pos_samples < current_position_samples:
                        start_pos_samples = current_position_samples
                    
                    # Audio an der richtigen Position einfügen
                    end_pos_samples = start_pos_samples + len(audio_clip)
                    if end_pos_samples > len(final_audio):
                        # Array bei Bedarf vergrößern
                        final_audio = np.pad(final_audio, (0, end_pos_samples - len(final_audio)), 'constant')
                    
                    final_audio[start_pos_samples:end_pos_samples] = audio_clip
                    
                    # Position aktualisieren
                    current_position_samples = end_pos_samples

                # Finales Audio auf tatsächlich verwendete Länge trimmen
                final_audio = final_audio[:current_position_samples]
                batch_end_time = time.time() - batch_start_time
                print(f"Batch {batch_idx+1}/{len(batches)} in {batch_end_time:.2f} Sekunden verarbeitet.")
        
            # Audio-Nachbearbeitung, falls kein Audio generiert wurde
            if len(final_audio) == 0:
                print("Kein Audio - Datei leer!")
                final_audio = np.zeros((1, 1000), dtype=np.float32)
                
        # Globale Normalisierung des gesamten Audios
        final_audio /= np.max(np.abs(final_audio)) + 1e-8  # Einheitliche Lautstärke
        final_audio = final_audio.astype(np.float32)  # In float32 konvertieren
        
        # Für torchaudio.save formatieren
        if final_audio.ndim == 1:
            final_audio = final_audio.reshape(1, -1)
        
        # Audio speichern
        torchaudio.save(output_path, torch.from_numpy(final_audio), sampling_rate)
        
        print(f"---------------------------")
        print(f"|<< TTS abgeschlossen!! >>|")
        print(f"---------------------------")

        tts_end_time = time.time() - tts_start_time
        logger.info(f"Step execution time: {(tts_end_time / 60):.0f}:{(tts_end_time):.3f} minutes")
        print(f"{(tts_end_time):.2f} Sekunden")
        print(f"{(tts_end_time / 60 ):.2f} Minuten")
        print(f"{(tts_end_time / 3600):.2f} Stunden")

        logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}")
    except Exception as e:
        logger.error(f"Fehler bei der TTS-Synthese: {str(e)}")
        raise

class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

def resample_to_44100_stereo(input_path, output_path, speed_factor):
    """
    Resample das Audio auf 44.100 Hz (Stereo), passe die Wiedergabegeschwindigkeit sowie die Lautstärke an.
    """
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}", exc_info=True)
            return

    try:
        print("-------------------------")
        print("|<< Starte ReSampling >>|")
        print("-------------------------")

        # Extrahiere die Originalsampling-Rate mit FFprobe für die Protokollierung
        try:
            probe_command = [
                "ffprobe", 
                "-v", "quiet", 
                "-show_entries", "stream=sample_rate", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                input_path
            ]
            original_sr = subprocess.check_output(probe_command).decode().strip()
            logger.info(f"Original-Samplingrate: {original_sr} Hz", exc_info=True)
        except Exception as e:
            logger.warning(f"Konnte Original-Samplingrate nicht ermitteln: {e}")
            original_sr = "unbekannt"
        
        # Erstelle den FFmpeg-Befehl mit allen Parametern
        # 1. Setze atempo-Filter für Geschwindigkeitsanpassung
        atempo_filter = create_atempo_filter_string(speed_factor)
        
        # 2. Bereite Audiofilter vor (Resample auf 44.100 Hz, Stereo-Konvertierung, Geschwindigkeit, Lautstärke)
        
        # Vollständige Filterkette erstellen
        filter_complex = (
            f"aresample=44100:resampler=soxr:precision=28," +  # Hochwertiges Resampling auf 44.100 Hz
            f"aformat=sample_fmts=s16:channel_layouts=stereo," +  # Ausgabeformat festlegen
            f"{atempo_filter},"  # Geschwindigkeitsanpassung
        )
        
        # FFmpeg-Befehl zusammenstellen
        command = [
            "ffmpeg",
            "-i", input_path,
            "-filter:a", filter_complex,
            "-y",  # Ausgabedatei überschreiben
            output_path
        ]
        
        # Befehl ausführen
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg-Fehler: {stderr.decode()}")
        
        logger.info("\n".join([
            f"Audio auf 44100 Hz (Stereo) resampled:",
            f"- Geschwindigkeitsfaktor: {speed_factor}",
            f"- Datei: {output_path}"
            ]), exc_info=True)
        print("--------------------------")
        print("|<< ReSampling beendet >>|")
        print("--------------------------")
    except Exception as e:
        logger.error(f"Fehler beim Resampling auf 44.100 Hz: {e}")

def create_atempo_filter_string(speed_factor):
    """
    Erstellt eine FFmpeg-Filterkette für die Geschwindigkeitsanpassung.
    Der atempo-Filter unterstützt nur Faktoren zwischen 0.5 und 2.0,
    daher müssen wir für extreme Werte mehrere Filter verketten.
    
    Args:
        speed_factor (float): Geschwindigkeitsfaktor
        
    Returns:
        str: FFmpeg-Filterkette für atempo
    """
    if 0.5 <= speed_factor <= 2.0:
        return f"atempo={speed_factor}"
    
    # Für Werte außerhalb des Bereichs verketten wir mehrere atempo-Filter
    atempo_chain = []
    remaining_factor = speed_factor
    
    # Für extreme Verlangsamung
    if remaining_factor < 0.5:
        while remaining_factor < 0.5:
            atempo_chain.append("atempo=0.5")
            remaining_factor /= 0.5
    
    # Für extreme Beschleunigung
    while remaining_factor > 2.0:
        atempo_chain.append("atempo=2.0")
        remaining_factor /= 2.0
    
    # Restfaktor hinzufügen
    if 0.5 <= remaining_factor <= 2.0:
        atempo_chain.append(f"atempo={remaining_factor}")
    
    return ",".join(atempo_chain)

def adjust_playback_speed(video_path, adjusted_video_path, speed_factor):
    """Passt die Wiedergabegeschwindigkeit des Originalvideos an und nutzt einen separaten Lautstärkefaktor für das Video."""
    if os.path.exists(adjusted_video_path):
        if not ask_overwrite(adjusted_video_path):
            logger.info(f"Verwende vorhandene Datei: {adjusted_video_path}", exc_info=True)
            return
    try:
        print("--------------------------------")
        print("|<< Videoanpassung gestartet >>|")
        print("--------------------------------")
        
        video_speed = 1 / speed_factor
        audio_speed = speed_factor
        
        temp_output = adjusted_video_path + ".temp.mp4"
        with gpu_context():
            (
            ffmpeg
            .input(video_path, hwaccel="cuda")
            .output(temp_output, vcodec="h264_nvenc", acodec="copy")
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            print("-----------------------------")
            print("|<< Teil 1 von 2 erledigt >>|")
            print("-----------------------------")
            (
            ffmpeg
            .input(temp_output)
            .output(
                    adjusted_video_path,
                    vf=f"setpts={video_speed}*PTS",
                    af=f"atempo={audio_speed}",
                    vcodec="h264_nvenc",
                    **{"max_muxing_queue_size": "1024"}
                    )
            .run(
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True
                )
            )
        
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print("------------------------------")
        print("|<< Videoanpassung beendet >>|")
        print("------------------------------")
        
        logger.info(
            f"Videogeschwindigkeit angepasst (Faktor={speed_factor}): {adjusted_video_path} ",
            exc_info=True
            #f"und Lautstärke={VOLUME_ADJUSTMENT_VIDEO}"
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "Keine stderr-Details verfügbar"
        logger.error(f"Fehler bei der Anpassung der Wiedergabegeschwindigkeit: {e}")
        logger.error(f"FFmpeg stderr-Ausgabe: {stderr}")

def combine_video_audio_ffmpeg(adjusted_video_path, translated_audio_path, final_video_path):
    """
    Kombiniert das angepasste Video mit dem neu erstellten Audio (z.B. TTS).
    Dabei wird zusätzlich ein Mixdown durchgeführt, wo beide Audiospuren (Video-Audio mit geringer Lautstärke und TTS-Audio) gemischt werden.
    """
    if not os.path.exists(adjusted_video_path):
        logger.error("Eingabevideo für das Kombinieren nicht gefunden.")
        return
    if not os.path.exists(translated_audio_path):
        logger.error("Übersetzte Audiodatei nicht gefunden.")
        return
    if os.path.exists(final_video_path):
        if not ask_overwrite(final_video_path):
            logger.info(f"Verwende vorhandene Datei: {final_video_path}", exc_info=True)
            return
    try:
        
        print("----------------------------------")
        print("|<< Starte Video-Audio-Mixdown >>|")
        print("----------------------------------")

        mix_start_time = time.time()
        
        filter_complex = (
            f"[0:a]volume={VOLUME_ADJUSTMENT_VIDEO}[a1];"  # Reduziere die Lautstärke des Originalvideos
            f"[1:a]volume={VOLUME_ADJUSTMENT_44100}[a2];"  # Halte die Lautstärke des TTS-Audios konstant
            "[a1][a2]amix=inputs=2:duration=longest"
        )
        
        video_path = ffmpeg.input(
                    adjusted_video_path,
                    hwaccel="cuda"
                    )
        audio_path = ffmpeg.input(
                    translated_audio_path,
                    hwaccel="cuda"
                    )
        with gpu_context():
            (
            ffmpeg
            .output(
                video_path.video,
                audio_path.audio,
                final_video_path,
                vcodec="h264_nvenc",
                acodec="aac",
                strict="experimental",
                filter_complex=filter_complex,
                map="0:v",
                map_metadata="-1"
                )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
            )
        print("-----------------------------------")
        print("|<< Video-Audio-Mixdown beendet >>|")
        print("-----------------------------------")
        logger.info(f"Finales Video erstellt und gemischt: {final_video_path}", exc_info=True)

        mix_end_time = time.time() - mix_start_time
        logger.info(f"Step execution time: {(mix_end_time / 60):.0f}:{(mix_end_time):.3f} minutes")
        print(f"{(mix_end_time):.2f} Sekunden")
        print(f"{(mix_end_time / 60 ):.2f} Minuten")
        print(f"{(mix_end_time / 3600):.2f} Stunden")

    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if hasattr(e, 'stderr') and e.stderr else "Keine stderr-Details verfügbar"
        logger.error(f"Fehler beim Kombinieren von Video und Audio: {e}")
        logger.error(f"FFmpeg stderr-Ausgabe: {stderr}")

# ==============================
# Hauptprogramm
# ==============================

def main():
    """
    Orchestriert alle Schritte, um das Video zu übersetzen, TTS-Audio zu erzeugen und schließlich ein fertiges Video zu erstellen.
    """
    # 1) Eingabevideo prüfen
    if not os.path.exists(VIDEO_PATH):
        logger.error(f"Eingabevideo nicht gefunden: {VIDEO_PATH}")
        return

    # 2) Audiospur aus dem Video extrahieren (Mono, 44.1 kHz)
    extract_audio_ffmpeg(VIDEO_PATH, ORIGINAL_AUDIO_PATH)

    # 3) Audioverarbeitung (Rauschunterdrückung und Lautheitsnormalisierung)
#    process_audio(ORIGINAL_AUDIO_PATH, PROCESSED_AUDIO_PATH)

    # 4) Audio resamplen auf 16 kHz, Mono (für TTS)
#    resample_to_16000_mono(PROCESSED_AUDIO_PATH, PROCESSED_AUDIO_PATH_SPEED, SPEED_FACTOR_RESAMPLE_16000)

    # 4.1) Spracherkennung (VAD) mit Silero VAD
#    detect_speech(PROCESSED_AUDIO_PATH_SPEED, ONLY_SPEECH)
    
    # 5) Optional: Erstellung eines Voice-Samples für die Stimmenklonung
    create_voice_sample(ORIGINAL_AUDIO_PATH, SAMPLE_PATH_1, SAMPLE_PATH_2, SAMPLE_PATH_3)
    
    # 6) Spracherkennung (Transkription) mit Whisper
    
    segments = transcribe_audio_with_timestamps(ORIGINAL_AUDIO_PATH, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return
    
    # 6.1) Wiederherstellung der Interpunktion
    restore_punctuation(TRANSCRIPTION_FILE, PUNCTED_TRANSCRIPTION_FILE)

    # 6.2) Grammatische Korrektur der Transkription
    correct_grammar_transcription(PUNCTED_TRANSCRIPTION_FILE, CORRECTED_TRANSCRIPTION_FILE, lang="en-US")
    
    # 7) Übersetzung der Segmente mithilfe von MADLAD400
    translated = translate_segments(CORRECTED_TRANSCRIPTION_FILE, TRANSLATION_FILE)
    if not translated:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden.")
        return
    
    restore_punctuation_de(TRANSLATION_FILE, PUNCTED_TRANSLATION_FILE)

    logger.info("Starte optionalen Schritt: Qualitätsprüfung der Übersetzung.")
    evaluate_translation_quality(
        source_csv_path=PUNCTED_TRANSCRIPTION_FILE,
        translated_csv_path=PUNCTED_TRANSLATION_FILE,
        report_path=TRANSLATION_QUALITY_REPORT,
        model_name=ST_QUALITY_MODEL,
        threshold=SIMILARITY_THRESHOLD
    )
    logger.info("Qualitätsprüfung der Übersetzung (optional) abgeschlossen.")

#   6.1) Zusammenführen von Transkript-Segmenten
    merge_transcript_chunks(
        input_file=PUNCTED_TRANSLATION_FILE,
        output_file=MERGED_TRANSLATION_FILE,
        min_dur=MIN_DUR,
        max_dur=MAX_DUR,
        max_gap=MAX_GAP,
        max_chars=MAX_CHARS,
        min_words=MIN_WORDS,
        iterations=ITERATIONS
    )

    logger.info("Starte optionalen Schritt: Generierung semantischer Embeddings.")
    generate_semantic_embeddings(
        input_csv_path=MERGED_TRANSLATION_FILE,
        output_npz_path=EMBEDDINGS_FILE_NPZ,
        output_csv_path=TRANSLATION_WITH_EMBEDDINGS_CSV,
        model_name=ST_EMBEDDING_MODEL_DE
    )
    logger.info("Generierung semantischer Embeddings (optional) abgeschlossen.")
    
    # TTS-optimierte Formatierung vor der Sprachsynthese
    format_translation_for_tts(
        MERGED_TRANSLATION_FILE,
        TTS_FORMATTED_TRANSLATION_FILE,
        lang="de-DE",
        use_embeddings=True,
        embeddings_file=EMBEDDINGS_FILE_NPZ,
        lt_path="D:\\LanguageTool-6.6"
    )
    

    # 8) Text-to-Speech (TTS) mit Stimmenklonung
    text_to_speech_with_voice_cloning(
        TTS_FORMATTED_TRANSLATION_FILE,
        SAMPLE_PATH_1,
        SAMPLE_PATH_2,
        SAMPLE_PATH_3,
        #SAMPLE_PATH_4,
        #SAMPLE_PATH_5,
        TRANSLATED_AUDIO_WITH_PAUSES
    )
    
    # 9) Audio resamplen auf 44.1 kHz, Stereo (für Mixdown), inkl. separatem Lautstärke- und Geschwindigkeitsfaktor
    resample_to_44100_stereo(TRANSLATED_AUDIO_WITH_PAUSES, RESAMPLED_AUDIO_FOR_MIXDOWN, SPEED_FACTOR_RESAMPLE_44100)

    # 10) Wiedergabegeschwindigkeit des Videos anpassen (separater Lautstärkefaktor für Video)
    
    adjust_playback_speed(VIDEO_PATH, ADJUSTED_VIDEO_PATH, SPEED_FACTOR_PLAYBACK)
    
    # 11) Kombination von angepasstem Video und übersetztem Audio
    combine_video_audio_ffmpeg(ADJUSTED_VIDEO_PATH, RESAMPLED_AUDIO_FOR_MIXDOWN, FINAL_VIDEO_PATH)


    total_time = time.time() - start_time
    print("-----------------------------------")
    print("|<< Video erfolgreich übersetzt >>|")
    print("-----------------------------------")
    print(f"|<< Gesamtprozessdauer: {(total_time / 60):.2f} Minuten -> {(total_time / 60 / 60):.2f} Stunden >>|")
    print(f"|<< Projekt abgeschlossen! Finale Ausgabedatei: {FINAL_VIDEO_PATH} >>|")
    print("-----------------------------------")
    logger.info(f"Projekt abgeschlossen! Finale Ausgabedatei: {FINAL_VIDEO_PATH}", exc_info=True)
    logger.info(f"|<< Gesamtprozessdauer: {(total_time / 60):.2f} Minuten -> {(total_time / 60 / 60):.2f} Stunden >>|")

    """
    cov.stop()
    cov.save()
    cov.html_report()
    """

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Ein nicht abgefangener Fehler ist in main aufgetreten: {e}", exc_info=True) # critical statt error
    finally:
        logger.info("Programm wird beendet.")
        torch.cuda.empty_cache()
        logging.shutdown()