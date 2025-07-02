import os
import logging

from sympy import true

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
import sys
import ctypes
import re
import gc
from typing import List, Dict, Tuple, Union
from pathlib import Path
from config import *
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
from spacy.language import Language
import ftfy
from ftfy import fix_encoding
import torch
from torch import autocast
torch.set_num_threads(6)
import tensor_parallel
import deepspeed
from deepspeed import init_inference, DeepSpeedConfig
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate import Accelerator
import shape as sh
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend
import ctypes
import torchaudio
from audiostretchy.stretch import stretch_audio
import pyrubberband
import time
from datetime import datetime, timedelta
import csv
import traceback
import psutil
import shutil
import language_tool_python
from language_tool_python import LanguageTool
from functools import partial
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from contextlib import contextmanager
from deepmultilingualpunctuation import PunctuationModel
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import sentencepiece as spm
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
import threading
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
#import whisper
from faster_whisper import WhisperModel, BatchedInferencePipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM
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

# Sofortige P-Core-Optimierung
def optimize_for_video_translation():
    """Optimiert CPU-Konfiguration für Video-Übersetzung"""
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            # P-Core-Affinität setzen
            current_process = psutil.Process(os.getpid())
            current_process.cpu_affinity(list(range(0, 12)))
            
            # PyTorch für P-Cores optimieren
            torch.set_num_threads(6)  # Ihre bestehende Zeile anpassen
            os.environ["OMP_NUM_THREADS"] = "6"  # Ihre bestehende Zeile anpassen
            
            print("✓ Video-Übersetzung für P-Cores optimiert")
            return True
    except:
        pass
    return False

# Direkte Ausführung
optimize_for_video_translation()

# Multiprocessing-Setup
mp.set_start_method('spawn', force=True)

torch.backends.cuda.matmul.allow_tf32 = True  # Schnellere Matrix-Multiplikationen
torch.backends.cudnn.allow_tf32 = True        # TF32 für cuDNN aktivieren
torch.backends.cudnn.benchmark = True         # Optimale Kernel-Auswahl
torch.backends.cudnn.deterministic = False    # Nicht-deterministische Optimierungen erlauben
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ['TORCH_BLAS_PREFER_CUBLASLT'] = '1'
os.environ["CT2_FLASH_ATTENTION"] = "1"
os.environ["CT2_VERBOSE"] = "1"
os.environ["CT2_USE_EXPERIMENTAL_PACKED_GEMM"] = "1"
os.environ["OMP_NUM_THREADS"] = "12"  # Nur P-Cores für CTranslate2
os.environ["MKL_NUM_THREADS"] = "12"  # MKL auf P-Cores beschränken
os.environ["CT2_CUDA_ALLOCATOR"] = "cuda_malloc_async"
os.environ["KMP_BLOCKTIME"] = "1"            # Optimale Thread-Wartezeit
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0" 
os.environ["CT2_CPU_ENABLE_MMAP"] = "1"  # Memory-Mapping für große Modelle
os.environ["CT2_CPU_PREFETCH"] = "32"    # Cache-Prefetching optimieren

# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.0   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.0   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 1.0     # Geschwindigkeitsfaktor für die Wiedergabe des Videos

# Lautstärkeanpassungen
VOLUME_ADJUSTMENT_44100 = 1.0   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.04   # Lautstärkefaktor für das Video

# ============================== 
# Hilfsfunktionen
# ==============================

try:
    spacy.require_gpu()
    print("✅ GPU für spaCy erfolgreich aktiviert.")
except:
    print("⚠️ GPU für spaCy konnte nicht aktiviert werden. Fallback auf CPU.")

# Gerät bestimmen (GPU bevorzugt, aber CPU als Fallback)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Verwende Gerät: {device}")
if device == "cpu":
    logger.warning("GPU/CUDA nicht verfügbar. Falle auf CPU zurück. Die Verarbeitung kann langsamer sein.")
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

def configure_cusolver_optimizations():
    """Konfiguriert cuSOLVER für RTX 4050 Mobile"""
    
    # cuSOLVER als bevorzugte Bibliothek setzen
    torch.backends.cuda.preferred_linalg_library("cusolver")
    
    # TensorFloat-32 für Ampere+ GPUs aktivieren
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    torch.backends.cuda.preferred_blas_library = "cublas"
    
    logger.info("cuSOLVER-Optimierungen aktiviert für RTX 4050")

def configure_attention_backends():
    """Optimiert Attention-Mechanismen für maximale Geschwindigkeit"""
    
    # Flash Attention aktivieren (falls verfügbar)
    torch.backends.cuda.enable_flash_sdp(True)
    
    # Memory-efficient Attention als Fallback
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Math-based Attention deaktivieren (langsamste Option)
    torch.backends.cuda.enable_math_sdp(False)
    
    logger.info("Optimierte Attention-Backends aktiviert")

def advanced_cuda_configuration():
    """Fortgeschrittene CUDA-Optimierungen für RTX 4050"""
    
    # cuFFT Cache für Transformer-FFN-Layers optimieren
    torch.backends.cuda.cufft_plan_cache.max_size = 32
    
    # Optimierte Reduced Precision für FP16
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # Memory Pool für konstante Allokationen
    #torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    
    # Multi-Head Attention Fast Path aktivieren
    torch.backends.mha.set_fastpath_enabled(True)

def load_whisper_model():
    """
    Lädt das Whisper-Modell in INT8-Quantisierung für schnellere GPU-Inferenz
    und richtet die gebatchte Pipeline ein.
    """
    model_size = "large-v3"
    # compute_type="int8_float16" nutzt INT8-Gewichte + FP16-Aktivierungen für Speed & geringen Speicher
    fw_model = WhisperModel(model_size, device="auto", compute_type="bfloat16", cpu_threads=6, local_files_only=True)
    pipeline = BatchedInferencePipeline(model=fw_model)
    return pipeline

def load_madlad400_translator_optimized(model_path=None, device="auto", compute_type="auto"):
    """
    Lädt das quantisierte MADLAD400-Übersetzungsmodell mit CTranslate2 korrekt.
    
    Args:
        model_path: Pfad zum quantisierten Modell
        device: "cuda", "cpu" oder "auto" für automatische Erkennung
        compute_type: "auto", "float32", "int8_float16" für RTX 4050 Mobile
    
    Returns:
        Tuple: (translator, tokenizer) - CTranslate2 Translator und SentencePiece Tokenizer
    """
    from huggingface_hub import snapshot_download
    
    # GPU-Speicher vor Modell-Laden bereinigen
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Modellpfad bestimmen oder herunterladen
    if model_path is None:
        model_path = "madlad400-3b-mt-8bit"
        
        if not os.path.exists(model_path):
            logger.info("Quantisiertes Modell wird von HuggingFace heruntergeladen...")
            try:
                # Verwende bereits quantisierte Version für bessere Performance
                model_path = snapshot_download("avans06/madlad400-7b-mt-bt-ct2-int8_float16")
                logger.info(f"Modell heruntergeladen nach: {model_path}")
            except Exception as e:
                logger.error(f"Fehler beim Herunterladen: {e}")
                raise
    
    # Automatische Device-Erkennung mit Fallback
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("GPU erkannt - verwende CUDA")
        else:
            device = "cpu"
            logger.warning("Keine GPU verfügbar - falle auf CPU zurück")
    
    # Automatische Compute-Type-Optimierung für RTX 4050 Mobile
    if compute_type == "auto":
        if device == "cuda":
            # RTX 4050 Mobile unterstützt INT8+FP16 optimal bei 6GB VRAM
            compute_type = "int8_float32"
        else:
            compute_type = "int8_bfloat16"
    
    logger.info(f"Lade CTranslate2 Translator von {model_path}...")
    logger.info(f"Device: {device}, Compute-Type: {compute_type}")
    
    # CTranslate2 Translator mit optimierten Einstellungen laden
    translator = ctranslate2.Translator(
        model_path, 
        device=device,  # KORREKTUR: Verwende GPU statt CPU!
        compute_type=compute_type,
        inter_threads=12,  # P-Core-optimiert für i7-13700H
        intra_threads=1   # Konservativ für RTX 4050 Mobile
    )
    """
    # SentencePiece Tokenizer laden (KORREKTUR: Nicht T5TokenizerFast!)
    tokenizer_path = os.path.join(model_path, "spiece.model")
    if not os.path.exists(tokenizer_path):
        # Fallback: Versuche spiece.model
        tokenizer_path = os.path.join(model_path, "sentencepiece.model")
    
    if os.path.exists(tokenizer_path):
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        logger.info("SentencePiece Tokenizer geladen")
    else:
        logger.error(f"SentencePiece Model nicht gefunden in {model_path}")
        raise FileNotFoundError("SentencePiece tokenizer model nicht gefunden")
    """
    tokenizer = T5TokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
        legacy=False,
        padding_side='left',  # Explizite Padding-Konfiguration
        truncation_side='right' # Explizite Truncation-Konfiguration
    )
    
    return translator, tokenizer

def load_xtts_v2():
    """
    Lädt Xtts v2 und konfiguriert DeepSpeed-Inferenz.
    """
    # 1) Konfiguration lesen
    config = XttsConfig()
    config.load_json("D:\\Modelle\\v203\\config.json")
    # 2) Modell initialisieren
    xtts_model = Xtts.init_from_config(
        config,
        vocoder_path=vocoder_pth,
        vocoder_config_path=vocoder_cfg
    )
    xtts_model.load_checkpoint(
        config,
        checkpoint_dir="D:\\Modelle\\v203",  # Pfad anpassen
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

def check_completion_status(output_file, expected_count=None, reference_file=None):
    """
    Überprüft, ob eine Ausgabedatei vollständig verarbeitet wurde.
    
    Args:
        output_file (str): Pfad zur zu prüfenden Ausgabedatei
        expected_count (int, optional): Erwartete Anzahl von Segmenten
        reference_file (str, optional): Referenzdatei zum Vergleich der Zeilenzahl
        
    Returns:
        tuple: (is_complete: bool, current_count: int, expected_count: int)
    """
    if not os.path.exists(output_file):
        return False, 0, expected_count or 0
    
    try:
        df = pd.read_csv(output_file, sep='|', dtype=str)
        current_count = len(df)
        
        if reference_file and os.path.exists(reference_file):
            ref_df = pd.read_csv(reference_file, sep='|', dtype=str)
            expected_count = len(ref_df)
        
        if expected_count is None:
            # Wenn keine erwartete Anzahl bekannt, als vollständig betrachten wenn Datei existiert
            return True, current_count, current_count
            
        is_complete = current_count >= expected_count
        return is_complete, current_count, expected_count
        
    except Exception as e:
        logger.error(f"Fehler beim Prüfen der Vollständigkeit von {output_file}: {e}")
        return False, 0, expected_count or 0

def save_progress_csv(segments_list, output_file, headers=['Startzeit', 'Endzeit', 'Text']):
    """
    Speichert Fortschritt in CSV-Datei mit atomarer Schreiboperation.
    
    Args:
        segments_list (list): Liste der zu speichernden Segmente
        output_file (str): Ausgabedatei
        headers (list): CSV-Header
    """
    if not segments_list:
        return
        
    try:
        # Temporäre Datei für atomare Schreiboperation
        temp_file = output_file + '.tmp'
        
        with open(temp_file, mode='w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='|')
            csv_writer.writerow(headers)
            
            for segment in segments_list:
                if isinstance(segment, dict):
                    if 'startzeit' in segment:
                        csv_writer.writerow([
                            segment['startzeit'], 
                            segment['endzeit'], 
                            segment['text']
                        ])
                    else:
                        start = str(timedelta(seconds=segment["start"])).split('.')[0]
                        end = str(timedelta(seconds=segment["end"])).split('.')[0]
                        csv_writer.writerow([start, end, segment["text"]])
                else:
                    csv_writer.writerow(segment)
        
        # Atomare Ersetzung der Originaldatei
        shutil.move(temp_file, output_file)
        logger.info(f"Fortschritt gespeichert: {len(segments_list)} Segmente in {output_file}")
        
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Fortschritts: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

def handle_continuation_logic(output_file, reference_file=None, expected_count=None):
    """
    Behandelt die Logik für Fortsetzung vs. Überschreibung, indem eine Liste von
    bereits verarbeiteten Segmenten zurückgegeben wird.
    
    Returns:
        tuple: (should_continue: bool, existing_segments: list)
    """
    is_complete, current_count, total_expected = check_completion_status(
        output_file, expected_count, reference_file
    )
    
    if not os.path.exists(output_file):
        logger.info(f"Neue Verarbeitung: {output_file} existiert nicht")
        return True, []
    
    if is_complete:
        # Datei ist vollständig - Benutzer fragen
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vollständige Datei: {output_file} ({current_count} Segmente)")
            return False, []
        else:
            logger.info(f"Überschreibe vollständige Datei: {output_file}")
            return True, []
    else:
        # Datei ist unvollständig - automatisch fortsetzen
        logger.info(f"Setze unvollständige Verarbeitung fort: {current_count}/{total_expected} Segmente")
        try:
            existing_segments = []
            df = pd.read_csv(output_file, sep='|', dtype=str)
            # Konvertiere DataFrame in die erwartete Diktionär-Struktur
            for _, row in df.iterrows():
                existing_segments.append(row.to_dict())
            return True, existing_segments
        except Exception as e:
            logger.error(f"Fehler beim Laden vorhandener Segmente aus {output_file}: {e}")
            return True, []

    # Logik zur Bestimmung, ob der Prozess abgeschlossen ist
    is_complete = False
    if reference_file and os.path.exists(reference_file):
        try:
            with open(reference_file, 'r', encoding='utf-8') as ref_file:
                # Zähle Zeilen in der Referenzdatei (ohne Header)
                total_expected = sum(1 for line in ref_file) - 1
                if len(processed_keys) >= total_expected:
                    is_complete = True
        except Exception as e:
            logger.warning(f"Konnte Referenzdatei {reference_file} nicht vollständig prüfen: {e}")

    if is_complete:
        # Datei scheint vollständig - Benutzer fragen
        if not ask_overwrite(output_file):
            logger.info(f"Verwende anscheinend vollständige Datei: {output_file} ({len(processed_keys)} Segmente)")
            return False, processed_keys # Nicht fortsetzen
        else:
            logger.info(f"Überschreibe vollständige Datei: {output_file}")
            return True, set() # Fortsetzen mit leerem Schlüsselset
    else:
        # Datei ist unvollständig - automatisch fortsetzen
        logger.info(f"Setze unvollständige Verarbeitung fort: {len(processed_keys)} Segmente bereits verarbeitet.")
        return True, processed_keys

def handle_key_based_continuation(output_file, reference_file=None, key_column_index=0):
    """
    Behandelt die Logik für Fortsetzung vs. Überschreibung auf Basis von eindeutigen Schlüsseln
    und gibt ein Set der verarbeiteten Schlüssel zurück. Speziell für die Übersetzung.
    
    Returns:
        tuple: (should_continue: bool, processed_keys: set)
    """
    processed_keys = set()
    
    if not os.path.exists(output_file):
        logger.info(f"Neue Verarbeitung: {output_file} existiert nicht.")
        return True, processed_keys

    # Lese vorhandene Schlüssel aus der Ausgabedatei
    try:
        with open(output_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile, delimiter='|')
            header = next(reader, None)
            if header is None: return True, processed_keys
            for row in reader:
                if len(row) > key_column_index:
                    processed_keys.add(row[key_column_index])
    except Exception as e:
        logger.error(f"Fehler beim Lesen der vorhandenen Ausgabedatei {output_file}: {e}")
        if ask_overwrite(output_file):
            return True, set()
        else:
            raise IOError(f"Konnte {output_file} nicht verarbeiten und Überschreiben wurde abgelehnt.") from e

    # Logik zur Bestimmung, ob der Prozess abgeschlossen ist
    is_complete = False
    if reference_file and os.path.exists(reference_file):
        try:
            with open(reference_file, 'r', encoding='utf-8') as ref_file:
                total_expected = sum(1 for line in ref_file) - 1
                if len(processed_keys) >= total_expected:
                    is_complete = True
        except Exception as e:
            logger.warning(f"Konnte Referenzdatei {reference_file} nicht vollständig prüfen: {e}")

    if is_complete:
        if not ask_overwrite(output_file):
            logger.info(f"Verwende anscheinend vollständige Datei: {output_file} ({len(processed_keys)} Segmente)")
            return False, processed_keys
        else:
            logger.info(f"Überschreibe vollständige Datei: {output_file}")
            return True, set()
    else:
        logger.info(f"Setze unvollständige Verarbeitung fort: {len(processed_keys)} Segmente bereits verarbeitet.")
        return True, processed_keys

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
def transcribe_audio_with_timestamps(audio_file, transcription_file, batch_save_interval=10):
    """
    Sichere Transkription mit Fortschrittsspeicherung, Wiederaufnahme, Live-Ausgabe und Prozessbalken.
    Korrigierte Version, die direkt über den Generator iteriert, um Hänger zu vermeiden.
    """
    should_continue, existing_segments = handle_continuation_logic(transcription_file)
    
    if not should_continue:
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

        processed_count = len(existing_segments)
        logger.info(f"Setze Transkription fort ab Segment {processed_count}")

        with gpu_context():
            pipeline = load_whisper_model()   # Laden des vortrainierten Whisper-Modells
            
            # VAD-Parameter für die Sprachsegmentierung definieren
            vad_params = {
                "threshold": 0.5,               # Niedriger Schwellwert für empfindlichere Spracherkennung
                "min_speech_duration_ms": 0,  # Minimale Sprachdauer in Millisekunden
                #"max_speech_duration_s": 30,    # Maximale Sprachdauer in Sekunden
                "min_silence_duration_ms": 0, # Minimale Stille-Dauer zwischen Segmenten
                #"speech_pad_ms": 400            # Polsterzeit vor und nach Sprachsegmenten
            }
            
            segments_generator, info = pipeline.transcribe(
                audio_file,
                batch_size=1,
                beam_size=15,
                patience=1.5,
                vad_filter=True,
                vad_parameters=vad_params,
                #chunk_length=25,
                #compression_ratio_threshold=2.8,    # Schwellenwert für Kompressionsrate
                #log_prob_threshold=-0.2,             # Schwellenwert für Log-Probabilität
                #no_speech_threshold=1.0,            # Schwellenwert für Stille
                #temperature=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5),      # Temperatur für Sampling
                temperature=1,                  # Temperatur für Sampling
                word_timestamps=True,               # Zeitstempel für Wörter
                hallucination_silence_threshold=0.5,  # Schwellenwert für Halluzinationen
                condition_on_previous_text=True,    # Bedingung an vorherigen Text
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                language="en",   
            )

        all_segments = existing_segments.copy()
        new_segments_batch = []
        
        print("\n--- [ Transkription Live-Ausgabe ] ---\n")
        
        # `tqdm` wird keine Gesamtanzahl anzeigen, aber den Fortschritt live darstellen.
        # Wir überspringen manuell die bereits verarbeiteten Segmente.
        
        # Initialisiere tqdm ohne die Gesamtanzahl
        progress_bar = tqdm(desc="\nTranskribiere Segmente", unit=" seg")
        
        for i, segment in enumerate(segments_generator):
            progress_bar.update(1) # Zähler des Fortschrittsbalkens manuell erhöhen
            
            # Logik zum Überspringen, falls wir den Prozess fortsetzen
            if i < processed_count:
                continue
                
            start = str(timedelta(seconds=segment.start)).split('.')[0]
            end = str(timedelta(seconds=segment.end)).split('.')[0]
            text = re.sub(r'\.\.\.$', '', segment.text.strip()).strip()
            
            # Die Live-Ausgabe bleibt erhalten und sollte nun sofort erscheinen
            print(f"[{start} --> {end}] {text}")
            
            adjusted_segment = {"startzeit": start, "endzeit": end, "text": text}
            new_segments_batch.append(adjusted_segment)
            all_segments.append(adjusted_segment)
            
            if len(new_segments_batch) >= batch_save_interval:
                save_progress_csv(all_segments, transcription_file)
                new_segments_batch = []
                logger.info(f"Zwischenspeicherung: {len(all_segments)} Segmente verarbeitet")

        progress_bar.close() # Schließe den Fortschrittsbalken am Ende

        if new_segments_batch or not existing_segments:
            save_progress_csv(all_segments, transcription_file)
        
        print("\n------------------------------------")
        print("|<< Transkription abgeschlossen! >>|")
        print("------------------------------------")
        logger.info("Transkription abgeschlossen!", exc_info=True)

        transcript_end_time = time.time() - transcription_start_time
        
        logger.info(f"Transkription abgeschlossen in {transcript_end_time:.2f} Sekunden")
        print(f"{(transcript_end_time):.2f} Sekunden")
        print(f"{(transcript_end_time / 60 ):.2f} Minuten")
        print(f"{(transcript_end_time / 3600):.2f} Stunden")
        
        return all_segments

    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}", exc_info=True)
        if 'all_segments' in locals() and all_segments:
            save_progress_csv(all_segments, transcription_file)
            logger.info(f"Notfall-Speicherung: {len(all_segments)} Segmente gesichert")
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
    Implementiert die 3-Wörter-Regel für Zeichenbegrenzungen.

    Args:
        input_file (str): Eingabedatei mit | als Trennzeichen.
        output_file (str): Zieldatei für Ergebnisse.
        min_dur (float): Ziel-Mindestdauer. Segmente darunter werden nach Möglichkeit zusammengeführt.
        max_dur (float): Maximale Segmentdauer in Sekunden (wird beim initialen Mergen beachtet).
        max_gap (float): Maximaler akzeptierter Zeitabstand für initiales Mergen.
        max_chars (int): Maximale Anzahl von Zeichen (3-Wörter-Regel wird angewendet).
        min_words (int): Ziel-Mindestwortzahl. Segmente darunter werden nach Möglichkeit zusammengeführt.
        iterations (int): Anzahl der Optimierungsdurchläufe.
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
        if not data_list: 
            return False

        merged_something = False
        i = 0
        while i < len(data_list):
            item = data_list[i]
            start_sec = parse_time(item['startzeit'])
            end_sec = parse_time(item['endzeit'])

            if start_sec is None or end_sec is None or start_sec > end_sec:
                logger.warning(f"Segment {i} hat ungültige Zeiten ({item['startzeit']} / {item['endzeit']}) - wird in force_merge übersprungen.")
                i += 1
                continue

            duration = round(end_sec - start_sec, 3)
            word_count = len(item['text'].split()) if item.get('text') else 0

            needs_merge = (duration < min_dur or word_count < min_words) and len(data_list) > 1

            if needs_merge:
                merged_this_iteration = False
                # Option 1: Mit Vorgänger zusammenführen (bevorzugt)
                if i > 0:
                    prev_item = data_list[i-1]
                    separator = " " if prev_item.get('text', '').strip() and item.get('text', '').strip() else ""
                    merged_text = (prev_item.get('text', '') + separator + item.get('text', '')).strip()

                    prev_item['endzeit'] = item['endzeit']
                    prev_item['text'] = merged_text
                    del data_list[i]
                    merged_something = True
                    merged_this_iteration = True
                    logger.debug(f"Segment {i+1} mit Vorgänger {i} zusammengeführt.")

                # Option 2: Mit Nachfolger zusammenführen
                elif i < len(data_list) - 1:
                    next_item = data_list[i+1]
                    separator = " " if item.get('text', '').strip() and next_item.get('text', '').strip() else ""
                    merged_text = (item.get('text', '') + separator + next_item.get('text', '')).strip()

                    item['endzeit'] = next_item['endzeit']
                    item['text'] = merged_text
                    del data_list[i+1]
                    merged_something = True
                    merged_this_iteration = True
                    logger.debug(f"Segment {i} mit Nachfolger {i+1} zusammengeführt.")

                if not merged_this_iteration:
                    i += 1
            else:
                i += 1

        return merged_something

    def split_text_with_3_word_rule(text, max_chars):
        """
        Teilt Text an Satz- oder Wortgrenzen auf, wendet aber die 3-Wörter-Regel an.
        Ignoriert max_chars, wenn das neue Segment weniger als 3 Wörter hätte.
        """
        if len(text) <= max_chars:
            return [text]

        # Versuche zuerst Aufteilung an Satzgrenzen
        sentences = split_into_sentences(text)
        
        if len(sentences) == 1:
            # Nur ein Satz - teile an Wortgrenzen mit 3-Wörter-Regel
            return split_words_with_3_word_rule(text, max_chars)
        else:
            # Mehrere Sätze - kombiniere unter Beachtung der Regeln
            return split_sentences_with_3_word_rule(sentences, max_chars)

    def split_words_with_3_word_rule(text, max_chars):
        """
        Teilt einen einzelnen Satz an Wortgrenzen mit 3-Wörter-Regel auf.
        """
        words = text.split()
        if len(words) <= 3:
            # Weniger als oder gleich 3 Wörter - nicht aufteilen
            return [text]

        chunks = []
        current_chunk = ""
        i = 0

        while i < len(words):
            word = words[i]
            test_chunk = current_chunk + (" " + word if current_chunk else word)

            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
                i += 1
            else:
                # Wort passt nicht mehr - prüfe 3-Wörter-Regel
                remaining_words = words[i:]
                
                if len(remaining_words) < 3:
                    # 3-Wörter-Regel: Füge verbleibende Wörter zum aktuellen Chunk hinzu
                    current_chunk += " " + " ".join(remaining_words) if current_chunk else " ".join(remaining_words)
                    chunks.append(current_chunk)
                    break
                else:
                    # Genug Wörter übrig - speichere aktuellen Chunk und starte neuen
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = word
                    i += 1

        # Letzten Chunk hinzufügen, falls noch vorhanden
        if current_chunk and current_chunk not in chunks:
            chunks.append(current_chunk)

        return chunks

    def split_sentences_with_3_word_rule(sentences, max_chars):
        """
        Kombiniert Sätze unter Beachtung der 3-Wörter-Regel.
        """
        chunks = []
        current_chunk = ""

        for j, sentence in enumerate(sentences):
            test_chunk = current_chunk + (" " + sentence if current_chunk else sentence)

            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                # Satz passt nicht mehr - prüfe 3-Wörter-Regel
                remaining_sentences = sentences[j:]
                remaining_text = " ".join(remaining_sentences)

                if len(remaining_text.split()) < 3:
                    # 3-Wörter-Regel: Füge verbleibende Sätze hinzu
                    current_chunk = test_chunk
                else:
                    # Genug Wörter übrig - speichere aktuellen Chunk
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence

        # Letzten Chunk hinzufügen
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    try:
        print(f"Starte Verarbeitung von: {input_file}")
        print(f"Parameter: min_dur={min_dur}s, max_dur={max_dur}s, max_gap={max_gap}s")
        print(f"max_chars={max_chars} (mit 3-Wörter-Regel), min_words={min_words}, iterations={iterations}")

        # Daten laden und validieren
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

        # Ungültige Zeiten entfernen
        invalid_mask = df['start_sec'].isna() | df['end_sec'].isna() | (df['start_sec'] > df['end_sec'])
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            print(f"Warnung: {invalid_count} Zeilen mit ungültigen Zeitangaben werden übersprungen.")
            df = df[~invalid_mask].copy()

        if df.empty:
            print("Keine gültigen Segmente nach initialer Zeitprüfung.")
            pd.DataFrame(columns=['startzeit', 'endzeit', 'text']).to_csv(output_file, sep='|', index=False)
            return pd.DataFrame(columns=['startzeit', 'endzeit', 'text'])

        df['duration'] = df['end_sec'] - df['start_sec']
        df = df.sort_values('start_sec').reset_index(drop=True)
        print(f"Nach Zeitvalidierung und Sortierung: {len(df)} gültige Segmente")

        current_df = df.copy()
        current_data_list = []

        # Iterative Optimierung
        for iteration in range(iterations):
            print(f"\n--- Optimierungsdurchlauf {iteration+1}/{iterations} ---")

            if iteration > 0:
                if not current_data_list:
                    print("Keine Segmente mehr für weiteren Durchlauf vorhanden.")
                    break
                
                # DataFrame aus aktueller Liste erstellen
                temp_df = pd.DataFrame(current_data_list)
                temp_df['start_sec'] = temp_df['startzeit'].apply(parse_time)
                temp_df['end_sec'] = temp_df['endzeit'].apply(parse_time)
                temp_df['text'] = temp_df['text'].astype(str).fillna('')

                invalid_mask = temp_df['start_sec'].isna() | temp_df['end_sec'].isna() | (temp_df['start_sec'] > temp_df['end_sec'])
                if invalid_mask.any():
                    print(f"Warnung (Durchlauf {iteration+1}): {invalid_mask.sum()} ungültige Zeiten entfernt.")
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

            # Phase 1: Zeitbasiertes Zusammenführen
            for _, row in current_df.iterrows():
                row_text = str(row['text']) if pd.notna(row['text']) else ''

                if current_chunk is None:
                    current_chunk = {
                        'start': row['start_sec'], 
                        'end': row['end_sec'],
                        'text': [row_text],
                        'original_start': row['startzeit'], 
                        'original_end': row['endzeit']
                    }
                else:
                    gap = row['start_sec'] - current_chunk['end']
                    potential_new_end = row['end_sec']
                    potential_duration = potential_new_end - current_chunk['start']
                    
                    # Berechne potenziellen Gesamttext
                    temp_texts = current_chunk['text'] + [row_text] if row_text else current_chunk['text']
                    potential_full_text = ""
                    first = True
                    for txt in temp_texts:
                        clean_txt = txt.strip()
                        if clean_txt:
                            if not first: 
                                potential_full_text += " "
                            potential_full_text += clean_txt
                            first = False
                    
                    potential_chars = len(potential_full_text)

                    # Zusammenführen wenn Kriterien erfüllt sind
                    if (gap >= 0 and gap <= max_gap) and \
                       (potential_duration <= max_dur) and \
                       (potential_chars <= max_chars):
                        current_chunk['end'] = potential_new_end
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
                                if not first: 
                                    final_text += " "
                                final_text += clean_txt
                                first = False
                        
                        if final_text:
                            merged_data.append({
                                'startzeit': current_chunk['original_start'],
                                'endzeit': current_chunk['original_end'],
                                'text': final_text
                            })
                        
                        # Beginne neuen Chunk
                        current_chunk = {
                            'start': row['start_sec'], 
                            'end': row['end_sec'],
                            'text': [row_text] if row_text else [],
                            'original_start': row['startzeit'], 
                            'original_end': row['endzeit']
                        }

            # Letzten Chunk hinzufügen
            if current_chunk:
                final_text = ""
                first = True
                for txt in current_chunk['text']:
                    clean_txt = txt.strip()
                    if clean_txt:
                        if not first: 
                            final_text += " "
                        final_text += clean_txt
                        first = False
                
                if final_text:
                    merged_data.append({
                        'startzeit': current_chunk['original_start'],
                        'endzeit': current_chunk['original_end'],
                        'text': final_text
                    })

            print(f"Nach Zeit-Zusammenführung: {len(merged_data)} Segmente")

            # Phase 2: Aufteilung zu langer Segmente mit 3-Wörter-Regel
            current_data_list = []
            segmente_aufgeteilt_max_chars = 0
            segments_protected_by_3_word_rule = 0

            for item in merged_data:
                text = item['text'].strip()
                start_time_sec = parse_time(item['startzeit'])
                end_time_sec = parse_time(item['endzeit'])

                if start_time_sec is None or end_time_sec is None:
                    logger.warning(f"Überspringe Segment wegen ungültiger Zeit: {item}")
                    current_data_list.append(item)
                    continue

                duration = end_time_sec - start_time_sec

                # Prüfe ob Text das Zeichenlimit überschreitet
                if len(text) <= max_chars:
                    current_data_list.append(item)
                else:
                    # Prüfe 3-Wörter-Regel bevor Aufteilung
                    word_count = len(text.split())
                    if word_count < 3:
                        # 3-Wörter-Regel greift - keine Aufteilung
                        segments_protected_by_3_word_rule += 1
                        print(f"Info: Segment mit {word_count} Wörtern wird wegen 3-Wörter-Regel nicht aufgeteilt " +
                              f"(Länge: {len(text)} > {max_chars}): {item['startzeit']}-{item['endzeit']}")
                        current_data_list.append(item)
                        continue

                    # Text kann aufgeteilt werden
                    segmente_aufgeteilt_max_chars += 1
                    print(f"Info: Segment wird aufgeteilt: {len(text)} > {max_chars} " +
                          f"({item['startzeit']}-{item['endzeit']})")

                    # Verwende die neue Aufteilungsfunktion mit 3-Wörter-Regel
                    text_chunks = split_text_with_3_word_rule(text, max_chars)
                    
                    print(f"-> Aufgeteilt in {len(text_chunks)} Segmente")

                    # Zeitverteilung proportional zur Textlänge
                    num_new_segments = len(text_chunks)
                    if num_new_segments == 0: 
                        continue

                    total_chars_in_split = sum(len(s) for s in text_chunks)
                    if total_chars_in_split == 0:
                        # Gleichmäßige Zeitverteilung bei fehlenden Zeichen
                        segment_duration = duration / num_new_segments if num_new_segments > 0 else 0
                        current_start_time = start_time_sec
                        
                        for i, segment_text in enumerate(text_chunks):
                            segment_end_time = current_start_time + segment_duration
                            if i == num_new_segments - 1: 
                                segment_end_time = end_time_sec
                            
                            new_item = {
                                'startzeit': format_time(current_start_time),
                                'endzeit': format_time(segment_end_time),
                                'text': segment_text
                            }
                            current_data_list.append(new_item)
                            current_start_time = segment_end_time
                        continue

                    # Proportionale Zeitverteilung nach Zeichenanzahl
                    current_start_time = start_time_sec
                    for i, segment_text in enumerate(text_chunks):
                        segment_chars = len(segment_text)
                        segment_proportion = segment_chars / total_chars_in_split
                        segment_duration = duration * segment_proportion
                        segment_end_time = current_start_time + segment_duration

                        # Korrektur für letztes Segment
                        if i == num_new_segments - 1:
                            segment_end_time = end_time_sec

                        new_item = {
                            'startzeit': format_time(current_start_time),
                            'endzeit': format_time(segment_end_time),
                            'text': segment_text
                        }
                        current_data_list.append(new_item)
                        current_start_time = segment_end_time

            print(f"Nach Längen-Aufteilung: {len(current_data_list)} Segmente")
            print(f"  Segmente aufgeteilt wegen max. Zeichen: {segmente_aufgeteilt_max_chars}")
            print(f"  Segmente durch 3-Wörter-Regel geschützt: {segments_protected_by_3_word_rule}")

        # Finale Phase: Erzwinge min_dur und min_words durch Zusammenführen
        print("\n--- Finale Bereinigung: Erzwinge Mindestdauer & Mindestwörter ---")
        force_merge_iterations = 0
        max_force_merge_iterations = len(current_data_list)
        
        while force_merge_iterations < max_force_merge_iterations:
            force_merge_iterations += 1
            print(f"Bereinigungsdurchlauf {force_merge_iterations}...")
            changed = force_merge_short_segments(current_data_list, min_dur, min_words)
            print(f"Segmente nach Durchlauf {force_merge_iterations}: {len(current_data_list)}")
            
            if not changed:
                print("Keine weiteren Zusammenführungen nötig.")
                break
                
        if force_merge_iterations == max_force_merge_iterations:
            print("Warnung: Maximalzahl an Bereinigungsdurchläufen erreicht.")

        # Abschluss und Speichern
        if not current_data_list:
            print("\n--- Verarbeitung abgeschlossen: KEINE finalen Segmente erzeugt ---")
            final_df = pd.DataFrame(columns=['startzeit', 'endzeit', 'text'])
            final_df.to_csv(output_file, sep='|', index=False)
            print(f"Leere Ergebnisdatei {output_file} gespeichert.")
            return final_df

        result_df = pd.DataFrame(current_data_list)
        result_df = result_df[['startzeit', 'endzeit', 'text']]
        result_df.to_csv(output_file, sep='|', index=False)

        # Abschlussbericht
        final_segment_count = len(result_df)
        print("\n--- Verarbeitungsstatistik (Final) ---")
        print(f"Originale Segmente gelesen:           {original_segment_count}")
        print(f"Gültige Segmente nach Init-Parse:     {len(df) if 'df' in locals() else 'N/A'}")
        print(f"Finale Segmente geschrieben:          {final_segment_count}")
        print(f"Durch 3-Wörter-Regel geschützt:       {segments_protected_by_3_word_rule}")
        print(f"Ergebnis in {output_file} gespeichert")
        print("--" * 20 + "\n")

        return result_df

    except FileNotFoundError:
        print(f"Fehler: Eingabedatei nicht gefunden: {input_file}")
        raise
    except ValueError as e:
        print(f"Fehler bei der Datenvalidierung: {str(e)}")
        traceback.print_exc()
        raise
    except Exception as e:
        print(f"Unerwarteter Fehler: {str(e)}")
        traceback.print_exc()
        raise

def safe_punctuate(text, model, logger):
    """
    Eine sichere Wrapper-Funktion, um die Interpunktion wiederherzustellen.
    Fängt Fehler ab, die durch leere oder problematische Texteingaben in der Bibliothek entstehen können.
    """
    # Prüfe, ob der Input ein gültiger, nicht-leerer String ist.
    if not isinstance(text, str) or not text.strip():
        # Wenn nicht, gib den Originaltext (z.B. leer oder None) zurück.
        return text
    try:
        # Versuche, die Interpunktion mit dem Modell wiederherzustellen.
        return model.restore_punctuation(text)
    except IndexError:
        # Fange den spezifischen IndexError ab, der bei leeren 'batches' auftritt.
        logger.warning(f"Konnte Interpunktion für das Segment nicht wiederherstellen, da es wahrscheinlich leer oder ungültig ist. Segment: '{text}'")
        # Gib den ursprünglichen Text zurück, um Datenverlust zu vermeiden.
        return text
    except Exception as e:
        # Fange alle anderen unerwarteten Fehler ab.
        logger.error(f"Ein unerwarteter Fehler ist bei der Interpunktion des Segments aufgetreten: '{text}'. Fehler: {e}")
        # Gib auch hier den Originaltext zurück.
        return text

def restore_punctuation(input_file, output_file):
    # Überprüft, ob die Ausgabedatei bereits existiert.
    if os.path.exists(output_file):
        # Fragt den Benutzer, ob die existierende Datei überschrieben werden soll.
        if not ask_overwrite(output_file):
            # Wenn nicht, wird die vorhandene Datei verwendet und die Funktion beendet.
            logger.info(f"Verwende vorhandene Interpunktions-Datei: {output_file}", exc_info=True)
            # Liest die bereits übersetzte/punktierte CSV-Datei und gibt sie zurück.
            return read_translated_csv(output_file)

    """Stellt die Interpunktion mit deepmultilingualpunctuation wieder her."""
    # Lese die Eingabedatei in einen pandas DataFrame ein, Spaltentrenner ist '|', alle Daten als String behandeln.
    df = pd.read_csv(input_file, sep='|', dtype=str)
    
    # Identifiziere die ursprüngliche Textspalte (ignoriere Groß/Kleinschreibung)
    text_col_original = None # Initialisiere die Variable für den Spaltennamen 'text'.
    for col in df.columns: # Durchlaufe alle Spaltennamen im DataFrame.
        if col.strip().lower() == 'text': # Prüfe, ob der bereinigte Spaltenname (in Kleinbuchstaben) 'text' ist.
            text_col_original = col # Speichere den originalen Spaltennamen (z.B. 'text', 'Text', ' text ').
            break # Beende die Schleife, sobald die Spalte gefunden wurde.
    if text_col_original is None: # Wenn nach der Schleife keine Textspalte gefunden wurde.
        raise ValueError("Keine Spalte 'text' (unabhängig von Groß/Kleinschreibung) in der Eingabedatei gefunden.") # Wirf einen Fehler.
    
    # Nutze den GPU-Kontextmanager für die Modell-Ausführung.
    with gpu_context():
        # Initialisiere das Interpunktionsmodell.
        model = PunctuationModel()
        
        # Wende das Modell mit unserer neuen, sicheren Funktion auf jede Zeile der Textspalte an und speichere das Ergebnis in einer NEUEN Spalte.
        df['punctuated_text'] = df[text_col_original].apply(lambda x: safe_punctuate(x, model, logger))

        # Lösche die ursprüngliche Textspalte, da wir nun die punktierte Version haben.
        df = df.drop(columns=[text_col_original])

        # Benenne die neue Spalte in den Standardnamen 'text' um (jetzt garantiert kleingeschrieben).
        df = df.rename(columns={'punctuated_text': 'text'})

        # Stelle sicher, dass die Spaltenreihenfolge sinnvoll ist (optional, aber gut für die Lesbarkeit).
        cols = df.columns.tolist() # Hole eine Liste aller Spaltennamen.
        # Prüfe, ob die Kernspalten für Zeitstempel und Text vorhanden sind.
        if 'startzeit' in cols and 'endzeit' in cols and 'text' in cols:
            # Definiere die gewünschte Reihenfolge der Kernspalten.
            core_cols = ['startzeit', 'endzeit', 'text']
            # Sammle alle anderen Spalten, die nicht zu den Kernspalten gehören.
            other_cols = [c for c in cols if c not in core_cols]
            # Ordne den DataFrame neu an: Zuerst die Kernspalten, dann der Rest.
            df = df[core_cols + other_cols]
        
    # Speichere den bearbeiteten DataFrame in der Ausgabedatei, mit '|' als Trennzeichen und ohne den Index.
    df.to_csv(output_file, sep='|', index=False)
    # Gib den Pfad zur erstellten Datei zurück.
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
        tool.close()
        
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

def ensure_transcription_quality_en(input_csv, output_csv):
    """
    Stellt sicher, dass die englische Transkription sauber segmentiert und konsistent ist.
    """
    if os.path.exists(output_csv):
        if not ask_overwrite(output_csv):
            print(f"Verwende vorhandene Datei: {output_csv}")
            return output_csv

    df = pd.read_csv(input_csv, sep='|', dtype=str)
    if 'Text' not in df.columns:
        print("Keine 'Text'-Spalte in der Eingabedatei gefunden.")
        return input_csv

    cleaned_rows = []
    for i, row in df.iterrows():
        text = row['Text']
        if not isinstance(text, str) or not text.strip():
            cleaned_rows.append(row)
            continue

        # Segmentiere und bereinige den Text
        doc = nlp_en(text)
        cleaned_text = " ".join([sent.text.strip() for sent in doc.sents if sent.text.strip()])
        row['Text'] = cleaned_text
        cleaned_rows.append(row)

    df_cleaned = pd.DataFrame(cleaned_rows)
    df_cleaned.to_csv(output_csv, sep='|', index=False, encoding='utf-8')
    print(f"Transkriptionsqualität gesichert und gespeichert in: {output_csv}")
    return output_csv

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

def setup_gpu_memory_optimization_rtx4050():
    """
    Konfiguriert GPU-Speicher optimal für RTX 4050 Mobile (6GB VRAM).
    Diese Funktion sollte ganz am Anfang des Skripts aufgerufen werden.
    """

    # RTX 4050 Mobile-spezifische Memory-Pool-Konfiguration
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
    
    # CTranslate2-spezifische Optimierungen für 6GB VRAM
    os.environ['CT2_CUDA_CACHING_ALLOCATOR_CONFIG'] = '4,3,10,104857600'  # 100MB Cache
    os.environ['CT2_CUDA_ALLOW_FP16'] = '1'
    os.environ['CT2_USE_EXPERIMENTAL_PACKED_GEMM'] = '1'
    
    # GPU-Optimierungen aktivieren
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True  
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Memory-Fragmentierung reduzieren
        torch.cuda.empty_cache()
        
        logger.info("RTX 4050 Mobile GPU-Optimierungen aktiviert")
        logger.info(f"Verfügbarer GPU-Speicher: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.warning("CUDA nicht verfügbar - GPU-Optimierungen übersprungen")

def calculate_optimal_batch_size_rtx4050(texts, tokenizer, max_length=512):
    """
    Berechnet optimale Batch-Größe für RTX 4050 Mobile basierend auf Textlängen.
    
    Args:
        texts: Liste der zu übersetzenden Texte
        tokenizer: SentencePiece Tokenizer
        max_length: Maximale Token-Länge pro Text
        
    Returns:
        int: Optimale Batch-Größe für RTX 4050 Mobile
    """
    if not torch.cuda.is_available():
        return 2  # CPU-Fallback
    
    # RTX 4050 Mobile: 6GB VRAM - konservative Schätzung
    available_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated()
    free_memory = available_memory - allocated_memory
    
    # Durchschnittliche Textlänge bestimmen
    sample_texts = texts[:min(5, len(texts))]
    avg_tokens = 0
    
    for text in sample_texts:
        if isinstance(text, dict):
            text = text.get('text', '')
        # Schätze Token-Anzahl (SentencePiece: ~4 Zeichen pro Token)
        estimated_tokens = min(len(str(text)) // 4, max_length)
        avg_tokens += estimated_tokens
    
    avg_tokens = avg_tokens / len(sample_texts) if sample_texts else max_length
    
    # Memory-Bedarf pro Sample schätzen (konservativ für MADLAD400-7B)
    # Basis: ~2 bytes pro Parameter * geschätzte Aktivierungen
    estimated_memory_per_sample = avg_tokens * 1024 * 8  # 8KB pro Token (konservativ)
    
    # Sicherheitspuffer: 70% des freien Speichers nutzen
    usable_memory = free_memory * 0.7
    
    # Batch-Größe berechnen
    if estimated_memory_per_sample > 0:
        calculated_batch_size = int(usable_memory / estimated_memory_per_sample)
        # RTX 4050 Mobile Limits: Min 1, Max 6 für Translation
        optimal_batch_size = max(1, min(calculated_batch_size, 2))
    else:
        optimal_batch_size = 2  # Sicherer Standard
    
    logger.info(f"Berechnete optimale Batch-Größe: {optimal_batch_size}")
    logger.info(f"Freier GPU-Speicher: {free_memory / 1024**3:.2f} GB")
    
    return optimal_batch_size

def estimate_memory_per_sample(
    avg_text_length: float, 
    max_length: int, 
    translator, 
    tokenizer,
    dtype_multiplier: float = 2.0  # bfloat16 = 2 bytes per parameter
) -> float:
    """
    Schätzt den Speicherverbrauch pro Sample basierend auf Modellgröße und Textlänge.
    
    Args:
        avg_text_length: Durchschnittliche Textlänge in Zeichen
        max_length: Maximale Token-Länge
        model: MADLAD400 Modell
        tokenizer: MADLAD400 Tokenizer
        dtype_multiplier: Multiplikator für Datentyp (bfloat16 = 2.0)
        
    Returns:
        Geschätzter Speicherverbrauch in Bytes
    """
    
    # Geschätzte Token-Anzahl (ca. 4 Zeichen pro Token für deutsche Texte)
    estimated_tokens = min(int(avg_text_length / 4), max_length)
    
    # Basis-Speicherverbrauch für MADLAD400-7B Modell
    # Input-Tensoren: batch_size * seq_length * hidden_size * dtype_bytes
    hidden_size = 1024  # Typisch für 7B Modelle
    input_memory = estimated_tokens * hidden_size * dtype_multiplier
    
    # Output-Tensoren und Zwischenergebnisse (Faktor 3-4x für Transformer)
    total_memory_per_sample = input_memory * 4
    
    # Zusätzlicher Speicher für Attention-Mechanismen
    attention_memory = estimated_tokens * estimated_tokens * dtype_multiplier
    
    # Gesamtspeicher pro Sample
    estimated_memory = total_memory_per_sample + attention_memory
    
    # Konservative Schätzung mit zusätzlichem Puffer für unvorhersehbare Allokationen
    return estimated_memory * 1.5

def prepend_previous_if_fragile(idx: int, segment_list: list) -> str:
    """
    Gibt den aktuellen Text zurück; beginnt er mit einem Gerundium/Partizip,
    wird automatisch der vorherige Satz vorangestellt (Mini-Kontext).

    Args:
        idx: Position des Segments im übergebenen segment_list.
        segment_list: Liste von Segment-Dicts bzw. Strings mit 'text'-Key.
    """
    # Rohtext des aktuellen Segments ermitteln
    curr = (segment_list[idx]["text"]
            if isinstance(segment_list[idx], dict)
            else str(segment_list[idx])).strip()

    # Muster: ing- oder ‑ed-Formen zu Satzbeginn (engl. Partizip/Gerundium)
    if re.match(r"^(?:[A-Za-z]+ing\b|[A-Za-z]+ed\b)", curr, flags=re.I):
        if idx > 0:
            prev = (segment_list[idx - 1]["text"]
                    if isinstance(segment_list[idx - 1], dict)
                    else str(segment_list[idx - 1])).strip()
            # max. 300 Zeichen Kontext, um VRAM zu schonen
            CTX_TOKEN = "[CTX]"            # darf im Quelltext NICHT vorkommen
            return f"{prev} {CTX_TOKEN} {curr}"[:300].strip()
    return curr

# Übersetzen
def translate_batch_madlad400(
    texts,
    translator,
    tokenizer,
    source_lang="en",
    target_lang="de",
    batch_size=None
):
    """
    Führt eine korrekte Batch-Übersetzung mit MADLAD400 und CTranslate2 durch 
    und bereinigt die Ausgabe robust von allen bekannten Modell-Artefakten.
    
    Args:
        texts (list): Liste von Texten oder Dictionaries mit einem 'text'-Schlüssel.
        translator (ctranslate2.Translator): Der geladene CTranslate2 Translator.
        tokenizer (spm.SentencePieceProcessor): Der geladene SentencePiece Tokenizer.
        source_lang (str): Kürzel der Quellsprache (z.B. "en").
        target_lang (str): Kürzel der Zielsprache (z.B. "de").
        batch_size (int, optional): Größe der Batches. Wird automatisch berechnet, wenn None.
        
    Returns:
        list[str]: Eine Liste der bereinigten, übersetzten Texte.
    """
    if not texts:
        logger.warning("Leere Textliste zur Übersetzung erhalten, gebe leere Liste zurück.")
        return []
    
    if batch_size is None:
        batch_size = calculate_optimal_batch_size_rtx4050(texts, tokenizer)
    
    all_translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Übersetze Batches", unit="batch"):
        batch_items = texts[i:i + batch_size]
        
        # Validierte Eingabetexte vorbereiten
        validated_source_texts = []
        for item in batch_items:
            if isinstance(item, dict):
                text_content = item.get('text', '')
            else:
                text_content = str(item).strip()
            
            if not isinstance(text_content, str):
                text_content = str(text_content)
            
            if not text_content.strip():
                text_content = ""
            
            # KORREKTUR: Entferne ALLE bestehenden Sprachpräfixe
            text_content = re.sub(r'^<2[a-z]{2}>\s*', '', text_content).strip()
            
            # KRITISCH: Erstelle korrekten MADLAD400-Input mit Zielsprache im Source
            # MADLAD400 Format: "<2de> source_text"
            prefixed_text = f"<2{target_lang}> {text_content}"
            validated_source_texts.append(prefixed_text)

        try:
            # Tokenisierung
            source_encoded = tokenizer(
                validated_source_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Token-Konvertierung für CTranslate2
            source_tokens = []
            for input_ids in source_encoded["input_ids"]:
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                while tokens and tokens[-1] in ["</s>", "<pad>"]:
                    tokens.pop()
                source_tokens.append(tokens)
            
            # Kein Target-Prefix bei MADLAD400!
            # Das Ziel ist bereits im Source-Text kodiert
            results = translator.translate_batch(
                source_tokens,
                beam_size=10,
                patience=5.0,
                length_penalty=1.0,
                repetition_penalty=1.05,
                no_repeat_ngram_size=2,
                max_decoding_length=256
            )

            # Dekodierung und Bereinigung
            batch_translations = []
            for result in results:
                if result.hypotheses:
                    translation_tokens = result.hypotheses[0]
                    translation_ids = tokenizer.convert_tokens_to_ids(translation_tokens)
                    translation = tokenizer.decode(
                        translation_ids, 
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    
                    # Bereinige eventuelle Restpräfixe
                    translation = clean_madlad_output(translation, target_lang)
                    batch_translations.append(translation)
                else:
                    batch_translations.append("ERROR: NO TRANSLATION")

            all_translations.extend(batch_translations)
            
        except Exception as e:
            logger.error(f"Fehler bei der Batch-Übersetzung: {e}", exc_info=True)
            # Fallback auf Einzelübersetzung
            individual_translations = []
            for single_text in validated_source_texts:
                try:
                    single_result = translate_single_text_fallback(
                        single_text, translator, tokenizer, target_lang
                    )
                    individual_translations.append(single_result)
                except Exception as se:
                    logger.error(f"Einzelübersetzung fehlgeschlagen: {se}")
                    individual_translations.append("ERROR: INDIVIDUAL TRANSLATION FAILED")
            
            all_translations.extend(individual_translations)
        
        # GPU-Speicher freigeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    return all_translations

def translate_single_text_fallback(text, translator, tokenizer, target_lang):
    """
    Fallback-Funktion für Einzeltextübersetzung bei MADLAD400-Fehlern.
    """
    try:
        source_encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        tokens = tokenizer.convert_ids_to_tokens(source_encoded["input_ids"][0])
        while tokens and tokens[-1] in ["</s>", "<pad>"]:
            tokens.pop()
        
        # Einzelübersetzung OHNE Target-Prefix
        result = translator.translate_batch(
            [tokens],
            # target_prefix=None,  # Kein Target-Prefix!
            beam_size=3,
            patience=1.0
        )
        
        if result and result[0].hypotheses:
            translation_tokens = result[0].hypotheses[0]
            translation_ids = tokenizer.convert_tokens_to_ids(translation_tokens)
            translation = tokenizer.decode(translation_ids, skip_special_tokens=True)
            return clean_madlad_output(translation, target_lang)
        else:
            return "ERROR: NO SINGLE TRANSLATION"
            
    except Exception as e:
        logger.error(f"Fallback-Übersetzung fehlgeschlagen: {e}")
        return "ERROR: FALLBACK FAILED"

def clean_madlad_output(text, target_lang):
    """
    Erweiterte Bereinigung von MADLAD400-Ausgaben mit verbesserter Sprachpräfix-Entfernung.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Entferne ALLE Sprachpräfixe (auch fehlerhafte)
    text = re.sub(rf'^<2{target_lang}>\s*', '', text).strip()
    text = re.sub(r'^<2[a-z]{2}>\s*', '', text).strip()
    text = re.sub(r'^<[0-9][a-z]{2}>\s*', '', text).strip()  # Varianten mit Zahlen
    
    # Entferne mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Entferne führende Satzzeichen
    text = re.sub(r'^[.!?,:;]\s*', '', text).strip()
    
    # Entferne eventuelle verbleibende Sprachkennzeichnungen
    text = re.sub(r'^(de|en|fr|es|it)\s*[.:]?\s*', '', text, flags=re.IGNORECASE).strip()
    
    # Deutsche Kapitalisierung
    if target_lang == "de" and text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text

def translate_segments_optimized_safe(transcription_file, translation_output_file, 
                                    cleaned_source_output_file, source_lang="en", 
                                    target_lang="de", batch_size=None):
    """
    Sichere Übersetzung mit Fortschrittsspeicherung, die robust gegen Wiederaufnahme ist,
    indem sie die korrekte schlüsselbasierte Logik verwendet.
    """
    # KORREKTUR: Ruft die neue, spezialisierte Funktion für die schlüsselbasierte Wiederaufnahme auf.
    should_continue_translation, processed_translation_keys = handle_key_based_continuation(
        translation_output_file, transcription_file, key_column_index=0
    )
    should_continue_source, processed_source_keys = handle_key_based_continuation(
        cleaned_source_output_file, transcription_file, key_column_index=0
    )
    
    if not should_continue_translation or not should_continue_source:
        logger.info("Eine der Ausgabedateien ist vollständig und wird nicht überschrieben. Überspringe Übersetzung.")
        return translation_output_file, cleaned_source_output_file

    all_input_segments = []
    with open(transcription_file, mode="r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter="|")
        next(csvreader, None)
        for row in csvreader:
            if len(row) >= 3:
                start_str, end_str, text_content = row[0], row[1], row[2]
                segment_data = {
                    "start": parse_time(start_str), "end": parse_time(end_str), 
                    "text": text_content, "start_str": start_str, "end_str": end_str
                }
                if segment_data["start"] is not None and segment_data["end"] is not None:
                    all_input_segments.append(segment_data)

    if not all_input_segments:
        logger.error("Keine gültigen Eingabesegmente nach der Validierung gefunden.")
        return None, None
    
    processed_keys = processed_translation_keys.intersection(processed_source_keys)
    segments_to_process = [
        seg for seg in all_input_segments if seg['start_str'] not in processed_keys
    ]
    
    logger.info(f"Übersetze {len(segments_to_process)} verbleibende Segmente von insgesamt {len(all_input_segments)}.")

    if not segments_to_process:
        logger.info("Alle Segmente bereits übersetzt.")
        return translation_output_file, cleaned_source_output_file

    existing_translations = []
    if os.path.exists(translation_output_file):
        df_existing = pd.read_csv(translation_output_file, sep='|', dtype=str)
        existing_translations = df_existing.to_dict('records')

    existing_source = []
    if os.path.exists(cleaned_source_output_file):
        df_existing_source = pd.read_csv(cleaned_source_output_file, sep='|', dtype=str)
        existing_source = df_existing_source.to_dict('records')

    all_translations = existing_translations
    all_cleaned_sources = existing_source

    try:
        translate_start_time = time.time()
        print("\n--- [ Übersetzung Live-Ausgabe ] ---\n")
        with gpu_context():
            translator, tokenizer = load_madlad400_translator_optimized()
            optimal_batch_size = batch_size or calculate_optimal_batch_size_rtx4050(
                segments_to_process, tokenizer
            )
            
            batch_iterator = range(0, len(segments_to_process), optimal_batch_size)
            for i in tqdm(batch_iterator, desc="\n\nÜbersetze Batches"):
                batch = segments_to_process[i:i + optimal_batch_size]
                
                translations = translate_batch_madlad400(
                    batch, translator, tokenizer, source_lang, target_lang, optimal_batch_size
                )
                
                new_translations_batch = []
                new_sources_batch = []

                for j, seg in enumerate(batch):
                    translated_text = translations[j] if j < len(translations) else ""
                    
                    print(f"Segment [{seg['start_str']} -> {seg['end_str']}] = {translated_text}")
                    
                    validation_segment = {"start": seg['start'], "end": seg['end'], "text": translated_text}
                    if is_valid_segment(validation_segment, check_text_content=True):
                        new_translations_batch.append({
                            'startzeit': seg["start_str"], 'endzeit': seg["end_str"],
                            'text': sanitize_for_csv_and_tts(translated_text)
                        })
                        new_sources_batch.append({
                            'startzeit': seg["start_str"], 'endzeit': seg["end_str"],
                            'text': sanitize_for_csv_and_tts(seg['text'])
                        })
                    else:
                        logger.warning(f"Ungültige Übersetzung verworfen für Segment bei {seg['start_str']}")
                
                if new_translations_batch:
                    all_translations.extend(new_translations_batch)
                    all_cleaned_sources.extend(new_sources_batch)
                    save_progress_csv(all_translations, translation_output_file, headers=['startzeit', 'endzeit', 'text'])
                    save_progress_csv(all_cleaned_sources, cleaned_source_output_file, headers=['startzeit', 'endzeit', 'text'])

        logger.info(f"Übersetzung abgeschlossen: {len(all_translations)} gültige Segmente.")
        print("\n---------------------------------")
        print("|<< Übersetzung abgeschlossen! >>|")
        print("---------------------------------")
        translate_end_time = time.time() - translate_start_time
        logger.info(f"Transkription abgeschlossen in {translate_end_time:.2f} Sekunden")
        print(f"Transkription abgeschlossen in {translate_end_time:.2f} Sekunden -> {(translate_end_time / 60):.2f} Minuten")
        return translation_output_file, cleaned_source_output_file

    except Exception as e:
        logger.error(f"Fehler bei der Übersetzung: {e}", exc_info=True)
        if 'all_translations' in locals() and all_translations: save_progress_csv(all_translations, translation_output_file, headers=['startzeit', 'endzeit', 'text'])
        if 'all_cleaned_sources' in locals() and all_cleaned_sources: save_progress_csv(all_cleaned_sources, cleaned_source_output_file, headers=['startzeit', 'endzeit', 'text'])
        logger.info("Notfall-Speicherung des Fortschritts durchgeführt.")
        return None, None

def is_valid_segment(segment, check_text_content=True):
    """
    Erweiterte Hilfsfunktion zur Segmentvalidierung mit detailliertem Logging.
    Prüft Zeitstempel und Textinhalt und protokolliert den genauen Grund für das Verwerfen.

    Args:
        segment (dict): Ein Dictionary, das ein Segment repräsentiert. Muss 'start', 'end', 'start_str', 'end_str' und 'text' enthalten.
        check_text_content (bool): Wenn True, wird auch der Textinhalt validiert.

    Returns:
        bool: True, wenn das Segment gültig ist, andernfalls False.
    """
    start = segment.get("start")
    end = segment.get("end")

    # Prüfung 1: Zeitstempel müssen gültig und in der korrekten Reihenfolge sein.
    if start is None or end is None or end <= start:
        logger.warning(
            f"Ungültiges Segment verworfen: Zeitstempel-Problem. "
            f"Start: {segment.get('start_str', 'N/A')}, Ende: {segment.get('end_str', 'N/A')}. "
            f"Geparsed als Start={start}, Ende={end}."
        )
        return False

    # Prüfung 2: Textinhalt muss vorhanden und sinnvoll sein.
    if check_text_content:
        text = segment.get("text", "").strip()
        
        # Verwirf Segmente, die leer sind oder nur aus sehr kurzem, generischem Text bestehen.
        forbidden_texts = {"de", "en", "fr", "es", "dt>", "de.", "en."}
        if not text or len(text) < 3 or text.lower() in forbidden_texts:
            logger.warning(
                f"Ungültiges Segment verworfen: Text-Problem. "
                f"Start: {segment.get('start_str', 'N/A')}, Text: '{text[:50]}...'"
            )
            return False
            
    # Wenn alle Prüfungen bestanden wurden, ist das Segment gültig.
    return True

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

# Lade das englische und deutsche Transformer-Modell einmal global
try:
    nlp_en = spacy.load("en_core_web_trf")
except OSError:
    print("Fehler: Das Modell 'en_core_web_trf' ist nicht installiert. Bitte mit 'python -m spacy download en_core_web_trf' nachinstallieren.")
    nlp_en = spacy.blank("en")
    nlp_en.add_pipe("sentencizer")

try:
# Laden Sie das spaCy-Modell, das nun auf der GPU laufen wird
    nlp_de = spacy.load("de_dep_news_trf")
except OSError:
    print("Fehler: Das Modell 'de_dep_news_trf' ist nicht installiert. Bitte mit 'python -m spacy download de_dep_news_trf' nachinstallieren.")
    nlp_de = spacy.blank("de")
    nlp_de.add_pipe("sentencizer")

def format_translation_for_tts(input_file, output_file, nlp, lang="de-DE", use_embeddings=False, embeddings_file=None, CHAR_LIMIT=None, lt_path="D:\\LanguageTool-6.6"):
    """
    Optimiert übersetzte Segmente für TTS, indem Grammatik korrigiert und der Text in Abschnitte
    mit maximal 150 Zeichen pro Zeitstempel aufgeteilt wird, es sei denn, das neue Segment hätte
    weniger als 3 Wörter. Implementiert eine offlinefähige Namenerkennung mit spaCy.
    
    Args:
        input_file (str): Pfad zur CSV-Datei mit den übersetzten Segmenten.
        output_file (str): Pfad zur Ausgabedatei mit formatierten Segmenten.
        nlp (spacy.Language): Das geladene spaCy-Sprachmodell (z.B. nlp_de oder nlp_en).
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
        
    format_for_tts_start = time.time()
    
    try:
        logger.info(f"Starte TTS-Formatierung für {lang}...")
        print("----------------------------------")
        print("|<< Starte TTS-Formatierung >>|")
        print("----------------------------------")
        
        # LanguageTool-Server starten aus dem angegebenen Pfad
        lt_jar_path = os.path.join(lt_path, "languagetool-server.jar")
        if not os.path.exists(lt_jar_path):
            logger.error(f"LanguageTool-Server JAR-Datei nicht gefunden unter: {lt_jar_path}")
            return input_file
        
        port = 8010  # Standardport für LanguageTool
        lt_process = subprocess.Popen(
            ["java", "-Xmx4g", "-cp", lt_jar_path, "org.languagetool.server.HTTPServer", "--port", str(port), "--allow-origin", "*"],
            cwd=lt_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,  # Verhindert Hängen
            text=True,
            bufsize=0  # Unbepuffert für sofortige Ausgabe
        )
        logger.info(f"LanguageTool-Server gestartet aus {lt_path} auf Port {port}.")
        time.sleep(10)  # Warten, bis der Server hochgefahren ist
        
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
        def protect_names_from_doc(doc):
            """
            Extrahiert Namen aus einem bereits verarbeiteten spaCy-Doc und erstellt Platzhalter.
            """
            protected_names = {}
            text = doc.text
            for ent in doc.ents:
                if ent.label_ == "PER":  # Personen-Namen
                    placeholder = f"__NAME_{ent.start_char}_{ent.end_char}__"
                    protected_names[placeholder] = ent.text
            modified_text = text
            for placeholder, name in protected_names.items():
                modified_text = modified_text.replace(name, placeholder)
            return modified_text, protected_names

        def restore_names(text, protected_names):
            """
            Stellt die maskierten Namen im Text wieder her.
            """
            for placeholder, name in protected_names.items():
                text = text.replace(placeholder, name)
            return text

        def ensure_text_consistency_de(nlp_model, text):
            """
            Stellt mit spaCy sicher, dass der deutsche Text sinnvoll segmentiert und konsistent ist.
            """
            doc = nlp_model(text)
            return " ".join([sent.text.strip() for sent in doc.sents if sent.text.strip()])
        
        def split_text_robustly(text: str, char_limit: int = 150, min_words_for_new_chunk: int = 3) -> list[str]:
            """
            Teilt Text robust in Abschnitte unter dem 'char_limit' auf.
            - Teilt primär an Satzgrenzen.
            - Wenn ein Satz das Limit überschreitet, wird er an Wortgrenzen geteilt.
            - Ein neuer Chunk wird nur erstellt, wenn die verbleibenden Wörter die Mindestanzahl erreichen.

            Args:
                text (str): Der zu teilende Text.
                char_limit (int): Die maximale Zeichenanzahl pro Chunk.
                min_words_for_new_chunk (int): Mindestanzahl an Wörtern für einen neuen Chunk.

            Returns:
                list[str]: Eine Liste mit den Text-Chunks.
            """
            chunks = []
            # Zuerst den Text in Sätze aufteilen (einfacher Regex)
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            
            current_chunk = ""
            for sentence in sentences:
                if not sentence:
                    continue
                    
                # Wenn der Satz allein schon zu lang ist, muss er intern geteilt werden
                if len(sentence) > char_limit:
                    # Aktuellen Chunk erst abschließen, falls vorhanden
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    
                    # Teile den langen Satz an Wortgrenzen
                    words = sentence.split()
                    sub_chunk = ""
                    for i, word in enumerate(words):
                        # Teste, ob das nächste Wort den sub_chunk überlaufen lässt
                        if len(sub_chunk) + len(word) + 1 > char_limit:
                            # Prüfe, ob die verbleibenden Wörter die 3-Wörter-Regel erfüllen
                            remaining_words = words[i:]
                            if len(remaining_words) < min_words_for_new_chunk:
                                # Regel verletzt -> hänge den Rest an und beende
                                sub_chunk += " " + " ".join(remaining_words)
                                break  # Schleife über Wörter beenden
                            else:
                                # Regel erfüllt -> schließe den sub_chunk ab und starte einen neuen
                                chunks.append(sub_chunk.strip())
                                sub_chunk = word
                        else:
                            sub_chunk = f"{sub_chunk} {word}" if sub_chunk else word
                    
                    # Füge den letzten sub_chunk des langen Satzes hinzu
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                        
                # Wenn der Satz zum aktuellen Chunk passt, füge ihn hinzu
                elif len(current_chunk) + len(sentence) + 1 <= char_limit:
                    current_chunk = f"{current_chunk} {sentence}" if current_chunk else sentence
                # Ansonsten schließe den alten Chunk ab und starte einen neuen mit diesem Satz
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            # Füge den allerletzten Chunk hinzu, falls er nicht leer ist
            if current_chunk:
                chunks.append(current_chunk.strip())
                
            return chunks
        
        def ensure_context_closure(segments: list) -> list:
            """
            Durchläuft die fertigen Segmente und sorgt dafür, dass jedes Segment
            mit einem vollständigen Satz endet. Beginnt ein Segment mit einem
            Partizip/Gerundium oder fehlt ein Schlusspunkt, wird der vorangehende
            Satz angehängt; Dubletten werden unmittelbar entfernt.
            """
            fixed = []
            for idx, seg in enumerate(segments):
                text = seg["text"].strip()

                fragile = (
                    re.match(r"^(?:[A-Za-zÄÖÜäöüß]+end|[A-Za-z]+nd)\b", text, re.I)  # grobe Partizip-Heuristik
                    or not text.endswith(('.', '!', '?'))
                )

                if fragile and idx > 0:
                    ctx = fixed[-1]["text"].strip()
                    candidate = f"{ctx} {text}".strip()
                    # Dublette entfernen, falls das Segment jetzt identisch wäre
                    if not candidate.startswith(ctx * 2):
                        text = candidate

                # Doppelten Kontext wieder abschneiden
                if fixed and text.startswith(fixed[-1]["text"]):
                    text = text[len(fixed[-1]["text"]):].lstrip(" .")

                # Sicherstellen, dass ein Satzende vorhanden ist
                if text and text[-1] not in ".!?":
                    text += "."

                seg["text"] = text
                fixed.append(seg)
            return fixed
        
        def merge_lowercase_continuations(rows: list[dict]) -> list[dict]:
            merged = []
            i = 0
            while i < len(rows):
                cur = rows[i].copy()
                while (i + 1 < len(rows)
                    and rows[i + 1]['text'][:1].islower()):
                    nxt = rows[i + 1]
                    cur['text'] = f"{cur['text']} {nxt['text']}"
                    cur['endzeit'] = nxt['endzeit']
                    i += 1
                merged.append(cur)
                i += 1
            return merged

        # Liste für formatierte Segmente
        formatted_segments = []
        total_segments = len(df)
        split_count = 0
        segments_exceeding_limit = 0

        # Schritt 1: Alle Texte aus dem DataFrame extrahieren
        texts_to_process = df['text'].astype(str).str.strip().tolist()
        
        # Schritt 2: Alle Texte in einem einzigen, GPU-beschleunigten Batch verarbeiten
        logger.info(f"Verarbeite {len(texts_to_process)} Texte mit spaCy auf der GPU...")
        # nlp.pipe ist für die Batch-Verarbeitung optimiert und nutzt die GPU effizient
        processed_docs = list(nlp.pipe(texts_to_process, batch_size=50)) # batch_size kann je nach VRAM angepasst werden
        logger.info("spaCy-Verarbeitung abgeschlossen.")

        # Schritt 3: Die verarbeiteten Dokumente in der Hauptschleife verwenden
        for i, row in tqdm(df.iterrows(), total=total_segments, desc="Formatiere Segmente"):
            # Zeiten parsen
            start_time = parse_time(row['startzeit'])
            end_time = parse_time(row['endzeit'])
            
            if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
                logger.warning(f"Ungültige Zeiten in Segment {i}: {row['startzeit']} - {row['endzeit']}")
                continue
            
            # Das bereits verarbeitete Dokument aus der Liste holen
            doc = processed_docs[i]
            
            if not doc.text.strip():
                logger.debug(f"Leeres Segment {i}, wird übersprungen.")
                continue
                
            # 1. Namen schützen (mit der neuen Funktion, die ein Doc entgegennimmt)
            protected_text, protected_names = protect_names_from_doc(doc)

            # 2. Grammatikkorrektur durchführen
            corrected_text = tool.correct(protected_text)

            # 3. Namen wiederherstellen
            corrected_text = restore_names(corrected_text, protected_names)

            # 4. Textkonsistenz sicherstellen (kann ebenfalls das Doc verwenden)
            final_text = ensure_text_consistency_de(nlp, corrected_text)
            
            # 5. Text in Abschnitte mit maximal 150 Zeichen aufteilen
            CHAR_LIMIT = CHAR_LIMIT
            text_chunks = split_text_robustly(final_text, char_limit=CHAR_LIMIT, min_words_for_new_chunk=3)
            
            # Prüfe, ob Segmente das Limit überschreiten (für Statistiken)
            for chunk in text_chunks:
                if len(chunk) > CHAR_LIMIT:
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
        
        formatted_segments = merge_lowercase_continuations(formatted_segments)
        
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
            format_for_tts = time.time() - format_for_tts_start
            logger.info(f"Step execution time: {(format_for_tts / 60):.0f}:{(format_for_tts):.3f} minutes")
            print(f"{(format_for_tts):.2f} Sekunden")
            print(f"{(format_for_tts / 60 ):.2f} Minuten")
            print(f"{(format_for_tts / 3600):.2f} Stunden")
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

def create_atempo_filter_string(speed_factor):
    """
    Erstellt eine robuste FFmpeg-Filterkette für die Geschwindigkeitsanpassung.
    Berücksichtigt den gültigen Bereich von atempo [0.5, 100.0] und verkettet 
    Filter für Werte außerhalb des Standardbereichs [0.5, 2.0].

    Args:
        speed_factor (float): Gewünschter Geschwindigkeitsfaktor.

    Returns:
        str: FFmpeg-Filterkette für atempo.
    """
    # Den Faktor auf den von FFmpeg unterstützten Gesamtbereich begrenzen
    speed_factor = max(0.5, min(speed_factor, 100.0))
    
    # Wenn der Faktor bereits im optimalen Einzel-Filter-Bereich liegt
    if 0.5 <= speed_factor <= 2.0:
        return f"atempo={speed_factor}"
    
    atempo_chain = []
    
    # Für extreme Beschleunigung (Faktor > 2.0)
    if speed_factor > 2.0:
        remaining_factor = speed_factor
        while remaining_factor > 2.0:
            atempo_chain.append("atempo=2.0")
            remaining_factor /= 2.0
        # Den verbleibenden Restfaktor hinzufügen
        if remaining_factor > 0.5: # Sicherstellen, dass der Restfaktor gültig ist
            atempo_chain.append(f"atempo={remaining_factor}")
            
    # Für extreme Verlangsamung (Faktor < 0.5)
    # Diese Logik ist theoretisch, da wir den Faktor bereits auf 0.5 begrenzen,
    # aber für die Vollständigkeit hier belassen.
    elif speed_factor < 0.5:
        atempo_chain.append("atempo=0.5") # Direkt auf das Minimum setzen

    return ",".join(atempo_chain)

# Synthetisieren
def text_to_speech_with_voice_cloning(
    translation_file,
    sample_path_1,
    sample_path_2,
    sample_path_3,
    #sample_path_4,
    #sample_path_5,
    output_path,
    batch_size=32
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
    
    # Pfade für temporäre Dateien definieren
    if not os.path.exists(TTS_TEMP_CHUNKS_DIR):
        os.makedirs(TTS_TEMP_CHUNKS_DIR)

    # Überprüfen, ob die Zieldatei existiert und den Benutzer fragen
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"TTS-Audio bereits vorhanden: {output_path}")
            return
        else:
            # Sicheres Aufräumen alter temporärer Dateien nur bei expliziter Überschreibungsanforderung
            logger.info("Überschreibe Zieldatei. Alte temporäre Dateien werden entfernt.")
            if os.path.exists(TTS_TEMP_CHUNKS_DIR): shutil.rmtree(TTS_TEMP_CHUNKS_DIR)
            if os.path.exists(TTS_PROGRESS_MANIFEST): os.remove(TTS_PROGRESS_MANIFEST)
            os.makedirs(TTS_TEMP_CHUNKS_DIR)

    # Flag zur Nachverfolgung des Erfolgs für eine sichere Bereinigung am Ende
    process_successful = False

    try:
        print(f"------------------")
        print(f"|<< Starte TTS >>|")
        print(f"------------------")

        tts_start_time = time.time()

        # GPU-Optimierungen aktivieren für maximale Leistung
        cuda_stream = setup_gpu_optimization()
        # TTS-Modell laden und auf GPU verschieben
        # --- Stufe 1: Ausfallsichere TTS-Synthese mit Batch-Verarbeitung ---
        
        # Fortschritt aus Manifest lesen, um bereits verarbeitete Segmente zu identifizieren
        processed_segments = {}
        if os.path.exists(TTS_PROGRESS_MANIFEST):
            logger.info(f"Lese Fortschritt aus {TTS_PROGRESS_MANIFEST}")
            with open(TTS_PROGRESS_MANIFEST, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='|')
                next(reader, None)  # Header sicher überspringen, auch wenn die Datei leer ist
                for row in reader:
                    if len(row) >= 5: # startzeit, endzeit, text, chunk_path, original_id
                        processed_segments[int(row[4])] = row[3] # Key: original_id -> Value: chunk_path
        
        # Alle zu verarbeitenden Segmente aus der Eingabedatei laden
        all_input_segments = []
        with open(translation_file, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader) # Header überspringen
            for i, row in enumerate(reader):
                if len(row) >= 3:
                    all_input_segments.append({'id': i, 'startzeit': row[0], 'endzeit': row[1], 'text': row[2].strip()})

        # Filtere die Segmente, die noch nicht synthetisiert wurden
        segments_to_process = [seg for seg in all_input_segments if seg['id'] not in processed_segments]

        if not segments_to_process:
            logger.info("Alle Segmente wurden bereits synthetisiert. Gehe direkt zur Audio-Assemblierung.")
        else:
            logger.info(f"{len(segments_to_process)} von {len(all_input_segments)} Segmenten werden synthetisiert.")
        with gpu_context():
            print(f"TTS-Modell wird initialisiert...")
            xtts_model = load_xtts_v2()
            
            # Bedingungen für die Stimme (Latent und Embedding) aus den Sprachbeispielen generieren
            with torch.cuda.stream(cuda_stream), torch.inference_mode():
                sample_paths = [
                    sample_path_1,
                    sample_path_2,
                    sample_path_3,
                    #sample_path_4,
                    #sample_path_5
                ]
                gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                    sound_norm_refs=False,
                    audio_path=sample_paths,
                    load_sr=22050
                )

            # DeepSpeed Inference für TTS initialisieren
            print(f"Initialisiere DeepSpeed Inference für TTS...")
            ds_config = {
                "enable_cuda_graph": False,  # CUDA Graphs sind bei variabler Inputlänge schwierig
                "dtype": torch.float16,      # Float32 für Kompatibilität
                "replace_with_kernel_inject": True  # Optimierung für Inferenz
            }
            ds_engine = deepspeed.init_inference(
                model=xtts_model,
                tensor_parallel={"tp_size": 1},  # Tensor-Parallelismus auf 1 setzen
                dtype=torch.float32,
                replace_with_kernel_inject=True
            )

            optimized_tts_model = ds_engine.module
            logger.info("DeepSpeed Inference für TTS initialisiert.")

            # Basismodell freigeben, um Speicher zu sparen
            del xtts_model

            # Synthese-Schleife
            for segment_info in tqdm(segments_to_process, desc="Synthetisiere Audio-Chunks"):
                with torch.cuda.stream(cuda_stream), torch.inference_mode():
                    result = optimized_tts_model.inference(
                        text=segment_info['text'],
                        language="de",
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        speed=1.2,
                        temperature=0.85,
                        repetition_penalty=10.0,
                        enable_text_splitting=False,
                        top_k=70,
                        top_p=0.9
                    )
                
                print(f"\nSegment {segment_info['id']} verarbeitet: {segment_info['text']}...\n")
                
                audio_clip = result.get("wav", np.zeros(1, dtype=np.float32))
                chunk_filename = f"chunk_{segment_info['id']}.wav"
                chunk_path = os.path.join(TTS_TEMP_CHUNKS_DIR, chunk_filename)
                sf.write(chunk_path, np.array(audio_clip), 24000)

                # Schreibe Fortschritt in die Manifest-Datei
                with open(TTS_PROGRESS_MANIFEST, mode='a', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter='|')
                    if os.path.getsize(TTS_PROGRESS_MANIFEST) == 0: writer.writerow(['startzeit', 'endzeit', 'text', 'chunk_path', 'original_id'])
                    writer.writerow([segment_info['startzeit'], segment_info['endzeit'], segment_info['text'], chunk_path, segment_info['id']])
            
            del optimized_tts_model, ds_engine, gpt_cond_latent, speaker_embedding

        # ======================================================================
        # PHASE 2: Zeitgenaue NumPy-basierte Audio-Assemblierung
        # ======================================================================
        print("\n|<< Starte finale Audio-Assemblierung >>|")
        
        # Lese das vollständige Manifest
        final_manifest = []
        with open(TTS_PROGRESS_MANIFEST, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|'); next(reader)
            for row in reader:
                if len(row) >= 4 and row[3].strip() and os.path.exists(row[3]):
                    final_manifest.append(row)
        
        if not final_manifest: raise RuntimeError("Keine gültigen Audio-Chunks zum Zusammenfügen gefunden.")

        # Sortiere nach Startzeit
        final_manifest.sort(key=lambda x: parse_time(x[0]))

        # Berechne die Gesamtlänge und allokiere das NumPy-Array
        sampling_rate = 24000
        last_segment = final_manifest[-1]
        max_length_seconds = parse_time(last_segment[1])
        # Lade den letzten Audio-Clip, um seine Dauer zu bekommen und die Gesamtlänge exakt zu bestimmen
        last_clip_data, _ = sf.read(last_segment[3])
        max_audio_length = int(max_length_seconds * sampling_rate) + len(last_clip_data) + sampling_rate # Sicherheitspuffer
        final_audio = np.zeros(max_audio_length, dtype=np.float32)

        # Füge jeden Chunk an der korrekten Position ein
        current_position_samples = 0
        for segment_data in tqdm(final_manifest, desc="Assembliere Audio-Chunks"):
            start_sec = parse_time(segment_data[0])
            chunk_path = segment_data[3]
            audio_clip, _ = sf.read(chunk_path, dtype='float32')

            start_pos_samples = int(start_sec * sampling_rate)
            # Verhindere Überlappung, indem der Startpunkt angepasst wird
            if start_pos_samples < current_position_samples:
                start_pos_samples = current_position_samples

            end_pos_samples = start_pos_samples + len(audio_clip)
            if end_pos_samples > len(final_audio):
                final_audio = np.pad(final_audio, (0, end_pos_samples - len(final_audio)), 'constant')
            
            final_audio[start_pos_samples:end_pos_samples] = audio_clip
            current_position_samples = end_pos_samples

        # Trimme das finale Audio auf die tatsächlich genutzte Länge
        final_audio = final_audio[:current_position_samples]

        # Normalisieren und Speichern
        if final_audio.size > 0:
            final_audio /= np.max(np.abs(final_audio)) + 1e-8
        else:
            logger.warning("Kein Audio generiert, erstelle leere Datei.")
            final_audio = np.zeros((1,), dtype=np.float32)

        sf.write(output_path, final_audio.astype(np.float32), sampling_rate)

        process_successful = True
        
        print(f"---------------------------")
        print(f"|<< TTS abgeschlossen!! >>|")
        print(f"---------------------------")

        tts_end_time = time.time() - tts_start_time
        logger.info(f"TTS abgeschlossen in {tts_end_time:.2f} Sekunden")
        print(f"{(tts_end_time):.2f} Sekunden")
        print(f"{(tts_end_time / 60 ):.2f} Minuten")
        print(f"{(tts_end_time / 3600):.2f} Stunden")

        logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}")
    except Exception as e:
        logger.error(f"Fehler bei der TTS-Synthese: {str(e)}")
    finally:
        # Sichere Bereinigung
        if process_successful:
            logger.info("Prozess erfolgreich, räume temporäre Dateien auf.")
            if os.path.exists(TTS_TEMP_CHUNKS_DIR): shutil.rmtree(TTS_TEMP_CHUNKS_DIR)
            if os.path.exists(TTS_PROGRESS_MANIFEST): os.remove(TTS_PROGRESS_MANIFEST)
        else:
            logger.warning("Prozess nicht erfolgreich. Temporäre Dateien werden für Wiederaufnahme beibehalten.")

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
        logger.info(f"Mixdown abgeschlossen in {mix_end_time:.2f} Sekunden")
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
    create_voice_sample(
        ORIGINAL_AUDIO_PATH,
        SAMPLE_PATH_1, 
        SAMPLE_PATH_2,
        SAMPLE_PATH_3,
        #SAMPLE_PATH_4,
        #SAMPLE_PATH_5
    )
    
    # 6) Spracherkennung (Transkription) mit Whisper
    
    segments = transcribe_audio_with_timestamps(ORIGINAL_AUDIO_PATH, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return
    
    """
    ensure_transcription_quality_en(TRANSCRIPTION_FILE, TRANSCRIPTION_CLEANED)
    
    # 6.1) Wiederherstellung der Interpunktion
    restore_punctuation(TRANSCRIPTION_CLEANED, PUNCTED_TRANSCRIPTION_FILE)
    
    # 6.2) Grammatische Korrektur der Transkription
    correct_grammar_transcription(PUNCTED_TRANSCRIPTION_FILE, CORRECTED_TRANSCRIPTION_FILE, lang="en-US")
    
    merge_transcript_chunks(
        input_file=CORRECTED_TRANSCRIPTION_FILE,
        output_file=MERGED_TRANSCRIPTION_FILE,
        min_dur=MIN_DUR,
        max_dur=MAX_DUR,
        max_gap=MAX_GAP,
        max_chars=MAX_CHARS,
        min_words=MIN_WORDS,
        iterations=ITERATIONS
    )
    """
    # 6.1) Wiederherstellung der Interpunktion
    restore_punctuation(TRANSCRIPTION_FILE, PUNCTED_TRANSCRIPTION_FILE_2)
    
    format_translation_for_tts(
        PUNCTED_TRANSCRIPTION_FILE_2,
        FORMATTED_TRANSKRIPTION_FILE,
        nlp_en,
        lang="en_US",
        use_embeddings=False,
        embeddings_file=None,
        CHAR_LIMIT=CHAR_LIMIT_TRANSCRIPTION,
        lt_path="D:\\LanguageTool-6.6"
    )
    
    # 7) Übersetzung der Segmente mithilfe von MADLAD400
    translation_file_path, cleaned_source_path = translate_segments_optimized_safe(
        transcription_file=FORMATTED_TRANSKRIPTION_FILE,
        translation_output_file=TRANSLATION_FILE,
        cleaned_source_output_file=CLEANED_SOURCE_FOR_QUALITY_CHECK, # NEUER PARAMETER
        source_lang="en",
        target_lang="de"
    )
    
    if not translation_file_path or not cleaned_source_path:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden. Breche ab.")
        return
    
    #restore_punctuation_de(TRANSLATION_FILE, PUNCTED_TRANSLATION_FILE)

    # 8) Qualitätsprüfung der Übersetzung
    logger.info("Starte optionalen Schritt: Qualitätsprüfung der Übersetzung.")
    evaluate_translation_quality(
        source_csv_path=cleaned_source_path, # WICHTIG: Der neue, bereinigte Pfad
        translated_csv_path=translation_file_path, # Der Pfad zur passenden Übersetzung
        report_path=TRANSLATION_QUALITY_REPORT,
        model_name=ST_QUALITY_MODEL,
        threshold=SIMILARITY_THRESHOLD
    )
    logger.info("Qualitätsprüfung der Übersetzung (optional) abgeschlossen.")

    # 9) Zusammenführen von Übersetzungs-Segmenten
    merge_transcript_chunks(
        input_file=translation_file_path, # Verwende die validierte Übersetzungsdatei
        output_file=MERGED_TRANSLATION_FILE,
        min_dur=MIN_DUR_TRANSLATION,
        max_dur=MAX_DUR_TRANSLATION,
        max_gap=MAX_GAP_TRANSLATION,
        max_chars=MAX_CHARS_TRANSLATION,
        min_words=MIN_WORDS_TRANSLATION,
        iterations=ITERATIONS_TRANSLATION
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
        nlp_de,
        lang="de-DE",
        use_embeddings=True,
        embeddings_file=EMBEDDINGS_FILE_NPZ,
        CHAR_LIMIT=CHAR_LIMIT_TRANSLATION,
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
        # Finale Bereinigung, falls nötig (z.B. offene Dateien, Prozesse)
        if 'optimum_trt_model' in globals() and globals().get('optimum_trt_model') is not None:
            logger.info("Versuche Notfall-Freigabe von optimum_trt_model.")
            del globals()['optimum_trt_model'] # Sicherstellen, dass es gelöscht wird
        torch.cuda.empty_cache()
        logging.shutdown()