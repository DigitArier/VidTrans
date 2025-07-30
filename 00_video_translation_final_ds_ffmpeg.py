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
    force=True,     # Wichtig für Python 3.8+: Stellt sicher, dass diese Konfiguration greift
    encoding='utf-8'
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
import socket
import re
import gc
from typing import Callable, Optional, Any, List, Dict, Tuple, Union, Set
import dataclasses
from dataclasses import dataclass
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
from spacy.tokens import Span
#from spacy.language import Language
import ftfy
from ftfy import fix_encoding
import torch
from torch import autocast
from torch.cuda import Stream
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

def setup_global_torch_optimizations():
    """
    NEUE VERSION: Konzentriert sich auf globale, statische Optimierungen für PyTorch und CUDA.
    Die dynamische Stream-Erstellung wird in den StreamManager ausgelagert.
    Zentrale Funktion zur Initialisierung aller globalen Performance-Optimierungen.
    Diese Funktion bündelt die Logik der zuvor ungenutzten Setup-Funktionen.
    Sie sollte als allererstes in der main() aufgerufen werden.
    """
    # P-Core-Affinität für Windows-Systeme mit Admin-Rechten.
    try:
        # Prüft, ob das Skript auf Windows läuft und Admin-Rechte besitzt.
        if sys.platform == "win32" and ctypes.windll.shell32.IsUserAnAdmin():
            current_process = psutil.Process(os.getpid()) # Holt das aktuelle Prozessobjekt.
            # Ermittelt die Indizes der physischen CPU-Kerne (Performance-Cores).
            p_cores = [i for i in range(psutil.cpu_count(logical=False))]
            current_process.cpu_affinity(p_cores) # Beschränkt den Prozess auf die P-Cores.
            thread_count = len(p_cores) # Setzt die Thread-Anzahl auf die Anzahl der P-Cores.
            torch.set_num_threads(thread_count) # Informiert PyTorch über die gewünschte Thread-Anzahl.
            os.environ["OMP_NUM_THREADS"] = str(thread_count) # Setzt die Thread-Anzahl für OpenMP.
            os.environ["MKL_NUM_THREADS"] = str(thread_count) # Setzt die Thread-Anzahl für die Intel MKL-Bibliothek.
            logger.info(f"Prozessaffinität auf {thread_count} P-Cores gesetzt.") # Loggt die erfolgreiche Zuweisung.
    except Exception as e:
        # Fängt mögliche Fehler ab (z.B. bei fehlenden Rechten).
        logger.warning(f"CPU-Affinität konnte nicht gesetzt werden: {e}")

    logger.info("Starte globale Performance-Optimierungen für PyTorch und CUDA...") # Log-Nachricht.
    
    # WICHTIG: Umgebungsvariablen für PyTorch CUDA müssen VOR dem ersten GPU-Zugriff gesetzt werden.
    # Konfiguriert den PyTorch CUDA Memory Allocator, um Speicherfragmentierung bei großen Modellen zu reduzieren.
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256,expandable_segments:True'
    # Setzt CTranslate2-spezifische Caching-Parameter für den Speicher, optimiert für Inferenz.
    os.environ['CT2_CUDA_CACHING_ALLOCATOR_CONFIG'] = '4,3,10,104857600'
    os.environ['CT2_CUDA_ALLOW_FP16'] = '1' # Erlaubt CTranslate2 die Nutzung von FP16 für schnellere Berechnungen.
    os.environ['CT2_USE_EXPERIMENTAL_PACKED_GEMM'] = '1' # Aktiviert einen experimentellen, schnelleren Algorithmus für Matrixmultiplikationen.
    logger.info("Umgebungsvariablen für GPU-Speicheroptimierung gesetzt.") # Bestätigungs-Log.

    if torch.cuda.is_available(): # Führt GPU-spezifische Optimierungen nur aus, wenn eine CUDA-GPU vorhanden ist.
        # Aktiviert die von NVIDIA bereitgestellte Flash Attention Implementierung für erhebliche Beschleunigung.
        torch.backends.cuda.enable_flash_sdp(True)
        # Aktiviert eine speichereffizientere, aber potenziell langsamere Attention-Implementierung als Fallback.
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # Deaktiviert eine mathematisch exaktere, aber langsamere Attention-Variante.
        torch.backends.cuda.enable_math_sdp(False)
        logger.info("Optimierte Attention-Backends (Flash Attention) aktiviert.") # Loggt die Aktivierung.

        # Erlaubt PyTorch, den TensorFloat-32 (TF32) Datentyp auf Ampere-GPUs (wie der RTX 4050) für MatMul zu nutzen. Beschleunigt stark.
        torch.backends.cuda.matmul.allow_tf32 = True
        # Erlaubt der cuDNN-Bibliothek ebenfalls die Nutzung von TF32 für Convolution-Operationen.
        torch.backends.cudnn.allow_tf32 = True
        # Aktiviert den cuDNN Auto-Tuner, der den schnellsten Algorithmus für die jeweilige Hardware und Datengröße sucht.
        torch.backends.cudnn.benchmark = True
        # Deaktiviert den deterministischen Modus, der für Reproduzierbarkeit sorgt, aber langsamer ist. Für Inferenz nicht nötig.
        torch.backends.cudnn.deterministic = False
        # Weist PyTorch an, die cuSOLVER-Bibliothek für lineare Algebra zu bevorzugen, die auf Ampere-Karten optimiert ist.
        torch.backends.cuda.preferred_linalg_library("cusolver")
        logger.info("TF32 und cuSOLVER-Optimierungen für Ampere-GPU (RTX 4050) aktiviert.") # Loggt die Aktivierung.

        # Setzt die maximale Größe des Caches für cuFFT-Pläne (schnelle Fourier-Transformation).
        torch.backends.cuda.cufft_plan_cache.max_size = 32
        # Erlaubt reduzierte Präzision bei FP16-Operationen, was die Performance verbessern kann.
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        # Aktiviert den "Fast Path" für Multi-Head Attention, falls von der PyTorch-Version unterstützt.
        if hasattr(torch.backends, 'mha') and hasattr(torch.backends.mha, 'set_fastpath_enabled'):
            torch.backends.mha.set_fastpath_enabled(True)
        logger.info("Fortgeschrittene CUDA-Konfigurationen (cuFFT, MHA) geladen.") # Loggt die Aktivierung.

        torch.cuda.empty_cache() # Gibt allen ungenutzten, zwischengespeicherten GPU-Speicher frei.
        logger.info("Initialer GPU-Speicher-Cache geleert.") # Bestätigungs-Log.
    else:
        # Fallback-Nachricht, wenn keine GPU gefunden wird.
        logger.warning("CUDA nicht verfügbar. GPU-spezifische Optimierungen werden übersprungen.")

# Multiprocessing-Setup
mp.set_start_method('spawn', force=True)
os.environ['TORCH_BLAS_PREFER_CUBLASLT'] = '1'
os.environ["CT2_FLASH_ATTENTION"] = "1"
os.environ["CT2_VERBOSE"] = "1"
os.environ["CT2_CUDA_ALLOCATOR"] = "cuda_malloc_async"
os.environ["KMP_BLOCKTIME"] = "1"            # Optimale Thread-Wartezeit
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0" 
os.environ["CT2_CPU_ENABLE_MMAP"] = "1"  # Memory-Mapping für große Modelle
os.environ["CT2_CPU_PREFETCH"] = "32"    # Cache-Prefetching optimieren

# Geschwindigkeits- und Lautstärkeeinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.0   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.0   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 1.0     # Geschwindigkeitsfaktor für die Wiedergabe des Videos
VOLUME_ADJUSTMENT_44100 = 1.0   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.04   # Lautstärkefaktor für das Video

# ============================== 
# Hilfsfunktionen
# ==============================

def wait_for_server_ready(port: int, timeout: int = 60) -> bool:
    """
    Wartet, bis ein TCP-Port auf localhost erreichbar ist.

    Args:
        port (int): Der zu prüfende Port.
        timeout (int): Maximale Wartezeit in Sekunden.

    Returns:
        bool: True, wenn der Server rechtzeitig bereit ist, sonst False.
    """
    logger.info(f"Warte auf LanguageTool-Server auf Port {port} (max. {timeout}s)...")
    start_time = time.monotonic()
    
    while time.monotonic() - start_time < timeout:
        try:
            # Erstelle einen Socket und versuche, eine Verbindung herzustellen.
            with socket.create_connection(("localhost", port), timeout=1):
                logger.info(f"✅ LanguageTool-Server auf Port {port} ist erreichbar.")
                return True
        except (ConnectionRefusedError, socket.timeout):
            # Server noch nicht bereit, warte eine Sekunde und versuche es erneut.
            time.sleep(1)
            print(".", end="", flush=True) # Visuelles Feedback für den Benutzer
            
    logger.error(f"❌ Zeitüberschreitung: LanguageTool-Server auf Port {port} nach {timeout}s nicht bereit.")
    return False

@contextmanager
def StreamManager():
    """
    NEUER, ZENTRALER CONTEXT MANAGER: Kapselt die Erstellung, Verwendung und
    Synchronisation eines hochprioren CUDA-Streams.
    Ersetzt alle bisherigen Stream-Funktionen.
    """
    if not torch.cuda.is_available():
        # Fallback für CPU-Umgebungen, gibt einen Null-Kontext zurück.
        yield None
        return

    # Prioritäts-Range ermitteln, um Fehler zu vermeiden.
    # -1 ist typischerweise eine hohe, 0 eine normale Priorität.
    try:
        _, high_priority = torch.cuda.stream_priority_range()
    except Exception:
        high_priority = -1 # Sicherer Fallback-Wert
        
    stream = torch.cuda.Stream(priority=high_priority)
    
    try:
        # Führt den Code-Block innerhalb des 'with'-Statements auf diesem Stream aus.
        with torch.cuda.stream(stream):
            logger.debug(f"Betrete CUDA Stream-Kontext (Priorität: {high_priority})")
            yield stream
    finally:
        # WICHTIG: Stellt sicher, dass alle Operationen auf dem Stream abgeschlossen sind,
        # bevor der Code fortfährt. Verhindert Race Conditions.
        stream.synchronize()
        logger.debug("CUDA Stream-Kontext verlassen und synchronisiert.")

@dataclass
class EntityInfo:
    """Erweiterte Entity-Information für bessere Verwaltung"""
    text: str
    label: str
    start: int
    end: int
    language: str

try:
    # Versucht, die GPU für spaCy zu aktivieren
    spacy.require_gpu()
    print("✅ GPU für spaCy erfolgreich aktiviert.")
except:
    # Fallback auf CPU, falls die GPU-Aktivierung fehlschlägt
    print("⚠️ GPU für spaCy konnte nicht aktiviert werden. Fallback auf CPU.")

# Gerät bestimmen (GPU bevorzugt, aber CPU als Fallback)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Verwende Gerät: {device}")
if device == "cpu":
    logger.warning("GPU/CUDA nicht verfügbar. Falle auf CPU zurück. Die Verarbeitung kann langsamer sein.")

# Startzeit für die Gesamtmessung
start_time = time.time()
step_times = {}

# Sprachkürzel für das gesamte Projekt
source_lang="en"
target_lang="de"

# Benötigte Installationen werden im Skript bereits importiert.

def start_language_tool_server(lt_path: str = "D:\\LanguageTool-6.6", port: int = 8010) -> Optional[subprocess.Popen]:
    """
    Startet den LanguageTool-Server als Subprozess und wartet, bis er bereit ist.

    Args:
        lt_path (str): Pfad zum LanguageTool-Verzeichnis.
        port (int): Port, auf dem der Server lauschen soll.

    Returns:
        Optional[subprocess.Popen]: Das Prozessobjekt des laufenden Servers oder None bei einem Fehler.
    """
    lt_jar_path = os.path.join(lt_path, "languagetool-server.jar")
    if not os.path.exists(lt_jar_path):
        logger.error(f"LanguageTool JAR-Datei nicht gefunden: {lt_jar_path}")
        return None

    command = [
        "java", "-Xmx4g", "-cp", lt_jar_path,
        "org.languagetool.server.HTTPServer", "--port", str(port), "--allow-origin", "*"
    ]
    
    logger.info(f"Starte LanguageTool-Server mit Befehl: {' '.join(command)}")
    
    # Leite Ausgaben in eine Log-Datei um für späteres Debugging
    with open("languagetool_server.log", "w") as lt_log:
        process = subprocess.Popen(
            command,
            stdout=lt_log,
            stderr=subprocess.STDOUT
        )

    logger.info(f"LanguageTool-Server-Prozess gestartet (PID: {process.pid}). Warte auf Bereitschaft...")

    # Robuster Health-Check
    if wait_for_server_ready(port=port, timeout=60):
        logger.info("✅ LanguageTool-Server ist bereit.")
        # WICHTIG: Die globale Variable LT_PORT muss für die refine_text_pipeline gesetzt werden.
        global LT_PORT
        LT_PORT = port
        return process
    else:
        logger.error("LanguageTool-Server konnte nicht gestartet werden. Überprüfen Sie 'languagetool_server.log'.")
        process.terminate()
        process.wait()
        return None

def stop_language_tool_server(process: subprocess.Popen):
    """
    Beendet den LanguageTool-Server-Prozess sicher.

    Args:
        process (subprocess.Popen): Das Prozessobjekt des Servers.
    """
    if process:
        logger.info(f"Beende LanguageTool-Server (PID: {process.pid})...")
        process.terminate()
        try:
            process.wait(timeout=10)
            logger.info("✅ LanguageTool-Server erfolgreich beendet.")
        except subprocess.TimeoutExpired:
            logger.warning("LanguageTool-Server reagiert nicht auf terminate, erzwinge Beendigung (kill)...")
            process.kill()
            process.wait()
            logger.info("✅ LanguageTool-Server wurde beendet (kill).")

# Konfigurationen für die Verwendung von CUDA
cuda_options = {
    "hwaccel": "cuda",
    "hwaccel_output_format": "cuda"
}

def run_command(command):
    """Führt einen Shell-Befehl aus und gibt ihn vorher aus."""
    print(f"Ausführung des Befehls: {command}")
    subprocess.run(command, shell=True, check=True)

def time_function(func, *args, **kwargs):
        """Misst und protokolliert die Ausführungszeit einer Funktion."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        execution_time = end - start
        logger.info(f"Execution time for {func.__name__}: {execution_time:.4f} seconds")
        return result, execution_time

# Modell-Ladefunktionen
def load_whisper_model():
    """Lädt das Faster-Whisper-Modell (large-v3) für die Transkription."""
    model_size = "large-v3"
    # compute_type="int8_float16" nutzt INT8-Gewichte + FP16-Aktivierungen für Speed & geringen Speicher
    fw_model = WhisperModel(model_size, device="auto", compute_type="bfloat16", cpu_threads=6, local_files_only=True)
    pipeline = BatchedInferencePipeline(model=fw_model)
    return pipeline

def load_madlad400_translator_optimized(model_path, device):
    """
    Lädt ein T5-basiertes Modell von Hugging Face und initialisiert es
    mit DeepSpeed Inference für maximale Performance.

    Args:
        model_name_or_path (str): Der Name oder Pfad des Hugging Face Modells.
        device (str): Das Zielgerät, z.B. 'cuda'.

    Returns:
        Das optimierte DeepSpeed-Engine-Modell und den Tokenizer.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("DeepSpeed Inference erfordert eine CUDA-fähige GPU.")

    logger.info(f"Lade T5-Modell '{model_path}' von Hugging Face...")

    # 1. Lade das Standard-PyTorch-Modell und den Tokenizer
    # Wichtig: Wir laden das Modell im bfloat16-Datentyp, der auf der RTX 4050
    # für eine gute Balance aus Performance und Präzision sorgt.
    tokenizer = T5TokenizerFast.from_pretrained(model_path)

    # 2. MANUELLE UND VOLLSTÄNDIGE KONFIGURATION ALLER SONDERZEICHEN NACH DEM LADEN
    sentinel_tokens = [f"<extra_id_{i}>" for i in range(100)]
    language_prefixes = [f"<2{lang}>" for lang in ["de", "en", "es", "fr", "ru"]]
    
    # Die Standard-Tokens sind bereits korrekt geladen, wir fügen nur die zusätzlichen hinzu.
    tokenizer.add_special_tokens({
        "additional_special_tokens": sentinel_tokens + language_prefixes
    })
    logger.info(f"{len(sentinel_tokens) + len(language_prefixes)} zusätzliche spezielle Tokens zum Tokenizer hinzugefügt.")

    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16  # Nutze bfloat16 für Ampere-Architektur (RTX 40xx)
    )

    model.resize_token_embeddings(len(tokenizer))

    # Modell in den Evaluationsmodus und auf die GPU verschieben
    model = model.eval().to(device)
    logger.info("Modell erfolgreich geladen und auf GPU verschoben.")

    # 2. Initialisiere die DeepSpeed Inference Engine
    logger.info("Initialisiere DeepSpeed Inference Engine mit Kernel-Injektion...")
    ds_engine = deepspeed.init_inference(
        model=model,
        dtype=torch.bfloat16,                 # Datentyp muss zum geladenen Modell passen
        tensor_parallel={"tp_size": 1},       # Für eine einzelne GPU immer 1
        replace_with_kernel_inject=True,      # HIER IST DER ENTSCHEIDENDE PUNKT!
        injection_policy=None                 # Für Standard-T5-Modelle kann DeepSpeed die Policy automatisch erkennen
    )
    
    logger.info("✅ DeepSpeed Inference Engine für T5-Modell erfolgreich initialisiert.")
    
    return ds_engine, tokenizer

def load_xtts_v2():
    """
    Lädt Xtts v2 und konfiguriert DeepSpeed-Inferenz.
    """
    # 1) Konfiguration lesen
    config = XttsConfig()
    config.load_json("D:\\Modelle\\v203_\\config.json")
    # 2) Modell initialisieren
    xtts_model = Xtts.init_from_config(
        config,
        vocoder_path=vocoder_pth,
        vocoder_config_path=vocoder_cfg
    )
    xtts_model.load_checkpoint(
        config,
        checkpoint_dir="D:\\Modelle\\v203_",  # Pfad anpassen
        use_deepspeed=False
    )
    xtts_model.to(torch.device(0))
    xtts_model.eval()

    return xtts_model

# Context Manager für GPU-Operationen
@contextmanager
def gpu_context():
    """Stellt sicher, dass der GPU-Speicher nach einer Operation freigegeben wird."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU-Speicher durch gpu_context bereinigt.")

def ask_overwrite(file_path):
    """
    Fragt den Benutzer, ob eine vorhandene Datei überschrieben werden soll

    Args:
        file_path (str): Pfad zur Datei
        
    Returns:
        bool: True wenn überschrieben werden soll, sonst False
    """
    while True:
        answer = input(f"Datei '{os.path.basename(file_path)}' existiert bereits. Überschreiben? (j/n, Enter für Nein): ").lower().strip()
        if answer in ['j', 'ja']:
            return True
        elif answer == "" or answer in ['n', 'nein']:
            return False
        print("Ungültige Eingabe. Bitte 'j' oder 'n' antworten.")

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

def create_voice_sample(
    audio_path,
    sample_path_1,
    sample_path_2,
    sample_path_3
    ):
    """Erstellt ein Voice-Sample aus dem verarbeiteten Audio für Stimmenklonung."""
    sample_paths = [
        sample_path_1,
        sample_path_2,
        sample_path_3
    ]
    for i, sample_path in enumerate(sample_paths, 1):
        if os.path.exists(sample_path):
            if not ask_overwrite(sample_path):
                logger.info(f"Verwende vorhandenes Sample #{i}: {sample_path}")
                return

    while True:
        start_time_str = input(f"Startzeit für Sample #{i} (in Sekunden, Enter zum Überspringen): ")
        if not start_time_str:
            logger.info(f"Erstellung von Sample #{i} übersprungen.")
            break
        
        end_time_str = input(f"Endzeit für Sample #{i} (in Sekunden): ")
        try:
            start_seconds = float(start_time_str)
            end_seconds = float(end_time_str)
            duration = end_seconds - start_seconds
            
            if duration <= 0:
                logger.warning("Endzeit muss nach der Startzeit liegen.")
                continue
            
            (
                ffmpeg.input(audio_path, ss=start_seconds, t=duration).output(
                    sample_path,
                    acodec='pcm_s16le',
                    ac=1,
                    ar=22050
                ).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Voice-Sample #{i} erfolgreich erstellt: {sample_path}")
            break
        except (ValueError, ffmpeg.Error) as e:
            logger.error(f"Fehler beim Erstellen von Sample #{i}: {e}")

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

def save_progress_csv(segments_list, output_file, headers=['startzeit', 'endzeit', 'text']):
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

def validate_text_for_tts_robust(
    text: str
    ) -> tuple[bool, str]:
    """
    Prüft Textsegmente vor der TTS-Verarbeitung auf TECHNISCHE Gültigkeit.
    KORRIGIERTE VERSION: Prüft NICHT mehr die Wortanzahl. Diese Logik
    gehört zum Merging-Prozess.

    Returns:
        (is_valid, reason) - True, wenn alle Prüfungen bestanden.
    """
    # Prüfung 1: Ist der Input ein gültiger, nicht-leerer String?
    if not isinstance(text, str) or not text.strip():
        return False, "Leerer oder ungültiger String"
    
    # Prüfung 2: Besteht der String nur aus Satzzeichen oder Symbolen?
    if not any(c.isalnum() for c in text):
        return False, "Enthält keine alphanumerischen Zeichen (nur Sonderzeichen)"
    
    return True, "OK"

def _protect_placeholders(text: str, items: set[str], placeholder_tag: str) -> tuple[str, dict]:
    """Ersetzt alle items durch Platzhalter, damit Regex-Splits nicht fehlschlagen."""
    mapping: dict[str, str] = {}
    for i, item in enumerate(items):
        if item in text:
            ph = f"__{placeholder_tag}_{i}__"
            mapping[ph] = item
            text = text.replace(item, ph)
    return text, mapping

def _restore_placeholders(text: str, mapping: dict) -> str:
    for ph, original in mapping.items():
        text = text.replace(ph, original)
    return text

def split_sentences_german(text: str) -> list[str]:
    """Abkürzungs-sichere Satzaufteilung für Deutsch."""
    protected, map_abbr = _protect_placeholders(text, GERMAN_ABBREVIATIONS, "ABBR")
    
    # Zahlen mit Dezimalpunkt schützen
    protected, map_num = _protect_placeholders(
        protected,
        set(re.findall(r"\b\d+\.\d+\b", protected)),
        "NUM"
    )
    
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])", protected.strip())
    
    # Platzhalter zurückwandeln
    return [_restore_placeholders(s.strip(), {**map_abbr, **map_num})
            for s in sentences if s.strip()]

def split_words_with_validation(text: str,
                                char_limit: int,
                                min_words: int = MIN_WORDS_GLOBAL) -> list[str]:
    """Wort-basiertes Splitting mit Mindestwortregeln."""
    words = text.split()
    if len(words) <= min_words:
        return [text]
    
    chunks, current = [], ""
    i = 0
    while i < len(words):
        nxt = f"{current} {words[i]}".strip()
        if len(nxt) <= char_limit:
            current = nxt
            i += 1
        else:
            remaining = len(words) - i
            if remaining < min_words:  # 2-Wort-Regel greift
                current += " " + " ".join(words[i:])
                break
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks

def split_text_robust_improved(
    text: str,
    char_limit: int = 150,
    min_words: int = MIN_WORDS_GLOBAL
    ) -> list[str]:
    """
    Drei-stufiges Chunking:
        1) Satz-Splits (Abkürzungs-sicher)
        2) Wort-Splits für überlange Sätze
        3) Post-Merge, falls Chunk < min_words
    """
    # ---------- Vorprüfung ----------
    if not text or len(text) <= char_limit:
        return [text.strip()]

    sentences: list[str] = split_sentences_german(text)
    chunks: list[str] = []
    current: str = ""

    # ---------- Hauptschleife ----------
    for sent in sentences:
        candidate: str = f"{current} {sent}".strip() if current else sent

        # Fall A: Der zusammengesetzte Satz passt ins Limit
        if len(candidate) <= char_limit:
            current = candidate
            continue

        # Fall B: Der zusammengesetzte Satz passt NICHT
        #  → bisherigen Inhalt als Chunk sichern,
        #    und den *aktuellen* Satz wortweise weiter zerhacken.
        if current:
            chunks.append(current)
            current = ""  # reset

        # Jetzt muss *sent* selbst ggf. weiter aufgeteilt werden
        inner_chunks = split_words_with_validation(
            sent,
            char_limit,
            min_words
        )
        chunks.extend(inner_chunks)

    # Letzten Rest anhängen
    if current:
        chunks.append(current)

    # ---------- Post-Merge-Pass ----------
    merged: list[str] = []
    for ch in chunks:
        if len(ch.split()) < min_words and merged:
            # Zu kurz → an vorherigen Chunk anhängen
            merged[-1] = f"{merged[-1]} {ch}".strip()
        else:
            merged.append(ch.strip())

    # Optionales Debug-Logging
    if any(len(c) > char_limit for c in merged):
        logger.debug(
            f"[split_text_robust_improved] Chunk überschreitet char_limit "
            f"nach Post-Merge: {[len(c) for c in merged]} – Text: {text[:60]}…"
        )

    return [c for c in merged if c]

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
                "threshold": 0.0,               # Niedriger Schwellwert für empfindlichere Spracherkennung
                "min_speech_duration_ms": 0,  # Minimale Sprachdauer in Millisekunden
                "max_speech_duration_s": 15,    # Maximale Sprachdauer in Sekunden
                "min_silence_duration_ms": 0, # Minimale Stille-Dauer zwischen Segmenten
                "speech_pad_ms": 300            # Polsterzeit vor und nach Sprachsegmenten
            }
            
            segments_generator, info = pipeline.transcribe(
                audio_file,
                batch_size=2,
                beam_size=15,
                patience=1.5,
                vad_filter=True,
                vad_parameters=vad_params,
                chunk_length=18,
                #compression_ratio_threshold=2.8,    # Schwellenwert für Kompressionsrate
                #log_prob_threshold=-0.2,             # Schwellenwert für Log-Probabilität
                #no_speech_threshold=1.0,            # Schwellenwert für Stille
                #temperature=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5),      # Temperatur für Sampling
                temperature=1,                  # Temperatur für Sampling
                word_timestamps=True,               # Zeitstempel für Wörter
                hallucination_silence_threshold=0.3,  # Schwellenwert für Halluzinationen
                condition_on_previous_text=True,    # Bedingung an vorherigen Text
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                language="en",
                #task="translate"
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
            print(f"\n[{start} --> {end}]:\n{text}")
            
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

def sanitize_for_csv_and_tts(text: str) -> str:
    """
    ERWEITERTE VERSION: Entfernt alle verbliebenen Entity-Platzhalter und normalisiert Text.
    
    Args:
        text: Zu bereinigender Text
        
    Returns:
        Bereinigter Text für CSV und TTS
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Entferne alle möglichen Entity-Platzhalter-Patterns
    entity_patterns = [
        r'__ENTITY_[A-Z_]+_\d+__',      # Standard Entity-Platzhalter
        r'__CUSTOM_PATTERN_\d+__',       # Custom Pattern-Platzhalter
        r'__NAME_\d+_\d+__',             # Name-Platzhalter
        r'__ABBR_\d+__',                 # Abkürzungs-Platzhalter
        r'__NUM_\d+__',                  # Zahlen-Platzhalter
        r'__[A-Z_]+_\d+__',              # Allgemeine Fallback-Pattern
        r'\[PLACEHOLDER\]',              # Standard Placeholder
        r'\[MASK\]',                     # Mask-Token
        r'<2[a-z]{2}>',                  # MADLAD400 Sprachpräfixe
        r'<[0-9][a-z]{2}>'               # Varianten mit Zahlen
    ]
    
    cleaned_text = text
    for pattern in entity_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Normalisiere Leerzeichen und Satzzeichen
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)           # Mehrfache Leerzeichen
    cleaned_text = re.sub(r'\s*[,\.]\s*[,\.]+', '.', cleaned_text)  # Doppelte Satzzeichen
    cleaned_text = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', cleaned_text)  # Führende/nachgestellte Zeichen
    
    # Deutsche Kapitalisierung
    cleaned_text = cleaned_text.strip()
    if cleaned_text and cleaned_text[0].islower():
        cleaned_text = cleaned_text[0].upper() + cleaned_text[1:]
    
    # CSV-konforme Bereinigung
    cleaned_text = cleaned_text.replace('|', ',')              # CSV-Delimiter
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)        # Finale Leerzeichen-Normalisierung
    
    return cleaned_text.strip()

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

# Übersetzen
def translate_batch_madlad400(
    texts,
    ds_engine,
    tokenizer,
    device:torch.device,
    target_lang="de",
    batch_size=None
):
    """
    NEUE VERSION: Führt eine Batch-Übersetzung mit einer DeepSpeed-optimierten Engine durch.
    Diese Funktion arbeitet auf der Tensor-Ebene.

    Args:
        texts (list): Liste von Texten (als Strings oder Dictionaries).
        ds_engine: Die initialisierte DeepSpeed Inference Engine.
        tokenizer: Der zugehörige Hugging Face Tokenizer.
        target_lang (str): Das Kürzel der Zielsprache.

    Returns:
        list[str]: Eine Liste der übersetzten und bereinigten Texte.
    """
    if not texts:
        return []

    # Schritt 1: Rohtexte extrahieren und MADLAD-Präfix hinzufügen
    source_texts = []
    for item in texts:
        text_content = item.get('text', '') if isinstance(item, dict) else str(item)
        source_texts.append(f"<2{target_lang}> {text_content.strip()}")

    # Schritt 2: Texte in PyTorch-Tensoren umwandeln
    inputs = tokenizer(
        source_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device) # Tensoren direkt auf das GPU-Gerät der Engine verschieben

    # Schritt 3: Inferenz mit der .generate()-Methode der DeepSpeed-Engine
    with torch.no_grad():
        output_ids = ds_engine.module.generate(
            **inputs,
            do_sample=False,
            num_beams=4,
            max_length=512,
            #patience=2.5,
            length_penalty=0.9,
            repetition_penalty=1.9,
            #max_decoding_length=1024,
            no_repeat_ngram_size=2,
            #early_stopping=True
        )

    # Schritt 4: Ausgabe-Tensoren zurück in Text-Strings dekodieren
    translations = tokenizer.batch_decode(output_ids, skip_special_tokens=False)

    # Schritt 5: Bekannte Artefakte bereinigen und Ergebnis zurückgeben
    cleaned_translations = []
    for t in translations:
        # ENTFERNE MANUELL NUR DIE UNERWÜNSCHTEN STEUERZEICHEN.
        # Unsere wichtigen Platzhalter bleiben so unberührt.
        cleaned = t.replace(tokenizer.pad_token, "").replace(tokenizer.eos_token, "").strip()

        if cleaned.startswith(tokenizer.unk_token):
            cleaned = cleaned[len(tokenizer.unk_token):].lstrip()

        # Führe die zusätzliche Bereinigung für Modell-Artefakte durch.
        cleaned = cleanup_translation_artifacts(cleaned)
        cleaned_translations.append(cleaned)

    return cleaned_translations

def translate_single_text_fallback(text, ds_engine, tokenizer, target_lang):
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
        result = ds_engine.translate_batch(
            [tokens],
            # target_prefix=None,  # Kein Target-Prefix!
            beam_size=3,
            patience=1.0
        )
        
        if result and result[0].hypotheses:
            translation_tokens = result[0].hypotheses[0]
            translation_ids = tokenizer.convert_tokens_to_ids(translation_tokens)
            translation = tokenizer.decode(translation_ids, skip_special_tokens=True)
            return cleanup_translation_artifacts(translation)
        else:
            return "ERROR: NO SINGLE TRANSLATION"
            
    except Exception as e:
        logger.error(f"Fallback-Übersetzung fehlgeschlagen: {e}")
        return "ERROR: FALLBACK FAILED"

def cleanup_translation_artifacts(text: str) -> str:
    """
    Zusätzliche Bereinigung für spezifische Übersetzungsartefakte.
    
    Args:
        text: Text mit möglichen Artefakten
        
    Returns:
        Bereinigter Text
    """
    # MADLAD400-spezifische Artefakte
    madlad_patterns = [
        r'^<2[a-z]{2}>\s*',              # Sprachpräfixe am Anfang
        r'\s*<2[a-z]{2}>\s*',            # Sprachpräfixe in der Mitte
        r'^(de|en|fr|es|it)\s*[.:]?\s*', # Sprach-Identifikatoren
    ]
    
    cleaned = text
    for pattern in madlad_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Wiederholte Phrasen entfernen (häufig bei NMT-Fehlern)
    words = cleaned.split()
    if len(words) > 4:
        # Erkenne Wiederholungen von 2-3 Wörtern
        for window_size in [3, 2]:
            i = 0
            while i < len(words) - window_size:
                phrase = words[i:i+window_size]
                next_phrase = words[i+window_size:i+2*window_size]
                if phrase == next_phrase:
                    # Entferne Wiederholung
                    words = words[:i+window_size] + words[i+2*window_size:]
                else:
                    i += 1
    
    return ' '.join(words)

def translate_segments_optimized_safe(
    refined_transcription_path: str,
    master_entity_map: Dict[int, Dict[str, EntityInfo]],
    translation_output_file: str,
    cleaned_source_output_file: str,
    source_lang: str = "en",
    target_lang: str = "de",
    batch_size: int = 1
) -> Tuple[str, str]:
    """
    KORRIGIERTE VERSION: Übersetzt Segmente mit verbesserter Entity-Behandlung.
    
    Args:
        refined_transcription_path (str): Pfad zur veredelten Quell-CSV.
        master_entity_map (Dict[int, ...]): Das von `refine_text_pipeline` erzeugte Mapping.
        translation_output_file (str): Zieldatei für die Übersetzung.
        cleaned_source_output_file (str): Zieldatei für den bereinigten Quelltext.
        source_lang (str): Quellsprache
        target_lang (str): Zielsprache
        batch_size (int): Batch-Größe für Übersetzung
        
    Returns:
        Tuple[str, str]: Pfade zu den erstellten Dateien.
    """
    should_continue_translation, processed_keys = handle_key_based_continuation(
        translation_output_file, refined_transcription_path, key_column_index=0
    )
    if not should_continue_translation:
        logger.info("Übersetzung wird übersprungen, da die Zieldatei vollständig ist.")
        return translation_output_file, cleaned_source_output_file

    # Lade die Segmente aus der veredelten Quelldatei
    df_source = pd.read_csv(refined_transcription_path, sep='|', dtype=str).fillna('')
    
    segments_to_process = []
    # KORREKTUR: Verwende das bereits vorhandene Entity-Mapping aus der CSV
    for i, row in df_source.iterrows():
        if row['startzeit'] not in processed_keys:
            # Lade das Entity-Mapping aus der CSV-Spalte
            if 'entity_map' in row and row['entity_map'] and row['entity_map'] != '{}':
                current_entity_map = _json_str_to_entity_map(row['entity_map'])
            else:
                # Fallback: Verwende das Master-Mapping oder erstelle neues
                if i in master_entity_map:
                    current_entity_map = master_entity_map[i]
                else:
                    # Als letzter Ausweg: Neue Entity-Erkennung
                    nlp_model = get_spacy_model("en_core_web_trf")
                    _, current_entity_map = entity_protection_final(row['text'], nlp_model)
            
            # NEU: Validierung des Mappings (leeres Dict erlauben, aber loggen)
            if not isinstance(current_entity_map, dict):
                logger.warning(f"Ungültiges Mapping für Segment {i}: {current_entity_map}. Setze auf leeres Dict.")
                current_entity_map = {}
            
            # VERBESSERUNG: Schutz direkt mit vorhandenem Mapping anwenden
            protected_text = apply_entity_protection_with_mapping(row['text'], current_entity_map)
            
            segments_to_process.append({
                "id": i,
                "startzeit": row['startzeit'],
                "endzeit": row['endzeit'],
                "original_text": row['text'],
                "protected_text": protected_text,
                "entity_mapping": current_entity_map  # Mapping für spätere Wiederherstellung
            })
            logger.debug(f"Segment {i} vorbereitet: {len(current_entity_map)} Entities gefunden.")

    if not segments_to_process:
        logger.info("Alle Segmente bereits übersetzt.")
        return translation_output_file, cleaned_source_output_file
    
    logger.info(f"Übersetze {len(segments_to_process)} verbleibende Segmente.")

    # Lade existierende Daten
    all_translations = [] if not os.path.exists(translation_output_file) else pd.read_csv(translation_output_file, sep='|', dtype=str).to_dict('records')
    all_cleaned_sources = [] if not os.path.exists(cleaned_source_output_file) else pd.read_csv(cleaned_source_output_file, sep='|', dtype=str).to_dict('records')
    
    translate_start_time = time.time()
    
    with gpu_context():
        ds_engine, tokenizer = load_madlad400_translator_optimized("madlad400-3b-mt", "cuda")
        device = next(ds_engine.parameters()).device

        for i in tqdm(range(0, len(segments_to_process), batch_size), desc="Übersetze geschützte Batches"):
            batch_data = segments_to_process[i:i + batch_size]
            texts_to_translate = [item['protected_text'] for item in batch_data]

            translated_protected_texts = translate_batch_madlad400(
                texts=texts_to_translate, ds_engine=ds_engine, tokenizer=tokenizer, 
                target_lang=target_lang, device=device
            )

            for j, translated_protected in enumerate(translated_protected_texts):
                segment_data = batch_data[j]
                
                # Entity-Wiederherstellung
                current_entity_map = segment_data.get('entity_mapping', {})
                if not current_entity_map:
                    logger.warning(f"Kein Mapping für Segment {segment_data['id']}. Verwende leeres Dict.")
                    current_entity_map = {}

                # Verwende robuste Wiederherstellung
                final_translated_text = restore_entities_robust(
                    translated_protected, 
                    current_entity_map,
                    target_language=target_lang
                )

                # Debug-Logging für Entity-Pipeline
                logger.debug(f"Segment {segment_data['id']}: {len(current_entity_map)} Entities, "
                            f"Übersetzung: {final_translated_text[:100]}...")

                print(f"\n[{segment_data['startzeit']} --> {segment_data['endzeit']}]:")
                #print(f"Original: {segment_data['original_text'][:100]}...")
                print(f"Übersetzt: {final_translated_text}")
                print(f"Entities: {len(current_entity_map)} gefunden\n")

                all_translations.append({
                    'startzeit': segment_data["startzeit"], 
                    'endzeit': segment_data["endzeit"], 
                    'text': sanitize_for_csv_and_tts(final_translated_text)
                })
                all_cleaned_sources.append({
                    'startzeit': segment_data["startzeit"], 
                    'endzeit': segment_data["endzeit"], 
                    'text': sanitize_for_csv_and_tts(segment_data['original_text'])
                })
        
        save_progress_csv(all_translations, translation_output_file)
        save_progress_csv(all_cleaned_sources, cleaned_source_output_file)

        logger.info(f"Übersetzung abgeschlossen: {len(all_translations)} gültige Segmente.")
        print("\n---------------------------------")
        print("|<< Übersetzung abgeschlossen! >>|")
        print("---------------------------------")
        
        translate_end_time = time.time() - translate_start_time
        logger.info(f"Übersetzung abgeschlossen in {translate_end_time:.2f} Sekunden")
        print(f"Übersetzung abgeschlossen in {translate_end_time:.2f} Sekunden -> {(translate_end_time / 60):.2f} Minuten")
        
        return translation_output_file, cleaned_source_output_file

def is_valid_segment(segment, check_text_content=True):
    """
    Erweiterte Hilfsfunktion zur Segmentvalidierung mit detailliertem Logging.
    Prüft Zeitstempel und Textinhalt und protokolliert den genauen Grund für das Verwerfen.

    Args:
        segment (dict): Ein Dictionary, das ein Segment repräsentiert. Muss 'start', 'end', 'startzeit', 'endzeit' und 'text' enthalten.
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
            f"Start: {segment.get('startzeit', 'N/A')}, Ende: {segment.get('endzeit', 'N/A')}. "
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
                f"Start: {segment.get('startzeit', 'N/A')}, Text: '{text[:50]}...'"
            )
            return False
            
    # Wenn alle Prüfungen bestanden wurden, ist das Segment gültig.
    return True

def evaluate_translation_quality(source_csv_path, translated_csv_path, report_path, summary_path, model_name, threshold):
    """
    Prüft die semantische Ähnlichkeit, erstellt einen detaillierten CSV-Bericht
    UND eine separate, saubere Zusammenfassungs-Datei.
    FINALE, KORRIGIERTE VERSION.
    """
    if os.path.exists(report_path) and not ask_overwrite(report_path):
        logger.info(f"Verwende vorhandenen Qualitätsbericht: {report_path}")
        return report_path

    logger.info(f"Starte Qualitätsprüfung der Übersetzung mit Modell: {model_name}")
    print("------------------------------------------")
    print("|<< Starte Qualitätsprüfung Übersetzung >>|")
    print("------------------------------------------")

    try:
        # Daten laden
        df_source = pd.read_csv(source_csv_path, sep='|', dtype=str).fillna('')
        df_translated = pd.read_csv(translated_csv_path, sep='|', dtype=str).fillna('')
        
        # Originalspaltennamen für den finalen Bericht sichern
        original_source_columns = df_source.columns.tolist()

        if len(df_source) != len(df_translated):
            logger.warning(f"Zeilenanzahl stimmt nicht überein ({len(df_source)} vs {len(df_translated)}). Qualitätsprüfung übersprungen."); return None
        
        # Spaltennamen intern für die Verarbeitung normalisieren
        df_source.columns = [col.strip().lower() for col in df_source.columns]
        df_translated.columns = [col.strip().lower() for col in df_translated.columns]
        
        if 'text' not in df_source.columns or 'text' not in df_translated.columns:
            logger.error(f"Benötigte Spalte 'text' (oder 'Text') nicht in den CSVs gefunden."); return None

        # KI-basierte Prüfung
        with gpu_context():
            model = SentenceTransformer(model_name, device=device)
            source_texts = df_source['text'].tolist()
            translated_texts = df_translated['text'].tolist()
            
            embeddings_source = model.encode(source_texts, convert_to_tensor=True, show_progress_bar=True)
            embeddings_translated = model.encode(translated_texts, convert_to_tensor=True, show_progress_bar=True)

            # Berechne die paarweise Ähnlichkeit aller Elemente
            pairwise_similarities = cos_sim(embeddings_source, embeddings_translated)
            # Extrahiere NUR die Diagonale (source[i] vs. translated[i]), dies gibt einen Vektor zurück
            similarities_vector = torch.diag(pairwise_similarities).tolist()

        # Ergebnisse aufbereiten
        results = []
        issues_found = 0
        for i in range(len(df_source)):
            # Greife direkt auf das Element im Vektor zu
            similarity = similarities_vector[i]
            
            flag = ""
            if similarity < threshold:
                status = f"Niedrig ({similarity:.3f})"
                flag = "CHECK MANUALLY"
                issues_found += 1
            else:
                status = f"OK ({similarity:.3f})"
            
            results.append({
                # Verwende die originalen Spaltennamen für den Bericht
                original_source_columns[0]: df_source.iloc[i].get(original_source_columns[0].lower(), 'N/A'),
                original_source_columns[1]: df_source.iloc[i].get(original_source_columns[1].lower(), 'N/A'),
                "Quelltext": df_source.iloc[i]['text'],
                "Uebersetzung": df_translated.iloc[i]['text'],
                "Aehnlichkeit": f"{similarity:.4f}",
                "Status": status,
                "Flag": flag
            })

        # Detaillierten CSV-Bericht speichern
        df_report = pd.DataFrame(results)
        df_report.to_csv(report_path, sep='|', index=False, encoding='utf-8')
        logger.info(f"Detaillierter Qualitätsbericht gespeichert: {report_path}")

        # Zusammenfassung in separater Datei speichern
        avg_similarity = sum(similarities_vector) / len(similarities_vector) if similarities_vector else 0.0
        summary_content = (
            f"Qualitätsprüfung der Übersetzung - Zusammenfassung\n"
            f"====================================================\n"
            f"Datum der Prüfung: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Quell-Segmente: {len(df_source)}\n"
            f"Gefundene Probleme (Ähnlichkeit < {threshold}): {issues_found}\n"
            f"Durchschnittliche semantische Ähnlichkeit: {avg_similarity:.4f}\n"
            f"====================================================\n"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
        
        logger.info(f"Zusammenfassender Bericht gespeichert: {summary_path}")
        print(f"Qualitätsprüfung abgeschlossen. Bericht: {report_path}, Zusammenfassung: {summary_path}")
        print(f"Gefundene Probleme: {issues_found}")
        print("-------------------------------------------")

        return report_path

    except Exception as e:
        logger.error(f"Fehler bei der Qualitätsprüfung der Übersetzung: {e}", exc_info=True)
        return None

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

# --------------------------------------------------------------------------
# HILFSFUNKTIONEN FÜR DIE PARALLELE VERARBEITUNG (NEU)
# --------------------------------------------------------------------------

def _process_single_text_correction(text: str, lang_code: str, port: int) -> str:
    """
    Worker-Funktion für einen einzelnen Prozess im Pool. Initialisiert eine eigene
    LanguageTool-Instanz und korrigiert einen einzelnen Text.
    WICHTIG: Muss auf der obersten Ebene des Moduls definiert sein, damit multiprocessing sie finden kann.
    """
    try:
        # Jeder Prozess erhält seine eigene, unabhängige Verbindung.
        lt_tool = language_tool_python.LanguageTool(lang_code, remote_server=f'http://localhost:{port}')
        return lt_tool.correct(text)
    except Exception as e:
        logger.error(f"Fehler im Worker-Prozess bei der Korrektur von Text: '{text[:50]}...'. Fehler: {e}")
        return text # Im Fehlerfall den unkorrigierten Text zurückgeben.

def correct_texts_in_parallel(texts: List[str], lang_code: str, port: int) -> List[str]:
    """
    Führt die Grammatikkorrektur für eine Liste von Texten parallel aus, indem es
    einen Pool von Worker-Prozessen nutzt. Jeder Prozess hat seine eigene
    LanguageTool-Instanz, um Thread-Safety-Probleme zu vermeiden.

    Args:
        texts (List[str]): Die Liste der zu korrigierenden Texte.
        lang_code (str): Der Sprachcode (z.B. 'de-DE').
        port (int): Der Port des laufenden LanguageTool-Servers.

    Returns:
        List[str]: Eine Liste der korrigierten Texte in der ursprünglichen Reihenfolge.
    """
    # Nutze die Anzahl der verfügbaren CPU-Kerne für maximale Parallelität.
    num_processes = cpu_count()
    
    # `partial` wird verwendet, um die festen Argumente (lang_code, port) an die
    # Worker-Funktion zu "binden", da pool.imap nur ein veränderliches Argument akzeptiert.
    partial_worker = partial(_process_single_text_correction, lang_code=lang_code, port=port)

    # Ein Pool von Prozessen wird erstellt.
    with Pool(processes=num_processes) as pool:
        logger.info(f"Starte parallele Grammatikkorrektur mit {num_processes} Prozessen...")
        # `pool.imap` ist wie `pool.map`, ermöglicht aber die Verwendung mit tqdm für eine Fortschrittsanzeige.
        # Es wendet die `partial_worker`-Funktion auf jedes Element in `texts` an.
        results = list(tqdm(pool.imap(partial_worker, texts), total=len(texts), desc="Grammatikkorrektur (parallel)"))
    
    return results

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
def load_spacy_model(model_name):
    """Lädt ein spaCy-Modell sicher."""
    try:
        return spacy.load(model_name)
    except OSError:
        logger.critical(f"spaCy-Modell '{model_name}' nicht gefunden. Bitte mit 'python -m spacy download {model_name}' installieren.")
        sys.exit(1)

_SPACY_MODELS_CACHE = {}

def get_spacy_model(model_name: str):
    """
    Lädt ein spaCy-Modell nur beim ersten Aufruf und speichert es im Cache.
    Dies verhindert, dass jeder neue Prozess das Modell erneut laden muss.
    """
    if model_name not in _SPACY_MODELS_CACHE:
        logger.info(f"Lade spaCy-Modell '{model_name}' zum ersten Mal...")
        try:
            # spacy.require_gpu() sollte idealerweise vor dem Laden aufgerufen werden.
            # Da wir es global bereits tun, ist es hier sicher.
            _SPACY_MODELS_CACHE[model_name] = spacy.load(model_name, disable=["textcat"])
            logger.info(f"Modell '{model_name}' erfolgreich geladen und im Cache gespeichert.")
        except OSError:
            logger.critical(f"spaCy-Modell '{model_name}' nicht gefunden. Bitte mit 'python -m spacy download {model_name}' installieren.")
            sys.exit(1)
    return _SPACY_MODELS_CACHE[model_name]

# Erweiterte Entity-Labels für umfassende NEE
TARGET_ENTITY_LABELS: Set[str] = {
    """
    "PERSON", #"PER", # Personen
    "ORG",           # Organisationen  
    "WORK_OF_ART",   # Kunstwerke, Shows, Filme
    "EVENT",         # Events, Veranstaltungen
    "PRODUCT",       # Produkte, Marken
    "LAW",           # Gesetze, Regelwerke
    "GPE",           # Geopolitische Entitäten
    "FACILITY",      # Gebäude, Orte
    "LANGUAGE",      # Sprachen
    "NORP", #"NOR"    # Nationalitäten, religiöse/politische Gruppen
    """
}

# Zusätzliche Patterns für Social Media Handles und spezielle Namen
SOCIAL_MEDIA_PATTERNS = [
    r'\b[A-Za-z0-9_]+K\d+\b',           # KittyK38, UserK123
#    r'\b@[A-Za-z0-9_]+\b',              # @username
#    r'\b#[A-Za-z0-9_]+\b',              # #hashtag
    r'\b[A-Za-z]+\d{1,3}\b',            # Name mit Zahlen
#    r'\b[A-Z][a-z]*[A-Z][a-z]*\d*\b'   # CamelCase mit optionalen Zahlen
]

# --------------------------------------------------------------------------
# HILFSFUNKTIONEN FÜR ENTITY-MAPPING
# --------------------------------------------------------------------------

def _json_str_to_entity_map(json_str: str) -> Dict[str, EntityInfo]:
    """
    Konvertiert einen JSON-String, der in der CSV-Datei gespeichert ist, zurück
    in ein Dictionary von Platzhaltern zu EntityInfo-Objekten.

    Args:
        json_str (str): Der JSON-String aus der CSV-Spalte.

    Returns:
        Dict[str, EntityInfo]: Das wiederhergestellte Mapping.
    """
    if not json_str or json_str == '{}':
        return {}
    
    try:
        data = json.loads(json_str)
        entity_map = {}
        for placeholder, info_dict in data.items():
            entity_map[placeholder] = EntityInfo(
                text=info_dict['text'],
                label=info_dict['label'],
                start=info_dict['start'],
                end=info_dict['end'],
                language=info_dict['language']
            )
        return entity_map
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Fehler beim Parsen von Entity-Mapping: {e} | Input: '{json_str[:100]}'")
        return {}

def _entity_map_to_json_str(entity_map: Dict[str, EntityInfo]) -> str:
    """
    Konvertiert ein EntityInfo-Dictionary in einen JSON-String zur Speicherung in einer CSV-Datei.

    Args:
        entity_map (Dict[str, EntityInfo]): Das Mapping von Platzhaltern zu Entity-Informationen.

    Returns:
        str: Ein JSON-formatierter String.
    """
    if not entity_map:
        return '{}'
    
    try:
        # Konvertiert das dataclass-Objekt in ein serialisierbares Dictionary
        serializable_map = {
            placeholder: dataclasses.asdict(entity_info)
            for placeholder, entity_info in entity_map.items()
        }
        return json.dumps(serializable_map, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Fehler beim Serialisieren von Entity-Mapping: {e}")
        return '{}'

# --------------------------------------------------------------------------
# FINALE, ROBUSTE ENTITY-SCHUTZ-FUNKTIONEN
# --------------------------------------------------------------------------

def entity_protection_final(text: str, nlp_model) -> Tuple[str, Dict[str, EntityInfo]]:
    """
    FINALE UND ROBUSTE VERSION: Ersetzt Entitäten durch T5-native Sentinel-Tokens
    (z.B. <extra_id_0>), die architektonisch für solche Aufgaben vorgesehen sind
    und die höchste Stabilität während der Generierung bieten.

    Args:
        text (str): Der zu verarbeitende Text.
        nlp_model: Das geladene spaCy-Sprachmodell.

    Returns:
        Tuple[str, Dict[str, EntityInfo]]: Ein Tupel mit dem geschützten Text und einem Mapping
        von Platzhaltern zu den ursprünglichen Entitätsinformationen.
    """
    if not text or not text.strip():
        return text, {}

    doc = nlp_model(text)
    entities_found: List[Span] = list(doc.ents)
    entity_mapping: Dict[str, EntityInfo] = {}

    entities_found.sort(key=lambda x: x.start_char, reverse=True)

    protected_text = text
    entity_counter = 0

    for ent in entities_found:
        if ent.label_ in TARGET_ENTITY_LABELS:
            placeholder = f"<extra_id_{entity_counter}>"
            
            protected_text = protected_text[:ent.start_char] + placeholder + protected_text[ent.end_char:]
            
            entity_mapping[placeholder] = EntityInfo(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                language=nlp_model.lang
            )
            entity_counter += 1

    return protected_text, entity_mapping

SENTINEL_REGEX = re.compile(r"<extra_id_(\d+)>")

def restore_entities_final(text: str, entity_mapping: Dict[str, EntityInfo]) -> str:
    """
    FINALE UND ROBUSTE VERSION: Setzt die durch Sentinel-Tokens geschützten Entitäten
    wieder in den übersetzten Text ein.

    Args:
        text (str): Der übersetzte Text mit Sentinel-Token-Platzhaltern.
        entity_mapping (Dict[str, EntityInfo]): Das Mapping von Platzhaltern zu Entitäten.

    Returns:
        str: Der Text mit wiederhergestellten Entitäten.
    """
    if not entity_mapping:
        return text

    def replace_match(match):
        placeholder = match.group(0)
        return entity_mapping.get(placeholder, placeholder).text

    restored_text = SENTINEL_REGEX.sub(replace_match, text)
    
    return re.sub(r'\s+', ' ', restored_text).strip()

# --------------------------------------------------------------------------
# KONTEXT- UND NACHBEARBEITUNGS-FUNKTIONEN
# --------------------------------------------------------------------------

def ensure_context_closure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analysiert Segmente und fügt Kontext vom vorherigen Segment hinzu, wenn ein
    Segment logisch unvollständig erscheint. Führt dabei auch die Entity-Mappings zusammen.
    """
    if df.empty:
        return df

    new_rows = []
    for i, row in df.iterrows():
        current_row_dict = row.to_dict()
        
        if i > 0 and new_rows:
            prev_row_dict = new_rows[-1]
            current_text = current_row_dict.get('text', '').strip()

            prev_text_clean = prev_row_dict.get("text", "").strip()
            ends_with_punctuation = prev_text_clean.endswith((".", "!", "?"))

            is_fragile = (
                current_text
                and (current_text[0].islower() or re.match(r'^(und|aber|oder|doch)\b', current_text, re.I))
            )
            starts_with_number = bool(re.match(r"^[0-9]+\S*", current_text))
            
            if is_fragile and not ends_with_punctuation and not starts_with_number:
                prev_row_dict['text'] = f"{prev_text_clean} {current_text}"
                
                prev_map = _json_str_to_entity_map(prev_row_dict.get('entity_map', '{}'))
                current_map = _json_str_to_entity_map(current_row_dict.get('entity_map', '{}'))
                combined_map = {**prev_map, **current_map}
                
                prev_row_dict['endzeit'] = current_row_dict.get('endzeit')
                prev_row_dict['entity_map'] = _entity_map_to_json_str(combined_map)
            else:
                new_rows.append(current_row_dict)
        else:
            new_rows.append(current_row_dict)

    return pd.DataFrame(new_rows)

# --------------------------------------------------------------------------
# DIE FINALE, HOCHPERFORMANTE HAUPTFUNKTION DER TEXTVEREDELUNG
# --------------------------------------------------------------------------

def refine_text_pipeline(
    input_file: str, 
    output_file: str, 
    spacy_model_name: str, 
    lang_code: str
) -> Tuple[str, Dict[int, Dict[str, EntityInfo]]]:
    """
    FINALE HOCHPERFORMANTE VERSION: Nutzt Batch-Verarbeitung und manuelle Parallelisierung
    für maximale Geschwindigkeit und respektiert die Benutzerentscheidung zum Überschreiben.
    """
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Versuche, veredelte Texte aus dem Cache zu laden: {output_file}")
            try:
                cached_df = pd.read_csv(output_file, sep='|', dtype=str).fillna('')
                if 'entity_map' not in cached_df.columns or 'text' not in cached_df.columns:
                    raise ValueError("Der Cache-Datei fehlen benötigte Spalten ('text' oder 'entity_map').")
                master_entity_map = {i: _json_str_to_entity_map(row['entity_map']) for i, row in cached_df.iterrows()}
                logger.info("Cache erfolgreich geladen. Veredelungs-Schritt wird übersprungen.")
                return output_file, master_entity_map
            except Exception as e:
                logger.critical(f"Laden der Cache-Datei '{output_file}' fehlgeschlagen: {e}")
                logger.critical("Das Programm kann nicht fortgesetzt werden, da das Überschreiben vom Benutzer abgelehnt wurde.")
                sys.exit(1)
        else:
            logger.info(f"Benutzer hat dem Überschreiben von '{output_file}' zugestimmt. Starte Veredelung neu.")
    else:
        logger.info(f"Zieldatei '{output_file}' nicht gefunden. Starte neue Veredelung.")

    logger.info(f"Starte HOCHPERFORMANTE Text-Veredelungs-Pipeline für {lang_code}...")
    print(f"\n--- Starte hochperformante Text-Veredelung ({lang_code}) ---")

    df = pd.read_csv(input_file, sep='|', dtype=str).fillna('')
    original_texts = df['text'].tolist()

    nlp_model = get_spacy_model(spacy_model_name)
    punctuation_model = PunctuationModel()

    logger.info("Stufe 1/4: Interpunktion im Batch wiederherstellen...")
    punctuated_texts = [safe_punctuate(text, punctuation_model, logger) for text in tqdm(original_texts, desc="Interpunktion")]

    logger.info("Stufe 2/4: Entitäten im Batch erkennen und schützen...")
    docs = list(tqdm(nlp_model.pipe(punctuated_texts), total=len(punctuated_texts), desc="spaCy NER"))
    
    protected_texts = []
    master_entity_map = {}
    for i, doc in enumerate(docs):
        protected_text, entity_map = entity_protection_final(doc.text, nlp_model)
        protected_texts.append(protected_text)
        master_entity_map[i] = entity_map

    # Stufe 3c: Grammatikkorrektur im BATCH (CPU-parallelisiert)
    logger.info("Stufe 3/4: Grammatik im Batch korrigieren (parallelisiert)...")
    corrected_texts = correct_texts_in_parallel(protected_texts, lang_code, LT_PORT)

    logger.info("Stufe 4/4: Entitäten wiederherstellen und Daten finalisieren...")
    processed_rows = []
    for i in range(len(df)):
        restored_text = restore_entities_final(corrected_texts[i], master_entity_map[i])
        processed_rows.append({
            'startzeit': df.iloc[i]['startzeit'],
            'endzeit': df.iloc[i]['endzeit'],
            'text': restored_text,
            'entity_map': _entity_map_to_json_str(master_entity_map[i])
        })

    final_df = pd.DataFrame(processed_rows)
    final_df = ensure_context_closure(final_df)
    final_df['text'] = final_df['text'].apply(sanitize_for_csv_and_tts)
    
    final_df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    
    total_entities = sum(len(mapping) for mapping in master_entity_map.values())
    logger.info(f"Hochperformante Text-Veredelung abgeschlossen. {total_entities} Entitäten verarbeitet.")
    logger.info(f"Ergebnis in: {output_file}")
    
    return output_file, master_entity_map

def refine_text_batch(batch: Dict[str, List], nlp_model, lt_tool, punctuation_model) -> Dict[str, List[str]]:
    """
    Diese Funktion wird mit `dataset.map` auf einen Batch von Daten angewendet.
    Sie führt die komplette Veredelungs-Logik GPU-effizient aus.

    Args:
        batch (Dict[str, List]): Ein Batch von Daten, wie von `datasets.map` übergeben.
        nlp_model: Das geladene spaCy-Modell.
        lt_tool: Die initialisierte LanguageTool-Instanz.
        punctuation_model: Die initialisierte PunctuationModel-Instanz.

    Returns:
        Dict[str, List[str]]: Ein Dictionary mit der neuen Spalte 'refined_text'.
    """
    original_texts = batch['text']
    
    # --- GPU-beschleunigte Batch-Schritte ---
    # 1. Interpunktion für den gesamten Batch wiederherstellen
    punctuated_texts = punctuation_model.restore_punctuation(original_texts)
    
    # 2. Entitäten für den gesamten Batch mit nlp.pipe erkennen
    # Dies ist um Größenordnungen schneller als einzelne Aufrufe.
    docs = list(nlp_model.pipe(punctuated_texts))
    
    refined_texts = []
    
    # --- CPU-gebundene Schritte (einzeln pro Eintrag im Batch) ---
    for i in range(len(original_texts)):
        doc = docs[i]
        text_to_process = doc.text

        # 2a. Entitäten schützen
        protected_text, entity_map = entity_protection_final(text_to_process, doc=doc) # Verwende den bereits prozessierten doc
        
        # 3. Grammatik korrigieren
        corrected_text = lt_tool.correct(protected_text)
        
        # 4. Entitäten wiederherstellen
        restored_text = restore_entities(corrected_text, entity_map)
        
        refined_texts.append(restored_text.strip())

    return {"refined_text": refined_texts}

def make_ph(label: str, idx: int) -> str:
    # U+F000–F0FF liegen im Private-Use-Bereich und werden von
    # Tokenizern nicht angefasst.
    return f"\uF000ENT_{label}_{idx}\uF0FF"

def restore_entities(text: str, entity_mapping: Dict[str, EntityInfo]) -> str:
    """
    KORRIGIERTE VERSION: Setzt die durch Sentinel-Tokens geschützten Entitäten
    wieder in den Text ein.

    Args:
        text (str): Der Text mit Sentinel-Token-Platzhaltern.
        entity_mapping (Dict[str, EntityInfo]): Das Mapping von Platzhaltern zu Entitäten.

    Returns:
        str: Der Text mit wiederhergestellten Entitäten.
    """
    if not entity_mapping:
        return text

    restored_text = text
    # Iteriere durch das Mapping und ersetze jeden Platzhalter.
    for placeholder, entity_info in entity_mapping.items():
        # Wir verwenden hier eine einfache .replace(), da die Marker einzigartig sind.
        # _get_smart_entity_replacement kann hier optional wieder eingefügt werden,
        # aber für maximale Stabilität beginnen wir mit der direkten Ersetzung.
        restored_text = restored_text.replace(placeholder, entity_info.text)

    return restored_text

def _get_smart_entity_replacement(entity_info: EntityInfo, target_language: str) -> str:
    """
    Intelligente Entity-Ersetzung basierend auf Typ und Zielsprache.
    """
    # Eigennamen: Immer Original beibehalten
    if entity_info.label in {"PERSON", "PER", "ORG", "ORGANIZATION", "NORP", "NOR"}:
        return entity_info.text
    
    # Geopolitische Entitäten: Original beibehalten
    if entity_info.label in {"GPE", "FACILITY", "LOCATION", "LOC"}:
        return entity_info.text
    
    # Produkte und Marken: Original beibehalten
    if entity_info.label in {"PRODUCT", "WORK_OF_ART"}:
        return entity_info.text
    
    # Fallback: Original beibehalten
    return entity_info.text

def apply_entity_protection_with_mapping(text: str, entity_mapping: Dict[str, EntityInfo]) -> str:
    """
    Wendet Entity-Schutz mit bereits vorhandenem Mapping an.
    """
    if not entity_mapping:
        return text
    
    protected_text = text
    
    # Sortiere Entitäten nach Startposition (rückwärts) für korrekte Ersetzung
    sorted_entities = sorted(
        entity_mapping.items(),
        key=lambda x: x[1].start,
        reverse=True
    )
    
    for placeholder, entity_info in sorted_entities:
        # Ersetze nur wenn die Entität tatsächlich im Text vorkommt
        if entity_info.text in protected_text:
            start_pos = protected_text.find(entity_info.text)
            if start_pos != -1:
                end_pos = start_pos + len(entity_info.text)
                protected_text = (
                    protected_text[:start_pos] + 
                    placeholder + 
                    protected_text[end_pos:]
                )
    
    return protected_text

def restore_entities_robust(
    text: str, 
    entity_mapping: Dict[str, EntityInfo],
    target_language: str = "de"
) -> str:
    """
    Erweiterte Version: Robuste Entity-Wiederherstellung.
    - Direktes Matching mit Fallback für verzerrte Tokens in einem Loop.
    - Optimiert für geringeren RAM-Verbrauch und schnellere Ausführung.
    """
    if not text or not entity_mapping:
        return text  # Frühzeitiger Exit, um unnötige Verarbeitung zu vermeiden.

    restored_text = text
    restoration_log = []  # Log für erfolgreiche Wiederherstellungen.

    import re  # Import für Regex, falls nicht global verfügbar.

    for placeholder, entity_info in entity_mapping.items():
        # Intelligente Ersetzung basierend auf Entity-Typ.
        replacement = _get_smart_entity_replacement(entity_info, target_language)
        
        # Primäres Matching: Exaktes Placeholder (einschließlich Unicode-Marker).
        if placeholder in restored_text:
            restored_text = restored_text.replace(placeholder, replacement)
            restoration_log.append(f"{placeholder} -> {replacement}")
            continue  # Erfolgreich ersetzt, nächsten Eintrag bearbeiten.

        # Fallback: Flexibles Regex für verzerrte Varianten (ohne Marker, Case-insensitive).
        core_pattern = placeholder.replace('\U0001F300', '').replace('\U0001F301', '')
        pattern = re.compile(rf"ENT_(PERSON|GPE|ORG|PRODUCT)_\d+", re.IGNORECASE)
        
        old_text = restored_text
        restored_text = pattern.sub(replacement, restored_text)
        
        if old_text != restored_text:
            restoration_log.append(f"Pattern {core_pattern} -> {replacement}")

    # Bereinige doppelte Leerzeichen.
    restored_text = re.sub(r'\s+', ' ', restored_text).strip()

    if restoration_log:
        logger.debug(f"Entity-Wiederherstellungen: {'; '.join(restoration_log)}")

    return restored_text

def validate_entity_pipeline(
    original_text: str,
    protected_text: str,
    translated_text: str,
    restored_text: str,
    entity_mapping: Dict[str, EntityInfo]
) -> List[str]:
    """
    Validiert die gesamte Entity-Pipeline und gibt Warnungen zurück.
    
    Args:
        original_text: Ursprünglicher Text
        protected_text: Text mit Platzhaltern
        translated_text: Übersetzter Text mit Platzhaltern
        restored_text: Finaler Text mit wiederhergestellten Entitäten
        entity_mapping: Entity-Mapping
        
    Returns:
        Liste von Validierungsfehlern/Warnungen
    """
    issues = []
    
    # Prüfe ob alle Entitäten geschützt wurden
    for placeholder, entity_info in entity_mapping.items():
        if entity_info.text in original_text and placeholder not in protected_text:
            issues.append(f"Entity '{entity_info.text}' nicht geschützt")
    
    # Prüfe ob Platzhalter durch Übersetzung erhalten blieben
    original_placeholders = set(re.findall(r'__ENTITY_[A-Z_]+_\d+__', protected_text))
    translated_placeholders = set(re.findall(r'__ENTITY_[A-Z_]+_\d+__', translated_text))
    
    lost_placeholders = original_placeholders - translated_placeholders
    if lost_placeholders:
        issues.append(f"Platzhalter in Übersetzung verloren: {lost_placeholders}")
    
    # Prüfe ob alle Platzhalter wiederhergestellt wurden
    remaining_placeholders = re.findall(r'__ENTITY_[A-Z_]+_\d+__', restored_text)
    if remaining_placeholders:
        issues.append(f"Nicht wiederhergestellte Platzhalter: {remaining_placeholders}")
    
    # Prüfe ob wichtige Entitäten im finalen Text vorhanden sind
    important_labels = {"PERSON", "ORG", "GPE", "PRODUCT"}
    for entity_info in entity_mapping.values():
        if (entity_info.label in important_labels and 
            entity_info.text not in restored_text):
            issues.append(f"Wichtige Entity möglicherweise verloren: {entity_info.text}")
    
    return issues

def format_for_tts_splitting(input_file, output_file, char_limit, min_words=3):
    """
    Teilt überlange Segmente und FÜHRT zu kurze Segmente zusammen.
    NEUE, KORREKTE IMPLEMENTIERUNG: Stellt sicher, dass kein Textinhalt durch
    zu kurze Segmente verloren geht, indem diese mit Nachbarn verschmolzen werden.

    Args:
        input_file (str): Pfad zur Eingabedatei.
        output_file (str): Pfad zur Ausgabedatei.
        char_limit (int): Maximale Zeichenlänge, bei der Segmente aufgeteilt werden.
        min_words (int): Minimale Wortanzahl; Segmente darunter werden zusammengeführt.
    """
    if os.path.exists(output_file) and not ask_overwrite(output_file):
        return output_file
        
    df = pd.read_csv(input_file, sep='|', dtype=str).fillna('')
    
    # --- Stufe 1: Überlange Segmente aufteilen ---
    split_segments = []
    for _, row in df.iterrows():
        text = row['text']
        start_sec, end_sec = parse_time(row['startzeit']), parse_time(row['endzeit'])
        
        # Teilen überlanger Segmente (Logik bleibt erhalten)
        chunks = split_text_robust_improved(text, char_limit)
        
        if len(chunks) == 1:
            split_segments.append(row.to_dict())
        else:
            total_duration = end_sec - start_sec
            total_chars = sum(len(c) for c in chunks) if sum(len(c) for c in chunks) > 0 else 1
            current_start = start_sec
            for i, chunk in enumerate(chunks):
                proportion = len(chunk) / total_chars
                chunk_duration = total_duration * proportion
                chunk_end = current_start + chunk_duration if i < len(chunks) - 1 else end_sec
                split_segments.append({
                    'startzeit': format_time(current_start),
                    'endzeit': format_time(chunk_end),
                    'text': chunk
                })
                current_start = chunk_end
    
    # --- Stufe 2: Zu kurze Segmente zusammenführen (Merge-Pass) ---
    if len(split_segments) <= 1:
        # Wenn es nichts zu mergen gibt, direkt speichern
        pd.DataFrame(split_segments).to_csv(output_file, sep='|', index=False)
        logger.info(f"Formatierung für TTS abgeschlossen. Ergebnis in: {output_file}")
        return output_file

    merged_segments = []
    # Konvertiere Liste von Dictionaries in eine modifizierbare Liste
    segments_to_merge = list(split_segments) 

    # Durchlauf 1: Vorwärts-Merge für das erste Element, falls es zu kurz ist
    while len(segments_to_merge[0]['text'].split()) < min_words and len(segments_to_merge) > 1:
        first_seg = segments_to_merge.pop(0)
        next_seg = segments_to_merge[0]
        
        # Kombiniere das erste Segment mit dem nachfolgenden
        next_seg['text'] = f"{first_seg['text']} {next_seg['text']}".strip()
        next_seg['startzeit'] = first_seg['startzeit'] # Behalte die früheste Startzeit

    # Durchlauf 2: Rückwärts-Merge für alle anderen Elemente
    # Wir bauen eine neue Liste auf, das ist stabiler als das Löschen aus der alten.
    for segment in segments_to_merge:
        if not merged_segments: # Das erste Element immer hinzufügen
            merged_segments.append(segment)
            continue
            
        # Wenn das aktuelle Segment zu kurz ist...
        if len(segment['text'].split()) < min_words:
            # ...füge es zum vorherigen in der *neuen* Liste hinzu.
            merged_segments[-1]['text'] = f"{merged_segments[-1]['text']} {segment['text']}".strip()
            merged_segments[-1]['endzeit'] = segment['endzeit'] # Aktualisiere auf die späteste Endzeit
        else:
            # Ansonsten füge es normal hinzu.
            merged_segments.append(segment)

    pd.DataFrame(merged_segments).to_csv(output_file, sep='|', index=False)
    logger.info(f"Formatierung für TTS (inkl. Merging kurzer Segmente) abgeschlossen. Ergebnis in: {output_file}")
    return output_file

def initialize_language_tool(lang: str, lt_path: str):
    """Initialisiert LanguageTool mit lokalem Server"""
    lt_jar_path = os.path.join(lt_path, "languagetool-server.jar")
    if not os.path.exists(lt_jar_path):
        raise FileNotFoundError(f"LanguageTool nicht gefunden: {lt_jar_path}")
    
    port = 8010
    lt_process = subprocess.Popen([
        "java", "-Xmx4g", "-cp", lt_jar_path,
        "org.languagetool.server.HTTPServer",
        "--port", str(port), "--allow-origin", "*"
    ], cwd=lt_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(15)  # Server-Start abwarten
    
    os.environ["LANGUAGE_TOOL_HOST"] = "localhost"
    os.environ["LANGUAGE_TOOL_PORT"] = str(port)
    
    return language_tool_python.LanguageTool(lang)

def load_existing_progress():
    """
    HILFSFUNKTION: Lädt bereits verarbeitete Segmente.
    """
    processed = {}
    if os.path.exists(TTS_PROGRESS_MANIFEST):
        try:
            with open(TTS_PROGRESS_MANIFEST, "r", encoding="utf-8") as csvf:
                reader = csv.reader(csvf, delimiter="|")
                next(reader, None)  # Header überspringen
                for row in reader:
                    if len(row) >= 5:
                        processed[int(row[4])] = row[3]
        except Exception as e:
            logger.warning(f"Fehler beim Laden des Fortschritts: {e}")
    return processed

def load_segments_from_file(translation_file):
    """
    HILFSFUNKTION: Lädt Segmente aus CSV-Datei.
    """
    segments = []
    try:
        with open(translation_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader)  # Header überspringen
            for i, row in enumerate(reader):
                if len(row) >= 3:
                    segments.append({
                        "id": i,
                        "start": row[0],
                        "end": row[1],
                        "text": row[2].strip()
                    })
    except Exception as e:
        logger.error(f"Fehler beim Laden der Segmente: {e}")
    return segments

def filter_new_segments(segments, processed):
    """
    HILFSFUNKTION: Filtert neue, nicht verarbeitete Segmente.
    """
    todo = []
    for seg in segments:
        is_valid, reason = validate_text_for_tts_robust(seg["text"])
        if not is_valid:
            logger.warning(f"Segment {seg['id']} übersprungen: {reason}")
        elif seg["id"] not in processed:
            todo.append(seg)
    return todo

def log_progress(segment_info, chunk_path):
    """
    HILFSFUNKTION: Protokolliert Fortschritt in Manifest-Datei.
    """
    try:
        with open(TTS_PROGRESS_MANIFEST, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter='|')
            
            # Header schreiben falls Datei neu ist
            if os.path.getsize(TTS_PROGRESS_MANIFEST) == 0:
                writer.writerow(['startzeit', 'endzeit', 'text', 'chunk_path', 'original_id'])
            
            writer.writerow([
                segment_info['start'], 
                segment_info['end'], 
                segment_info['text'], 
                chunk_path, 
                segment_info['id']
            ])
    except Exception as e:
        logger.error(f"Fehler beim Protokollieren des Fortschritts: {e}")

def assemble_final_audio_ffmpeg(output_path: str, sampling_rate: int = 24000):
    """
    Assembliert die finale Audiodatei aus den synthetisierten Chunks mithilfe von FFmpeg.
    NEUE, VERBESSERTE VERSION: Diese Methode verhindert das Abschneiden von Audio
    und eliminiert Klicks durch intelligentes Überblenden (Crossfading) bei Überlappungen
    und an den Nahtstellen der Segmente.

    Args:
        output_path (str): Der Pfad zur finalen Ausgabedatei.
        sampling_rate (int): Die Abtastrate des Audios (muss mit den Chunks übereinstimmen).
    """
    logger.info("Starte zeitachsenbasierte Audio-Montage mit FFmpeg und Crossfade.")
    print("\n|<< Starte finale Audio-Assemblierung (FFmpeg mit Crossfade) >>|")

    # Konfigurationen
    stretched_chunks_dir = os.path.join(TTS_TEMP_CHUNKS_DIR, "stretched")
    concat_list_path = "ffmpeg_concat_list.txt"
    MIN_GAP_SEC = 0.2  # Die feste Pause, die bei einer Überlappung eingefügt wird.

    # 1. Lade das Manifest mit allen zu verarbeitenden Chunks
    if not os.path.exists(TTS_PROGRESS_MANIFEST):
        logger.error("Fortschritts-Manifest nicht gefunden. Assemblierung nicht möglich.")
        raise FileNotFoundError(f"Manifest-Datei '{TTS_PROGRESS_MANIFEST}' nicht gefunden.")

    try:
        print("\n|<< Starte finale Audio-Assemblierung >>|")
        
        # Lese das vollständige Manifest
        final_manifest = []
        if os.path.exists(TTS_PROGRESS_MANIFEST):
            with open(TTS_PROGRESS_MANIFEST, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter="|")
                # Sicherstellen, dass die Datei nicht leer ist, bevor der Header gelesen wird
                try:
                    header = next(reader)
                    for row in reader:
                        # Validieren, dass die Zeile die erwartete Struktur hat
                        if len(row) >= 4 and row[3].strip(): # Muss 4 Spalten haben und der Pfad darf nicht leer sein
                            final_manifest.append(row)
                        else:
                            logger.warning(f"Ungültige oder unvollständige Zeile im Manifest übersprungen: {row}")
                except StopIteration:
                    logger.warning(f"Manifest-Datei {TTS_PROGRESS_MANIFEST} war leer oder enthielt nur einen Header.")
        
        if not final_manifest: raise RuntimeError("Keine gültigen Audio-Chunks zum Zusammenfügen gefunden.")

        final_manifest.sort(key=lambda x: convert_time_to_seconds(x[0]))
        
        files_for_concat_list = []
        running_total_duration_sec = 0.0
        sampling_rate = 24000

        for i, segment_data in enumerate(tqdm(final_manifest, desc="Stufe 1: Bearbeite Chunks & Pausen")):
            # Die Validierung fand bereits statt, daher sind die Zugriffe hier sicher.
            start_sec = convert_time_to_seconds(segment_data[0])
            end_sec = convert_time_to_seconds(segment_data[1])
            chunk_path = segment_data[3]
            
            if not os.path.exists(chunk_path): 
                logger.warning(f"Chunk-Datei aus Manifest nicht gefunden, wird übersprungen: {chunk_path}")
                continue

            # Pause berechnen und erzeugen
            gap_duration = start_sec - running_total_duration_sec
            if gap_duration > 0.02:
                silence_path = os.path.join(stretched_chunks_dir, f"silence_{i}.wav")
                silence_command = ['ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=r={sampling_rate}:cl=mono', '-t', str(gap_duration), '-acodec', 'pcm_s16le', '-y', silence_path]
                subprocess.run(silence_command, check=True, capture_output=True)
                files_for_concat_list.append(silence_path)
                running_total_duration_sec += gap_duration

            # Audio-Chunk stretchen
            audio_info = sf.info(chunk_path)
            actual_duration = audio_info.duration
            target_duration = end_sec - start_sec
            
            stretched_chunk_path = os.path.join(stretched_chunks_dir, f"stretched_{i}.wav")

            if target_duration > 0.01 and actual_duration > 0.01:
                speed_factor = actual_duration / target_duration
                capped_speed_factor = max(0.9, min(speed_factor, 1.3))
                
                if abs(capped_speed_factor - speed_factor) > 0.01:
                    logger.warning(f"Stretch-Faktor ({speed_factor:.2f}) für Segment bei {segment_data[0]} auf {capped_speed_factor:.2f} begrenzt.")
                
                atempo_filter = create_atempo_filter_string(capped_speed_factor)
                stretch_command = ['ffmpeg', '-i', chunk_path, '-filter:a', atempo_filter, '-acodec', 'pcm_s16le', '-y', stretched_chunk_path]
                subprocess.run(stretch_command, check=True, capture_output=True)
            else:
                shutil.copy(chunk_path, stretched_chunk_path)
            
            # Berechne die tatsächliche Dauer des bearbeiteten Chunks
            stretched_duration = sf.info(stretched_chunk_path).duration
            
            # Bestimme die Startzeit des aktuellen Segments in der *neuen* Timeline
            # Wenn der ursprüngliche Startpunkt eine Überlappung verursachen würde, passe ihn an.
            if start_sec < running_total_duration_sec:
                logger.warning(
                    f"Überlappung erkannt bei Segment {i} (Start: {segment_data[0]}). "
                    f"Ursprünglicher Start: {start_sec:.2f}s, Aktuelles Audio-Ende: {running_total_duration_sec:.2f}s. "
                    f"Füge feste Pause von {MIN_GAP_SEC}s ein."
                )
                # Setze den neuen Startpunkt auf das Ende des vorherigen Clips plus eine feste Pause
                effective_start_sec = running_total_duration_sec + MIN_GAP_SEC
            else:
                # Keine Überlappung, der ursprüngliche Zeitstempel kann verwendet werden
                effective_start_sec = start_sec

            # Berechne die notwendige Stille, um zum neuen, effektiven Startpunkt zu gelangen
            gap_duration = effective_start_sec - running_total_duration_sec
            if gap_duration > 0.02:
                silence_path = os.path.join(stretched_chunks_dir, f"silence_{i}.wav")
                silence_command = ['ffmpeg', '-f', 'lavfi', '-i', f'anullsrc=r={sampling_rate}:cl=mono', '-t', str(gap_duration), '-acodec', 'pcm_s16le', '-y', silence_path]
                subprocess.run(silence_command, check=True, capture_output=True)
                files_for_concat_list.append(silence_path)
                running_total_duration_sec += gap_duration

            # Füge den gestretchten Sprach-Chunk hinzu
            files_for_concat_list.append(stretched_chunk_path)
            running_total_duration_sec += stretched_duration

        # Stufe 2.2: Verketten
        print("Stufe 2: Verketten der bearbeiteten Chunks via FFmpeg...")
        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for file_path in files_for_concat_list:
                f.write(f"file '{os.path.abspath(file_path).replace(os.sep, '/')}'\n")

        concat_command = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-acodec', 'pcm_s16le', '-y', output_path]
        subprocess.run(concat_command, check=True, capture_output=False)
        
        final_duration_info = sf.info(output_path)
        logger.info(f"Finale Audiodatei erfolgreich erstellt. Dauer: {str(timedelta(seconds=final_duration_info.duration))}")
        print(f"Finale Audiodatei erfolgreich erstellt. Dauer: {str(timedelta(seconds=final_duration_info.duration))}")

    except Exception as e:
        logger.error(f"Ein kritischer Fehler ist aufgetreten: {e}", exc_info=True)
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"FFmpeg stderr: {e.stderr.decode('utf-8', errors='ignore')}")

def assemble_final_audio_numpy(output_path: str, sampling_rate: int = 24000):
    """
    Assembliert die finale Audiodatei aus den synthetisierten Chunks mithilfe einer
    zeitachsenbasierten NumPy-Montage. Diese Methode verhindert das Abschneiden von
    Audio durch intelligentes Überblenden (Crossfading) bei Überlappungen.
    Args:
        output_path (str): Der Pfad zur finalen Ausgabedatei.
        sampling_rate (int): Die Abtastrate des Audios (muss mit den Chunks übereinstimmen).
    """
    logger.info("Starte zeitachsenbasierte Audio-Montage mit NumPy.")
    print("\n|<< Starte finale Audio-Assemblierung (NumPy-basiert) >>|")

    # 1. Lade das Manifest mit allen zu verarbeitenden Chunks
    if not os.path.exists(TTS_PROGRESS_MANIFEST):
        logger.error("Fortschritts-Manifest nicht gefunden. Assemblierung nicht möglich.")
        return

    manifest = []
    with open(TTS_PROGRESS_MANIFEST, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        next(reader)  # Header überspringen
        for row in reader:
            if len(row) >= 4 and os.path.exists(row[3]):
                manifest.append({
                    "start_sec": convert_time_to_seconds(row[0]),
                    "end_sec": convert_time_to_seconds(row[1]),
                    "text": row[2],
                    "path": row[3]
                })

    if not manifest:
        logger.error("Keine gültigen Audio-Chunks im Manifest zum Assemblieren gefunden.")
        return

    # Nach Startzeit sortieren, um die korrekte Reihenfolge sicherzustellen
    manifest.sort(key=lambda x: x["start_sec"])

    # 2. Erstelle die stille "Leinwand"
    # Die Gesamtlänge wird durch die Endzeit des letzten Segments bestimmt.
    total_duration_sec = manifest[-1]["end_sec"]
    total_samples = int(total_duration_sec * sampling_rate) + sampling_rate # +1 Sekunde Puffer
    final_audio_canvas = np.zeros(total_samples, dtype=np.float32)

    previous_end_sample = 0
    crossfade_duration_sec = 0.1  # 100ms für einen weichen Übergang
    crossfade_samples = int(crossfade_duration_sec * sampling_rate)

    # 3. Montiere jeden Chunk auf die Leinwand
    for segment in tqdm(manifest, desc="Montiere Audio-Segmente"):
        # Lade den Audio-Chunk
        try:
            audio_clip, sr = sf.read(segment["path"], dtype='float32')
            if sr != sampling_rate:
                # Hier könnte bei Bedarf ein Resampling stattfinden
                logger.warning(f"Samplerate-Mismatch in {segment['path']}. Erwartet: {sampling_rate}, Gefunden: {sr}")
                continue
        except Exception as e:
            logger.error(f"Fehler beim Laden von Chunk {segment['path']}: {e}")
            continue

        # Berechne die Start- und Endposition in Samples auf der Leinwand
        start_sample = int(segment["start_sec"] * sampling_rate)
        end_sample = start_sample + len(audio_clip)


        # Überprüfen, ob die Leinwand groß genug ist, und ggf. erweitern
        if end_sample > len(final_audio_canvas):
            padding_needed = end_sample - len(final_audio_canvas)
            final_audio_canvas = np.pad(final_audio_canvas, (0, padding_needed), 'constant')
            logger.info(f"Audio-Leinwand um {padding_needed} Samples erweitert.")

        if start_sample < previous_end_sample:
            overlap_samples = previous_end_sample - start_sample
            
            # KORREKTUR: Die tatsächliche Fade-Länge darf niemals länger sein als die Überlappung,
            # der gewünschte Crossfade oder die Länge des neuen Clips selbst.
            fade_len = min(overlap_samples, crossfade_samples, len(audio_clip))

            # Zusätzliche Sicherheitsprüfung: Wenn fade_len negativ oder null ist, überspringe Crossfade.
            if fade_len > 0:
                logger.warning(
                    f"Überlappung erkannt bei {segment['start_sec']:.2f}s. "
                    f"Dauer: {overlap_samples / sampling_rate:.3f}s. Wende {fade_len / sampling_rate:.3f}s Crossfade an."
                )

                fade_out = np.linspace(1.0, 0.0, fade_len)
                fade_in = np.linspace(0.0, 1.0, fade_len)

                # Slices für die Operationen definieren. Diese haben jetzt garantiert die gleiche Länge `fade_len`.
                canvas_fade_region = final_audio_canvas[start_sample : start_sample + fade_len]
                clip_fade_region = audio_clip[:fade_len]

                # Mischen
                final_audio_canvas[start_sample : start_sample + fade_len] = \
                    canvas_fade_region * fade_out + clip_fade_region * fade_in

                # Füge den Rest des neuen Clips nach der Fade-Region ein
                remaining_clip_start_index = fade_len
                final_audio_canvas[start_sample + remaining_clip_start_index : end_sample] = audio_clip[remaining_clip_start_index:]
            else:
                # Kein sinnvoller Crossfade möglich, füge den Clip einfach hinzu (additiv).
                final_audio_canvas[start_sample:end_sample] += audio_clip

        else:
            # Kein Overlap, füge den Clip direkt ein (additiv, um eventuelle vorherige Fades nicht zu überschreiben)
            final_audio_canvas[start_sample:end_sample] += audio_clip

        # Die Endposition des aktuellen Clips wird zur neuen Referenz
        previous_end_sample = end_sample

    final_audio_canvas = final_audio_canvas[:previous_end_sample]

    peak_amplitude = np.max(np.abs(final_audio_canvas))
    if peak_amplitude > 0.98: # Normalisierungsschwelle leicht gesenkt
        normalization_factor = 0.98 / peak_amplitude
        final_audio_canvas *= normalization_factor
        logger.info(f"Audio normalisiert (Faktor: {normalization_factor:.2f}), da Spitzenamplitude {peak_amplitude:.2f} war.")
        
    sf.write(output_path, final_audio_canvas, sampling_rate)
    final_duration = len(final_audio_canvas) / sampling_rate
    logger.info(f"Finale Audiodatei erfolgreich montiert: {output_path} (Dauer: {final_duration:.2f}s)")
    print(f"Finale Audiodatei erfolgreich montiert. Dauer: {timedelta(seconds=final_duration)}")

def transfer_conditioning_to_device(gpt_cond_latent, speaker_embedding, target_device):
    """
    SPEZIALISIERTE FUNKTION: Sichere Übertragung von Conditioning-Tensoren.
    """
    print(f"🔄 Übertrage Conditioning-Tensoren auf {target_device}")
    
    # GPT Conditioning übertragen
    if gpt_cond_latent is not None:
        original_device = gpt_cond_latent.device
        gpt_cond_latent = gpt_cond_latent.to(target_device)
        print(f"✅ GPT Conditioning: {original_device} → {gpt_cond_latent.device}")
    
    # Speaker Embedding übertragen
    if speaker_embedding is not None:
        original_device = speaker_embedding.device
        speaker_embedding = speaker_embedding.to(target_device)
        print(f"✅ Speaker Embedding: {original_device} → {speaker_embedding.device}")
    
    return gpt_cond_latent, speaker_embedding

@contextmanager
def synchronized_cuda_stream(stream, wait_for_streams=None):
    """
    ERWEITERTE VERSION: Context Manager mit automatischer Synchronisation.
    
    Args:
        stream: Zu verwendender Stream
        wait_for_streams: Liste von Streams auf die gewartet werden soll
    """
    if not torch.cuda.is_available():
        yield
        return
    
    # Vor Stream-Wechsel: Synchronisation mit anderen Streams
    if wait_for_streams:
        for wait_stream in wait_for_streams:
            if wait_stream != stream:
                stream.wait_stream(wait_stream)
    
    # Stream-Context ausführen
    with torch.cuda.stream(stream):
        yield

def safe_tts_inference_with_device_handling(
    model, 
    text, 
    gpt_cond_latent, 
    speaker_embedding, 
    **kwargs
):
    """
    SICHERE TTS-INFERENZ: Mit umfassender Device-Behandlung.
    """
    try:
        # Inferenz mit Device-Monitoring
        with torch.inference_mode():
            result = model.inference(
                text=text,
                language="de",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **kwargs
            )
        
        return result
        
    except RuntimeError as e:
        if "same device" in str(e).lower():
            logger.error(f"Device-Mismatch während Inferenz: {e}")
            
            # Automatische Korrektur versuchen
            model_device = next(model.parameters()).device
            logger.info(f"Versuche automatische Korrektur auf {model_device}")
            
            corrected_gpt = gpt_cond_latent.to(model_device) if gpt_cond_latent is not None else None
            corrected_speaker = speaker_embedding.to(model_device) if speaker_embedding is not None else None
            
            # Retry mit korrigierten Tensoren
            with torch.inference_mode():
                result = model.inference(
                    text=text,
                    language="de",
                    gpt_cond_latent=corrected_gpt,
                    speaker_embedding=corrected_speaker,
                    **kwargs
                )
            
            logger.info("✅ Automatische Device-Korrektur erfolgreich")
            return result
        else:
            raise
    
    except Exception as e:
        logger.error(f"Unerwarteter Fehler bei TTS-Inferenz: {e}")
        raise

# Synthetisieren
def text_to_speech_with_voice_cloning(
    translation_file,
    sample_path_1,
    sample_path_2,
    sample_path_3,
    #sample_path_4,
    #sample_path_5,
    output_path
):
    """
    Optimiert Text-to-Speech mit Voice Cloning und verschiedenen Beschleunigungen.

    Args:
        translation_file: Pfad zur CSV-Datei mit übersetzten Texten.
        sample_path_1, sample_path_2: Pfade zu Sprachbeispielen für die Stimmenklonung.
        output_path: Ausgabepfad für die generierte Audiodatei.
        batch_size: Größe des Batches für parallele Verarbeitung.
    """
    stretched_chunks_dir = os.path.join(TTS_TEMP_CHUNKS_DIR, "stretched")
    concat_list_path = "ffmpeg_concat_list.txt"

    # === SCHRITT 1: Lade und filtere Segmente, um festzustellen, ob Arbeit zu tun ist ===
    processed = load_existing_progress()
    all_segments = load_segments_from_file(translation_file)
    segments_to_process = filter_new_segments(all_segments, processed)
    
    # === SCHRITT 2: Prüfen, ob überhaupt etwas zu tun ist ===
    if not segments_to_process:
        logger.info("Keine neuen Segmente zu synthetisieren.")
        # Wenn die Ausgabedatei bereits existiert, ist alles in Ordnung.
        if os.path.exists(output_path):
            logger.info(f"Finale TTS-Audiodatei '{output_path}' existiert bereits und es gibt nichts Neues zu tun.")
        pass # Funktion sicher beenden.

    logger.info(f"{len(segments_to_process)} neue Segmente werden synthetisiert…")

    # === SCHRITT 3: JETZT, da wir wissen, dass Arbeit zu tun ist, prüfen wir die Ausgabedatei ===
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            # Der Benutzer möchte nicht überschreiben, also brechen wir ab.
            logger.info(f"Benutzer hat das Überschreiben der existierenden Datei '{output_path}' abgelehnt. TTS-Synthese wird abgebrochen.")
            return
        else:
            # Der Benutzer möchte überschreiben, also räumen wir auf.
            logger.info(f"Benutzer hat dem Überschreiben von '{output_path}' zugestimmt. Alte temporäre Dateien werden entfernt.")
            if os.path.exists(TTS_TEMP_CHUNKS_DIR): shutil.rmtree(TTS_TEMP_CHUNKS_DIR)
            if os.path.exists(TTS_PROGRESS_MANIFEST): os.remove(TTS_PROGRESS_MANIFEST)

    # Stelle sicher, dass die temporären Verzeichnisse existieren
    os.makedirs(TTS_TEMP_CHUNKS_DIR, exist_ok=True)
    os.makedirs(stretched_chunks_dir, exist_ok=True)

    with gpu_context():
        try:
            print("------------------")
            print("|<< Starte TTS >>|")
            print("------------------")
            tts_start_time = time.time()
            
            logger.info(f"TTS wird auf Gerät ausgeführt: {device}")

            # Modell laden und auf das Zielgerät verschieben
            xtts_model = load_xtts_v2()
        
            print(f"✅ Modell auf {next(xtts_model.parameters()).device} geladen")
        
            # Conditioning-Latents direkt auf dem Zielgerät erstellen
            sample_paths = [
                sample_path_1,
                sample_path_2,
                sample_path_3
                ]
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=sample_paths, load_sr=22050, sound_norm_refs=True
            )
            gpt_cond_latent = gpt_cond_latent.to(device)
            speaker_embedding = speaker_embedding.to(device)
            
            logger.info("TTS-Modell und Conditioning-Latents erfolgreich geladen.")

            print("🚀 DeepSpeed-Initialisierung...")
            ds_engine = deepspeed.init_inference(
                model=xtts_model,
                tensor_parallel={"tp_size": 1},
                dtype=torch.float32,
                replace_with_kernel_inject=False
            )
            optimized_tts_model = ds_engine.module
            logger.info("✅ DeepSpeed erfolgreich initialisiert")

            for segment_info in tqdm(segments_to_process, desc="Synthetisiere Audio-Chunks"):
                try:
                    with torch.inference_mode():
                        # Die Inferenz wird nun auf dem vom StreamManager verwalteten Stream ausgeführt.
                        result = optimized_tts_model.inference(
                            text=segment_info['text'],
                            language="de",
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            speed=1.1,
                            temperature=0.9,
                            repetition_penalty=5.0,
                            top_k=50,
                            top_p=0.80,
                            enable_text_splitting=False
                        )
                        print(f"\n[{segment_info['start']} --> {segment_info['end']}]:\n{segment_info['text']}\n")
                    audio_clip = result.get("wav")
                    if audio_clip is None or audio_clip.size == 0:
                        logger.warning(f"Leeres Audio für Segment {segment_info['id']} erhalten.")
                        continue
                
                    # Audio speichern (nachdem der Stream synchronisiert wurde)
                    chunk_filename = f"chunk_{segment_info['id']}.wav"
                    chunk_path = os.path.join(TTS_TEMP_CHUNKS_DIR, chunk_filename)
                    sf.write(chunk_path, audio_clip, 24000)
                    log_progress(segment_info, chunk_path)

                except Exception as e:
                    logger.error(f"Fehler bei der TTS-Synthese für Segment {segment_info['id']}: {e}", exc_info=True)
                    continue # Zum nächsten Segment springen

            # ======================================================================
            # PHASE 4: Audio-Assemblierung
            # ======================================================================
            try:
                logger.info("Alle Chunks synthetisiert. Starte finale Audio-Montage.")
                assemble_final_audio_ffmpeg(output_path)
            except Exception as e:
                logger.critical(f"Die finale Audio-Montage ist fehlgeschlagen: {e}", exc_info=True)

            logger.info(f"TTS-Prozess abgeschlossen. Finale Audiodatei: {output_path}")

            
            print(f"---------------------------")
            print(f"|<< TTS abgeschlossen!! >>|")
            print(f"---------------------------")
            
            tts_end_time = time.time() - tts_start_time
            logger.info(f"TTS abgeschlossen in {tts_end_time:.2f} Sekunden")
            print(f"{(tts_end_time):.2f} Sekunden")
            print(f"{(tts_end_time / 60):.2f} Minuten")
            print(f"{(tts_end_time / 3600):.2f} Stunden")
            logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}")
            logger.info("🎉 Stream-optimierte TTS-Synthese abgeschlossen")
            
        except Exception as e:
            logger.critical(f"Ein schwerwiegender Fehler ist im TTS-Prozess aufgetreten: {e}", exc_info=True)
        # Am Ende des `with gpu_context()`-Blocks wird der Speicher automatisch geleert.

def create_conditioning_tensors_on_device(
    xtts_model, 
    sample_paths, 
    target_device=None, 
    load_sr=22050
):
    """
    KORRIGIERTE VERSION: Erstellt Conditioning-Tensoren direkt auf dem Ziel-Device.
    """
    if target_device is None:
        target_device = next(xtts_model.parameters()).device
    
    print(f"🔄 Erstelle Conditioning-Tensoren auf Device: {target_device}")
    
    try:
        # WICHTIG: Modell muss sich im eval-Modus befinden
        xtts_model.eval()
        
        with torch.inference_mode():
            # Conditioning-Tensoren erstellen
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=sample_paths,
                load_sr=load_sr
            )
            
            # KRITISCH: Explizite Device-Übertragung
            if gpt_cond_latent is not None:
                gpt_cond_latent = gpt_cond_latent.to(target_device)
                print(f"✅ GPT Conditioning auf {gpt_cond_latent.device} übertragen")
            
            if speaker_embedding is not None:
                speaker_embedding = speaker_embedding.to(target_device)
                print(f"✅ Speaker Embedding auf {speaker_embedding.device} übertragen")
            
            # Validierung
            if gpt_cond_latent is not None and speaker_embedding is not None:
                model_device = next(xtts_model.parameters()).device
                
                if (gpt_cond_latent.device == model_device and 
                    speaker_embedding.device == model_device):
                    print("✅ Alle Conditioning-Tensoren korrekt auf Modell-Device")
                else:
                    print(f"❌ Device-Mismatch: Modell={model_device}, "
                        f"GPT={gpt_cond_latent.device}, Speaker={speaker_embedding.device}")
            
            return gpt_cond_latent, speaker_embedding
    
    except Exception as e:
        print(f"❌ Fehler bei Conditioning-Erstellung: {e}")
        return None, None

def ensure_tensor_on_device(tensor, target_device):
    """
    HILFSFUNKTION: Stellt sicher, dass Tensor auf dem richtigen Device ist.
    """
    if tensor is None:
        return None
    
    if hasattr(tensor, 'device') and tensor.device != target_device:
        print(f"🔄 Übertrage Tensor von {tensor.device} auf {target_device}")
        return tensor.to(target_device)
    
    return tensor

def ensure_memory_safety_with_streams(tensor, target_stream):
    """
    KRITISCHE FUNKTION: Verhindert Memory-Corruption bei Multi-Stream-Usage.
    
    Args:
        tensor: PyTorch Tensor
        target_stream: Stream auf dem Tensor verwendet wird
    """
    if not torch.cuda.is_available() or tensor is None:
        return
    
    if hasattr(tensor, 'record_stream'):
        # Markiert Tensor als "verwendet" auf diesem Stream
        # Verhindert vorzeitige Speicher-Freigabe
        tensor.record_stream(target_stream)
        
        logger.debug(f"Memory-Safety: Tensor auf Stream {target_stream} registriert")
    else:
        logger.warning("Tensor unterstützt record_stream nicht - Memory-Corruption möglich")

def tts_tensor_postprocessing(tensor):
    """
    ECHTE OPERATION: TTS-spezifische Tensor-Nachbearbeitung.
    """
    if tensor is None:
        return None
    
    # Audio-Tensor-Normalisierung
    if tensor.dtype == torch.float32:
        # RMS-Normalisierung
        rms = torch.sqrt(torch.mean(tensor**2))
        if rms > 0:
            target_rms = 0.15
            tensor = tensor * (target_rms / rms)
        
        # Peak-Limiting
        peak = torch.max(torch.abs(tensor))
        if peak > 0.95:
            tensor = tensor * (0.95 / peak)
    
    return tensor

def tts_conditioning_transfer(conditioning_tensor):
    """
    ECHTE OPERATION: TTS-Conditioning-Tensor-Transfer.
    """
    if conditioning_tensor is None:
        return None
    
    # Sicherstellen dass Conditioning-Tensor korrekt formatiert ist
    if conditioning_tensor.dim() == 2:
        # Batch-Dimension hinzufügen falls nötig
        if conditioning_tensor.shape[0] != 1:
            conditioning_tensor = conditioning_tensor.unsqueeze(0)
    
    return conditioning_tensor.contiguous()

def transfer_to_stream(tensor, target_stream):
    """
    HILFSFUNKTION: Tensor sicher auf anderen Stream transferieren.
    """
    if tensor is None or not torch.cuda.is_available():
        return tensor
    
    with torch.cuda.stream(target_stream):
        # Kopiere Tensor auf aktuellen Stream
        transferred = tensor.detach().clone()
        
        # WICHTIG: record_stream für Memory-Safety
        if hasattr(tensor, 'record_stream'):
            tensor.record_stream(target_stream)
        
        return transferred

def transfer_audio_to_cpu_async(audio_tensor, stream):
    """
    HILFSFUNKTION: Asynchroner Transfer von GPU zu CPU.
    """
    with torch.cuda.stream(stream):
        # non_blocking=True für asynchronen Transfer
        cpu_tensor = audio_tensor.to('cpu', non_blocking=True)
        
        # record_stream für Memory-Safety
        if hasattr(audio_tensor, 'record_stream'):
            audio_tensor.record_stream(stream)
        
        return cpu_tensor

def cleanup_cuda_streams(streams):
    """
    HILFSFUNKTION: Sichere Stream-Bereinigung.
    """
    for stream in streams:
        if stream is not None:
            try:
                stream.synchronize()
            except Exception as e:
                logger.warning(f"Stream-Cleanup-Fehler: {e}")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

def resample_to_44100_stereo(input_path, output_path, speed_factor):
    """
    Resample das Audio auf 44.100 Hz (Stereo), passe die Wiedergabegeschwindigkeit sowie die Lautstärke an.
    """
    if os.path.exists(output_path) and not ask_overwrite(output_path):
        logger.info(f"Verwende vorhandene resampelte Datei: {output_path}")
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
    # ======================================================================
    # Performance-Optimierungen als allererster Schritt aktivieren
    setup_global_torch_optimizations()
    # ======================================================================

    # LanguageTool-Server einmalig für das gesamte Skript starten
    lt_process = None
    try:
        lt_jar_path = os.path.join("D:\\LanguageTool-6.6", "languagetool-server.jar")
        if not os.path.exists(lt_jar_path):
            raise FileNotFoundError(f"LanguageTool nicht gefunden: {lt_jar_path}")
        
        global LT_PORT
        LT_PORT = 8010
        lt_process = subprocess.Popen(
            ["java", "-Xmx4g", "-cp", lt_jar_path, "org.languagetool.server.HTTPServer", "--port", str(LT_PORT), "--allow-origin", "*"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
        )
        logger.info(f"LanguageTool-Server-Prozess gestartet (PID: {lt_process.pid}).")
        
        # ## ÄNDERUNG: Ersetze time.sleep durch die robuste "Health Check"-Schleife ##
        if not wait_for_server_ready(port=LT_PORT, timeout=60):
            # Wenn der Server nicht startet, beende das Skript sauber.
            lt_process.terminate()
            raise RuntimeError("LanguageTool-Server konnte nicht gestartet werden.")

        # --------------------------------------------------------------------------
        # SCHRITT 1: VORBEREITUNG & TRANSKRIPTION
        # --------------------------------------------------------------------------
        if not os.path.exists(VIDEO_PATH):
            logger.error(f"Eingabevideo nicht gefunden: {VIDEO_PATH}"); return

        extract_audio_ffmpeg(VIDEO_PATH, ORIGINAL_AUDIO_PATH)
        create_voice_sample(
            ORIGINAL_AUDIO_PATH,
            SAMPLE_PATH_1,
            SAMPLE_PATH_2,
            SAMPLE_PATH_3
        )
        transcribe_audio_with_timestamps(ORIGINAL_AUDIO_PATH, TRANSCRIPTION_FILE)

        # --------------------------------------------------------------------------
        # SCHRITT 2: VEREDELUNG DER ENGLISCHEN TRANSKRIPTION
        # --------------------------------------------------------------------------
        refined_transcription_path, master_entity_map_en = refine_text_pipeline(
            input_file=TRANSCRIPTION_FILE,
            output_file=REFINED_TRANSCRIPTION_FILE,
            #spacy_model_name="ru_core_news_lg",
            #lang_code='ru-RU',
            spacy_model_name="en_core_web_trf",
            lang_code='en-US'
        )
        if not refined_transcription_path:
            logger.error("Veredelung der Transkription fehlgeschlagen."); return

        # --------------------------------------------------------------------------
        # SCHRITT 3: ÜBERSETZUNG
        # --------------------------------------------------------------------------
        translation_file_path, cleaned_source_path = translate_segments_optimized_safe(
            refined_transcription_path=refined_transcription_path,
            master_entity_map=master_entity_map_en, # <-- HIER wird das Mapping übergeben
            translation_output_file=TRANSLATION_FILE,
            cleaned_source_output_file=CLEANED_SOURCE_FOR_QUALITY_CHECK,
            source_lang="en",
            target_lang="de",
            batch_size=1
        )
        if not translation_file_path:
            logger.error("Übersetzung fehlgeschlagen."); return

        # --------------------------------------------------------------------------
        # SCHRITT 4: QUALITÄTSPRÜFUNG
        # --------------------------------------------------------------------------
        evaluate_translation_quality(
            source_csv_path=cleaned_source_path,
            translated_csv_path=translation_file_path,
            report_path=TRANSLATION_QUALITY_REPORT,
            summary_path=TRANSLATION_QUALITY_SUMMARY,
            model_name=ST_QUALITY_MODEL,
            threshold=SIMILARITY_THRESHOLD
        )

        # --------------------------------------------------------------------------
        # SCHRITT 5: VEREDELUNG DER DEUTSCHEN ÜBERSETZUNG
        # --------------------------------------------------------------------------
        refined_translation_path, _ = refine_text_pipeline(
            input_file=translation_file_path,
            output_file=REFINED_TRANSLATION_FILE,
            spacy_model_name="de_dep_news_trf",
            lang_code='de-DE'
        )
        if not refined_translation_path:
            logger.error("Veredelung der Übersetzung fehlgeschlagen."); return
            
        # --------------------------------------------------------------------------
        # SCHRITT 6: TTS-FORMATIERUNG & SYNTHESE
        # --------------------------------------------------------------------------
        # Die alte format_translation_for_tts wird durch eine einfachere Version ersetzt
        # die nur noch das finale Splitting durchführt.
        tts_formatted_file = format_for_tts_splitting(
            input_file=refined_translation_path,
            output_file=TTS_FORMATTED_TRANSLATION_FILE,
            char_limit=CHAR_LIMIT_TRANSLATION
        )
        
        text_to_speech_with_voice_cloning(
            tts_formatted_file,
            SAMPLE_PATH_1,
            SAMPLE_PATH_2,
            SAMPLE_PATH_3,
            TRANSLATED_AUDIO_WITH_PAUSES
        )

        # --------------------------------------------------------------------------
        # SCHRITT 7: FINALES VIDEO ERSTELLEN
        # --------------------------------------------------------------------------
        resample_to_44100_stereo(TRANSLATED_AUDIO_WITH_PAUSES, RESAMPLED_AUDIO_FOR_MIXDOWN, SPEED_FACTOR_RESAMPLE_44100)
        adjust_playback_speed(VIDEO_PATH, ADJUSTED_VIDEO_PATH, SPEED_FACTOR_PLAYBACK)
        combine_video_audio_ffmpeg(ADJUSTED_VIDEO_PATH, RESAMPLED_AUDIO_FOR_MIXDOWN, FINAL_VIDEO_PATH)

        total_time = time.time() - start_time
        logger.info(f"Projekt erfolgreich abgeschlossen in {(total_time / 60):.2f} Minuten.")
        print(f"\n\n--- PROJEKT ABGESCHLOSSEN ---")
        print(f"Gesamtdauer: {(total_time / 60):.2f} Minuten")
        print(f"Finale Ausgabedatei: {FINAL_VIDEO_PATH}")

    except Exception as e:
        logger.critical(f"Ein nicht abgefangener Fehler ist in main aufgetreten: {e}", exc_info=True)
    finally:
        if lt_process:
            logger.info("Beende LanguageTool-Server...")
            lt_process.terminate()
            lt_process.wait()
            logger.info("LanguageTool-Server beendet.")

if __name__ == "__main__":
    lt_process = start_language_tool_server()
    
    if lt_process:
        try:
            main()
        except Exception as e:
            logger.critical(f"Ein nicht abgefangener Fehler ist aufgetreten: {e}", exc_info=True)
        finally:
            stop_language_tool_server(lt_process)
            logger.info("Programm wird beendet.")
            logging.shutdown()
    else:
        logger.critical("Programm wird beendet, da der LanguageTool-Server nicht gestartet werden konnte.")
        logging.shutdown()