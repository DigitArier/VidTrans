import os
import io
import re
from pathlib import Path
import subprocess
import ffmpeg
import logging
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import pandas as pd
import json
import scipy.signal as signal
import torch
torch.set_num_threads(1)
import shape as sh
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import pyrubberband
import time
from datetime import datetime, timedelta
import csv
from config import *
from tqdm import tqdm
from contextlib import contextmanager
from scipy.interpolate import interp1d
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MarianConfig,
    MarianPreTrainedModel,
    MarianMTModel,
    MarianTokenizer,
    PreTrainedModel,
    AutoModelForCausalLM,
    GenerationMixin,
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    UnivNetFeatureExtractor,
    UnivNetModel
    )
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.vocoder.configs.univnet_config import UnivnetConfig
from TTS.vocoder.models.univnet_generator import UnivnetGenerator
from TTS.tts.models.xtts import Xtts
from bigvgan.inference import inference
import whisper
from faster_whisper import vad, WhisperModel
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


# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.0   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.0   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 1.0      # Geschwindigkeitsfaktor für die Wiedergabe des Videos

# Lautstärkeanpassungen
VOLUME_ADJUSTMENT_44100 = 1.0   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.05   # Lautstärkefaktor für das Video

# ============================== 
# Globale Konfigurationen und Logging
# ==============================
logging.basicConfig(filename='video_translation_final.log', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
_WHISPER_MODEL = None
_TRANSLATE_MODEL = None
_TTS_MODEL = None

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
    'hwaccel': 'cuda',
    'hwaccel_output_format': 'cuda'
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

def ask_overwrite(file_path):
    """Fragt den Benutzer, ob eine bestehende Datei überschrieben werden soll."""
    while True:
        choice = input(f"Die Datei '{file_path}' existiert bereits. Überschreiben? (j/n): ").strip().lower()
        if choice in ["j", "ja"]:
            return True
        elif choice == "" or choice in ["n", "nein"]:
            return False

step_start_time = time.time()
def get_whisper_model():
    global _WHISPER_MODEL
    if not _WHISPER_MODEL:
#        _WHISPER_MODEL = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(device)
        _WHISPER_MODEL = WhisperModel("large-v3-turbo", device=device)
#        _WHISPER_MODEL.to(torch.device("cuda"))
        torch.cuda.empty_cache()
    return _WHISPER_MODEL

def get_translate_model():
    global _TRANSLATE_MODEL
    if not _TRANSLATE_MODEL:
        _TRANSLATE_MODEL = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
#       _TRANSLATE_MODEL = T5ForConditionalGeneration.from_pretrained("t5-large")
        _TRANSLATE_MODEL.to(torch.device("cuda"))
        torch.cuda.empty_cache()
    return _TRANSLATE_MODEL


#def get_tts_model():
#    global _TTS_MODEL
#    if not _TTS_MODEL:
#        config = XttsConfig(
#            model="xtts_v2.0.2"
#        )
#        config.load_json(r"C:\Users\regme\Desktop\Translate\VidTrans\VidTrans\XTTS\config.json")
#        _TTS_MODEL = Xtts.init_from_config(config)
#        _TTS_MODEL.load_checkpoint(config, checkpoint_dir=r"C:\Users\regme\Desktop\Translate\VidTrans\VidTrans\XTTS\2.0.2")
# tts_models/de/thorsten/vits"
# tts_models/de/thorsten/tacotron2-DDC
# tts_models/multilingual/multi-dataset/xtts_v2
#        _TTS_MODEL.to(torch.device("cuda"))
#        #torch.load(_TTS_MODEL, weights_only=True)
#        torch.cuda.empty_cache()
#    return _TTS_MODEL


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
        ffmpeg.input(video_path).output(
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
        sampling_rate=SAMPLING_RATE
        logger.info("Lade Silero VAD-Modell...")
        vad_model = load_silero_vad(onnx=USE_ONNX)
        logger.info("Starte Sprachaktivitätserkennung...")
        wav = read_audio(audio_path, sampling_rate)


        # Initialize output_audio as a 2D tensor [1, samples]
        output_audio = torch.zeros((1, len(wav)), dtype=wav.dtype, device=wav.device)
        
        speech_timestamps = get_speech_timestamps(
            wav,                                # Audio-Daten
            vad_model,                          # Silero VAD-Modell
            sampling_rate=SAMPLING_RATE,        # 16.000 Hz
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
            {"start": int(ts["start"] * SAMPLING_RATE), "end": int(ts["end"] * SAMPLING_RATE)}
            for ts in speech_timestamps
        ]
        # Speichere nur die Sprachsegmente als separate WAV-Datei
        save_audio(
            'only_speech_silero.wav',
            collect_chunks(speech_timestamps_samples, wav),
                    sampling_rate=SAMPLING_RATE
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

def transcribe_audio_with_timestamps(audio_file, transcription_file):
    """Führt eine Spracherkennung mit Whisper durch und speichert die Transkription (inkl. Zeitstempel) in einer JSON-Datei."""
    if os.path.exists(transcription_file):
        if not ask_overwrite(transcription_file):
            logger.info(f"Verwende vorhandene Transkription: {transcription_file}", exc_info=True)
            return read_transcripted_csv(transcription_file)
    try:
        logger.info("Lade Whisper-Modell (large-v3-turbo)...", exc_info=True)
        logger.info("Starte Transkription...", exc_info=True)
        print("----------------------------")
        print("|<< Starte Transkription >>|")
        print("----------------------------")
        model = get_whisper_model()                                                 # Laden des vortrainierten Whisper-Modells
        segments, info = model.transcribe(
            audio_file,                         # Audio-Datei
            beam_size=10,
            patience=2.0,
            vad_filter=True,
#            chunk_length=60,
            compression_ratio_threshold=1.8,    # Schwellenwert für Kompressionsrate
#            log_prob_threshold=-0.2,             # Schwellenwert für Log-Probabilität
#            no_speech_threshold=2.0,            # Schwellenwert für Stille
            temperature=(0.05, 0.1, 0.2),      # Temperatur für Sampling
            word_timestamps=True,               # Zeitstempel für Wörter
#            hallucination_silence_threshold=0.35,  # Schwellenwert für Halluzinationen
            condition_on_previous_text=True,    # Bedingung an vorherigen Text
            no_repeat_ngram_size=3,
#            repetition_penalty=1.5,
#            verbose=True,                       # Ausführliche Ausgabe
            language="en",                       # Englische Sprache
#            task="translate",                    # Übersetzung aktivieren
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
        return segments_list

    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}", exc_info=True)
        return []

def parse_time(time_str):
    """
    Konvertiert h:mm:ss-Strings in Sekunden
    """
    try:
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    except ValueError:
        raise ValueError(f"Ungültiges Zeitformat: {time_str}")

def format_time(seconds):
    """
    Konvertiert Sekunden zurück in h:mm:ss-Format
    """
    return str(timedelta(seconds=int(seconds)))

def merge_transcript_chunks(input_file, output_file, min_dur=1, max_dur=15, max_gap=5):
    """
    Führt Transkript-Segmente unter Berücksichtigung der spezifizierten Regeln zusammen
    
    Args:
        input_file (str): Eingabedatei mit | als Trennzeichen
        output_file (str): Zieldatei für Ergebnisse
        min_dur (int): Minimale Segmentdauer in Sekunden
        max_dur (int): Maximale Segmentdauer in Sekunden
        max_gap (int): Maximaler akzeptierter Zeitabstand zwischen Segmenten
    """
    if os.path.exists(output_file):
        if not ask_overwrite(output_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {output_file}", exc_info=True)
            return read_translated_csv(output_file)
        
    try:
        # CSV mit | als Trennzeichen einlesen
        df = pd.read_csv(
            input_file,
            sep='|',
            dtype=str
        )

        # Debugging: Zeige erkannte Spalten
        print("Original Spalten:", df.columns.tolist())

        # Spaltennamen normalisieren
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '')
        print("Normalisierte Spalten:", df.columns.tolist())

        # Erforderliche Spalten validieren
        required_columns = {'startzeit', 'endzeit', 'text'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Fehlende Spalten: {', '.join(missing)}")

        print(f"Verarbeite {len(df)} Segmente aus {input_file}...")

        # Zeitkonvertierung mit Fehlerprotokollierung
        def safe_parse(time_str):
            try:
                return parse_time(time_str)
            except ValueError:
                print(f"Ungültige Zeitangabe: {time_str}")
                return None

        # Zeitkonvertierung
        df['start_sec'] = df['startzeit'].apply(safe_parse)
        df['end_sec'] = df['endzeit'].apply(safe_parse)
        df['duration'] = df['end_sec'] - df['start_sec']

        # Ungültige Zeilen filtern
        original_count = len(df)
        df = df.dropna(subset=['start_sec', 'end_sec'])
        if len(df) < original_count:
            print(f"{original_count - len(df)} ungültige Zeilen entfernt")
            
        # Sortierung nach Startzeit
        df = df.sort_values('start_sec').reset_index(drop=True)

        merged_data = []
        current_chunk = None

        for _, row in df.iterrows():
            if not current_chunk:
                # Neues Segment starten
                current_chunk = {
                    'start': row['start_sec'],
                    'end': row['end_sec'],
                    'text': [row['text']]
                }
            else:
                gap = row['start_sec'] - current_chunk['end']
                
                # Entscheidungslogik
                if (gap <= max_gap) and ((row['end_sec'] - current_chunk['start']) <= max_dur):
                    # Segment erweitern
                    current_chunk['end'] = row['end_sec']
                    current_chunk['text'].append(row['text'])
                else:
                    # Aktuelles Segment speichern
                    merged_data.append({
                        'startzeit': format_time(current_chunk['start']),
                        'endzeit': format_time(current_chunk['end']),
                        'text': ' '.join(current_chunk['text'])
                    })
                    # Neues Segment beginnen
                    current_chunk = {
                        'start': row['start_sec'],
                        'end': row['end_sec'],
                        'text': [row['text']]
                    }

        # Letztes Segment hinzufügen
        if current_chunk:
            merged_data.append({
                'startzeit': format_time(current_chunk['start']),
                'endzeit': format_time(current_chunk['end']),
                'text': ' '.join(current_chunk['text'])
            })

        # Nachbearbeitung: Segmente unter Mindestdauer
        final_data = []
        for item in merged_data:
            duration = parse_time(item['endzeit']) - parse_time(item['startzeit'])
            if duration >= min_dur:
                final_data.append(item)
            else:
                print(f"Segment {item['startzeit']}-{item['endzeit']} ({duration}s) unter Mindestdauer")

        # Ergebnis speichern
        result_df = pd.DataFrame(final_data)
        result_df.to_csv(output_file, sep='|', index=False)
        print(f"Ergebnis mit {len(result_df)} Segmenten in {output_file} gespeichert")
        
    except Exception as e:
        print(f"Kritischer Fehler: {str(e)}")
        raise

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

def translate_segments(transcription_file, translation_file, source_lang="en", target_lang="de"):
    """Übersetzt die bereits transkribierten Segmente mithilfe von MarianMT."""
    if os.path.exists(translation_file):
        if not ask_overwrite(translation_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {translation_file}", exc_info=True)
            return read_translated_csv(translation_file)
#                return json.load(file)
            
    try:
        print(f"--------------------------")
        print(f"|<< Starte Übersetzung >>|")
        print(f"--------------------------")
        # 1) Lese die CSV-Transkription ein
        segments = []
        with open(transcription_file, mode='r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter='|')
            next(csv_reader)  # Header überspringen
            for row in csv_reader:
                if len(row) == 3:
                    start = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[0].split(':'))))
                    end = sum(float(x) * 60 ** i for i, x in enumerate(reversed(row[1].split(':'))))
                    segments.append({
                        "start": start,
                        "end": end,
                        "text": row[2]
                    })
        if not segments:
            logger.error("Keine Segmente gefunden!")
            return []
        model_name = (f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
#        model_name = "t5-large"
        logger.info(f"Lade Übersetzungsmodell: {model_name}", exc_info=True)
        tokenizer = MarianTokenizer.from_pretrained(model_name)
#        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = get_translate_model()
#        translation_prefix = "translate English to German: "
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        translated_segments = []
        for segment in segments:
#            input_text = translation_prefix + segment["text"]
            input_text = segment["text"]
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
            #   inputs = {k: v.to(device) for k, v in inputs.items()}  # Verschiebe alle Eingaben auf das Gerät
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                num_beams=7,
                repetition_penalty=1.3,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
#                return_dict=True, 
                temperature=0.2,
                no_repeat_ngram_size=3,
#                return_dict_in_generate=True,
#                output_scores=True,
                min_length=5,
                max_length=60
            )
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translated_text = clean_translation(translated_text)  # Bereinigung anwenden!
            
            translated_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text 
            })
            with open(translation_file, mode='w', encoding='utf-8', newline='') as csvfile:
                csv_writer = csv.writer(csvfile, delimiter='|')
                csv_writer.writerow(['Startpunkt Zeitstempel', 'Endpunkt Zeitstempel', 'Text'])  # Header
                for seg in translated_segments:
                    start = str(timedelta(seconds=seg["start"])).split('.')[0]
                    ende = str(timedelta(seconds=seg["end"])).split('.')[0]
                    csv_writer.writerow([start, ende, seg["text"]])
            logger.info("Übersetzung abgeschlossen!")            
        print(f"-----------------------------------")
        print(f"|<< Übersetzung abgeschlossen!! >>|")
        print(f"-----------------------------------")
        return translated_segments
    except Exception as e:
        logger.error(f"Fehler bei der Übersetzung: {e}")
    return []

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
    """Konvertiert Zeitstempel im Format HH:MM:SS in Sekunden."""
    parts = list(map(float, time_str.split(':')))
    if len(parts) == 3:  # Stunden:Minuten:Sekunden
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # Minuten:Sekunden
        return parts[0] * 60 + parts[1]
    else:  # Nur Sekunden
        return parts[0]

def check_and_replace_last_char(file_path):
    # CSV-Datei lesen
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    print("------------------------")
    print("|<< Zeilen aufräumen >>|")
    print("------------------------")
    # Zeilen modifizieren, wenn nötig
    modified_lines = []
    for line in lines:
        line = line.strip()  # Zeilenendenzeichen entfernen
        
        # Klammern und Anführungszeichen entfernen
        line = line.replace("(", "").replace(")", "").replace('"', '').replace('"', '')
        
        # Bindestriche entfernen, wenn ein Leerzeichen vorhanden ist
        line = line.replace(" - ", " ").replace("–", " ")
        
        # Alle "-" entfernen, die nicht von Leerzeichen umgeben sind
        line = ''.join([c if c != '-' or (c == '-' and (i > 0 and line[i-1].isspace()) and (i < len(line) - 1 and line[i+1].isspace())) else ' ' for i, c in enumerate(line)])
        
        # Zeile in zwei Teile teilen: Zeitangaben und Rest
        prefix = line[:16]
        rest_of_line = line[16:]
        
        # Sicherstellen, dass ab dem 16. Zeichen ein Buchstabe am Anfang steht
        while rest_of_line and not rest_of_line[0].isalnum():
            rest_of_line = rest_of_line[1:]  # Erstes Zeichen entfernen, wenn es kein Buchstabe ist
        
        if rest_of_line:  # Stelle sicher, dass der Rest der Zeile nicht leer ist
            last_char = rest_of_line[-1]  # Letzten Charakter der Zeile holen
            
            # Vermeide mehrere Punkte hintereinander
            if last_char == '.':
                rest_of_line = rest_of_line[:-1]
            
            if last_char.isalnum():  # Wenn der letzte Charakter ein Buchstabe ist
                rest_of_line += '.'  # Einen Punkt ans Ende hängen
            elif last_char in [';', ':', '-']:  # Wenn es ein Komma oder ähnliches Satzzeichen ist
                rest_of_line = rest_of_line[:-1] + '.'  # Letzten Charakter durch einen Punkt ersetzen
            elif last_char not in [' ', ',', '.', '?', '!']:  # Wenn es ein anderes Satzzeichen ist, aber nicht ?, ! oder .
                rest_of_line = rest_of_line[:-1] + '.'  # Letzten Charakter durch einen Punkt ersetzen
            
            # Vermeide Leerzeichen vor Punkten
            rest_of_line = rest_of_line.replace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(',', ',')
        
        # Füge die Zeitangaben und den modifizierten Text wieder zusammen
        modified_lines.append(prefix + rest_of_line + '\n')
    
    # Modifizierte Zeilen zurück in die Datei schreiben
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)

def text_to_speech_with_voice_cloning(translation_file, sample_path_1, sample_path_2, sample_path_3, output_path):
    """Führt die Umwandlung von Text zu Sprache durch (TTS), inklusive Stimmenklonung basierend auf sample.wav."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"TTS-Audio bereits vorhanden: {output_path}", exc_info=True)
            return
    try:
        print(f"------------------")
        print(f"|<< Starte TTS >>|")
        print(f"------------------")
        logger.info(f"Lade TTS-Modell ...", exc_info=True)
        #tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=True).to(device)
        #tts = TTS(model_name="tts_models/de/thorsten/vits", progress_bar=True).to(device)
        final_audio = np.array([], dtype=np.float32)                # Array für die kombinierten Audio-Segmente
        sampling_rate = 24000
        
        config = XttsConfig(model_param_stats=True)
        config.load_json(r"D:\AllTalk\alltalk_tts\models\xtts\v203_10_4_63\config.json")
                
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=r"D:\AllTalk\alltalk_tts\models\xtts\v203_10_4_63", checkpoint_path=r"D:\AllTalk\alltalk_tts\models\xtts\v203_10_4_63\model.pth", use_deepspeed=False)
        model.to(torch.device("cuda"))
        
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(sound_norm_refs=True, audio_path=[sample_path_1, sample_path_2, sample_path_3])
        # Lese die CSV-Datei mit den Zeitstempeln und Texten
        with open(translation_file, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            next(reader)  # Header überspringen
            for row in reader:
                if len(row) < 3:
                    continue
                # Extrahiere Startzeit, Endzeit und Text
                start = convert_time_to_seconds(row[0])
                end = convert_time_to_seconds(row[1])
                text = row[2].strip("-:")
                text = row[2].strip()
#                text = text.strip()

                print(f"🔍 Bearbeite Segment mit: {start}-{end}s mit Text:\n{text}\n")

                try:
#                    audio_clip = audio_clip[wav]
                    result = model.inference(
                        gpt_cond_latent=gpt_cond_latent,
                        speaker_embedding=speaker_embedding,
                        text=text,                          # Übersetzter Text
                        language="de",
#                        speaker_wav = sample_path_1,          # Stimmenklon-Sample #1
                        num_beams = 2,
                        speed = 0.9,                          # Sprechgeschwindigkeit
                        temperature = 0.7,
                        length_penalty = 1.1,
                        repetition_penalty = 10.0,
#                        do_sample=True,
                        enable_text_splitting=True,
                        top_k=55,
                        top_p=0.90,
                    )

                    # Extrahiere das Audio aus dem Dictionary
                    if isinstance(result, dict):
                        audio_clip = result.get("wav", None)
                        if not isinstance(audio_clip, (np.ndarray, torch.Tensor)):
                            print(f"⚠️ Warnung: `inference()` hat kein Audio für Text: {text} generiert.")
                            audio_clip = np.zeros(1000, dtype=np.float32)  # Ersetze mit Stille
                    audio_clip = np.array(audio_clip, dtype=np.float32)
                    audio_clip = np.squeeze(audio_clip) if audio_clip.ndim > 1 else audio_clip      # In float32 konvertieren
                    
                    if audio_clip.size == 0:
                        print(f"⚠️ Warnung: `audio_clip` ist leer. Ersetze mit Stille.")
                        audio_clip = np.zeros(1000, dtype=np.float32)
                    
#                    peak_val = np.max(np.abs(audio_clip)) if np.any(audio_clip) else 1.0
                    peak_val = np.max(np.abs(audio_clip)) + 1e-8
                    audio_clip /= peak_val
                    
#                    audio_clip = audio_clip.astype(np.float32)
#                    audio_clip /= peak_val  # Abschließende Normalisierung
                    current_length = len(final_audio) / sampling_rate                           # Aktuelle Länge des Audios
                    silence_duration = max(0.0, start - current_length)                         # Stille vor dem Segment   
                    silence_samples = int(silence_duration * sampling_rate)                    # Stille in Samples
                    silence_segment = np.zeros(silence_samples, dtype=np.float32)
                    final_audio = np.concatenate([final_audio, silence_segment, audio_clip])                # Alle Segmente zusammenfügen
                except Exception as segment_error:
                    logger.error(f"Fehler in Segment {start}-{end}s: {segment_error}", exc_info=True)
                    continue
                                    # Speichere das finale Audio  

        if len(final_audio) == 0:
            print("Kein Audio - Datei leer!")
            final_audio = np.zeros((1, 1000), dtype=np.float32)

#        final_audio = apply_denoising(final_audio, sampling_rate)                       # Rauschfilter anwenden
        final_audio = final_audio.astype(np.float32)                                    # In float32 konvertieren
        
        if final_audio.ndim == 1:
            final_audio = final_audio.reshape(1, -1)  # [1, samples] statt [samples]
        
        torchaudio.save(output_path, torch.from_numpy(final_audio), sampling_rate)
        
        print(f"---------------------------")
        print(f"|<< TTS abgeschlossen!! >>|")
        print(f"---------------------------")
        
        logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}", exc_info=True)
    except Exception as e:
        logger.error(f"Fehler: {str(e)}", exc_info=True)
        raise

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
        # Lade Audio in mono (falls im Original), dann dupliziere ggf. auf 2 Kanäle
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        audio = np.vstack([audio, audio])             # Duplicate mono channel to create stereo
        audio = np.vstack([audio, audio])             # Duplicate mono channel to create stereo
        logger.info(f"Original-Samplingrate: {sr} Hz", exc_info=True)

        target_sr = 44100

        # Resample auf 44.100 Hz
        if sr != target_sr:
            audio_resampled = np.vstack([
                librosa.resample(audio[channel], res_type="kaiser_best", orig_sr=sr, target_sr=target_sr, scale=True, fix=True)
                for channel in range(audio.shape[0])
            ])
        else:
            audio_resampled = audio

        # Wiedergabegeschwindigkeit anpassen
        if speed_factor != 1.0:
            stretched_channels = [
                librosa.effects.time_stretch(audio_resampled[channel], rate=speed_factor)
                for channel in range(audio_resampled.shape[0])
            ]
            audio_stretched = np.vstack(stretched_channels)
        else:
            audio_stretched = audio_resampled

        # Wandle Daten in np.int16 um und speichere als WAV mit PCM-16
        audio_int16 = (audio_stretched * 32767).astype(np.int16)  # Convert to int16 range

        # Hier format und subtype explizit angeben:
        sf.write(
            output_path,
            audio_int16.T,  # Transpose to match (samples, channels) format
            samplerate=target_sr,
            format="WAV",
            subtype="PCM_16"
        )
        logger.info("\n".join([
            f"Audio auf {target_sr} Hz (Stereo) resampled:",
            f"- Geschwindigkeitsfaktor: {speed_factor}",
            f"- Lautstärkeanpassung: {VOLUME_ADJUSTMENT_44100}",
            f"- Datei: {output_path}"
            ]), exc_info=True)
        print("--------------------------")
        print("|<< ReSampling beendet >>|")
        print("--------------------------")
    except Exception as e:
        logger.error(f"Fehler beim Resampling auf 44.100 Hz: {e}")

def adjust_playback_speed(video_path, adjusted_video_path, speed_factor):
    """Passt die Wiedergabegeschwindigkeit des Originalvideos an und nutzt einen separaten Lautstärkefaktor für das Video."""
    if os.path.exists(adjusted_video_path):
        if not ask_overwrite(adjusted_video_path):
            logger.info(f"Verwende vorhandene Datei: {adjusted_video_path}", exc_info=True)
            return
    try:
        video_speed = 1 / speed_factor
        audio_speed = speed_factor
#        ffmpeg.input(video_path, **cuda_options).output(
#        ffmpeg.input(video_path, **cuda_options).output(
        ffmpeg.input(video_path).output(
            adjusted_video_path,
            vf=f"setpts={video_speed}*PTS",
                        af=f"atempo={audio_speed}"
        ).run(overwrite_output=True)
        logger.info(
            f"Videogeschwindigkeit angepasst (Faktor={speed_factor}): {adjusted_video_path} ",
            exc_info=True
            #f"und Lautstärke={VOLUME_ADJUSTMENT_VIDEO}"
        )
    except ffmpeg.Error as e:
        logger.error(f"Fehler bei der Anpassung der Wiedergabegeschwindigkeit: {e}")

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
        filter_complex = (
            f"[0:a]volume={VOLUME_ADJUSTMENT_VIDEO}[a1];"  # Reduziere die Lautstärke des Originalvideos
            f"[1:a]volume={VOLUME_ADJUSTMENT_44100}[a2];"  # Halte die Lautstärke des TTS-Audios konstant
            "[a1][a2]amix=inputs=2:duration=longest"
        )
#        video_input = ffmpeg.input(adjusted_video_path, **cuda_options)
#        video_input = ffmpeg.input(adjusted_video_path, **cuda_options)
        video_input = ffmpeg.input(adjusted_video_path)
        audio_input = ffmpeg.input(translated_audio_path)

        ffmpeg.output(
            video_input.video,
            audio_input.audio,
            final_video_path,
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            filter_complex=filter_complex,
            map="0:v",
            map_metadata="-1"
        ).overwrite_output().run()
        logger.info(f"Finales Video erstellt und gemischt: {final_video_path}", exc_info=True)
    except ffmpeg.Error as e:
        logger.error(f"Fehler beim Kombinieren von Video und Audio: {e}")

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
    process_audio(ORIGINAL_AUDIO_PATH, PROCESSED_AUDIO_PATH)

    # 4) Audio resamplen auf 16 kHz, Mono (für TTS)
    resample_to_16000_mono(PROCESSED_AUDIO_PATH, PROCESSED_AUDIO_PATH_SPEED, SPEED_FACTOR_RESAMPLE_16000)

    # 4.1) Spracherkennung (VAD) mit Silero VAD
#    detect_speech(PROCESSED_AUDIO_PATH_SPEED, ONLY_SPEECH)
    
    # 5) Optional: Erstellung eines Voice-Samples für die Stimmenklonung
    create_voice_sample(ORIGINAL_AUDIO_PATH, SAMPLE_PATH_1, SAMPLE_PATH_2, SAMPLE_PATH_3)

    # 6) Spracherkennung (Transkription) mit Whisper
    segments = transcribe_audio_with_timestamps(PROCESSED_AUDIO_PATH, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return

    # 6.1) Zusammenführen von Transkript-Segmenten
    merge_transcript_chunks(
        input_file=TRANSCRIPTION_FILE,
        output_file=MERGED_TRANSCRIPTION_FILE,
        min_dur=1,
        max_dur=15,
        max_gap=5
    )

    # 7) Übersetzung der Segmente mithilfe von MarianMT
    translated = translate_segments(MERGED_TRANSCRIPTION_FILE, TRANSLATION_FILE)
    if not translated:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden.")
        return

    check_and_replace_last_char(TRANSLATION_FILE)
    # 8) Text-to-Speech (TTS) mit Stimmenklonung
    text_to_speech_with_voice_cloning(TRANSLATION_FILE, SAMPLE_PATH_1, SAMPLE_PATH_2, SAMPLE_PATH_3, TRANSLATED_AUDIO_WITH_PAUSES)

    # 9) Audio resamplen auf 44.1 kHz, Stereo (für Mixdown), inkl. separatem Lautstärke- und Geschwindigkeitsfaktor
    resample_to_44100_stereo(TRANSLATED_AUDIO_WITH_PAUSES, RESAMPLED_AUDIO_FOR_MIXDOWN, SPEED_FACTOR_RESAMPLE_44100)

    # 10) Wiedergabegeschwindigkeit des Videos anpassen (separater Lautstärkefaktor für Video)
    adjust_playback_speed(VIDEO_PATH, ADJUSTED_VIDEO_PATH, SPEED_FACTOR_PLAYBACK)

    # 11) Kombination von angepasstem Video und übersetztem Audio
    combine_video_audio_ffmpeg(ADJUSTED_VIDEO_PATH, RESAMPLED_AUDIO_FOR_MIXDOWN, FINAL_VIDEO_PATH)

    total_time = time.time() - start_time
    print(f"\nGesamtprozessdauer: {(total_time / 60):.2f} Minuten -> {(total_time / 60 / 60):.2f} Stunden")

    logger.info(f"Projekt abgeschlossen! Finale Ausgabedatei: {FINAL_VIDEO_PATH}", exc_info=True)

if __name__ == "__main__":
    main()