import os
import subprocess
import ffmpeg
import logging
import librosa
import soundfile as sf
import numpy as np
import json
import scipy.signal as signal
import torch
torch.set_num_threads(1)
import torchaudio
import pyrubberband
import time
from datetime import datetime, timedelta
import csv
from config import *
from tqdm import tqdm
from contextlib import contextmanager
from transformers import AutoTokenizer, MarianMTModel
from TTS.api import TTS
import whisper
from pydub import AudioSegment
from pydub.effects import high_pass_filter, low_pass_filter, normalize
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio, collect_chunks


# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.0   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.0   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 1.0        # Geschwindigkeitsfaktor für die Wiedergabe des Videos

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
        return result, end - start

def ask_overwrite(file_path):
    """Fragt den Benutzer, ob eine bestehende Datei überschrieben werden soll."""
    while True:
        choice = input(f"Die Datei '{file_path}' existiert bereits. Überschreiben? (j/n): ").strip().lower()
        if choice in ["j", "ja"]:
            return True
        elif choice in ["n", "nein"]:
            return False

step_start_time = time.time()
def get_whisper_model():
    global _WHISPER_MODEL
    if not _WHISPER_MODEL:
        _WHISPER_MODEL = whisper.load_model("large-v3-turbo")
        _WHISPER_MODEL.to(torch.device("cuda"))
        torch.cuda.empty_cache()
    return _WHISPER_MODEL

def get_translate_model():
    global _TRANSLATE_MODEL
    if not _TRANSLATE_MODEL:
        _TRANSLATE_MODEL = MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
        _TRANSLATE_MODEL.to(torch.device("cuda"))
        torch.cuda.empty_cache()
    return _TRANSLATE_MODEL

def get_tts_model():
    global _TTS_MODEL
    if not _TTS_MODEL:
        _TTS_MODEL = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2",   # XTTS-Modell
                    progress_bar=True,                                                  # Fortschrittsbalken
                    vocoder_path=vocoder_pth,                                           # Vocoder-Modell
                    vocoder_config_path=vocoder_cfg                                     # Vocoder-Konfiguration
                    )
        _TTS_MODEL.to(torch.device("cuda"))
        torch.cuda.empty_cache()
    return _TTS_MODEL

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
        
    # Helper function to save and log intermediate steps
    def save_step(audio_segment, filename):
        audio_segment.export(filename, format="wav")
        logger.info(f"Zwischenschritt gespeichert: {filename} - Größe: {os.path.getsize(filename)} Bytes", exc_info=True)
    
    # 1. Load the audio file
    audio = AudioSegment.from_wav(input_file)
    #save_step(audio, "process_original.wav")
    
    # 2. High-Pass Filter für klare Stimme (z.B. 80-100 Hz)
    #    Filtert tieffrequentes Dröhnen/Brummen heraus [1][2].
    audio_hp = high_pass_filter(audio, cutoff=100)
    save_step(audio_hp, "process_high_pass.wav")
    
    # 3. Noise Gate, um Atem und Hintergrundrauschen zu unterdrücken
    #    Threshold je nach Sprechpegel, z.B. -48 dB [2][7].
    def noise_gate(audio_segment, threshold_db=-20):
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
        
        def compress(signal_band, threshold=0.1, ratio=4.0):
            # Soft-Knee-ähnliche Funktion
            compressed = np.where(
                np.abs(signal_band) > threshold,
                threshold + (signal_band - threshold) / ratio,
                signal_band
            )
            return compressed
        
        low_band_comp = compress(low_band, threshold=0.1, ratio=3.0)
        mid_band_comp = compress(mid_band, threshold=0.1, ratio=4.0)
        high_band_comp = compress(high_band, threshold=0.1, ratio=4.0)
        
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
    audio_eq = high_pass_filter(audio_comp, cutoff=100)   # tiefe Frequenzen raus
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
            min_silence_duration_ms=50,         # Minimale Stille-Dauer
            speech_pad_ms=100,                   # Padding für Sprachsegmente
            return_seconds=True,                # Rückgabe in Sekunden
            threshold=0.4                       # Schwellenwert für Sprachaktivität
        )
        # Überprüfe, ob Sprachaktivität gefunden wurde
        if not speech_timestamps:
            logger.warning("Keine Sprachaktivität gefunden. Das Audio enthält möglicherweise nur Stille oder Rauschen.")
            return []
        
        # Setze die sprachaktiven Abschnitte in das leere Audio-Array
        prev_end = 0                                        # Startzeit des vorherigen Segments
        max_silence_samples = int(2.0 * sampling_rate)      # Maximal 2 Sekunden Stille

        for segment in speech_timestamps:
            start = int(segment['start'] * sampling_rate - 1.0)   # Sekunden in Samples umrechnen
            end = int(segment['end'] * sampling_rate + 1.0)       # Sekunden in Samples umrechnen
    
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
        
        # Nahe beieinander liegende Segmente zusammenführen
        #merged_segments = []
        #prev_end = 0
        #max_silence_gap = 500
        #if not speech_timestamps:
        #    return []
        #merged_segments = [speech_timestamps[0].copy()]  # Erstes Segment kopieren
        #prev_end = merged_segments[0]['end']
    
        #for segment in speech_timestamps[1:]:
        #    current_start = segment['start']
        #    if current_start - prev_end <= max_silence_gap:
        #        # Segmente zusammenführen
        #        merged_segments[-1]['end'] = segment['end']
        #    else:
        #        # Neues Segment hinzufügen
        #        merged_segments.append(segment.copy())
        #    prev_end = segment['end']
        #with open("merged_segments.json", "w", encoding="utf-8") as file:
        #    json.dump(merged_segments, file, ensure_ascii=False, indent=4)
        
        # Konvertiere Zeitstempel in Samples
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
        return speech_timestamps
    except Exception as e:
        logger.error(f"Fehler bei der Sprachaktivitätserkennung: {e}", exc_info=True)
        return []

def create_voice_sample(audio_path, sample_path):

    """Erstellt ein Voice-Sample aus dem verarbeiteten Audio für Stimmenklonung."""
    if os.path.exists(sample_path):
        choice = input("Eine sample.wav existiert bereits. Möchten Sie eine neue erstellen? (j/n, ENTER zum Überspringen): ").strip().lower()
        if choice == "" or choice in ["n", "nein"]:
            logger.info("Verwende vorhandene sample.wav.", exc_info=True)
            return

    while True:
        start_time = input("Startzeit für das Sample (in Sekunden): ")
        end_time = input("Endzeit für das Sample (in Sekunden): ")
        
        if start_time == "" or end_time == "":
            logger.info("Erstellung der sample.wav übersprungen.", exc_info=True)
            return
        
        try:
            start_seconds = float(start_time)
            end_seconds = float(end_time)
            duration = end_seconds - start_seconds
            
            if duration <= 0:
                logger.warning("Endzeit muss nach der Startzeit liegen.")
                continue
            
            ffmpeg.input(audio_path, ss=start_seconds, t=duration).output(
                sample_path,
                acodec='pcm_s16le',
                ac=1,
                ar=22050
            ).overwrite_output().run(capture_stdout=True, capture_stderr=True)
            
            logger.info(f"Voice sample erstellt: {sample_path}", exc_info=True)
            break
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
        model = get_whisper_model()
        logger.info("Starte Transkription...", exc_info=True)
        result = model.transcribe(
            audio_file,                         # Audio-Datei
            condition_on_previous_text=True,    # Bedingung an vorherigen Text
            verbose=True,                       # Ausführliche Ausgabe
            language="en"                       # Englische Sprache
            )
        #segments = result["segments"]
        for segment in result["segments"]:
            segment["start"] = max(segment["start"] - 0, 0)  # Startzeit anpassen
            segment["end"] = max(segment["end"] - 2, 0)      # Endzeit anpassen
            
#        # JSON-Datei speichern
#        with open(transcription_file, "w", encoding="utf-8") as file:
#            json.dump(result["segments"], file, ensure_ascii=False, indent=4)
            
        # CSV-Datei speichern  
        transcription_file = transcription_file.replace('.json', '.csv')
        with open(transcription_file, mode='w', encoding='utf-8', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter='|')
            csv_writer.writerow(['Startpunkt Zeitstempel', 'Endpunkt Zeitstempel', 'Text'])  # Header
            
            for segment in result["segments"]:
                start = str(timedelta(seconds=segment["start"])).split('.')[0]
                ende = str(timedelta(seconds=segment["end"])).split('.')[0]
                text = segment["text"]
                csv_writer.writerow([start, ende, text])
                
        logger.info("Transkription abgeschlossen!", exc_info=True)
        return result["segments"]
    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}", exc_info=True)
        return []

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

def translate_segments(transcription_file, translation_file, source_lang="en", target_lang="de"):
    """Übersetzt die bereits transkribierten Segmente mithilfe von MarianMT."""
    if os.path.exists(translation_file):
        if not ask_overwrite(translation_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {translation_file}", exc_info=True)
            return read_translated_csv(translation_file)
#                return json.load(file)
            
    try:
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
        logger.info(f"Lade Übersetzungsmodell: {model_name}", exc_info=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = get_translate_model()
        
        translated_segments = []
        for segment in segments:
            inputs = tokenizer(segment["text"], return_tensors="pt").to(device)
            #inputs = {k: v.to(device) for k, v in inputs.items()}  # Verschiebe alle Eingaben auf das Gerät
            outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100).to(device)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

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
        sample_rate=sr, 
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

def text_to_speech_with_voice_cloning(translation_file, sample_path, output_path):
    """Führt die Umwandlung von Text zu Sprache durch (TTS), inklusive Stimmenklonung basierend auf sample.wav."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"TTS-Audio bereits vorhanden: {output_path}", exc_info=True)
            return
    try:
        logger.info("Lade TTS-Modell (multilingual/multi-dataset/xtts_v2)...", exc_info=True)
        #tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=True).to(device)
        #tts = TTS(model_name="tts_models/de/thorsten/vits", progress_bar=True).to(device)
        tts = get_tts_model()
        sampling_rate = tts.synthesizer.output_sample_rate          # 24.000 Hz
        final_audio = np.array([], dtype=np.float32)                # Array für die kombinierten Audio-Segmente
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
                text = row[2].strip()

                try:
                    audio_clip = tts.tts(   
                        text=text,                          # Übersetzter Text
                        speaker_wav=sample_path,            # Stimmenklon-Sample
                        language="de",                      # Zielsprache
                        temperature=0.2,                    # Temperatur für Sampling
                        repetition_penalty=10.0,            # Wiederholungspenalität
                        speed=1.2,                          # Sprechgeschwindigkeit
                        sound_norm_refs=True                # Normalisierung
                        )

                    audio_clip = np.squeeze(audio_clip)                                         # In float32 konvertieren
                    if audio_clip.size == 0:
                        raise ValueError("Leeres Audio-Segment nach Squeeze")
                    peak_val = np.max(np.abs(audio_clip))
                    if peak_val < 1e-6:
                        raise ValueError("Stummes Audio-Segment erkannt")
                    audio_clip = audio_clip.astype(np.float32)
                    audio_clip /= peak_val  # Abschließende Normalisierung
                    current_length = len(final_audio) / sampling_rate                           # Aktuelle Länge des Audios
                    silence_duration = max(0.0, start - current_length - 2.0)                         # Stille vor dem Segment   
                    silence_samples = int(silence_duration * sampling_rate)                    # Stille in Samples
                    silence_segment = np.zeros(silence_samples, dtype=np.float32)
                    final_audio = np.concatenate([final_audio, silence_segment, audio_clip])                # Alle Segmente zusammenfügen
            
                except Exception as segment_error:
                    logger.error(f"Fehler in Segment {start}-{end}s: {segment_error}", exc_info=True)
                    continue
        
        final_audio = apply_denoising(final_audio, sampling_rate)                       # Rauschfilter anwenden
        final_audio = final_audio.astype(np.float32)                                    # In float32 konvertieren
        final_audio = final_audio.reshape(1, -1)  # [1, samples] statt [samples]
        torchaudio.save(output_path, torch.from_numpy(final_audio), sampling_rate)
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
        # Lade Audio in mono (falls im Original), dann dupliziere ggf. auf 2 Kanäle
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        audio = np.vstack([audio, audio])             # Duplicate mono channel to create stereo
        logger.info(f"Original-Samplingrate: {sr} Hz", exc_info=True)

        target_sr = 44100

        # Resample auf 44.100 Hz
        if sr != target_sr:
            audio_resampled = np.vstack([
                librosa.resample(audio[channel], orig_sr=sr, target_sr=target_sr)
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

    # 4) Audio resamplen auf 22.050 Hz, Mono (für TTS)
    resample_to_16000_mono(PROCESSED_AUDIO_PATH, PROCESSED_AUDIO_PATH_SPEED, SPEED_FACTOR_RESAMPLE_16000)

    # 4.1) Spracherkennung (VAD) mit Silero VAD
    detect_speech(PROCESSED_AUDIO_PATH_SPEED, ONLY_SPEECH)
    
    # 5) Optional: Erstellung eines Voice-Samples für die Stimmenklonung
    create_voice_sample(ORIGINAL_AUDIO_PATH, SAMPLE_PATH)

    # 6) Spracherkennung (Transkription) mit Whisper
    segments = transcribe_audio_with_timestamps(ONLY_SPEECH, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return
    #with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
    #    json.dump(segments, f, ensure_ascii=False, indent=4)

    # 7) Übersetzung der Segmente mithilfe von MarianMT
    translated = translate_segments(TRANSCRIPTION_FILE, TRANSLATION_FILE)
    if not translated:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden.")
        return
    #with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
    #    json.dump(translated, f, ensure_ascii=False, indent=4)

    # 8) Text-to-Speech (TTS) mit Stimmenklonung
    text_to_speech_with_voice_cloning(TRANSLATION_FILE, SAMPLE_PATH, TRANSLATED_AUDIO_WITH_PAUSES)

    # 9) Audio resamplen auf 44.100 Hz, Stereo (für Mixdown), inkl. separatem Lautstärke- und Geschwindigkeitsfaktor
    resample_to_44100_stereo(TRANSLATED_AUDIO_WITH_PAUSES, RESAMPLED_AUDIO_FOR_MIXDOWN, SPEED_FACTOR_RESAMPLE_44100)

    # 10) Wiedergabegeschwindigkeit des Videos anpassen (separater Lautstärkefaktor für Video)
    adjust_playback_speed(VIDEO_PATH, ADJUSTED_VIDEO_PATH, SPEED_FACTOR_PLAYBACK)

    # 11) Kombination von angepasstem Video und übersetztem Audio
    combine_video_audio_ffmpeg(ADJUSTED_VIDEO_PATH, RESAMPLED_AUDIO_FOR_MIXDOWN, FINAL_VIDEO_PATH)

    total_time = time.time() - start_time
    print(f"\nGesamtprozessdauer: {(total_time / 60):.2f} Minuten -> {(total_time / 60 / 60):.2f} Stunden")
    print("\nZeiten für einzelne Zwischenschritte:")

    logger.info(f"Projekt abgeschlossen! Finale Ausgabedatei: {FINAL_VIDEO_PATH}", exc_info=True)

if __name__ == "__main__":
    main()
