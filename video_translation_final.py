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
import time
#import whisperx
from config import *
from pprint import pprint
#from speechbrain.inference import interfaces
from transformers import AutoTokenizer, MarianMTModel
from TTS.api import TTS
import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.effects import high_pass_filter, low_pass_filter, normalize
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, save_audio, collect_chunks
from IPython.display import Audio
from scipy.signal import butter, lfilter
#from pyloudnorm import Meter, normalize
#import webrtcvad
#import wave
#import collections

# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_16000 = 1.1   # Geschwindigkeitsfaktor für 22.050 Hz (Mono)
SPEED_FACTOR_RESAMPLE_44100 = 1.3   # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 0.85        # Geschwindigkeitsfaktor für die Wiedergabe des Videos

# Lautstärkeanpassungen
VOLUME_ADJUSTMENT_44100 = 1.0   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.05   # Lautstärkefaktor für das Video

# ============================== 
# Globale Konfigurationen und Logging
# ==============================
logging.basicConfig(filename='video_translation_final.log', format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================== 
# Hilfsfunktionen
# ==============================

start_time = time.time()
step_times = {}

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
    """Resample the audio to 24.000 Hz (Mono)."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}", exc_info=True)
            return

    try:
        # Load the audio file
        audio, sr = librosa.load(input_path, sr=None, mono=True)  # Load with original sampling rate
        logger.info(f"Original sampling rate: {sr} Hz", exc_info=True)

        # Adjust playback speed if necessary
        if speed_factor != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)

        # Resample to 16.000 Hz if needed
        target_sr = 16000
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio

        # Save the resampled audio
        sf.write(output_path, audio_resampled, samplerate=target_sr)
        logger.info(f"Audio successfully resampled to {target_sr} Hz (Mono): {output_path}", exc_info=True)

    except Exception as e:
        logger.error(f"Error during resampling to 22.050 Hz: {e}")

def detect_speech(audio_path, output_json_path):
    """Führt eine Sprachaktivitätserkennung (VAD) mit Silero VAD durch."""
    try:
        logger.info("Lade Silero VAD-Modell...")
        vad_model = load_silero_vad(onnx=USE_ONNX)
        logger.info("Starte Sprachaktivitätserkennung...")
        wav = read_audio(audio_path, SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            sampling_rate=SAMPLING_RATE,
            return_seconds=True,
            visualize_probs=True
        )
        with open("timestamps_raw.json", "w", encoding="utf-8") as file:
            json.dump(speech_timestamps, file, ensure_ascii=False, indent=4)

        # Nahe beieinander liegende Segmente zusammenführen
        merged_segments = []
        prev_end = 0
        max_silence_gap = 500
        if not speech_timestamps:
            return []
    
        merged_segments = [speech_timestamps[0].copy()]  # Erstes Segment kopieren
        prev_end = merged_segments[0]['end']
    
        for segment in speech_timestamps[1:]:
            current_start = segment['start']
            if current_start - prev_end <= max_silence_gap:
                # Segmente zusammenführen
                merged_segments[-1]['end'] = segment['end']
            else:
                # Neues Segment hinzufügen
                merged_segments.append(segment.copy())
            prev_end = segment['end']
        with open("merged_segments.json", "w", encoding="utf-8") as file:
            json.dump(merged_segments, file, ensure_ascii=False, indent=4)
        

        # Konvertiere Zeitstempel in Samples
        speech_timestamps_samples = [
            {"start": int(ts["start"] * SAMPLING_RATE), "end": int(ts["end"] * SAMPLING_RATE)}
            for ts in speech_timestamps
        ]

        # Speichere nur die Sprachsegmente als separate WAV-Datei
        save_audio('only_speech.wav', collect_chunks(speech_timestamps_samples, wav), sampling_rate=SAMPLING_RATE)
        Audio('only_speech.wav')
        
        # Konvertiere die Zeitstempel in ein serialisierbares Format
        serializable_timestamps = [
            {"start": float(ts["start"]), "end": float(ts["end"])}
            for ts in speech_timestamps
        ]
        # Schreibe die Ergebnisse in eine JSON-Datei
        with open(output_json_path, "w", encoding="utf-8") as file:
            json.dump(serializable_timestamps, file, ensure_ascii=False, indent=4)

            logger.info("Sprachaktivitätserkennung abgeschlossen!")
            return serializable_timestamps
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
            with open(transcription_file, "r", encoding="utf-8") as file:
                return json.load(file)
    try:
        logger.info("Lade Whisper-Modell (large-v3-turbo)...", exc_info=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model("large-v3-turbo").to(device)
        logger.info("Starte Transkription...", exc_info=True)
        result = model.transcribe(audio_file, verbose=True, language="en")
        #segments = result["segments"]
        for segment in result["segments"]:
            segment["start"] = max(segment["start"] - 0, 0)  # Startzeit anpassen
            segment["end"] = max(segment["end"] - 2, 0)      # Endzeit anpassen
        with open(transcription_file, "w", encoding="utf-8") as file:
            json.dump(result["segments"], file, ensure_ascii=False, indent=4)
        logger.info("Transkription abgeschlossen!", exc_info=True)
        return result["segments"]
    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}", exc_info=True)
        return []

def translate_segments(segments, translation_file, source_lang="en", target_lang="de"):
    """Übersetzt die bereits transkribierten Segmente mithilfe von MarianMT."""
    if os.path.exists(translation_file):
        if not ask_overwrite(translation_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {translation_file}", exc_info=True)
            with open(translation_file, "r", encoding="utf-8") as file:
                return json.load(file)
            
    try:
        if not segments:
            logger.error("Keine Segmente für die Übersetzung gefunden.")
            return []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = (f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}")
        logger.info(f"Lade Übersetzungsmodell: {model_name}", exc_info=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)

        for segment in segments:
            inputs = tokenizer(segment["text"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Verschiebe alle Eingaben auf das Gerät
            outputs = model.generate(**inputs, max_length=512, num_beams=8, early_stopping=True)
            segment["translated_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with open(translation_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)
            logger.info("Übersetzung abgeschlossen!")
        return segments

    except Exception as e:
        logger.error(f"Fehler bei der Übersetzung: {e}")
        return []

def text_to_speech_with_voice_cloning(segments, sample_path, output_path):
    """Führt die Umwandlung von Text zu Sprache durch (TTS), inklusive Stimmenklonung basierend auf sample.wav."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"TTS-Audio bereits vorhanden: {output_path}", exc_info=True)
            return
    try:
        logger.info("Lade TTS-Modell (multilingual/multi-dataset/xtts_v2)...", exc_info=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        sampling_rate = tts.synthesizer.output_sample_rate
        final_audio_segments = [np.array([])]
        for segment in segments:
            translated_text = segment.get("translated_text", "")
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            logger.info(f"Erzeuge Audio für Segment ({start_time:.2f}-{end_time:.2f}s): {translated_text}", exc_info=True)
            
            # Adaptive Geschwindigkeit
            base_speed = 1.2
            if len(translated_text) > 50:
                speed = min(base_speed * 1.5, 2.0)
            else:
                speed = base_speed
            # TTS mit dynamischer Geschwindigkeit
            audio_clip = tts.tts(
                text=segment["translated_text"],
                speaker_wav=sample_path,
                language="de",
                speed=speed
                )
            text_length_factor = len(translated_text) * 0.1  # Kürzere Texte = kürzere Pausen
            clip_length = len(audio_clip) / sampling_rate
            pause_duration = max(0, duration - clip_length - text_length_factor)
            #pause_duration = max(0, duration - clip_length -0.2)
            silence = np.zeros(int(pause_duration * sampling_rate))
            final_audio_segments.append(audio_clip)
            final_audio_segments.append(silence)
        final_audio = np.concatenate(final_audio_segments)
        torchaudio.save(output_path, torch.tensor(final_audio).unsqueeze(0), 24000)
        sf.write(output_path, final_audio, samplerate=sampling_rate)
        logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}", exc_info=True)
    except Exception as e:
        logger.error(f"Fehler bei Text-to-Speech mit Stimmenklonen: {e}")

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
        audio = np.vstack([audio, audio])  # Duplicate mono channel to create stereo
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

    # 3) Audio resamplen auf 22.050 Hz, Mono (für TTS)
    process_audio(ORIGINAL_AUDIO_PATH, PROCESSED_AUDIO_PATH)
    resample_to_16000_mono(PROCESSED_AUDIO_PATH, PROCESSED_AUDIO_PATH_SPEED, SPEED_FACTOR_RESAMPLE_16000)

    # 4) Audioverarbeitung (Rauschunterdrückung und Lautheitsnormalisierung)
    

    # 4.1) Spracherkennung (VAD) mit Silero VAD
    detect_speech(PROCESSED_AUDIO_PATH_SPEED, SPEECH_TIMESTAMPS)
    
    # 5) Optional: Erstellung eines Voice-Samples für die Stimmenklonung
    create_voice_sample(ORIGINAL_AUDIO_PATH, SAMPLE_PATH)

    # 6) Spracherkennung (Transkription) mit Whisper
    segments = transcribe_audio_with_timestamps(SPEECH_TIMESTAMPS_WAV, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return
    with open(TRANSCRIPTION_FILE, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)

    # 7) Übersetzung der Segmente mithilfe von MarianMT
    translated_segments = translate_segments(segments, TRANSLATION_FILE)
    if not translated_segments:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden.")
        return
    with open(TRANSLATION_FILE, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=4)

    # 8) Text-to-Speech (TTS) mit Stimmenklonung
    text_to_speech_with_voice_cloning(translated_segments, SAMPLE_PATH, TRANSLATED_AUDIO_WITH_PAUSES)

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
