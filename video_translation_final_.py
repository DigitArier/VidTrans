import os
import ffmpeg
import logging
import librosa
import soundfile as sf
import numpy as np
import json
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import whisper
from pydub import AudioSegment
from pyloudnorm import Meter, normalize

# ============================== 
# Globale Konfigurationen und Logging
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Geschwindigkeitseinstellungen
SPEED_FACTOR_RESAMPLE_44100 = 1.25  # Geschwindigkeitsfaktor für 44.100 Hz (Stereo)
SPEED_FACTOR_PLAYBACK = 0.8     # Geschwindigkeitsfaktor für die Wiedergabe des Videos

# Lautstärkeanpassungen
VOLUME_ADJUSTMENT_44100 = 1.5   # Lautstärkefaktor für 44.100 Hz (Stereo)
VOLUME_ADJUSTMENT_VIDEO = 0.5   # Lautstärkefaktor für das Video

# Dateipfade
VIDEO_PATH = "40 Bible verses that prove the True Shape of the Earth_HD.mp4"
ORIGINAL_AUDIO_PATH = "original_audio.wav"
PROCESSED_AUDIO_PATH = "processed_audio.wav"
SAMPLE_PATH = "sample.wav"
DOWNSAMPLED_AUDIO_PATH = "downsampled_audio.wav"
TRANSCRIPTION_FILE = "transcription.json"
TRANSLATION_FILE = "translation.json"
TRANSLATED_AUDIO_WITH_PAUSES = "translated_audio_with_pauses.wav"
RESAMPLED_AUDIO_FOR_MIXDOWN = "resampled_audio_44100.wav"
ADJUSTED_VIDEO_PATH = "adjusted_video.mp4"
FINAL_VIDEO_PATH = "40 Bible verses that prove the True Shape of the Earth_HD_deutsch.mp4"

# ============================== 
# Hilfsfunktionen
# ==============================
def ask_overwrite(file_path):
    """Fragt den Benutzer, ob eine bestehende Datei überschrieben werden soll."""
    while True:
        choice = input(f"Die Datei '{file_path}' existiert bereits. Überschreiben? (j/n): ").strip().lower()
        if choice in ["j", "ja"]:
            return True
        elif choice in ["n", "nein"]:
            return False

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
            ar="44100"  # 44.100 Hz
        ).run()
        logger.info(f"Audio extrahiert: {audio_output}")
    except ffmpeg.Error as e:
        logger.error(f"Fehler bei der Audioextraktion: {e}")

def process_audio(input_path, output_path):
    """Führt Rauschunterdrückung und Lautheitsnormalisierung durch."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}")
            return
    try:
        # Lade das Audio
        audio = AudioSegment.from_wav(input_path)
        
        # Rauschunterdrückung (vereinfachtes Beispiel)
        noise_reduced = audio.compress_dynamic_range(threshold=-20, ratio=4.0, attack=5, release=50)
        
        # Konvertiere zu numpy array für pyloudnorm
        samples = np.array(noise_reduced.get_array_of_samples()).astype(np.int16)
        samples = samples / np.iinfo(samples.dtype).max
        
        # Lautheitsnormalisierung nach EBU R128
        meter = Meter(44100)  # Annahme: 44.1 kHz Samplerate
        loudness = meter.integrated_loudness(samples)
        normalized_audio = normalize.loudness(samples, loudness, -23.0)
        
        # Speichere das verarbeitete Audio
        sf.write(output_path, normalized_audio, 44100)
        logger.info(f"Audio verarbeitet und gespeichert: {output_path}")
    except Exception as e:
        logger.error(f"Fehler bei der Audioverarbeitung: {e}")

def create_voice_sample(audio_path, sample_path):
    """Erstellt ein Voice-Sample aus dem verarbeiteten Audio für Stimmenklonung."""
    if os.path.exists(sample_path):
        choice = input("Eine sample.wav existiert bereits. Möchten Sie eine neue erstellen? (j/n, ENTER zum Überspringen): ").strip().lower()
        if choice == "" or choice in ["n", "nein"]:
            logger.info("Verwende vorhandene sample.wav.")
            return

    while True:
        start_time = input("Startzeit für das Sample (in Sekunden): ")
        end_time = input("Endzeit für das Sample (in Sekunden): ")
        
        if start_time == "" or end_time == "":
            logger.info("Erstellung der sample.wav übersprungen.")
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
            
            logger.info(f"Voice sample erstellt: {sample_path}")
            break
        except ValueError:
            logger.error("Ungültige Eingabe. Bitte gültige Zahlen eintragen.")
        except ffmpeg.Error as e:
            logger.error(f"Fehler beim Erstellen des Voice Samples: {e}")

def resample_to_22050_mono(input_path, output_path):
    """Resample das Audio auf 22.050 Hz (Mono)."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}")
            return
    try:
        audio, sr = librosa.load(input_path, sr=None)
        logger.info(f"Original-Samplingrate: {sr} Hz")
        target_sr = 22050
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio
        sf.write(output_path, audio_resampled, samplerate=target_sr)
        logger.info(f"Audio erfolgreich auf {target_sr} Hz (Mono) resampled: {output_path}")
    except Exception as e:
        logger.error(f"Fehler beim Resampling auf 22.050 Hz: {e}")

def transcribe_audio_with_timestamps(audio_file, transcription_file):
    """Führt eine Spracherkennung mit Whisper durch und speichert die Transkription (inkl. Zeitstempel) in einer JSON-Datei."""
    if os.path.exists(transcription_file):
        if not ask_overwrite(transcription_file):
            logger.info(f"Verwende vorhandene Transkription: {transcription_file}")
            with open(transcription_file, "r", encoding="utf-8") as file:
                return json.load(file)
    try:
        logger.info("Lade Whisper-Modell (large-v2)...")
        model = whisper.load_model("large-v2")
        logger.info("Starte Transkription...")
        result = model.transcribe(audio_file, verbose=True, language="en")
        segments = result.get("segments", [])
        with open(transcription_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)
        logger.info("Transkription abgeschlossen!")
        return segments
    except Exception as e:
        logger.error(f"Fehler bei der Transkription: {e}")
        return []

def translate_segments(segments, translation_file, source_lang="en", target_lang="de"):
    """Übersetzt die bereits transkribierten Segmente mithilfe von MarianMT."""
    if os.path.exists(translation_file):
        if not ask_overwrite(translation_file):
            logger.info(f"Verwende vorhandene Übersetzungen: {translation_file}")
            with open(translation_file, "r", encoding="utf-8") as file:
                return json.load(file)
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        logger.info(f"Lade Übersetzungsmodell: {model_name}")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        for segment in segments:
            text = segment["text"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            segment["translated_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        with open(translation_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)
        logger.info("Übersetzung abgeschlossen!")
        return segments
    except Exception as e:
        logger.error(f"Fehler bei der Übersetzung: {e}")
        return []

def text_to_speech_with_voice_cloning(segments, sample_path, output_path, speed=1.0):
    """Führt die Umwandlung von Text zu Sprache durch (TTS), inklusive Stimmenklonung basierend auf sample.wav."""
    if os.path.exists(output_path):
        logger.info(f"TTS-Audio bereits vorhanden: {output_path}")
        return
    try:
        logger.info("Lade TTS-Modell (multilingual/multi-dataset/xtts_v2)...")
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
        sampling_rate = tts.synthesizer.output_sample_rate
        final_audio_segments = []
        for segment in segments:
            translated_text = segment.get("translated_text", "")
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time
            logger.info(f"Erzeuge Audio für Segment ({start_time:.2f}-{end_time:.2f}s): {translated_text}")
            audio_clip = tts.tts(
                text=translated_text,
                speaker_wav=sample_path,
                emotion="Neutral",
                language="de",
                speed=speed
            )
            clip_length = len(audio_clip) / sampling_rate
            pause_duration = max(0, duration - clip_length)
            silence = np.zeros(int(pause_duration * sampling_rate))
            final_audio_segments.append(audio_clip)
            final_audio_segments.append(silence)
        final_audio = np.concatenate(final_audio_segments)
        sf.write(output_path, final_audio, samplerate=sampling_rate)
        logger.info(f"TTS-Audio mit geklonter Stimme erstellt: {output_path}")
    except Exception as e:
        logger.error(f"Fehler bei Text-to-Speech mit Stimmenklonen: {e}")

def resample_to_44100_stereo(input_path, output_path, speed_factor):
    """Resample das Audio auf 44.100 Hz (Stereo), passe die Wiedergabegeschwindigkeit sowie die Lautstärke an."""
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            logger.info(f"Verwende vorhandene Datei: {output_path}")
            return
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        logger.info(f"Original-Samplingrate: {sr} Hz")
        target_sr = 44100
        if sr != target_sr:
            if len(audio.shape) == 1:
                audio = np.vstack([audio, audio])
                logger.info("Mono-Audio wurde auf 2 Kanäle dupliziert.")
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=1)
        else:
            audio_resampled = audio
        if speed_factor != 1.0:
            if len(audio_resampled.shape) == 2:
                stretched_channels = []
                for channel_data in audio_resampled:
                    stretched_channels.append(librosa.effects.time_stretch(channel_data, rate=speed_factor))
                audio_stretched = np.vstack(stretched_channels)
            else:
                audio_stretched = librosa.effects.time_stretch(audio_resampled, rate=speed_factor)
        else:
            audio_stretched = audio_resampled
        audio_adjusted = audio_stretched * VOLUME_ADJUSTMENT_44100
        sf.write(output_path, audio_adjusted, samplerate=target_sr)
        logger.info(
            f"Audio auf {target_sr} Hz (Stereo) resampled, Geschwindigkeitsfaktor={speed_factor}, "
            f"Lautstärkeanpassung={VOLUME_ADJUSTMENT_44100}. Datei: {output_path}"
        )
    except Exception as e:
        logger.error(f"Fehler beim Resampling auf 44.100 Hz: {e}")

def adjust_playback_speed(video_path, adjusted_video_path, speed_factor):
    """Passt die Wiedergabegeschwindigkeit des Originalvideos an und nutzt einen separaten Lautstärkefaktor für das Video."""
    if os.path.exists(adjusted_video_path):
        if not ask_overwrite(adjusted_video_path):
            logger.info(f"Verwende vorhandene Datei: {adjusted_video_path}")
            return
    try:
        video_speed = 1 / speed_factor
        audio_speed = speed_factor
        ffmpeg.input(video_path).output(
            adjusted_video_path,
            vf=f"setpts={video_speed}*PTS",
                        af=f"atempo={audio_speed},volume={VOLUME_ADJUSTMENT_VIDEO}"
        ).run(overwrite_output=True)
        logger.info(
            f"Videogeschwindigkeit angepasst (Faktor={speed_factor}) "
            f"und Lautstärke={VOLUME_ADJUSTMENT_VIDEO}: {adjusted_video_path}"
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
            logger.info(f"Verwende vorhandene Datei: {final_video_path}")
            return
    try:
        filter_complex = (
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
            map="0:v"
        ).overwrite_output().run()
        logger.info(f"Finales Video erstellt und gemischt: {final_video_path}")
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

    # 4) Optional: Erstellung eines Voice-Samples für die Stimmenklonung
    create_voice_sample(PROCESSED_AUDIO_PATH, SAMPLE_PATH)

    # 5) Audio resamplen auf 22.050 Hz, Mono (für TTS)
    resample_to_22050_mono(PROCESSED_AUDIO_PATH, DOWNSAMPLED_AUDIO_PATH)

    # 6) Spracherkennung (Transkription) mit Whisper
    segments = transcribe_audio_with_timestamps(DOWNSAMPLED_AUDIO_PATH, TRANSCRIPTION_FILE)
    if not segments:
        logger.error("Transkription fehlgeschlagen oder keine Segmente gefunden.")
        return

    # 7) Übersetzung der Segmente mithilfe von MarianMT
    translated_segments = translate_segments(segments, TRANSLATION_FILE)
    if not translated_segments:
        logger.error("Übersetzung fehlgeschlagen oder keine Segmente vorhanden.")
        return

    # 8) Text-to-Speech (TTS) mit Stimmenklonung
    text_to_speech_with_voice_cloning(translated_segments, SAMPLE_PATH, TRANSLATED_AUDIO_WITH_PAUSES, speed=1.0)

    # 9) Audio resamplen auf 44.100 Hz, Stereo (für Mixdown), inkl. separatem Lautstärke- und Geschwindigkeitsfaktor
    resample_to_44100_stereo(TRANSLATED_AUDIO_WITH_PAUSES, RESAMPLED_AUDIO_FOR_MIXDOWN, SPEED_FACTOR_RESAMPLE_44100)

    # 10) Wiedergabegeschwindigkeit des Videos anpassen (separater Lautstärkefaktor für Video)
    adjust_playback_speed(VIDEO_PATH, ADJUSTED_VIDEO_PATH, SPEED_FACTOR_PLAYBACK)

    # 11) Kombination von angepasstem Video und übersetztem Audio
    combine_video_audio_ffmpeg(ADJUSTED_VIDEO_PATH, RESAMPLED_AUDIO_FOR_MIXDOWN, FINAL_VIDEO_PATH)

    logger.info(f"Projekt abgeschlossen! Finale Ausgabedatei: {FINAL_VIDEO_PATH}")

if __name__ == "__main__":
    main()
