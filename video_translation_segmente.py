import os
import ffmpeg
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import whisper
import librosa
import soundfile as sf
import numpy as np
import json

# Funktion: Benutzerabfrage für vorhandene Dateien
def ask_overwrite(file_path):
    while True:
        choice = input(f"Die Datei '{file_path}' existiert bereits. Überschreiben? (j/n): ").strip().lower()
        if choice in ["j", "ja"]:
            return True
        elif choice in ["n", "nein"]:
            return False

# Funktion: Audiospur aus Video extrahieren
def extract_audio_ffmpeg(video_path, audio_output):
    if os.path.exists(audio_output):
        if not ask_overwrite(audio_output):
            print(f"Verwende vorhandene Datei: {audio_output}")
            return
    try:
        ffmpeg.input(video_path).output(audio_output, acodec="pcm_s16le", ac=1, ar="44100").run()
        print(f"Audio extrahiert: {audio_output}")
    except ffmpeg.Error as e:
        print(f"Fehler bei der Audioextraktion: {e}")

# Funktion: Audio resamplen auf 22.050 Hz (für TTS)
def resample_to_22050_mono(input_path, output_path):
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            print(f"Verwende vorhandene Datei: {output_path}")
            return
    try:
        audio, sr = librosa.load(input_path, sr=None)
        print(f"Original-Samplingrate: {sr} Hz")
        
        # Resample auf 22.050 Hz und Mono
        target_sr = 22050
        if sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio
        
        sf.write(output_path, audio_resampled, samplerate=target_sr)
        print(f"Audio erfolgreich resampled auf {target_sr} Hz (Mono): {output_path}")
    except Exception as e:
        print(f"Fehler beim Resampling: {e}")

# Funktion: Spracherkennung mit Whisper (inklusive Zeitstempel)
def transcribe_audio_with_timestamps(audio_file, transcription_file):
    if os.path.exists(transcription_file):
        if not ask_overwrite(transcription_file):
            print(f"Transkription bereits vorhanden: {transcription_file}")
            with open(transcription_file, "r", encoding="utf-8") as file:
                return json.load(file)
    try:
        print("Lade Whisper-Modell...")
        model = whisper.load_model("large-v2")
        print("Starte Transkription...")
        result = model.transcribe(audio_file, language="en")
        segments = result.get("segments", [])
        
        with open(transcription_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)

        print("Transkription abgeschlossen!")
        return segments
    except Exception as e:
        print(f"Fehler bei der Transkription: {e}")
        return []

# Funktion: Text übersetzen (pro Segment)
def translate_segments(segments, translation_file, source_lang="en", target_lang="de"):
    if os.path.exists(translation_file):
       if not ask_overwrite(translation_file):
            print(f"Übersetzung bereits vorhanden: {translation_file}")
            with open(translation_file, "r", encoding="utf-8") as file:
                return json.load(file)
    try:
        model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        for segment in segments:
            text = segment["text"]
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            segment["translated_text"] = tokenizer.decode(outputs[0], skip_special_tokens=True)

        with open(translation_file, "w", encoding="utf-8") as file:
            json.dump(segments, file, ensure_ascii=False, indent=4)

        print("Übersetzung abgeschlossen!")
        return segments
    except Exception as e:
        print(f"Fehler bei der Übersetzung: {e}")
        return []

# Funktion: Text-to-Speech mit Pausen (basierend auf Zeitstempeln)
def text_to_speech_with_pauses(segments, tts_model_name, output_path):
    if os.path.exists(output_path):
        print(f"TTS-Audio bereits vorhanden: {output_path}")
        return
    try:
        tts = TTS(model_name=tts_model_name, progress_bar=True)
        
        sampling_rate = tts.output_sample_rate if hasattr(tts, "output_sample_rate") else 22050
        
        final_audio = []

        for segment in segments:
            translated_text = segment.get("translated_text", "")
            start_time = segment["start"]
            end_time = segment["end"]
            duration = end_time - start_time

            print(f"Erzeuge Audio für Segment ({start_time}-{end_time}s): {translated_text}")

            audio_clip = tts.tts(translated_text, speed=1.2)

            silence_duration = max(0, duration - len(audio_clip) / sampling_rate)
            silence = np.zeros(int(silence_duration * sampling_rate))

            final_audio.append(audio_clip)
            final_audio.append(silence)

        final_audio_np = np.concatenate(final_audio)

        sf.write(output_path, final_audio_np, samplerate=sampling_rate)
        
        print(f"TTS-Audio erstellt: {output_path}")

    except Exception as e:
        print(f"Fehler bei Text-to-Speech: {e}")

# Funktion: Audio resamplen auf 44.100 Hz und Stereo (für Video-Mixdown)
def resample_to_44100_stereo(input_path, output_path):
    if os.path.exists(output_path):
        if not ask_overwrite(output_path):
            print(f"Verwende vorhandene Datei: {output_path}")
            return
    try:
        audio, sr = librosa.load(input_path, sr=None, mono=False)
        
        target_sr = 44100
        if sr != target_sr:
            audio_resampled = librosa.resample(audio.T if len(audio.shape) == 2 else audio,
                                               orig_sr=sr,
                                               target_sr=target_sr)
        
            sf.write(output_path,
                     audio_resampled.T if len(audio.shape) == 2 else audio_resampled,
                     samplerate=target_sr)
        
    except Exception as e:
         print(f"Fehler beim Upsampling")

def combine_video_audio_ffmpeg(video_path, translated_audio_path, final_video_path):
    if not os.path.exists(video_path):
        print("Fehler: Eingabevideo nicht gefunden.")
        return

    if not os.path.exists(translated_audio_path):
        print("Fehler: Übersetzte Audiodatei nicht gefunden.")
        return
    if os.path.exists(final_video_path):
        if not ask_overwrite(final_video_path):
            return
    try:
        # Filter-Komplex für den Mixdown
        filter_complex = (
            "[0:a]volume=0.05[a1];"
            "[1:a]volume=2.0[a2];"
            "[a1][a2]amix=inputs=2:duration=longest"
        )

        # Input streams
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(translated_audio_path)

        # Kombinieren von Video und gemischtem Audio
        stream = ffmpeg.output(
            video_input.video,
            audio_input.audio,
            final_video_path,
            vcodec="copy",
            acodec="aac",
            strict="experimental",
            filter_complex=filter_complex,
            map='0:v',
            map_metadata="-1"
        ).overwrite_output()

        stream.run()
        
        print(f"Finales Video erstellt: {final_video_path}")
    except ffmpeg.Error as e:
        print(f"Fehler beim Kombinieren von Video und Audio: {e}")


# Hauptprogramm 
def main():
    # Pfade für Eingabe- und Ausgabedateien
    video_path = "input_video.mp4"  # Eingabevideo
    original_audio_path = "original_audio.wav"  # Extrahierte Audiospur aus dem Video
    downsampled_audio_path = "downsampled_audio.wav"  # Für TTS auf 22.050 Hz resamplte Audiodatei
    transcription_file = "transcription.json"  # JSON-Datei für die Transkription (Debugging)
    translation_file = "translation.json"  # JSON-Datei für die Übersetzung (Debugging)
    translated_audio_with_pauses = "translated_audio_with_pauses.wav"  # TTS-Audio mit Sprechpausen
    resampled_audio_for_mixdown = "resampled_audio_44100.wav"  # Für Mixdown auf 44.100 Hz resampltes Audio
    final_video_path = "final_video.mp4"  # Finale kombinierte Videoausgabe

    # Prüfen, ob das Eingabevideo existiert
    if not os.path.exists(video_path):
        print(f"Eingabevideo nicht gefunden: {video_path}")
        return

    # Schritt 1: Audiospur aus dem Video extrahieren
    extract_audio_ffmpeg(video_path, original_audio_path)

    # Schritt 2: Audio resamplen auf 22.050 Hz und Mono (für TTS)
    resample_to_22050_mono(original_audio_path, downsampled_audio_path)

    # Schritt 3: Spracherkennung (Transkription) mit Whisper
    segments = transcribe_audio_with_timestamps(downsampled_audio_path, transcription_file)

    if not segments:
        print("Transkription fehlgeschlagen.")
        return

    # Schritt 4: Übersetzung der Segmente (JSON-Datei für Debugging)
    translated_segments = translate_segments(segments, translation_file)

    # Schritt 5: Text-to-Speech (TTS) mit Pausen basierend auf Zeitstempeln
    text_to_speech_with_pauses(translated_segments, "tts_models/de/thorsten/vits", translated_audio_with_pauses)

    # Schritt 6: Audio resamplen auf 44.100 Hz und Stereo (für Mixdown)
    resample_to_44100_stereo(translated_audio_with_pauses, resampled_audio_for_mixdown)

    # Schritt 7: Video und Audio kombinieren (mit Lautstärkeanpassung und Mixdown)
    combine_video_audio_ffmpeg(video_path, resampled_audio_for_mixdown, final_video_path)

    print(f"Projekt abgeschlossen! Finale Datei: {final_video_path}")

if __name__ == "__main__":
    main()