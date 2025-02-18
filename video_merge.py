from moviepy import VideoFileClip, concatenate_videoclips
import os

input_videos = [
    "its Worse than You think full documentary_000_deutsch.mp4",
    "its Worse than You think full documentary_001_deutsch.mp4",
    "its Worse than You think full documentary_002_deutsch.mp4",
    "its Worse than You think full documentary_003_deutsch.mp4",
    "its Worse than You think full documentary_004_deutsch.mp4",
    "its Worse than You think full documentary_005_deutsch.mp4"
]
output_video = "its Worse than You think full documentary_deutsch.mp4"

def merge_mp4_files(input_files, output_file):
    """
    Führt mehrere MP4-Dateien zu einer einzigen Datei zusammen.

    :param input_files: Liste der Eingabe-MP4-Dateien (mit vollständigen Pfaden).
    :param output_file: Pfad zur Ausgabedatei (MP4).
    """
    # Überprüfe, ob alle Dateien existieren
    for file in input_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Datei nicht gefunden: {file}")
    
    try:
        print("Lade Videodateien...")

        # Videoclips laden
        clips = [VideoFileClip(file) for file in input_files]

        # Videos zusammenfügen
        final_clip = concatenate_videoclips(clips, method="compose")

        print(f"Exportiere zusammengeführtes Video nach: {output_file}")

        # Video exportieren
        final_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        # Ressourcen freigeben
        for clip in clips:
            clip.close()
        final_clip.close()

        print("Video erfolgreich zusammengeführt!")
    
    except Exception as e:
        print(f"Fehler beim Zusammenführen der Videos: {e}")

merge_mp4_files(input_videos, output_video)