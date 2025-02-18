from moviepy.video.io.VideoFileClip import VideoFileClip
import os

# Globale Variable für das Input-Video
input_video = "its Worse than You think full documentary.mp4"  # <-- Hier einfach den Pfad zum Video anpassen


def split_video_by_parts(video_path, output_folder, num_parts):
    """Teilt ein MP4-Video in eine angegebene Anzahl von gleich großen Teilen mit 3 Sekunden Überlappung."""
    video = VideoFileClip(video_path)
    duration = video.duration

    # Dauer pro Teil
    part_duration = duration / num_parts
    overlap = 3  # Überlappungszeit in Sekunden

    for i in range(num_parts):
        start_time = max(0, i * part_duration - (overlap if i > 0 else 0))
        end_time = min(duration, (i + 1) * part_duration)

        output_path = os.path.join(output_folder, f"{input_video.removesuffix(".mp4")}_{i + 1}.mp4")
        print(f"Erstelle Teil {i + 1}: {start_time:.2f}s - {end_time:.2f}s")

        video.subclipped(start_time, end_time).write_videofile(output_path, codec="libx264", audio_codec="aac")

    video.close()
    print("Video erfolgreich in gleich große Teile geteilt!")


def split_video_by_length(video_path, output_folder, segment_length):
    """Teilt ein MP4-Video in Teile mit einer angegebenen Länge (in Sekunden) und 3 Sekunden Überlappung."""
    video = VideoFileClip(video_path)
    duration = video.duration
    overlap = 3  # Überlappungszeit in Sekunden

    num_parts = int(duration // (segment_length - overlap)) + 1

    for i in range(num_parts):
        start_time = max(0, i * (segment_length - overlap))
        end_time = min(duration, start_time + segment_length)

        if start_time >= duration:
            break

        output_path = os.path.join(output_folder, f"{input_video.removesuffix(".mp4")}_{i + 1}.mp4")
        print(f"Erstelle Segment {i + 1}: {start_time:.2f}s - {end_time:.2f}s")

        video.subclipped(start_time, end_time).write_videofile(output_path, codec="libx264", audio_codec="aac")

    video.close()
    print("Video erfolgreich in Segmente geteilt!")


def main():
    """Hauptfunktion zur Auswahl der Teilungsmethode."""
    if not os.path.exists(input_video):
        print(f"Die Datei {input_video} wurde nicht gefunden!")
        return

    output_folder = "video_parts"
    os.makedirs(output_folder, exist_ok=True)

    print("Wie möchten Sie das Video teilen?")
    print("1 - In eine bestimmte Anzahl gleich großer Teile")
    print("2 - In Teile mit einer bestimmten Länge (in Sekunden)")
    choice = input("Bitte wählen Sie (1 oder 2): ")

    if choice == "1":
        num_parts = int(input("Geben Sie die Anzahl der Teile an: "))
        if num_parts > 0:
            split_video_by_parts(input_video, output_folder, num_parts)
        else:
            print("Ungültige Anzahl von Teilen!")

    elif choice == "2":
        segment_length = int(input("Geben Sie die Videolänge pro Segment (in Sekunden) an: "))
        if segment_length > 0:
            split_video_by_length(input_video, output_folder, segment_length)
        else:
            print("Ungültige Videolänge!")

    else:
        print("Ungültige Auswahl!")

    print("Fertig!")


if __name__ == "__main__":
    main()
