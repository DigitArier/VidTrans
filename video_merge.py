import os

input_videos = [
    "30 Wildkraeuter + Heilwirkung Teil 1-3_HD.mp4",
    "30 Wildkraeuter + Heilwirkung Teil 2-3_HD.mp4",
    "30 Wildkraeuter + Heilwirkung Teil 3-3_HD.mp4",
]
output_video = "30 Wildkraeuter + Heilwirkung_HD.mp4"

def merge_mp4_files(input_files, output_file):
    """Führt mehrere MP4-Dateien zu einer einzigen Datei zusammen."""

    list_file = "merge_list.txt"

    # Textdatei für FFmpeg erstellen
    with open(list_file, "w") as f:
        for video in input_files:
            f.write(f"file '{video}'\n")

    # FFmpeg-Kommando ausführen
    os.system(f'ffmpeg -f concat -safe 0 -i {list_file} -c copy "{output_file}"')

    print(f"✅ Video erfolgreich zusammengefügt: {output_file}")

merge_mp4_files(input_videos, output_video)