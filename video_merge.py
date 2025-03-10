import os

input_videos = [
    "its Worse than You think full documentary_1_deutsch.mp4",
    "its Worse than You think full documentary_2_deutsch.mp4",
    "its Worse than You think full documentary_3_deutsch.mp4",
    "its Worse than You think full documentary_4_deutsch.mp4"
]
output_video = "its Worse than You think full documentary_deutsch.mp4"

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