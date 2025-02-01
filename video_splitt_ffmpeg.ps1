# Videodauer ermitteln
$duration = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "its Worse than You think full documentary.mp4"

# Anzahl der Segmente festlegen
$segments = 6

# Segmentdauer berechnen
$segment_time = [math]::Round($duration / $segments, 2)

# FFmpeg-Befehl ausführen
ffmpeg -i "its Worse than You think full documentary.mp4" -c copy -f segment -segment_time $segment_time -reset_timestamps 1 "its Worse than You think full documentary_%03d.mp4"