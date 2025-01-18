        # Anpassung der Timestamps (z.B. Verkürzung um 2 Sekunden)
for segment in result["segments"]:
    segment["start"] = max(segment["start"] - 0, 0)  # Startzeit anpassen
    segment["end"] = max(segment["end"] - 2, 0)      # Endzeit anpassen