import ctranslate2

print(f"Unterstützung durch CUDA:")
print(ctranslate2.get_supported_compute_types("cuda"))

print(f"\nUnterstützung durch CPU:")
print(ctranslate2.get_supported_compute_types("cpu"))
