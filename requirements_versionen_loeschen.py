import sys
import os

def process_line(line):
    """Verarbeitet eine Zeile: Entfernt '==' und alles Dahinter."""
    if '==' in line:
        return line.split('==', 1)[0].rstrip() + '\n'
    return line.rstrip() + '\n'

if __name__ == "__main__":
    #if len(sys.argv) != 3:
        #print("Verwendung: python script.py eingabe.txt ausgabe.txt")
        #sys.exit(1)
    
    input_file = "req_310.txt"
    output_file = "req_310_cleaned.txt"
    
    if not os.path.exists(input_file):
        print(f"Fehler: Datei '{input_file}' nicht gefunden.")
        sys.exit(1)
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            processed = process_line(line)
            outfile.write(processed)
    
    print(f"Verarbeitung abgeschlossen. Ausgabe in '{output_file}' geschrieben.")
