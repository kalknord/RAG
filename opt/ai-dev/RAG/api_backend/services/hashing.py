# =====================================
# KRYPTOGRAPHISCHE DATEI-HASH-BERECHNUNG
# =====================================

import hashlib

def calculate_file_hash(filepath, algo='sha256'):
    """
    Berechnet kryptographischen Hash einer Datei mit speicher-effizienter Chunk-Verarbeitung.
    
    Parameter:
        filepath (str): Absoluter oder relativer Pfad zur Ziel-Datei
                       Muss existieren und lesbar sein für erfolgreiche Hash-Berechnung
        
        algo (str): Hash-Algorithmus-Bezeichnung aus hashlib-Bibliothek
                   - 'sha256' (Default): Optimal für Security + Performance Balance
    Rückgabe:
        str: Hexadezimal-repräsentierter Hash-Wert als String
    """
    
    # HASH-OBJEKT-INITIALISIERUNG: Dynamische Algorithmus-Auswahl
    h = hashlib.new(algo)
    
    # DATEI-VERARBEITUNG: Context Manager für sicheres Resource-Management
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            
            # HASH-UPDATE: Inkrementelle Hash-Berechnung
            h.update(chunk)
    
    # FINALE HASH-EXTRAKTION: Hexadezimal-Repräsentation für String-Kompatibilität
    return h.hexdigest()