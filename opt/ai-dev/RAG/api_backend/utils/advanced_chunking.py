# =====================================
# ADVANCED CHUNKING
# =====================================

from typing import List, Dict, Any    
import re                            

def process_parsed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    
    Diese Funktion implementiert die 1:1-Beziehung zwischen
    Text-Chunks und ihren Embedding-Vektoren.
    
    Parameter:
        chunks (List[Dict[str, Any]]): Raw Chunks vom Document Parser
                                      Struktur: [{"text": str, "metadata": dict}, ...]
                                      Kann heterogene Formate enthalten
    
    Rückgabe:
        List[Dict[str, Any]]: Enhanced Chunks mit angereicherten Metadaten
                             Struktur: [{"text": str, "metadata": enhanced_dict}, ...]
    """
    # EARLY-RETURN: Leere Input-Liste → Leere Output-Liste (triviale Count-Preservation)
    if not chunks:
        return []
    
    # OUTPUT-CONTAINER: Pre-sized für exakte Count-Kontrolle
    processed_chunks = []
    
    # CHUNK-ITERATION: Jeder Chunk wird individuell verarbeitet ohne Skip-Möglichkeit
    for i, chunk in enumerate(chunks):
        
        # TYPE-SAFETY: Sicherstellung dass Chunk als Dictionary verarbeitet werden kann
        # Parser können verschiedene Chunk-Formate liefern (Objekte, Strings, etc.)
        if not isinstance(chunk, dict):
            # FALLBACK-CONVERSION: Unbekannte Formate in Standard-Dictionary konvertieren
            # Verhindert TypeErrors und ermöglicht einheitliche Verarbeitung
            chunk = {
                "text": str(chunk),                     # String-Konvertierung für beliebige Typen
                "metadata": {"chunk_index": i}          # Minimal-Metadaten für Tracking
            }
        
        # TEXT-EXTRACTION: Sichere Extraktion mit Fallback auf leeren String
        text = chunk.get("text", "")
        
        # METADATA-EXTRACTION: Bestehende Metadaten extrahieren für Anreicherung
        metadata = chunk.get("metadata", {})
        
        # CONTENT-VALIDATION: Kritische Entscheidung für leere/unbrauchbare Chunks
        # WICHTIG: Chunks werden NIEMALS übersprungen, immer Placeholder verwenden
        if not text or len(text.strip()) < 5:
            # PLACEHOLDER-STRATEGY: Minimal-Content statt Chunk-Deletion
            # Erhält Count-Invariant und ermöglicht Debugging
            text = f"[Empty chunk {i}]"
        
        # METADATA-ENHANCEMENT: Anreicherung mit berechneten Qualitäts-Metriken
        # Spread-Operator (...) erhält bestehende Metadaten und fügt neue hinzu
        enhanced_metadata = {
            **metadata,                                                       # Bestehende Metadaten erhalten
            "word_count": len(text.split()) if text else 0,                   # Wort-Anzahl für Lesbarkeits-Metriken
            "char_count": len(text),                                          # Zeichen-Anzahl für Längen-Analyse
            "estimated_reading_time": estimate_reading_time(text),            # Geschätzte Lesezeit in Minuten
            "contains_table": detect_table_content(text),                     # Tabellen-Erkennung für Content-Type-Classification
            "contains_list": detect_list_content(text),                       # Listen-Erkennung für Struktur-Analyse
            "text_quality_score": calculate_text_quality(text),               # Multi-Faktor Qualitätsbewertung
            "chunk_index": i                                                  # Explizite Index-Zuordnung für Debugging
        }
        
        # CHUNK-RECONSTRUCTION: Standard-Format mit angereicherten Metadaten
        # KRITISCH: IMMER append() - niemals conditional skip
        processed_chunks.append({
            "text": text,                           # Bereinigter oder Placeholder-Text
            "metadata": enhanced_metadata           # Vollständig angereicherte Metadaten
        })
    
    # INVARIANT-VERIFICATION: Finale Validierung der Count-Preservation
    # Dies ist eine KRITISCHE Sicherheitsmaßnahme für System-Integrität
    if len(processed_chunks) != len(chunks):
        # CRITICAL ERROR: Count-Mismatch ist ein systemweiter Fehler
        # ValueError mit detaillierter Fehlermeldung für Debugging
        raise ValueError(f"CRITICAL: Chunk count changed from {len(chunks)} to {len(processed_chunks)}")
    
    return processed_chunks

def clean_and_enhance_text(text: str) -> str:
    """
    Bereinigt und normalisiert Text OHNE Inhaltsverlust oder Leerstring-Erzeugung.

    Parameter:
        text (str): Zu bereinigender Rohtext aus Document-Parsing
    
    Rückgabe:
        str: Bereinigter Text, garantiert nicht leer (minimum: "[Cleaned empty text]")
    """
    # NULL-SAFETY: Frühe Rückkehr bei null/undefined Input
    if not text:
        return "[Empty text]"  # Placeholder statt leerer String
    
    # WHITESPACE-NORMALISIERUNG: Multiple Spaces zu Single Space
    # \s+ matched alle Whitespace-Charaktere (Space, Tab, Newline, etc.)
    text = re.sub(r'\s+', ' ', text)

    # LINEBREAK-OPTIMIZATION: Excessive Newlines reduzieren
    # Behält Paragraph-Struktur bei aber entfernt redundante Leerzeilen
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # SMART-QUOTE-NORMALISIERUNG: Typographische Zeichen → ASCII
    # Linke/rechte single quotes → Standard apostrophe
    text = text.replace("'", "'").replace("'", "'")
    
    # SMART-QUOTE-NORMALISIERUNG: Typographische Anführungszeichen → ASCII
    # Linke/rechte double quotes → Standard double quote
    text = text.replace(""", '"').replace(""", '"')
    
    # DASH-NORMALISIERUNG: Verschiedene Dash-Typen → Standard Hyphen
    # En-dash und Em-dash → Standard ASCII hyphen für Konsistenz
    text = text.replace('–', '-').replace('—', '-')

    # EDGE-TRIMMING: Leading/Trailing Whitespace-Entfernung
    # Entfernt Spaces/Tabs am Anfang und Ende des Texts
    text = text.strip()
    
    # FINAL-SAFETY-CHECK: Verhindert leere Strings nach Bereinigung
    # KRITISCH: Leere Strings können Count-Preservation-Logik durcheinander bringen
    if not text:
        return "[Cleaned empty text]"

    return text

def detect_table_content(text: str) -> bool:
    """
    Erkennt Tabellen-Inhalte durch Pattern-Matching für Content-Classification.
    
    Parameter:
        text (str): Zu analysierender Text-Chunk
    
    Rückgabe:
        bool: True wenn Tabellen-Strukturen erkannt werden, False sonst
    """
    # NULL-SAFETY: Frühe Rückkehr bei leerem Input
    if not text:
        return False
        
    # TABELLEN-PATTERN-DEFINITION: Verschiedene Tabellen-Formate abdecken
    table_indicators = [
        r'\|.*\|',                          # Pipe-separierte Tabellen (Markdown-Style)
        r'\t.*\t',                          # Tab-separierte Inhalte (TSV-Style)
        r'^\s*[-+|=\s]+$',                  # Tabellen-Trennlinien (ASCII-Art-Tables)
        r'[A-Za-z]+\s*:\s*[0-9]+',         # Key-Value Paare mit numerischen Werten
    ]
    
    # PATTERN-ITERATION: Jeden Tabellen-Indikator testen
    for pattern in table_indicators:
        # MULTILINE-MATCHING: ^ und $ matchen Zeilenanfang/-ende
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    # NO-MATCH: Keine Tabellen-Pattern gefunden
    return False

def detect_list_content(text: str) -> bool:
    """
    Erkennt Listen-Strukturen durch Pattern-Matching für Content-Classification.
    
    Parameter:
        text (str): Zu analysierender Text-Chunk
    
    Rückgabe:
        bool: True wenn Listen-Strukturen erkannt werden, False sonst
    """
    # NULL-SAFETY: Frühe Rückkehr bei leerem Input
    if not text:
        return False
        
    # LISTEN-PATTERN-DEFINITION: Verschiedene Listen-Marker abdecken
    list_patterns = [
        r'^\s*[-•*]\s+',                    # Bullet points (verschiedene Marker)
        r'^\s*\d+\.\s+',                    # Nummerierte Listen (1., 2., 3.)
        r'^\s*[a-zA-Z]\)\s+',               # Alphabetische Listen (a), b), c))
        r'^\s*[ivxIVX]+\.\s+',              # Römische Numerierung (i., ii., iii., I., II., III.)
    ]
    
    # PATTERN-ITERATION: Jeden Listen-Indikator testen
    for pattern in list_patterns:
        # MULTILINE-MATCHING: ^ matcht Zeilenanfang für echte Listen-Items
        if re.search(pattern, text, re.MULTILINE):
            return True
    
    # NO-MATCH: Keine Listen-Pattern gefunden
    return False

def calculate_text_quality(text: str) -> float:
    """
    Berechnet multi-dimensionalen Qualitäts-Score für Text-Chunks (0.1-1.0).
    
    Parameter:
        text (str): Zu bewertender Text-Chunk
    
    Rückgabe:
        float: Qualitäts-Score zwischen 0.1 (minimal) und 1.0 (perfekt)
               Minimum 0.1 gewährleistet dass keine Chunks komplett ausgeschlossen werden
    """
    # NULL-SAFETY: Minimal-Qualität für leeren Text
    if not text:
        return 0.1  # Minimal quality for empty text, but never 0
    
    # BASIS-SCORE: Startwert für multiplikative Reduktion
    score = 1.0
    
    # LÄNGEN-BEWERTUNG: Optimale Wort-Anzahl für LLM-Verarbeitung
    # Basiert auf empirischen Erfahrungen mit Transformer-Modellen
    word_count = len(text.split())
    if word_count < 5:
        score *= 0.3  # Sehr kurze Texte: niedrige Qualität aber nicht null
    elif word_count < 10:
        score *= 0.5  # Kurze Texte: mittlere Qualität
    elif word_count > 1000:
        score *= 0.8  # Sehr lange Texte: schwer für LLM zu verarbeiten
    # Optimal range 10-1000 Wörter: score bleibt 1.0
    
    # SONDERZEICHEN-RATIO-ANALYSE: OCR-Fehler und Formatierungs-Artefakte
    special_chars = len(re.findall(r'[^\w\s]', text))
    char_ratio = special_chars / len(text) if text else 0
    if char_ratio > 0.3:  # Mehr als 30% Sonderzeichen
        score *= 0.7  # Wahrscheinlich OCR-Fehler oder beschädigte Daten
    
    # GROßBUCHSTABEN-RATIO-ANALYSE: ALL-CAPS-Text und OCR-Artefakte
    upper_ratio = len(re.findall(r'[A-Z]', text)) / len(text) if text else 0
    if upper_ratio > 0.5:  # Mehr als 50% Großbuchstaben
        score *= 0.8  # OCR-Fehler oder schlecht formatierter Text
    
    # LEXICAL-DIVERSITY-ANALYSE: Wiederholende Wörter reduzieren Qualität
    words = text.lower().split()
    if len(words) > 0:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.5:  # Weniger als 50% unique Wörter
            score *= 0.9  # Hohe Wiederholung deutet auf schlechte Segmentierung hin
    
    # MINIMUM-SCORE-GARANTIE: Niemals komplett wertlose Chunks erzeugen
    # KRITISCH: Score 0 würde zu Division-by-Zero und anderen Problemen führen
    return max(round(score, 2), 0.1)

def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    Schätzt Lesezeit für Text-Chunk basierend auf durchschnittlicher Lesegeschwindigkeit.
    
    Rückgabe:
        float: Geschätzte Lesezeit in Minuten, minimum 0.1
    """
    # NULL-SAFETY: Minimal-Zeit für leeren Text
    if not text:
        return 0.1
        
    # WORT-COUNT: Einfache Space-basierte Segmentierung
    word_count = len(text.split())
    
    # ZEIT-BERECHNUNG: Linear basierend auf Wort-Anzahl
    # max() gewährleistet Minimum-Zeit für Konsistenz
    return round(max(word_count / words_per_minute, 0.1), 1)

def get_chunk_statistics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Berechnet umfassende Statistiken für verarbeitete Chunk-Collections.
    
    Parameter:
        chunks (List[Dict[str, Any]]): Verarbeitete Chunks mit angereicherten Metadaten
    
    Rückgabe:
        Dict[str, Any]: Strukturierte Statistiken für Dashboard-Konsumierung
                       Alle numerischen Werte sind gerundet für bessere Lesbarkeit
    """
    # EARLY-RETURN: Leere Collection → Null-Statistiken
    if not chunks:
        return {
            "total_chunks": 0,
            "total_words": 0,
            "average_words_per_chunk": 0,
            "average_quality_score": 0,
            "chunks_with_tables": 0,
            "chunks_with_lists": 0,
            "total_reading_time": 0
        }
    
    # METADATEN-EXTRAKTION: Sichere Navigation durch Chunk-Metadaten
    # get() mit Default-Werten verhindert KeyError bei inconsistenten Strukturen
    word_counts = [chunk["metadata"].get("word_count", 0) for chunk in chunks]
    quality_scores = [chunk["metadata"].get("text_quality_score", 0) for chunk in chunks]
    
    # STATISTIK-AGGREGATION: Strukturierte Berechnung aller Metriken
    return {
        # VOLUMEN-METRIKEN: Basis-Statistiken über Chunk-Collection
        "total_chunks": len(chunks),                                                    # Absolute Anzahl verarbeiteter Chunks
        "total_words": sum(word_counts),                                               # Gesamt-Wort-Anzahl für Kapazitäts-Planning
        
        # DURCHSCHNITTS-METRIKEN: Qualitäts-Indikatoren mit Null-Division-Schutz
        "average_words_per_chunk": round(sum(word_counts) / len(word_counts), 1) if word_counts else 0,           # Durchschnittliche Chunk-Länge
        "average_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0,   # Durchschnittliche Qualitätsbewertung
        
        # CONTENT-ANALYSE: Strukturelle Dokument-Eigenschaften
        "chunks_with_tables": sum(1 for chunk in chunks if chunk["metadata"].get("contains_table", False)),      # Anzahl Chunks mit Tabellen-Content
        "chunks_with_lists": sum(1 for chunk in chunks if chunk["metadata"].get("contains_list", False)),        # Anzahl Chunks mit Listen-Content
        
        # USABILITY-METRIKEN: Geschätzte Zeiten für User-Experience-Planning
        "total_reading_time": round(sum(chunk["metadata"].get("estimated_reading_time", 0) for chunk in chunks), 1)  # Gesamt-Lesezeit in Minuten
    }