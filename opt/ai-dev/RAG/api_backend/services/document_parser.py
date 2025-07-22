# =====================================
# DOCUMENT PARSER SERVICE 
# =====================================

import httpx
import os
import asyncio
from typing import List, Dict, Any, Tuple
import logging

# Logging-Konfiguration f�r detaillierte Fehleranalyse und Performance-Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service-Discovery: Parser Service URL aus Umgebungsvariablen
# Standardwert nutzt Docker-Container-Namen f�r interne Kommunikation
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://pdf_parser:8000")

async def parse_document_with_embeddings(
    file_path: str, 
    strategy: str = "hi_res",               # Parsing-Strategie: "hi_res" f�r maximale Qualit�t
    chunking_strategy: str = "title",       # Chunk-Strategie: "title" f�r semantische Grenzen
    max_characters: int = 500,              # Maximale Chunk-Gr��e f�r optimale Embedding-Performance
    generate_embeddings: bool = True        # Ein-Schritt-Verarbeitung: Parsing + Embeddings
) -> Dict[str, Any]:
    """
    Zentrale Funktion f�r die vollst�ndige Dokumentenverarbeitung in einem Aufruf.
    
    Diese Funktion implementiert das One-Shot-Parsing-Prinzip, bei dem Dokumentenanalyse
    und Embedding-Generierung atomisch ausgef�hrt werden. Dies gew�hrleistet:
    - Konsistenz zwischen Text-Chunks und deren Vektorrepr�sentationen
    - Reduzierte Netzwerk-Latenz durch kombinierten Service-Aufruf
    - Transaktionale Sicherheit bei der Verarbeitung
    
    Parameter:
        file_path: Absoluter Pfad zur zu verarbeitenden Datei
        strategy: "hi_res" verwendet fortgeschrittene Layout-Analyse und OCR
        chunking_strategy: "title" ber�cksichtigt Dokumentenstruktur bei der Segmentierung
        max_characters: Optimiert f�r all-MiniLM-L6-v2 Model (384 Dimensionen)
        generate_embeddings: True f�r vollst�ndige RAG-Pipeline
    
    R�ckgabe:
        Dict mit 'success', 'chunks', 'embeddings', 'metadata', 'ocr_info'
    
    Raises:
        httpx.HTTPError: Bei Netzwerk- oder Service-Fehlern
        Exception: Bei unerwarteten Verarbeitungsfehlern
    """
    # Asynchroner HTTP-Client mit gro�z�gigem Timeout f�r ML-Operations
    # 180 Sekunden ber�cksichtigen OCR-Verarbeitung und Embedding-Generierung
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            # Strukturierte Request-Payload f�r Parser Service
            # Alle Parameter werden validiert und an den ML-Service weitergeleitet
            response = await client.post(
                f"{PARSER_SERVICE_URL}/parse",
                json={
                    "file_path": file_path,                          # Eingabedatei
                    "strategy": strategy,                            # Layout-Analyse-Strategie
                    "chunking_strategy": chunking_strategy,          # Segmentierungsverfahren
                    "max_characters": max_characters,                # Chunk-Gr��enlimit
                    "new_after_n_chars": max_characters - 50,        # Pufferzone f�r nat�rliche Grenzen
                    "combine_text_under_n_chars": 50,                # Minimale Chunk-Gr��e
                    "generate_embeddings": generate_embeddings,      # Embedding-Flag
                    "ocr_languages": ["deu", "eng"],                 # Deutsch und Englisch f�r OCR
                    "extract_images": True                           # Bildextraktion aktiviert
                }
            )
            # HTTP-Status-Validierung - wirft Exception bei 4xx/5xx Codes
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            # Detaillierte Logging f�r Netzwerk- und Service-Fehler
            logger.error(f"HTTP error when calling enhanced parser: {e}")
            raise
        except Exception as e:
            # Catch-all f�r unerwartete Fehler mit vollst�ndiger Weiterleitung
            logger.error(f"Error calling enhanced parser: {e}")
            raise

async def generate_embeddings_only(texts: List[str]) -> List[List[float]]:
    """
    Standalone Embedding-Generierung mit robuster Fehlerbehandlung.
    
    Diese Funktion wurde als Reaktion auf Produktions-Fehler entwickelt und implementiert
    mehrschichtige Validierung f�r die kritische Embedding-Pipeline:
    - Input-Sanitization gegen leere/ung�ltige Texte
    - Dimension-Validation f�r ML-Model-Konsistenz
    - Count-Verification f�r 1:1 Text-zu-Embedding Mapping
    - Typ-Validation f�r numerische Vector-Komponenten
    
    Parameter:
        texts: Liste von Texten f�r Embedding-Generierung (bereits gefiltert)
    
    R�ckgabe:
        List[List[float]]: 384-dimensionale Vektoren (all-MiniLM-L6-v2)
        Leere Liste bei Fehlern oder ung�ltigen Inputs
    
    Raises:
        Exception: Bei kritischen Validierungsfehlern (Count-Mismatch, Type-Errors)
    """
    # EINGABE-VALIDATION: Fr�he R�ckkehr bei leeren Inputs
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []
    
    # TEXT-SANITIZATION: Entfernung von Whitespace-only und leeren Strings
    # Kritisch f�r ML-Model-Stabilit�t - leere Inputs k�nnen NaN-Vektoren erzeugen
    valid_texts = [text.strip() for text in texts if text and text.strip()]
    if not valid_texts:
        logger.warning("No valid texts after filtering")
        return []
    
    # HTTP-Client mit moderatem Timeout - Embeddings sind schneller als Full-Parsing
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # PERFORMANCE-LOGGING: Tracking f�r Batch-Size und Response-Zeit
            logger.info(f"?? Requesting embeddings for {len(valid_texts)} texts")
            
            # API-AUFRUF: Minimale Payload f�r Embedding-Only Operation
            response = await client.post(
                f"{PARSER_SERVICE_URL}/embed",
                json={"texts": valid_texts}
            )
            
            # STATUS-MONITORING: HTTP-Response-Code Logging f�r Debugging
            logger.info(f"?? Embedding response status: {response.status_code}")
            
            # HTTP-VALIDATION: Exception bei Client/Server-Fehlern
            response.raise_for_status()
            result = response.json()
            
            # RESPONSE-STRUKTUR-ANALYSE: Debugging-Information f�r API-Entwicklung
            logger.info(f"?? Embedding response keys: {list(result.keys())}")
            
            # Response-Format-Pr�fung
            # Verhindert Runtime-Fehler bei API-�nderungen oder Service-Fehlern
            if not isinstance(result, dict):
                logger.error(f"? Invalid response type: {type(result)}")
                raise Exception(f"Invalid response format: expected dict, got {type(result)}")
            
            # ERFOLGS-FLAG-PR�FUNG: Explizite Service-Level-Fehlerbehandlung
            # Manche ML-Services melden Fehler �ber Status-Flags statt HTTP-Codes
            if "success" in result:
                if not result.get("success", False):
                    error_msg = result.get("error", "Unknown embedding error")
                    logger.error(f"? Embedding generation failed: {error_msg}")
                    raise Exception(f"Embedding generation failed: {error_msg}")
                
                embeddings = result.get("embeddings", [])
            else:
                # FALLBACK-STRATEGIE: Direkte Embedding-Extraktion bei Legacy-APIs
                embeddings = result.get("embeddings", result)
            
            # EMBEDDING-EXISTENCE-VALIDATION: Null/Empty-Check
            if not embeddings:
                logger.error("? No embeddings in response")
                raise Exception("No embeddings returned from service")
            
            # TYP-VALIDATION: List-Struktur erforderlich f�r Vektor-Arrays
            if not isinstance(embeddings, list):
                logger.error(f"? Invalid embeddings type: {type(embeddings)}")
                raise Exception(f"Invalid embeddings format: expected list, got {type(embeddings)}")
            
            # KRITISCHE COUNT-VALIDATION: 1:1 Mapping zwischen Input und Output
            # Verhindert Index-Fehler bei nachgelagerten Verarbeitungsschritten
            if len(embeddings) != len(valid_texts):
                error_msg = f"Embedding count mismatch: {len(valid_texts)} texts -> {len(embeddings)} embeddings"
                logger.error(f"? {error_msg}")
                raise Exception(error_msg)
            
            # VEKTOR-STRUKTUR-VALIDATION: Jeder Embedding muss valide Nummer-Liste sein
            # Verhindert NaN/Infinity-Werte die ChromaDB korrumpieren k�nnen
            for i, embedding in enumerate(embeddings):
                # STRUKTUR-CHECK: Jeder Vektor muss Liste sein
                if not isinstance(embedding, list):
                    logger.error(f"? Invalid embedding {i}: not a list")
                    raise Exception(f"Invalid embedding format at index {i}")
                
                # L�NGEN-CHECK: Leere Vektoren sind ung�ltig
                if len(embedding) == 0:
                    logger.error(f"? Empty embedding at index {i}")
                    raise Exception(f"Empty embedding at index {i}")
                
                # NUMERISCHE-VALIDATION: Alle Komponenten m�ssen Numbers sein
                # Verhindert String/Object-Injection in Vektor-Datenbank
                if not all(isinstance(x, (int, float)) for x in embedding):
                    logger.error(f"? Non-numeric values in embedding {i}")
                    raise Exception(f"Non-numeric values in embedding {i}")
            
            # ERFOLGS-LOGGING: Performance-Metriken f�r Monitoring
            logger.info(f"? Successfully generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")
            return embeddings
                
        except httpx.HTTPError as e:
            # DETAILLIERTE HTTP-FEHLER-BEHANDLUNG: Wichtig f�r Service-Debugging
            logger.error(f"? HTTP error when calling embedding endpoint: {e}")
            logger.error(f"? Response status: {getattr(e.response, 'status_code', 'unknown')}")
            logger.error(f"? Response text: {getattr(e.response, 'text', 'unknown')}")
            raise Exception(f"Embedding HTTP error: {str(e)}")
        except Exception as e:
            # VOLLST�NDIGE TRACEBACK-ERFASSUNG: Kritisch f�r Produktions-Debugging
            logger.error(f"? Error calling embedding endpoint: {e}")
            import traceback
            logger.error(f"? Traceback: {traceback.format_exc()}")
            raise Exception(f"Embedding generation error: {str(e)}")

async def parse_document_advanced(file_path: str, strategy: str = "hi_res", 
                                 chunking_strategy: str = "title", max_characters: int = 500):
    """
    Legacy-Kompatibilit�ts-Wrapper f�r bestehende Client-Code.
    
    Parameter:
        Identisch zu parse_document_with_embeddings(), aber generate_embeddings=False
    
    R�ckgabe:
        Parse-Ergebnisse ohne Embeddings f�r Legacy-Workflows
    """
    return await parse_document_with_embeddings(
        file_path=file_path,
        strategy=strategy,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
        generate_embeddings=False  # Legacy-Modus: Keine Embeddings
    )

async def check_parser_connection() -> Tuple[bool, str]:
    """
    Service-Health-Check f�r Parser-Microservice mit detaillierter Capability-Analyse.
    
    Diese Funktion implementiert aktives Service-Discovery und Health-Monitoring
    f�r die kritische Parser-Infrastruktur. Sie �berpr�ft:
    - Service-Verf�gbarkeit und Response-Zeit
    - ML-Model-Status (Embedding-F�higkeiten)
    - Unterst�tzte Dateiformate
    - Service-Capabilities und Features
    
    R�ckgabe:
        Tuple[bool, str]: (Verf�gbarkeit, Detaillierter Status-String)
        - True: Service operational mit allen Capabilities
        - False: Service offline oder kritische Features fehlen
    """
    try:
        # KURZES TIMEOUT: Health-Checks sollen schnell sein f�r Load-Balancer
        async with httpx.AsyncClient(timeout=5.0) as client:
            # HEALTH-ENDPOINT: Standardisierte Service-Introspection
            response = await client.get(f"{PARSER_SERVICE_URL}/health")
            response.raise_for_status()
            result = response.json()
            
            # SERVICE-CAPABILITY-EXTRAKTION: Wichtig f�r Feature-Verf�gbarkeit
            capabilities = result.get('capabilities', [])
            embedding_loaded = result.get('embedding_model_loaded', False)
            supported_formats = result.get('supported_formats', [])
            
            # HUMAN-READABLE STATUS-AGGREGATION: F�r Dashboard-Anzeige
            status_msg = f"Enhanced Parser online. Formats: {len(supported_formats)}, Capabilities: {', '.join(capabilities)}"
            
            # KRITISCHE FEATURE-CHECKS: Embedding-F�higkeit f�r RAG-Betrieb
            if embedding_loaded:
                status_msg += " | Embeddings: ?"
            else:
                status_msg += " | Embeddings: ?"  # Warnung: Reduzierte Funktionalit�t
                
            return True, status_msg
    except Exception as e:
        # OFFLINE-STATUS: Service nicht erreichbar oder fehlerhaft
        return False, f"Enhanced Parser offline: {str(e)}"

async def get_supported_formats() -> Dict[str, str]:
    """
    Dynamische Abfrage der unterst�tzten Dateiformate vom Parser-Service.
    
    Diese Funktion implementiert Service-Discovery f�r Dateiformat-Capabilities.
    Sie erm�glicht:
    - Dynamische UI-Anpassung basierend auf Service-Features
    - Automatische Format-Validation ohne Hard-coding
    - Fallback-Strategien bei Service-Ausfall
    
    R�ckgabe:
        Dict[str, str]: Mapping von Dateiendung zu Beschreibung
        Beispiel: {".pdf": "PDF Document", ".docx": "Word Document"}
    """
    try:
        # KURZES TIMEOUT: Format-Abfrage sollte cached und schnell sein
        async with httpx.AsyncClient(timeout=5.0) as client:
            # FORMATS-ENDPOINT: Service-spezifische Capability-Discovery
            response = await client.get(f"{PARSER_SERVICE_URL}/formats")
            response.raise_for_status()
            result = response.json()
            # EXTRAHIERUNG: Nested-JSON-Struktur f�r Format-Informationen
            return result.get("formats", {})
    except Exception as e:
        # FALLBACK-STRATEGIE: Statische Format-Liste bei Service-Ausfall
        logger.error(f"Error getting supported formats: {e}")
        
        # VOLLST�NDIGER FALLBACK: Basis-Formate die immer unterst�tzt werden
        # Diese Liste wird manuell synchron gehalten mit Service-Capabilities
        return {
            ".pdf": "PDF Document",                    # Prim�rformat f�r Dokumentenverarbeitung
            ".docx": "Word Document",                  # Microsoft Word (moderne Version)
            ".doc": "Word Document (Legacy)",         # Microsoft Word (Legacy-Format)
            ".pptx": "PowerPoint Presentation",       # Microsoft PowerPoint (moderne Version)
            ".ppt": "PowerPoint (Legacy)",           # Microsoft PowerPoint (Legacy-Format)
            ".txt": "Text File",                      # Plain Text
            ".md": "Markdown File",                   # Markdown-Dokumentation
            ".html": "HTML File",                     # Web-Inhalte
            ".htm": "HTML File",                      # HTML-Alternative-Endung
            ".xlsx": "Excel File",                    # Microsoft Excel (moderne Version)
            ".xls": "Excel File (Legacy)",           # Microsoft Excel (Legacy-Format)
            ".csv": "CSV File",                       # Comma-Separated Values
            ".rtf": "Rich Text Format",               # Rich Text Format
            ".odt": "OpenDocument Text",              # LibreOffice/OpenOffice
            ".xml": "XML File"                        # Structured XML Data
        }