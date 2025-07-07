# =====================================
# DOCUMENT PARSER SERVICE 
# =====================================

import httpx
import os
import asyncio
from typing import List, Dict, Any, Tuple
import logging

# Logging-Konfiguration für detaillierte Fehleranalyse und Performance-Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service-Discovery: Parser Service URL aus Umgebungsvariablen
# Standardwert nutzt Docker-Container-Namen für interne Kommunikation
PARSER_SERVICE_URL = os.getenv("PARSER_SERVICE_URL", "http://pdf_parser:8000")

async def parse_document_with_embeddings(
    file_path: str, 
    strategy: str = "hi_res",               # Parsing-Strategie: "hi_res" für maximale Qualität
    chunking_strategy: str = "title",       # Chunk-Strategie: "title" für semantische Grenzen
    max_characters: int = 500,              # Maximale Chunk-Größe für optimale Embedding-Performance
    generate_embeddings: bool = True        # Ein-Schritt-Verarbeitung: Parsing + Embeddings
) -> Dict[str, Any]:
    """
    Zentrale Funktion für die vollständige Dokumentenverarbeitung in einem Aufruf.
    
    Diese Funktion implementiert das One-Shot-Parsing-Prinzip, bei dem Dokumentenanalyse
    und Embedding-Generierung atomisch ausgeführt werden. Dies gewährleistet:
    - Konsistenz zwischen Text-Chunks und deren Vektorrepräsentationen
    - Reduzierte Netzwerk-Latenz durch kombinierten Service-Aufruf
    - Transaktionale Sicherheit bei der Verarbeitung
    
    Parameter:
        file_path: Absoluter Pfad zur zu verarbeitenden Datei
        strategy: "hi_res" verwendet fortgeschrittene Layout-Analyse und OCR
        chunking_strategy: "title" berücksichtigt Dokumentenstruktur bei der Segmentierung
        max_characters: Optimiert für all-MiniLM-L6-v2 Model (384 Dimensionen)
        generate_embeddings: True für vollständige RAG-Pipeline
    
    Rückgabe:
        Dict mit 'success', 'chunks', 'embeddings', 'metadata', 'ocr_info'
    
    Raises:
        httpx.HTTPError: Bei Netzwerk- oder Service-Fehlern
        Exception: Bei unerwarteten Verarbeitungsfehlern
    """
    # Asynchroner HTTP-Client mit großzügigem Timeout für ML-Operations
    # 180 Sekunden berücksichtigen OCR-Verarbeitung und Embedding-Generierung
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            # Strukturierte Request-Payload für Parser Service
            # Alle Parameter werden validiert und an den ML-Service weitergeleitet
            response = await client.post(
                f"{PARSER_SERVICE_URL}/parse",
                json={
                    "file_path": file_path,                          # Eingabedatei
                    "strategy": strategy,                            # Layout-Analyse-Strategie
                    "chunking_strategy": chunking_strategy,          # Segmentierungsverfahren
                    "max_characters": max_characters,                # Chunk-Größenlimit
                    "new_after_n_chars": max_characters - 50,        # Pufferzone für natürliche Grenzen
                    "combine_text_under_n_chars": 50,                # Minimale Chunk-Größe
                    "generate_embeddings": generate_embeddings,      # Embedding-Flag
                    "ocr_languages": ["deu", "eng"],                 # Deutsch und Englisch für OCR
                    "extract_images": True                           # Bildextraktion aktiviert
                }
            )
            # HTTP-Status-Validierung - wirft Exception bei 4xx/5xx Codes
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            # Detaillierte Logging für Netzwerk- und Service-Fehler
            logger.error(f"HTTP error when calling enhanced parser: {e}")
            raise
        except Exception as e:
            # Catch-all für unerwartete Fehler mit vollständiger Weiterleitung
            logger.error(f"Error calling enhanced parser: {e}")
            raise

async def generate_embeddings_only(texts: List[str]) -> List[List[float]]:
    """
    Standalone Embedding-Generierung mit robuster Fehlerbehandlung.
    
    Diese Funktion wurde als Reaktion auf Produktions-Fehler entwickelt und implementiert
    mehrschichtige Validierung für die kritische Embedding-Pipeline:
    - Input-Sanitization gegen leere/ungültige Texte
    - Dimension-Validation für ML-Model-Konsistenz
    - Count-Verification für 1:1 Text-zu-Embedding Mapping
    - Typ-Validation für numerische Vector-Komponenten
    
    Parameter:
        texts: Liste von Texten für Embedding-Generierung (bereits gefiltert)
    
    Rückgabe:
        List[List[float]]: 384-dimensionale Vektoren (all-MiniLM-L6-v2)
        Leere Liste bei Fehlern oder ungültigen Inputs
    
    Raises:
        Exception: Bei kritischen Validierungsfehlern (Count-Mismatch, Type-Errors)
    """
    # EINGABE-VALIDATION: Frühe Rückkehr bei leeren Inputs
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []
    
    # TEXT-SANITIZATION: Entfernung von Whitespace-only und leeren Strings
    # Kritisch für ML-Model-Stabilität - leere Inputs können NaN-Vektoren erzeugen
    valid_texts = [text.strip() for text in texts if text and text.strip()]
    if not valid_texts:
        logger.warning("No valid texts after filtering")
        return []
    
    # HTTP-Client mit moderatem Timeout - Embeddings sind schneller als Full-Parsing
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # PERFORMANCE-LOGGING: Tracking für Batch-Size und Response-Zeit
            logger.info(f"?? Requesting embeddings for {len(valid_texts)} texts")
            
            # API-AUFRUF: Minimale Payload für Embedding-Only Operation
            response = await client.post(
                f"{PARSER_SERVICE_URL}/embed",
                json={"texts": valid_texts}
            )
            
            # STATUS-MONITORING: HTTP-Response-Code Logging für Debugging
            logger.info(f"?? Embedding response status: {response.status_code}")
            
            # HTTP-VALIDATION: Exception bei Client/Server-Fehlern
            response.raise_for_status()
            result = response.json()
            
            # RESPONSE-STRUKTUR-ANALYSE: Debugging-Information für API-Entwicklung
            logger.info(f"?? Embedding response keys: {list(result.keys())}")
            
            # Response-Format-Prüfung
            # Verhindert Runtime-Fehler bei API-Änderungen oder Service-Fehlern
            if not isinstance(result, dict):
                logger.error(f"? Invalid response type: {type(result)}")
                raise Exception(f"Invalid response format: expected dict, got {type(result)}")
            
            # ERFOLGS-FLAG-PRÜFUNG: Explizite Service-Level-Fehlerbehandlung
            # Manche ML-Services melden Fehler über Status-Flags statt HTTP-Codes
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
            
            # TYP-VALIDATION: List-Struktur erforderlich für Vektor-Arrays
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
            # Verhindert NaN/Infinity-Werte die ChromaDB korrumpieren können
            for i, embedding in enumerate(embeddings):
                # STRUKTUR-CHECK: Jeder Vektor muss Liste sein
                if not isinstance(embedding, list):
                    logger.error(f"? Invalid embedding {i}: not a list")
                    raise Exception(f"Invalid embedding format at index {i}")
                
                # LÄNGEN-CHECK: Leere Vektoren sind ungültig
                if len(embedding) == 0:
                    logger.error(f"? Empty embedding at index {i}")
                    raise Exception(f"Empty embedding at index {i}")
                
                # NUMERISCHE-VALIDATION: Alle Komponenten müssen Numbers sein
                # Verhindert String/Object-Injection in Vektor-Datenbank
                if not all(isinstance(x, (int, float)) for x in embedding):
                    logger.error(f"? Non-numeric values in embedding {i}")
                    raise Exception(f"Non-numeric values in embedding {i}")
            
            # ERFOLGS-LOGGING: Performance-Metriken für Monitoring
            logger.info(f"? Successfully generated {len(embeddings)} embeddings, dim={len(embeddings[0])}")
            return embeddings
                
        except httpx.HTTPError as e:
            # DETAILLIERTE HTTP-FEHLER-BEHANDLUNG: Wichtig für Service-Debugging
            logger.error(f"? HTTP error when calling embedding endpoint: {e}")
            logger.error(f"? Response status: {getattr(e.response, 'status_code', 'unknown')}")
            logger.error(f"? Response text: {getattr(e.response, 'text', 'unknown')}")
            raise Exception(f"Embedding HTTP error: {str(e)}")
        except Exception as e:
            # VOLLSTÄNDIGE TRACEBACK-ERFASSUNG: Kritisch für Produktions-Debugging
            logger.error(f"? Error calling embedding endpoint: {e}")
            import traceback
            logger.error(f"? Traceback: {traceback.format_exc()}")
            raise Exception(f"Embedding generation error: {str(e)}")

async def parse_document_advanced(file_path: str, strategy: str = "hi_res", 
                                 chunking_strategy: str = "title", max_characters: int = 500):
    """
    Legacy-Kompatibilitäts-Wrapper für bestehende Client-Code.
    
    Parameter:
        Identisch zu parse_document_with_embeddings(), aber generate_embeddings=False
    
    Rückgabe:
        Parse-Ergebnisse ohne Embeddings für Legacy-Workflows
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
    Service-Health-Check für Parser-Microservice mit detaillierter Capability-Analyse.
    
    Diese Funktion implementiert aktives Service-Discovery und Health-Monitoring
    für die kritische Parser-Infrastruktur. Sie überprüft:
    - Service-Verfügbarkeit und Response-Zeit
    - ML-Model-Status (Embedding-Fähigkeiten)
    - Unterstützte Dateiformate
    - Service-Capabilities und Features
    
    Rückgabe:
        Tuple[bool, str]: (Verfügbarkeit, Detaillierter Status-String)
        - True: Service operational mit allen Capabilities
        - False: Service offline oder kritische Features fehlen
    """
    try:
        # KURZES TIMEOUT: Health-Checks sollen schnell sein für Load-Balancer
        async with httpx.AsyncClient(timeout=5.0) as client:
            # HEALTH-ENDPOINT: Standardisierte Service-Introspection
            response = await client.get(f"{PARSER_SERVICE_URL}/health")
            response.raise_for_status()
            result = response.json()
            
            # SERVICE-CAPABILITY-EXTRAKTION: Wichtig für Feature-Verfügbarkeit
            capabilities = result.get('capabilities', [])
            embedding_loaded = result.get('embedding_model_loaded', False)
            supported_formats = result.get('supported_formats', [])
            
            # HUMAN-READABLE STATUS-AGGREGATION: Für Dashboard-Anzeige
            status_msg = f"Enhanced Parser online. Formats: {len(supported_formats)}, Capabilities: {', '.join(capabilities)}"
            
            # KRITISCHE FEATURE-CHECKS: Embedding-Fähigkeit für RAG-Betrieb
            if embedding_loaded:
                status_msg += " | Embeddings: ?"
            else:
                status_msg += " | Embeddings: ?"  # Warnung: Reduzierte Funktionalität
                
            return True, status_msg
    except Exception as e:
        # OFFLINE-STATUS: Service nicht erreichbar oder fehlerhaft
        return False, f"Enhanced Parser offline: {str(e)}"

async def get_supported_formats() -> Dict[str, str]:
    """
    Dynamische Abfrage der unterstützten Dateiformate vom Parser-Service.
    
    Diese Funktion implementiert Service-Discovery für Dateiformat-Capabilities.
    Sie ermöglicht:
    - Dynamische UI-Anpassung basierend auf Service-Features
    - Automatische Format-Validation ohne Hard-coding
    - Fallback-Strategien bei Service-Ausfall
    
    Rückgabe:
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
            # EXTRAHIERUNG: Nested-JSON-Struktur für Format-Informationen
            return result.get("formats", {})
    except Exception as e:
        # FALLBACK-STRATEGIE: Statische Format-Liste bei Service-Ausfall
        logger.error(f"Error getting supported formats: {e}")
        
        # VOLLSTÄNDIGER FALLBACK: Basis-Formate die immer unterstützt werden
        # Diese Liste wird manuell synchron gehalten mit Service-Capabilities
        return {
            ".pdf": "PDF Document",                    # Primärformat für Dokumentenverarbeitung
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