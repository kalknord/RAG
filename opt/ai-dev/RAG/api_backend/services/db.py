# =====================================
# RAG-SYSTEM DATENBANK-LAYER
# =====================================

import psycopg2
import psycopg2.extras
import psycopg2.pool
from psycopg2.extensions import register_adapter, AsIs
from enum import Enum
import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import logging

# Konfiguriere Logging für Datenbank-Operationen und Performance-Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection_params() -> Dict[str, Any]:
    """
    Hole Datenbankverbindungsparameter aus Umgebungsvariablen
    
    Verwendet Docker-Container-Namen als Standard-Hosts für interne Kommunikation
    Umgebungsvariablen ermöglichen flexible Konfiguration für verschiedene Deployment-Szenarien
    
    Returns:
        Dict mit Verbindungsparametern für psycopg2
    """
    return {
        "host": os.getenv("POSTGRES_HOST", "rag_postgres"),          # Docker-Service-Name
        "dbname": os.getenv("POSTGRES_DB", "rag_metadata"),         # Datenbank für Metadaten
        "user": os.getenv("POSTGRES_USER", "rag_user"),             # Dedizierter RAG-Benutzer
        "password": os.getenv("POSTGRES_PASSWORD", "rag_pw"),       # Passwort aus Environment
        "port": int(os.getenv("POSTGRES_PORT", "5432")),            # Standard PostgreSQL-Port
        "connect_timeout": 10,                                      # Timeout für Verbindungsaufbau
        "application_name": "rag_api_backend"                       # Identifikation in PostgreSQL-Logs
    }

# Diese Enums definieren die erlaubten Werte für verschiedene Kategorien
# Sie entsprechen exakt den ENUM-Typen in der PostgreSQL-Datenbank
# Dies gewährleistet Typsicherheit und verhindert ungültige Statuswerte

class FileStatus(Enum):
    """
    Dateistatus-Enum für Lebenszyklus-Management
    Definiert alle möglichen Zustände einer Datei im System
    """
    UPLOADED = 'uploaded'       # Datei hochgeladen, wartet auf Verarbeitung
    PROCESSING = 'processing'   # Wird gerade verarbeitet (Parser läuft)
    CHUNKED = 'chunked'         # Erfolgreich in Chunks aufgeteilt, bereit für Suche
    ERROR = 'error'             # Fehler bei Verarbeitung aufgetreten
    DELETED = 'deleted'         # Soft-gelöscht (für Recovery verfügbar)

class DocumentType(Enum):
    """
    Dokumenttyp-Enum für Format-spezifische Verarbeitung
    Ermöglicht optimierte Parser-Strategien je nach Dateityp
    """
    PDF = 'pdf'                 # PDF-Dokumente (Hauptfokus des Systems)
    WORD = 'word'               # Microsoft Word-Dokumente (.docx/.doc)
    POWERPOINT = 'powerpoint'   # PowerPoint-Präsentationen (.pptx/.ppt)
    EXCEL = 'excel'             # Excel-Tabellen (.xlsx/.xls)
    TEXT = 'text'               # Reine Textdateien (.txt)
    HTML = 'html'               # HTML-Dokumente
    MARKDOWN = 'markdown'       # Markdown-Dateien (.md)
    CSV = 'csv'                 # Comma-Separated Values

class ProcessingMethod(Enum):
    """
    Verarbeitungsmethoden-Enum für Qualitäts- und Performance-Tracking
    Dokumentiert, welche Parsing-Strategie verwendet wurde
    """
    DIRECT_TEXT = 'direct_text'     # Direkter Text ohne OCR (beste Qualität)
    OCR_STANDARD = 'ocr_standard'   # Standard-OCR für eingescannte Dokumente
    OCR_GPU = 'ocr_gpu'             # GPU-beschleunigtes OCR für bessere Performance
    TABLE_EXTRACT = 'table_extract' # Spezielle Tabellen-Extraktion
    HYBRID = 'hybrid'               # Kombination mehrerer Methoden (Standard)

# Registriere Custom-Adapter für Python-Enums, damit sie korrekt in PostgreSQL geschrieben werden
# Dies ermöglicht die nahtlose Verwendung von Python-Enums in SQL-Queries

def adapt_enum(enum_value):
    """
    Konvertiert Python-Enum zu PostgreSQL-ENUM-String
    Wrapped den Enum-Wert in Single-Quotes für SQL-Kompatibilität
    """
    return AsIs(f"'{enum_value.value}'")

# Registriere Adapter für alle verwendeten Enum-Typen
register_adapter(FileStatus, adapt_enum)
register_adapter(DocumentType, adapt_enum)
register_adapter(ProcessingMethod, adapt_enum)

# Dataclasses definieren die Datenstrukturen für das gesamte System
# Sie bieten Typsicherheit, automatische Serialisierung und klare Interfaces

@dataclass
class FileInfo:
    """
    Vollständige Datei-Metadaten-Struktur
    
    Enthält alle Informationen über eine Datei im System:
    - Identifikation und Speicherort
    - Verarbeitungsstatus und -metriken
    - Qualitätsbewertung und Analytics
    - Zeitstempel für Lifecycle-Tracking
    """
    id: int                                   # Eindeutige Datenbank-ID
    file_name: str                            # Ursprünglicher Dateiname
    file_hash: str                            # SHA256-Hash für Duplikat-Erkennung
    file_path: str                            # Absoluter Pfad zur physischen Datei
    file_size: int                            # Dateigröße in Bytes
    file_extension: str                       # Dateierweiterung (.pdf, .docx, etc.)
    document_type: DocumentType               # Kategorisierter Dokumenttyp
    status: FileStatus                        # Aktueller Verarbeitungsstatus
    chunk_count: int                          # Anzahl erstellter Text-Chunks
    upload_date: str                          # ISO-Zeitstempel des Uploads
    last_chunked: Optional[str] = None        # Letzter Verarbeitungszeitpunkt
    error_message: Optional[str] = None       # Fehlermeldung bei Problemen
    metadata: Dict[str, Any] = None           # Zusätzliche Metadaten (JSON)
    content_quality_score: float = 0.0        # Qualitätsbewertung (0.0-1.0)
    processing_duration_ms: Optional[int] = None  # Verarbeitungszeit in Millisekunden
    last_accessed: Optional[str] = None       # Letzter Zugriff für Analytics

    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert FileInfo zu Dictionary für API-Responses
        Wandelt Enum-Werte zu Strings für JSON-Serialisierung
        """
        result = asdict(self)
        result['document_type'] = self.document_type.value
        result['status'] = self.status.value
        return result

@dataclass
class ChunkInfo:
    """
    Text-Chunk-Metadaten-Struktur
    
    Repräsentiert einen einzelnen Text-Abschnitt mit allen Metadaten:
    - Position im Originaldokument
    - Inhaltsanalyse und Qualitätsbewertung
    - Verarbeitungsdetails und Performance-Metriken
    """
    id: int                                  # Eindeutige Chunk-ID
    file_id: int                             # Referenz zur Eltern-Datei
    text: str                                # Extrahierter Text-Inhalt
    page_number: int                         # Seitennummer im Originaldokument
    chunk_index: int                         # Index innerhalb der Datei
    chunk_quality_score: float               # Qualitätsbewertung des Chunks (0.0-1.0)
    contains_table: bool                     # Flag für Tabellen-Inhalt
    contains_list: bool                      # Flag für Listen-Strukturen
    text_length: Optional[int] = None        # Textlänge in Zeichen (auto-generiert)
    word_count: Optional[int] = None         # Wortanzahl (auto-generiert)
    element_type: Optional[str] = None       # Element-Typ (Title, Text, Table, etc.)
    processing_method: Optional[ProcessingMethod] = None  # Verwendete Verarbeitungsmethode
    metadata: Dict[str, Any] = None          # Erweiterte Metadaten (JSON)
    created_at: Optional[str] = None         # Erstellungszeitpunkt

    def to_dict(self) -> Dict[str, Any]:
        """
        Konvertiert ChunkInfo zu Dictionary für API-Responses
        Behandelt Enum-zu-String-Konvertierung
        """
        result = asdict(self)
        if self.processing_method:
            result['processing_method'] = self.processing_method.value
        return result

@dataclass
class SystemMetrics:
    """
    System-Performance-Metriken-Struktur
    
    Aggregiert wichtige Kennzahlen für Monitoring und Analytics:
    - Datei-Verarbeitungsstatistiken
    - Speicher- und Performance-Metriken
    - Benutzer-Aktivitätsanalyse
    """
    files_pending: int                       # Dateien in Warteschlange
    files_processing: int                    # Aktuell in Verarbeitung
    files_ready: int                         # Bereit für Suche
    files_error: int                         # Fehlgeschlagene Verarbeitungen
    total_storage: str                       # Gesamter Speicherverbrauch (human-readable)
    total_files: int                         # Gesamtanzahl Dateien im System
    total_chunks: int                        # Gesamtanzahl Text-Chunks
    avg_processing_time: Optional[float]     # Durchschnittliche Verarbeitungszeit
    max_processing_time: Optional[float]     # Maximale Verarbeitungszeit
    avg_file_quality: Optional[float]        # Durchschnittliche Datequalität
    files_uploaded_today: int                # Heute hochgeladene Dateien
    files_accessed_today: int                # Heute auf Dateien zugegriffen

# Connection-Pool-Management

class DatabaseManager:
    """
    Zentraler Datenbank-Connection-Pool-Manager
    
    Verwaltet einen Thread-sicheren Pool von PostgreSQL-Verbindungen:
    - Automatische Verbindungserstellung und -wiederverwendung
    - Fehlerbehandlung mit automatischem Rollback
    - Graceful Shutdown und Ressourcen-Cleanup
    - Context Manager für sichere Transaktionsbehandlung
    """
    
    def __init__(self):
        """
        Initialisiert den DatabaseManager und erstellt den Connection Pool
        """
        self.connection_pool = None
        self._init_connection_pool()
    
    def _init_connection_pool(self):
        """
        Initialisiert Connection Pool mit optimalen Einstellungen
        
        Konfiguration:
        - minconn=2: Mindestens 2 Verbindungen für Responsivität
        - maxconn=20: Maximum 20 Verbindungen um PostgreSQL nicht zu überlasten
        - ThreadedConnectionPool: Thread-sichere Implementierung
        """
        try:
            connection_params = get_connection_params()
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,              # Minimale Anzahl offener Verbindungen
                maxconn=20,             # Maximale Anzahl Verbindungen im Pool
                **connection_params     # Verbindungsparameter
            )
            logger.info("✅ Database connection pool initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        Context Manager für sichere Verbindungsnutzung
        
        Implementiert das "Resource Acquisition Is Initialization" (RAII) Pattern:
        1. Holt Verbindung aus Pool
        2. Deaktiviert Autocommit für Transaktionskontrolle
        3. Führt User-Code aus
        4. Commitet bei Erfolg, rollback bei Fehlern
        5. Gibt Verbindung zurück an Pool
        
        Yields:
            psycopg2.connection: Datenbankverbindung für Transaktionen
        """
        conn = None
        try:
            # Hole Verbindung aus Pool
            conn = self.connection_pool.getconn()
            conn.autocommit = False          # Explizite Transaktionskontrolle
            yield conn                       # Gib Verbindung an User-Code
            conn.commit()                    # Commite bei erfolgreichem Durchlauf
        except Exception as e:
            # Fehlerbehandlung mit Rollback
            if conn:
                try:
                    conn.rollback()
                    logger.warning(f"Transaction rolled back due to error: {e}")
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
            logger.error(f"Database error: {e}")
            raise
        finally:
            # Cleanup: Gib Verbindung zurück an Pool
            if conn:
                try:
                    conn.rollback()          # Sicherheits-Rollback
                    self.connection_pool.putconn(conn)
                except Exception as e:
                    logger.error(f"Error returning connection to pool: {e}")
                    # Bei defekter Verbindung: Schließe und lasse Pool neue erstellen
                    try:
                        conn.close()
                    except:
                        pass

    def close_all_connections(self):
        """
        Schließt alle Verbindungen im Pool (für Graceful Shutdown)
        Wird beim Application-Shutdown aufgerufen
        """
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("All database connections closed")

# Globale Instanz des DatabaseManagers für systemweite Nutzung
db_manager = DatabaseManager()

def test_db_connection() -> tuple[bool, str]:
    """
    Testet Datenbankverbindung und Schema-Integrität
    
    Prüft:
    1. Grundlegende Verbindung zur Datenbank
    2. Existenz der benötigten Tabellen
    3. Schema-Vollständigkeit
    
    Returns:
        Tuple[bool, str]: (Erfolg, Statusmeldung)
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Teste grundlegende Verbindung
            cur.execute("SELECT 1")
            result = cur.fetchone()
            
            # Teste Schema-Existenz - prüfe kritische Tabellen
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('uploaded_files', 'pdf_chunks', 'chroma_collections')
            """)
            table_count = cur.fetchone()[0]
            
            cur.close()
            
            if table_count >= 3:
                return True, "✅ Database connection and schema OK"
            else:
                return False, f"❌ Missing tables. Found {table_count}/3 required tables"
                
    except Exception as e:
        return False, f"❌ Connection failed: {str(e)}"

# CRUD-Operationen für Dateien
# mit erweiterten Features wie Duplikat-Erkennung und Qualitäts-Tracking

def register_uploaded_file(file_info: Dict[str, Any]) -> Optional[int]:
    """
    Registriert neue Datei mit erweiterten Metadaten und ENUM-Unterstützung
    
    Implementiert intelligente Dateiregistrierung:
    - Konvertiert String-Dokumenttypen zu Enums
    - Extrahiert Suchschlüsselwörter aus Dateinamen
    - Speichert erweiterte Metadaten als JSON
    - Setzt sichere Defaults für alle Felder
    
    Args:
        file_info: Dictionary mit Datei-Metadaten
    
    Returns:
        Optional[int]: Datei-ID bei Erfolg, None bei Fehler
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Konvertiere Dokumenttyp zu Enum mit Fallback
            doc_type_str = file_info.get('document_type', 'text')
            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                logger.warning(f"Unknown document type '{doc_type_str}', defaulting to TEXT")
                doc_type = DocumentType.TEXT
            
            # Extrahiere Suchschlüsselwörter aus Dateiname für bessere Suche
            keywords = extract_keywords_from_filename(file_info['file_name'])
            
            # INSERT mit erweiterten Metadaten
            cur.execute("""
                INSERT INTO uploaded_files 
                (file_name, file_hash, file_path, file_size, file_extension, 
                 document_type, mime_type, status, metadata, search_keywords,
                 has_images, has_tables)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                file_info['file_name'],                          # Ursprünglicher Dateiname
                file_info['file_hash'],                          # SHA256-Hash für Duplikat-Erkennung
                file_info['file_path'],                          # Absoluter Pfad zur Datei
                file_info['file_size'],                          # Dateigröße in Bytes
                file_info['file_extension'],                     # Dateierweiterung
                doc_type,                                        # Enum-Dokumenttyp
                file_info.get('mime_type'),                      # MIME-Type (optional)
                FileStatus.UPLOADED,                             # Initial-Status
                json.dumps(file_info.get('metadata', {})),       # JSON-Metadaten
                keywords,                                        # Array von Suchschlüsselwörtern
                file_info.get('has_images', False),              # Bilder-Flag
                file_info.get('has_tables', False)               # Tabellen-Flag
            ))
            
            # Hole generierte ID
            file_id = cur.fetchone()[0]
            cur.close()
            
            logger.info(f"✅ Registered file {file_info['file_name']} with ID {file_id}")
            return file_id
            
    except Exception as e:
        logger.error(f"❌ Error registering file: {e}")
        return None

def update_file_status(file_id: int, status: Union[FileStatus, str], 
                      chunk_count: int = 0, error_message: str = None,
                      processing_duration_ms: int = None) -> bool:
    """
    Aktualisiert Dateistatus mit erweiterten Metriken
    
    Implementiert intelligente Statusaktualisierung:
    - Unterschiedliche Update-Strategien je nach Status
    - Automatische Zeitstempel-Verwaltung
    - Retry-Counter für Fehler-Tracking
    - Performance-Metriken-Erfassung
    
    Args:
        file_id: Eindeutige Datei-ID
        status: Neuer Status (Enum oder String)
        chunk_count: Anzahl erstellter Chunks (bei CHUNKED)
        error_message: Fehlermeldung (bei ERROR)
        processing_duration_ms: Verarbeitungszeit in Millisekunden
    
    Returns:
        bool: True bei erfolgreichem Update
    """
    try:
        # Konvertiere String zu Enum falls nötig
        if isinstance(status, str):
            status = FileStatus(status)
        
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Status-spezifische Update-Strategien
            if status == FileStatus.CHUNKED:
                # Erfolgreiche Verarbeitung: Setze alle relevanten Felder
                cur.execute("""
                    UPDATE uploaded_files 
                    SET status = %s, chunk_count = %s, last_chunked = NOW(),
                        processing_duration_ms = %s, updated_at = NOW(),
                        error_message = NULL, retry_count = 0
                    WHERE id = %s
                """, (status, chunk_count, processing_duration_ms, file_id))
            elif status == FileStatus.ERROR:
                # Fehler-Tracking: Inkrementiere Retry-Counter
                cur.execute("""
                    UPDATE uploaded_files 
                    SET status = %s, error_message = %s, updated_at = NOW(),
                        retry_count = retry_count + 1
                    WHERE id = %s
                """, (status, error_message, file_id))
            else:
                # Standard-Update: Nur Status und Zeitstempel
                cur.execute("""
                    UPDATE uploaded_files 
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                """, (status, file_id))
            
            # Prüfe ob Update erfolgreich war
            affected_rows = cur.rowcount
            cur.close()
            
            if affected_rows > 0:
                logger.info(f"✅ Updated file {file_id} status to {status.value}")
                return True
            else:
                logger.warning(f"⚠️ No file found with ID {file_id}")
                return False
            
    except Exception as e:
        logger.error(f"❌ Error updating file status: {e}")
        return False

def get_file_by_id(file_id: int) -> Optional[FileInfo]:
    """
    Holt Datei anhand ID mit vollständigen Metadaten
    
    Lädt komplette Datei-Informationen aus der Datenbank:
    - Alle Basis-Metadaten
    - Verarbeitungsstatistiken
    - Qualitäts- und Performance-Metriken
    - Zeitstempel-Informationen
    
    Args:
        file_id: Eindeutige Datei-ID
    
    Returns:
        Optional[FileInfo]: FileInfo-Objekt oder None falls nicht gefunden
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Vollständige Datei-Metadaten abfragen (ohne soft-gelöschte)
            cur.execute("""
                SELECT id, file_name, file_hash, file_path, file_size, 
                       file_extension, document_type, status, chunk_count,
                       upload_date, last_chunked, last_accessed, error_message, 
                       metadata, content_quality_score, processing_duration_ms
                FROM uploaded_files 
                WHERE id = %s AND status != 'deleted'
            """, (file_id,))
            
            file_row = cur.fetchone()
            cur.close()
            
            if file_row:
                # Konvertiere Datenbank-Row zu FileInfo-Objekt
                return FileInfo(
                    id=file_row['id'],
                    file_name=file_row['file_name'],
                    file_hash=file_row['file_hash'],
                    file_path=file_row['file_path'],
                    file_size=file_row['file_size'],
                    file_extension=file_row['file_extension'],
                    document_type=DocumentType(file_row['document_type']),       # String zu Enum
                    status=FileStatus(file_row['status']),                       # String zu Enum
                    chunk_count=file_row['chunk_count'],
                    upload_date=file_row['upload_date'].isoformat() if file_row['upload_date'] else None,
                    last_chunked=file_row['last_chunked'].isoformat() if file_row['last_chunked'] else None,
                    last_accessed=file_row['last_accessed'].isoformat() if file_row['last_accessed'] else None,
                    error_message=file_row['error_message'],
                    metadata=file_row['metadata'] or {},                         # JSON zu Dict
                    content_quality_score=file_row['content_quality_score'] or 0.0,
                    processing_duration_ms=file_row['processing_duration_ms']
                )
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting file by ID {file_id}: {e}")
        return None

def get_file_by_hash(file_hash: str) -> Optional[FileInfo]:
    """
    Holt Datei anhand SHA256-Hash für Duplikat-Erkennung
    
    Implementiert effiziente Hash-basierte Suche:
    - Verwendet optimierten Index auf file_hash
    - Filtert soft-gelöschte Dateien aus
    - Liefert vollständige Metadaten
    
    Args:
        file_hash: SHA256-Hash der Datei
    
    Returns:
        Optional[FileInfo]: FileInfo-Objekt oder None falls nicht gefunden
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Hash-basierte Suche (nutzt idx_uploaded_files_hash Index)
            cur.execute("""
                SELECT id, file_name, file_hash, file_path, file_size, 
                       file_extension, document_type, status, chunk_count,
                       upload_date, last_chunked, error_message, metadata,
                       content_quality_score
                FROM uploaded_files 
                WHERE file_hash = %s AND status != 'deleted'
            """, (file_hash,))
            
            file_row = cur.fetchone()
            cur.close()
            
            if file_row:
                return FileInfo(
                    id=file_row['id'],
                    file_name=file_row['file_name'],
                    file_hash=file_row['file_hash'],
                    file_path=file_row['file_path'],
                    file_size=file_row['file_size'],
                    file_extension=file_row['file_extension'],
                    document_type=DocumentType(file_row['document_type']),
                    status=FileStatus(file_row['status']),
                    chunk_count=file_row['chunk_count'],
                    upload_date=file_row['upload_date'].isoformat() if file_row['upload_date'] else None,
                    last_chunked=file_row['last_chunked'].isoformat() if file_row['last_chunked'] else None,
                    error_message=file_row['error_message'],
                    metadata=file_row['metadata'] or {},
                    content_quality_score=file_row['content_quality_score'] or 0.0
                )
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting file by hash: {e}")
        return None

def get_all_uploaded_files() -> List[FileInfo]:
    """
    Holt alle Dateien mit optimierter Abfrage über Performance-View
    
    Verwendet die file_overview View für optimierte Performance:
    - Vorgenerierte Statistiken und Metriken
    - Sortierung nach Upload-Datum (neueste zuerst)
    - Erweiterte Qualitäts- und Performance-Indikatoren
    
    Returns:
        List[FileInfo]: Liste aller Dateien im System
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Nutze optimierte file_overview View mit aggregierten Statistiken
            cur.execute("""
                SELECT id, file_name, file_hash, file_path, file_size, 
                       file_extension, document_type, status, chunk_count,
                       upload_date, last_chunked, error_message, metadata,
                       content_quality_score, processing_duration_ms,
                       avg_chunk_quality, total_word_count, chunks_with_tables,
                       display_status, processing_speed, days_since_last_access
                FROM file_overview 
                ORDER BY upload_date DESC
            """)
            
            files = cur.fetchall()
            cur.close()
            
            # Konvertiere alle Rows zu FileInfo-Objekten
            result = []
            for file_row in files:
                file_info = FileInfo(
                    id=file_row['id'],
                    file_name=file_row['file_name'],
                    file_hash=file_row['file_hash'],
                    file_path=file_row['file_path'],
                    file_size=file_row['file_size'],
                    file_extension=file_row['file_extension'],
                    document_type=DocumentType(file_row['document_type']),
                    status=FileStatus(file_row['status']),
                    chunk_count=file_row['chunk_count'],
                    upload_date=file_row['upload_date'].isoformat() if file_row['upload_date'] else None,
                    last_chunked=file_row['last_chunked'].isoformat() if file_row['last_chunked'] else None,
                    error_message=file_row['error_message'],
                    metadata=file_row['metadata'] or {},
                    content_quality_score=file_row['content_quality_score'] or 0.0,
                    processing_duration_ms=file_row['processing_duration_ms']
                )
                result.append(file_info)
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Error getting all files: {e}")
        return []

def delete_uploaded_file(file_id: int) -> bool:
    """
    Soft-Delete einer Datei (setzt Status auf 'deleted')
    
    Implementiert sicheres Soft-Delete:
    - Bewahrt alle Daten für Recovery auf
    - Aktualisiert Zeitstempel
    - Filtert bereits gelöschte Dateien aus
    - Ermöglicht einfache Wiederherstellung
    
    Args:
        file_id: Eindeutige Datei-ID
    
    Returns:
        bool: True bei erfolgreichem Soft-Delete
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Soft Delete: Bewahrt Daten für Recovery auf
            cur.execute("""
                UPDATE uploaded_files 
                SET status = %s, updated_at = NOW()
                WHERE id = %s AND status != 'deleted'
            """, (FileStatus.DELETED, file_id))
            
            affected_rows = cur.rowcount
            cur.close()
            
            if affected_rows > 0:
                logger.info(f"✅ Soft deleted file {file_id}")
                return True
            else:
                logger.warning(f"⚠️ File {file_id} not found or already deleted")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error deleting file {file_id}: {e}")
        return False

def permanently_delete_file(file_id: int) -> bool:
    """
    Permanente Löschung einer Datei und aller Chunks (CASCADE)
    
    Implementiert vollständige Datenlöschung:
    1. Holt Datei-Informationen vor Löschung
    2. Löscht Datenbank-Einträge (CASCADE löscht Chunks automatisch)
    3. Entfernt physische Datei vom Dateisystem
    4. Ist nicht rückgängig machbar
    
    Args:
        file_id: Eindeutige Datei-ID
    
    Returns:
        bool: True bei erfolgreicher permanenter Löschung
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Hole Datei-Info vor Löschung für Cleanup
            cur.execute("SELECT file_name, file_path FROM uploaded_files WHERE id = %s", (file_id,))
            file_info = cur.fetchone()
            
            if not file_info:
                return False
            
            file_name, file_path = file_info
            
            # Lösche aus Datenbank (CASCADE behandelt Chunks automatisch)
            cur.execute("DELETE FROM uploaded_files WHERE id = %s", (file_id,))
            
            # Lösche physische Datei falls vorhanden
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete physical file {file_path}: {e}")
            
            cur.close()
            logger.info(f"✅ Permanently deleted file {file_name}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error permanently deleting file {file_id}: {e}")
        return False


# Verwalten von Text-Chunks 

def insert_chunk_metadata(file_id: int, chunk_data: Dict[str, Any]) -> bool:
    """
    KORRIGIERT: Fügt Chunk OHNE auto-generierte Spalten ein
    
    KRITISCHER FIX: Diese Funktion wurde korrigiert um:
    - Korrekte Signatur zu verwenden (file_id zuerst, dann chunk_data)
    - Auto-generierte Spalten auszuschließen (text_length, word_count, sentence_count)
    - ON CONFLICT für Upsert-Verhalten zu implementieren
    - Robuste Enum-Konvertierung zu gewährleisten
    
    Args:
        file_id: Eindeutige Datei-ID (KORRIGIERTE REIHENFOLGE)
        chunk_data: Dictionary mit Chunk-Daten
    
    Returns:
        bool: True bei erfolgreichem Insert
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Konvertiere Processing-Method zu Enum falls vorhanden
            processing_method = chunk_data.get('processing_method')
            if processing_method and isinstance(processing_method, str):
                try:
                    processing_method = ProcessingMethod(processing_method)
                except ValueError:
                    processing_method = ProcessingMethod.HYBRID  # Sicherer Fallback
            
            # Hole Text für Validation (aber füge generated columns nicht ein)
            text = chunk_data['text']
            
            # PostgreSQL berechnet text_length, word_count, sentence_count automatisch
            cur.execute("""
                INSERT INTO pdf_chunks 
                (file_id, file_name, file_hash, page_number, chunk_index, text,
                 element_type, contains_table, contains_list, contains_image_reference, 
                 contains_code, chunk_quality_score, readability_score, processing_method, 
                 ocr_confidence, metadata, language_detected, section_title)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_id, page_number, chunk_index) DO UPDATE SET
                text = EXCLUDED.text,
                chunk_quality_score = EXCLUDED.chunk_quality_score,
                processing_method = EXCLUDED.processing_method,
                metadata = EXCLUDED.metadata
            """, (
                file_id,                                       # file_id zuerst
                chunk_data.get('file_name', ''),               # Legacy-Kompatibilität
                chunk_data.get('file_hash', ''),               # Legacy-Kompatibilität
                chunk_data.get('page_number', 1),              # Seitennummer im Dokument
                chunk_data.get('chunk_index', 0),              # Index innerhalb der Datei
                text,                                          # Chunk-Text-Inhalt
                chunk_data.get('element_type', 'Text'),        # Element-Typ (Title, Text, etc.)
                chunk_data.get('contains_table', False),       # Tabellen-Flag
                chunk_data.get('contains_list', False),        # Listen-Flag
                chunk_data.get('contains_image_reference', False),  # Bild-Referenz-Flag
                chunk_data.get('contains_code', False),        # Code-Flag
                chunk_data.get('chunk_quality_score', 0.0),    # Qualitätsbewertung
                chunk_data.get('readability_score'),           # Lesbarkeits-Score
                processing_method,                             # Verarbeitungsmethode (Enum)
                chunk_data.get('ocr_confidence'),              # OCR-Konfidenz
                json.dumps(chunk_data.get('metadata', {})),    # JSON-Metadaten
                chunk_data.get('language_detected', 'de'),     # Erkannte Sprache
                chunk_data.get('section_title')                # Abschnitts-Titel
            ))
            
            cur.close()
            return True
            
    except Exception as e:
        logger.error(f"❌ Error inserting chunk for file {file_id}: {e}")
        return False

# LEGACY-FUNKTION FÜR RÜCKWÄRTSKOMPATIBILITÄT BEWAHRT
def insert_chunk_metadata_legacy(file_name: str, file_hash: str, page_number: int, 
                                chunk_index: int, text: str) -> bool:
    """
    Legacy-Funktionssignatur - konvertiert zu neuem Format
    
    Diese Funktion bewahrt Rückwärtskompatibilität zu älteren Code-Teilen:
    - Warnt vor Verwendung der veralteten Signatur
    - Findet file_id anhand des Hash
    - Konvertiert zu neuer insert_chunk_metadata-Signatur
    - Setzt sinnvolle Defaults für fehlende Parameter
    
    Args:
        file_name: Dateiname (Legacy-Parameter)
        file_hash: SHA256-Hash der Datei
        page_number: Seitennummer
        chunk_index: Chunk-Index
        text: Text-Inhalt
    
    Returns:
        bool: True bei erfolgreichem Insert
    """
    logger.warning("⚠️ Using legacy insert_chunk_metadata signature")
    
    # Finde file_id anhand Hash
    file_info = get_file_by_hash(file_hash)
    if not file_info:
        logger.error(f"❌ Cannot find file with hash {file_hash}")
        return False
    
    # Konvertiere zu neuem Format
    chunk_data = {
        'file_name': file_name,
        'file_hash': file_hash,
        'page_number': page_number,
        'chunk_index': chunk_index,
        'text': text,
        'chunk_quality_score': 0.5,    # Standard-Qualität
        'processing_method': 'hybrid'   # Standard-Verarbeitungsmethode
    }
    
    # Verwende neue Signatur
    return insert_chunk_metadata(file_info.id, chunk_data)

def get_chunks_by_file_id(file_id: int, limit: int = None, quality_threshold: float = 0.0) -> List[ChunkInfo]:
    """
    Holt Chunks anhand Datei-ID mit Qualitätsfilterung
    
    Implementiert optimierte Chunk-Abfrage:
    - Nutzt Index auf (file_id, page_number, chunk_index)
    - Filtert nach Qualitätsschwelle
    - Sortiert nach natürlicher Reihenfolge
    - Unterstützt optionale Limitierung
    
    Args:
        file_id: Eindeutige Datei-ID
        limit: Optionale Begrenzung der Ergebnisse
        quality_threshold: Minimale Chunk-Qualität (0.0-1.0)
    
    Returns:
        List[ChunkInfo]: Liste der Chunks mit Metadaten
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Basis-Query mit Qualitätsfilterung
            query = """
                SELECT id, file_id, text, page_number, chunk_index, text_length,
                       word_count, element_type, contains_table, contains_list,
                       chunk_quality_score, processing_method, metadata, created_at
                FROM pdf_chunks 
                WHERE file_id = %s AND chunk_quality_score >= %s
                ORDER BY page_number, chunk_index
            """
            
            params = [file_id, quality_threshold]
            
            # Optionale Limitierung
            if limit:
                query += " LIMIT %s"
                params.append(limit)
            
            cur.execute(query, params)
            chunks = cur.fetchall()
            cur.close()
            
            # Konvertiere zu ChunkInfo-Objekten
            result = []
            for chunk_row in chunks:
                chunk_info = ChunkInfo(
                    id=chunk_row['id'],
                    file_id=chunk_row['file_id'],
                    text=chunk_row['text'],
                    page_number=chunk_row['page_number'],
                    chunk_index=chunk_row['chunk_index'],
                    text_length=chunk_row['text_length'],         # Auto-generiert
                    word_count=chunk_row['word_count'],           # Auto-generiert
                    element_type=chunk_row['element_type'],
                    contains_table=chunk_row['contains_table'],
                    contains_list=chunk_row['contains_list'],
                    chunk_quality_score=chunk_row['chunk_quality_score'],
                    processing_method=ProcessingMethod(chunk_row['processing_method']) if chunk_row['processing_method'] else None,
                    metadata=chunk_row['metadata'] or {},
                    created_at=chunk_row['created_at'].isoformat() if chunk_row['created_at'] else None
                )
                result.append(chunk_info)
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Error getting chunks for file {file_id}: {e}")
        return []

def search_chunks_fulltext(search_term: str, file_ids: List[int] = None,
                          limit: int = 10, quality_threshold: float = 0.3) -> List[ChunkInfo]:
    """
    Volltextsuche mit deutscher Sprachunterstützung und Ranking
    
    Implementiert fortgeschrittene PostgreSQL-Volltextsuche:
    - to_tsvector('german', text) für deutsche Wortanalyse
    - plainto_tsquery('german', search_term) für Suchanfragen
    - ts_rank() für Relevanz-Bewertung
    - Qualitäts- und Datei-Filterung
    - Sortierung nach Relevanz und Qualität
    
    Args:
        search_term: Suchbegriff oder -phrase
        file_ids: Optional Liste von Datei-IDs für Filterung
        limit: Maximale Anzahl Ergebnisse
        quality_threshold: Minimale Chunk-Qualität
    
    Returns:
        List[ChunkInfo]: Relevanz-sortierte Suchergebnisse
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Volltext-Query mit deutscher Sprachunterstützung
            base_query = """
                SELECT id, file_id, text, page_number, chunk_index, text_length,
                       word_count, element_type, contains_table, contains_list,
                       chunk_quality_score, processing_method, metadata,
                       ts_rank(to_tsvector('german', text), 
                               plainto_tsquery('german', %s)) as relevance_rank
                FROM pdf_chunks 
                WHERE to_tsvector('german', text) @@ plainto_tsquery('german', %s)
                AND chunk_quality_score >= %s
            """
            
            params = [search_term, search_term, quality_threshold]
            
            # Optionale Datei-Filterung
            if file_ids:
                base_query += " AND file_id = ANY(%s)"
                params.append(file_ids)
            
            # Sortierung nach Relevanz und Qualität
            base_query += """
                ORDER BY relevance_rank DESC, chunk_quality_score DESC 
                LIMIT %s
            """
            params.append(limit)
            
            cur.execute(base_query, params)
            chunks = cur.fetchall()
            cur.close()
            
            # Konvertiere zu ChunkInfo-Objekten
            result = []
            for chunk_row in chunks:
                chunk_info = ChunkInfo(
                    id=chunk_row['id'],
                    file_id=chunk_row['file_id'],
                    text=chunk_row['text'],
                    page_number=chunk_row['page_number'],
                    chunk_index=chunk_row['chunk_index'],
                    text_length=chunk_row['text_length'],
                    word_count=chunk_row['word_count'],
                    element_type=chunk_row['element_type'],
                    contains_table=chunk_row['contains_table'],
                    contains_list=chunk_row['contains_list'],
                    chunk_quality_score=chunk_row['chunk_quality_score'],
                    processing_method=ProcessingMethod(chunk_row['processing_method']) if chunk_row['processing_method'] else None,
                    metadata=chunk_row['metadata'] or {}
                )
                result.append(chunk_info)
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Error in fulltext search: {e}")
        return []

# System-Performance-Überwachung und Benutzer-Analytics

def log_query(query_text: str, results_count: int, response_time_ms: int,
              embedding_time_ms: int = None, retrieval_time_ms: int = None,
              collection_name: str = None, user_session: str = None,
              source_ip: str = None) -> bool:
    """
    Protokolliert Abfrage für Analytics mit detailliertem Timing
    
    Erfasst umfassende Analytics-Daten:
    - Query-Text und Hash für Deduplizierung
    - Performance-Metriken (Response-, Embedding-, Retrieval-Zeit)
    - Benutzer-Kontext (Session, IP, Collection)
    - Ergebnis-Qualität (Anzahl Treffer)
    
    Args:
        query_text: Original-Suchanfrage
        results_count: Anzahl gefundener Ergebnisse
        response_time_ms: Gesamtantwortzeit in Millisekunden
        embedding_time_ms: Zeit für Embedding-Generierung
        retrieval_time_ms: Zeit für Vektorsuche
        collection_name: Verwendete ChromaDB-Collection
        user_session: Session-ID für User-Tracking
        source_ip: Client-IP-Adresse
    
    Returns:
        bool: True bei erfolgreichem Logging
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Generiere Query-Hash für Deduplizierungs-Analyse
            query_hash = hashlib.sha256(query_text.encode()).hexdigest()
            
            # Vollständiges Analytics-Logging
            cur.execute("""
                INSERT INTO query_logs 
                (query_text, query_hash, results_count, response_time_ms,
                 embedding_time_ms, retrieval_time_ms, collection_used, 
                 user_session, source_ip)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                query_text,         # Vollständiger Query-Text
                query_hash,         # SHA256-Hash für Deduplizierung
                results_count,      # Anzahl Ergebnisse
                response_time_ms,   # Gesamtantwortzeit
                embedding_time_ms,  # Embedding-Generierungszeit
                retrieval_time_ms,  # Vektorsuche-Zeit
                collection_name,    # ChromaDB-Collection
                user_session,       # Benutzer-Session-ID
                source_ip           # Client-IP für Geo-Analytics
            ))
            
            cur.close()
            return True
            
    except Exception as e:
        logger.error(f"❌ Error logging query: {e}")
        return False

def get_system_performance() -> Optional[SystemMetrics]:
    """
    Holt umfassende System-Performance-Metriken
    
    Nutzt die optimierte system_performance View für:
    - Datei-Verarbeitungsstatistiken
    - Speicher- und Performance-Metriken
    - Qualitäts-Indikatoren
    - Benutzeraktivität
    
    Returns:
        Optional[SystemMetrics]: Vollständige Systemmetriken oder None bei Fehler
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Nutze optimierte Performance-View
            cur.execute("SELECT * FROM system_performance")
            result = cur.fetchone()
            cur.close()
            
            if result:
                return SystemMetrics(
                    files_pending=result['files_pending'] or 0,          # Warteschlange
                    files_processing=result['files_processing'] or 0,    # In Bearbeitung
                    files_ready=result['files_ready'] or 0,              # Suchbereit
                    files_error=result['files_error'] or 0,              # Fehlgeschlagen
                    total_storage=result['total_storage'] or '0 bytes',  # Speicherverbrauch
                    total_files=result['total_files'] or 0,              # Gesamtdateien
                    total_chunks=result['total_chunks'] or 0,            # Gesamtchunks
                    avg_processing_time=result['avg_processing_time'],   # Ø Verarbeitungszeit
                    max_processing_time=result['max_processing_time'],   # Max Verarbeitungszeit
                    avg_file_quality=result['avg_file_quality'],         # Ø Datequalität
                    files_uploaded_today=result['files_uploaded_today'] or 0,   # Heute hochgeladen
                    files_accessed_today=result['files_accessed_today'] or 0    # Heute zugegriffen
                )
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting system performance: {e}")
        return None

def get_query_analytics(days: int = 30) -> Dict[str, Any]:
    """
    Holt Abfrage-Analytics für die letzten N Tage
    
    Berechnet umfassende Analytics-Metriken:
    - Gesamte und einzigartige Abfragen
    - Performance-Statistiken (Antwortzeiten)
    - Erfolgsquoten und Problembereiche
    - Langsame Abfragen und Zero-Result-Queries
    
    Args:
        days: Anzahl Tage für Analytics-Zeitraum
    
    Returns:
        Dict[str, Any]: Analytics-Daten oder leerer Dict bei Fehler
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Umfassende Analytics-Abfrage mit Aggregationen
            cur.execute("""
                SELECT 
                    COUNT(*) as total_queries,                                     -- Gesamtabfragen
                    COUNT(DISTINCT query_hash) as unique_queries,                  -- Einzigartige Abfragen
                    AVG(response_time_ms) as avg_response_time,                   -- Ø Antwortzeit
                    AVG(results_count) as avg_results_count,                      -- Ø Ergebnisanzahl
                    MAX(response_time_ms) as max_response_time,                   -- Max Antwortzeit
                    MIN(response_time_ms) as min_response_time,                   -- Min Antwortzeit
                    COUNT(*) FILTER (WHERE response_time_ms > 5000) as slow_queries,      -- Langsame Abfragen
                    COUNT(*) FILTER (WHERE results_count = 0) as zero_result_queries      -- Erfolglose Abfragen
                FROM query_logs 
                WHERE created_at >= NOW() - INTERVAL '%s days'
            """, (days,))
            
            result = cur.fetchone()
            cur.close()
            
            return dict(result) if result else {}
            
    except Exception as e:
        logger.error(f"❌ Error getting query analytics: {e}")
        return {}

def get_database_health() -> Dict[str, Any]:
    """
    Holt Datenbank-Gesundheitsmetriken
    
    Sammelt wichtige PostgreSQL-Metriken:
    - Tabellengröße und Speicherverbrauch
    - Index-Nutzungsstatistiken
    - Verbindungsstatistiken
    - Performance-Indikatoren
    
    Returns:
        Dict[str, Any]: Datenbank-Gesundheitsdaten
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Tabellengröße abfragen (größte zuerst)
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """)
            table_sizes = cur.fetchall()
            
            # Index-Nutzungsstatistiken (meistgenutzte zuerst)
            cur.execute("""
                SELECT 
                    indexrelname as index_name,
                    idx_scan as scans,
                    idx_tup_read as tuples_read,
                    idx_tup_fetch as tuples_fetched
                FROM pg_stat_user_indexes 
                WHERE idx_scan > 0
                ORDER BY idx_scan DESC
                LIMIT 10
            """)
            index_usage = cur.fetchall()
            
            # Verbindungsstatistiken für aktuele Datenbank
            cur.execute("""
                SELECT 
                    COUNT(*) as total_connections,
                    COUNT(*) FILTER (WHERE state = 'active') as active_connections,
                    COUNT(*) FILTER (WHERE state = 'idle') as idle_connections
                FROM pg_stat_activity 
                WHERE datname = current_database()
            """)
            connection_stats = cur.fetchone()
            
            cur.close()
            
            return {
                "table_sizes": [dict(row) for row in table_sizes],      # Tabellengröße
                "index_usage": [dict(row) for row in index_usage],      # Index-Performance
                "connections": dict(connection_stats)                   # Verbindungs-Stats
            }
            
    except Exception as e:
        logger.error(f"❌ Error getting database health: {e}")
        return {}

# Hilfsfunktionen für Wartung, Suche und Datenverarbeitung

def extract_keywords_from_filename(filename: str) -> List[str]:
    """
    Extrahiert durchsuchbare Schlüsselwörter aus Dateinamen
    
    Implementiert intelligente Keyword-Extraktion:
    - Entfernt Dateierweiterung
    - Trennt an verschiedenen Separatoren
    - Filtert kurze Wörter und Stopwords
    - Normalisiert zu Kleinbuchstaben
    
    Args:
        filename: Ursprünglicher Dateiname
    
    Returns:
        List[str]: Liste von Suchschlüsselwörtern
    """
    import re
    
    # Entferne Dateierweiterung
    name_without_ext = os.path.splitext(filename)[0]
    
    # Trenne an verschiedenen Separatoren und bereinige
    keywords = re.split(r'[_\-\s\.\(\)\[\]]+', name_without_ext.lower())
    
    # Filtere kurze Keywords und häufige Wörter
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'pdf', 'doc', 'file'}
    keywords = [kw.strip() for kw in keywords if len(kw.strip()) > 2 and kw.strip() not in stop_words]
    
    return keywords

def cleanup_old_logs(days: int = 90) -> int:
    """
    Bereinigt alte Query-Logs um Tabellen-Bloat zu verhindern
    
    Löscht Query-Logs älter als die angegebene Anzahl Tage:
    - Verhindert unkontrolliertes Wachstum der query_logs Tabelle
    - Bewahrt Performance-relevante Daten auf
    - Gibt Anzahl gelöschter Einträge zurück
    
    Args:
        days: Aufbewahrungszeit in Tagen (Standard: 90 Tage)
    
    Returns:
        int: Anzahl gelöschter Log-Einträge
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Lösche alte Logs basierend auf created_at
            cur.execute("""
                DELETE FROM query_logs 
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days,))
            
            deleted_count = cur.rowcount
            cur.close()
            
            logger.info(f"✅ Cleaned up {deleted_count} old query logs")
            return deleted_count
            
    except Exception as e:
        logger.error(f"❌ Error cleaning up logs: {e}")
        return 0

def recompute_collection_stats() -> bool:
    """
    Berechnet Collection-Statistiken für Genauigkeit neu
    
    Aktualisiert ChromaDB-Collection-Statistiken:
    - Dokument-Anzahl basierend auf aktuellen Chunks
    - Gesamtgröße aller Text-Inhalte
    - Durchschnittliche Chunk-Länge
    - Last-Updated-Zeitstempel
    
    Returns:
        bool: True bei erfolgreicher Neuberechnung
    """
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Aktualisiere alle Collection-Statistiken basierend auf aktuellen Daten
            cur.execute("""
                UPDATE chroma_collections cc
                SET 
                    document_count = (
                        SELECT COUNT(*) 
                        FROM chunk_collections ccm 
                        WHERE ccm.collection_name = cc.collection_name
                    ),
                    total_size_bytes = (
                        SELECT COALESCE(SUM(LENGTH(pc.text)), 0) 
                        FROM chunk_collections ccm 
                        JOIN pdf_chunks pc ON ccm.chunk_id = pc.id 
                        WHERE ccm.collection_name = cc.collection_name
                    ),
                    avg_chunk_length = (
                        SELECT COALESCE(AVG(pc.text_length), 0) 
                        FROM chunk_collections ccm 
                        JOIN pdf_chunks pc ON ccm.chunk_id = pc.id 
                        WHERE ccm.collection_name = cc.collection_name
                    ),
                    last_updated = NOW()
            """)
            
            cur.close()
            logger.info("✅ Collection statistics recomputed")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error recomputing collection stats: {e}")
        return False

def database_maintenance() -> Dict[str, str]:
    """
    Führt umfassende Datenbank-Wartung aus
    
    Implementiert vollständige Wartungsroutine:
    1. Bereinigt alte Query-Logs (90+ Tage)
    2. Berechnet Collection-Statistiken neu
    3. Aktualisiert Tabellen-Statistiken für Query-Planner
    4. Prüft auf ungenutzte Indizes (optional)
    
    Returns:
        Dict[str, str]: Wartungsergebnisse und Statusmeldungen
    """
    results = {}
    
    try:
        # Schritt 1: Bereinige alte Logs
        deleted_logs = cleanup_old_logs(90)
        results['cleanup_logs'] = f"{deleted_logs} old query logs deleted"
        
        # Schritt 2: Berechne Statistiken neu
        recompute_collection_stats()
        results['recompute_stats'] = "Collection statistics updated"
        
        # Schritt 3: Analysiere Tabellen für Query-Planner
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            # Aktualisiere Statistiken für alle wichtigen Tabellen
            tables = ['uploaded_files', 'pdf_chunks', 'chunk_collections', 'query_logs']
            for table in tables:
                cur.execute(f"ANALYZE {table}")
            
            cur.close()
        
        results['analyze_tables'] = "Table statistics refreshed"
        
        # Schritt 4: Wartung abgeschlossen
        results['maintenance_complete'] = "Database maintenance finished successfully"
        
        logger.info("✅ Database maintenance completed")
        return results
        
    except Exception as e:
        logger.error(f"❌ Database maintenance failed: {e}")
        results['error'] = str(e)
        return results

# Rückwärtskompatibilität zu älteren Code-Versionen
# während sie moderne, optimierte Implementierungen verwenden

def get_conn():
    """
    Legacy-Funktion - verwende db_manager.get_connection() stattdessen
    
    DEPRECATED: Diese Funktion ist veraltet und sollte nicht mehr verwendet werden
    Neue Code sollte den DatabaseManager verwenden für bessere Performance
    """
    logger.warning("⚠️ Using deprecated get_conn(). Use db_manager.get_connection() instead")
    return psycopg2.connect(**get_connection_params())

def get_chunks_by_file(file_name: str) -> List[Dict[str, Any]]:
    """
    Legacy-Funktion - aktualisiert für optimierte Abfragen
    
    DEPRECATED: Verwende get_chunks_by_file_id() für bessere Performance
    Diese Funktion konvertiert Dateiname zu ID und nutzt moderne Implementierung
    
    Args:
        file_name: Name der Datei (langsame Suche)
    
    Returns:
        List[Dict[str, Any]]: Chunk-Daten als Dictionary-Liste
    """
    logger.warning("⚠️ Using deprecated get_chunks_by_file. Use get_chunks_by_file_id instead")
    
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Finde file_id zuerst (langsam!)
            cur.execute("SELECT id FROM uploaded_files WHERE file_name = %s AND status != 'deleted'", (file_name,))
            file_result = cur.fetchone()
            
            if not file_result:
                return []
            
            file_id = file_result['id']
            
            # Nutze optimierte Chunk-Abfrage mit Preview
            cur.execute("""
                SELECT id, file_name, file_hash, page_number, chunk_index, 
                       LEFT(text, 200) as text_preview,     -- Nur Preview für Performance
                       text_length,
                       chunk_quality_score,
                       created_at
                FROM pdf_chunks 
                WHERE file_id = %s 
                ORDER BY page_number, chunk_index
            """, (file_id,))
            
            results = cur.fetchall()
            cur.close()
            
            return [dict(row) for row in results]
            
    except Exception as e:
        logger.error(f"❌ Error getting chunks by file name: {e}")
        return []

def get_all_files() -> List[Dict[str, Any]]:
    """
    Legacy-Funktion - gibt Dictionary-Format für Kompatibilität zurück
    
    Konvertiert moderne FileInfo-Objekte zu Legacy-Dictionary-Format
    für Rückwärtskompatibilität mit älteren Code-Teilen
    """
    files = get_all_uploaded_files()
    return [file_info.to_dict() for file_info in files]

def delete_file_chunks(file_name: str, file_hash: str = None) -> bool:
    """
    Legacy-Funktion - konvertiert zu neuem Soft-Delete-Ansatz
    
    DEPRECATED: Diese Funktion nutzt veraltete Parameter-Struktur
    Moderne Code sollte delete_uploaded_file() mit file_id verwenden
    
    Args:
        file_name: Dateiname (ineffizient)
        file_hash: Optional SHA256-Hash (effizienter)
    
    Returns:
        bool: True bei erfolgreichem Soft-Delete
    """
    logger.warning("⚠️ Using deprecated delete_file_chunks")
    
    try:
        with db_manager.get_connection() as conn:
            cur = conn.cursor()
            
            if file_hash:
                # Finde Datei per Hash und lösche soft
                file_info = get_file_by_hash(file_hash)
                if file_info:
                    return delete_uploaded_file(file_info.id)
            else:
                # Finde Datei per Name und lösche soft (langsam!)
                cur.execute("SELECT id FROM uploaded_files WHERE file_name = %s", (file_name,))
                result = cur.fetchone()
                if result:
                    return delete_uploaded_file(result[0])
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in legacy delete function: {e}")
        return False

def check_file_exists(file_name: str, file_hash: str) -> bool:
    """
    Legacy-Funktion - aktualisiert für neues Schema
    
    Prüft Datei-Existenz basierend auf Hash (nicht Name für Performance)
    """
    file_info = get_file_by_hash(file_hash)
    return file_info is not None and file_info.status != FileStatus.DELETED

def get_database_stats() -> Dict[str, Any]:
    """
    Legacy-Funktion - gibt Basis-Statistiken zurück
    
    Konvertiert moderne SystemMetrics zu Legacy-Format
    für Rückwärtskompatibilität
    """
    metrics = get_system_performance()
    if not metrics:
        return {}
    
    return {
        "total_chunks": metrics.total_chunks,
        "total_files": metrics.total_files,
        "unique_files": metrics.total_files,     # Legacy-Kompatibilität
        "average_chunk_length": 0,               # Würde separate Abfrage benötigen
        "total_text_length": 0                   # Würde separate Abfrage benötigen
    }

# Initialisierung

def init_enhanced_tables():
    """
    Initialisiert erweiterte Tabellen - jetzt über init_extended.sql behandelt
    
    Diese Funktion ist hauptsächlich für Legacy-Kompatibilität
    Die eigentliche Tabellen-Initialisierung erfolgt über das SQL-Skript
    """
    logger.info("✅ Enhanced tables should be initialized via init_extended.sql")
    return True

# Teste Verbindung beim Modul-Import für sofortiges Feedback
try:
    connection_ok, connection_msg = test_db_connection()
    if connection_ok:
        logger.info(f"✅ Database module initialized: {connection_msg}")
    else:
        logger.error(f"❌ Database connection failed: {connection_msg}")
except Exception as e:
    logger.error(f"❌ Database module initialization failed: {e}")

# Cleanup-Funktion für sauberes Herunterfahren
import atexit

def cleanup_connections():
    """
    Bereinigt Datenbankverbindungen beim Exit
    
    Wird automatisch aufgerufen wenn Python-Prozess beendet wird:
    - Schließt alle Connection-Pool-Verbindungen
    - Verhindert "connection leaked" Warnungen
    - Ermöglicht sauberen PostgreSQL-Shutdown
    """
    try:
        db_manager.close_all_connections()
        logger.info("✅ Database connections cleaned up")
    except Exception as e:
        logger.error(f"❌ Error cleaning up connections: {e}")

# Cleanup-Funktion für automatisches Aufrufen beim Exit
atexit.register(cleanup_connections)