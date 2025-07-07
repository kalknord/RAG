# =====================================
# FILE MANAGER SERVICE - ZENTRALE DATEIVERWALTUNG
# =====================================

import os
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from services.db import (
    FileStatus, DocumentType, ProcessingMethod,     
    FileInfo, ChunkInfo, SystemMetrics,             
    insert_chunk_metadata, log_query, get_system_performance,  
    search_chunks_fulltext, get_query_analytics, get_database_health,
    database_maintenance, test_db_connection, get_connection_params,
    db_manager                                      
)
from services.document_parser import (
    parse_document_with_embeddings,                 
    check_parser_connection,                        
    get_supported_formats as parser_get_formats     
)
from services.retrieval import add_documents_to_chroma, remove_documents_from_chroma
from utils.advanced_chunking import process_parsed_chunks, get_chunk_statistics
from services.hashing import calculate_file_hash
import psycopg2
import psycopg2.extras

# Logger für strukturierte Protokollierung
logger = logging.getLogger(__name__)

class FileManager:
    """
    ZENTRALE DATEIVERWALTUNG für RAG-Systeme in produktiven Umgebungen.
    
    Diese Klasse implementiert eine kompatible FileManager-Version, die mit
    bestehenden Database-Schemas arbeitet und Enterprise-Features bietet:
    
    """
    
    def __init__(self):
        """
        Initialisiert FileManager mit Lazy-Loading für Service-Dependencies.
        
        LAZY LOADING RATIONALE:
        Unterstützte Dateiformate werden zur Laufzeit vom Parser-Service abgerufen,
        nicht zur Compile-Zeit festgelegt. Dies ermöglicht:
        - Hot-Updates der Format-Unterstützung ohne Client-Neustart
        - Graceful Degradation bei Service-Ausfällen
        - Service-Discovery für dynamische Microservice-Umgebungen
        """
        self.supported_formats = None           # Cache für unterstützte Dateiformate
        self._formats_initialized = False       # Flag für Lazy-Loading-Status
    
    async def _initialize_formats(self):
        """
        Initialisiert unterstützte Dateiformate vom Parser-Service.
        
        Diese Funktion implementiert Service-Discovery für Dateiformat-Capabilities
        mit Fallback-Strategien für hohe Verfügbarkeit:
        
        FEHLERBEHANDLUNG:
        1. Erfolgreiche Service-Abfrage ? Vollständige Format-Liste
        2. Service-Ausfall ? Fallback auf Basis-Formate  
        3. Netzwerk-Timeout ? Cached Formate (falls verfügbar)
        
        CACHING-STRATEGIE:
        Formate werden einmalig geladen und für die Session gecacht,
        da sich unterstützte Formate selten ändern.
        """
        # Verhindert mehrfache Initialisierung in Concurrent-Szenarien
        if self._formats_initialized:
            return
            
        try:
            # Service-Discovery: Dynamische Format-Abfrage vom Parser
            self.supported_formats = await parser_get_formats()
            self._formats_initialized = True
            logger.info(f"? Loaded {len(self.supported_formats)} supported formats")
        except Exception as e:
            # FALLBACK-STRATEGIE: Basis-Formate bei Service-Ausfall
            logger.error(f"? Failed to load supported formats: {e}")
            
            # Kritische Basis-Formate die immer verfügbar sein müssen
            self.supported_formats = {
                '.pdf': 'PDF Document',           # Primär-Format für Geschäftsdokumente
                '.txt': 'Text Document',          # Plain Text für Notizen/Scripts
                '.docx': 'Word Document',         # Microsoft Word (moderne Version)
                '.doc': 'Word Document (Legacy)'  # Microsoft Word (Legacy-Support)
            }
            self._formats_initialized = True
    
    async def _ensure_formats_loaded(self):
        """
        Stellt sicher dass Dateiformate vor Verwendung geladen sind.
        
        THREAD-SAFETY:
        Diese Funktion ist idempotent und kann sicher von mehreren
        Async-Tasks parallel aufgerufen werden.
        """
        if not self._formats_initialized:
            await self._initialize_formats()
    
    def is_supported_file_type(self, filename: str) -> bool:
        """
        Prüft ob Dateityp unterstützt wird - synchrone Fallback-Version.
        
        Parameter:
            filename: Dateiname mit Erweiterung für Typ-Erkennung
            
        Rückgabe:
            bool: True wenn Dateityp unterstützt wird, False sonst
        """
        if not self.supported_formats:
            # EMERGENCY FALLBACK: Basis-Formate wenn Service nicht verfügbar
            fallback_formats = {'.pdf', '.txt', '.docx', '.doc', '.pptx', '.xlsx'}
            file_ext = os.path.splitext(filename.lower())[1]
            return file_ext in fallback_formats
        
        # STANDARD PATH: Validierung gegen vollständige Format-Liste
        file_ext = os.path.splitext(filename.lower())[1]
        return file_ext in self.supported_formats
    
    async def register_file(self, file_info: Dict[str, Any]) -> Optional[int]:
        """
        Registriert neue Datei mit bestehender Database-Struktur.
        
        Diese Funktion implementiert den kritischen Pfad für Datei-Aufnahme
        ins RAG-System mit umfassender Validierung und Fehlerbehandlung:
        
        VERARBEITUNGSSCHRITTE:
        1. Format-Validierung gegen Service-Discovery
        2. SHA256-Hash-basierte Duplikat-Erkennung  
        3. Soft-deleted Files: Reaktivierung statt Neuanlage
        4. Document-Type-Mapping für Database-ENUM-Compliance
        5. Transaktionale Database-Insertion mit Rollback-Fähigkeit
        
        DUPLIKAT-HANDLING:
        - Aktive Datei gefunden ? Fehler, keine Duplikate
        - Soft-deleted Datei ? Reaktivierung mit Status "uploaded"
        - Neue Datei ? Normale Registrierung
        
        Parameter:
            file_info: Dictionary mit Datei-Metadaten (name, hash, path, size, etc.)
            
        Rückgabe:
            Optional[int]: Database-ID der registrierten Datei oder None bei Fehlern
        """
        try:
            # VORAUSSETZUNG: Sicherstellen dass Formate geladen sind
            await self._ensure_formats_loaded()
            
            # DUPLIKAT-PRÜFUNG: SHA256-Hash als eindeutige Identifikation
            existing_file = await self.get_file_by_hash(file_info["file_hash"])
            
            if existing_file:
                if existing_file.get("status") == "deleted":
                    # RECOVERY-SZENARIO: Reaktivierung soft-gelöschter Dateien
                    logger.info(f"?? Reactivating soft-deleted file: {existing_file['file_name']}")
                    await self._update_file_status_direct(
                        existing_file["id"], 
                        "uploaded",  # Setze Status zurück auf "uploaded" für Neuverarbeitung
                        error_message=None  # Lösche vorherige Fehler-Messages
                    )
                    return existing_file["id"]
                else:
                    # KONFLIKT: Datei bereits aktiv im System
                    logger.warning(f"?? File already exists: {existing_file['file_name']} (status: {existing_file.get('status')})")
                    return None
            
            # DOCUMENT-TYPE-MAPPING: Dateiendung ? Database-ENUM-Werte
            file_ext = file_info.get("file_extension", "").lower()
            
            # Mapping-Tabelle für unterstützte Dokumenttypen
            # WICHTIG: Muss synchron mit Database-ENUM gehalten werden
            document_type_mapping = {
                '.pdf': 'pdf',                    # PDF-Dokumente (Primärformat)
                '.docx': 'word',                  # Microsoft Word (moderne Version)
                '.doc': 'word',                   # Microsoft Word (Legacy)
                '.pptx': 'powerpoint',            # Microsoft PowerPoint (moderne Version)
                '.ppt': 'powerpoint',             # Microsoft PowerPoint (Legacy)
                '.xlsx': 'excel',                 # Microsoft Excel (moderne Version)
                '.xls': 'excel',                  # Microsoft Excel (Legacy)
                '.txt': 'text',                   # Plain Text Files
                '.md': 'markdown',                # Markdown-Dokumentation
                '.html': 'html',                  # HTML-Seiten
                '.csv': 'csv'                     # Comma-Separated Values
            }
            
            # Fallback auf 'text' für unbekannte Dateitypen
            document_type = document_type_mapping.get(file_ext, 'text')
            
            # TRANSAKTIONALE DATABASE-INSERTION
            # Verwendet direktes SQL für Kompatibilität mit bestehender Database
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                # ATOMARE INSERTION: Alle Felder in einer Transaktion
                cur.execute("""
                    INSERT INTO uploaded_files 
                    (file_name, file_hash, file_path, file_size, file_extension, 
                     document_type, status, chunk_count, processing_method, upload_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    file_info["file_name"],       # Original-Dateiname
                    file_info["file_hash"],       # SHA256-Hash für Eindeutigkeit
                    file_info["file_path"],       # Absoluter Pfad im Filesystem
                    file_info["file_size"],       # Dateigröße in Bytes
                    file_ext,                     # Normalisierte Dateiendung
                    document_type,                # Gemappter ENUM-Wert
                    "uploaded",                   # Initial-Status für Verarbeitung
                    0,                           # Chunk-Count initial 0
                    "hybrid",                    # Standard-Verarbeitungsmethode
                    datetime.now()               # Zeitstempel für Upload
                ))
                
                # RÜCKGABE-WERT: Auto-generierte Database-ID
                file_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
            
            # ERFOLGS-LOGGING: Strukturierte Protokollierung für Monitoring
            if file_id:
                logger.info(f"? Registered file: {file_info['file_name']} (ID: {file_id})")
                return file_id
            else:
                logger.error(f"? Failed to register file: {file_info['file_name']}")
                return None
                
        except psycopg2.IntegrityError as e:
            # CONSTRAINT-VERLETZUNGEN: Spezifische Behandlung für Database-Konflikte
            if "duplicate key" in str(e):
                # Hash-Collision oder Race-Condition bei parallelen Uploads
                logger.warning(f"?? File with hash already exists: {file_info['file_name']}")
                return None
            else:
                # Andere Integritäts-Verletzungen (Foreign Keys, Check Constraints)
                logger.error(f"? Database integrity error: {e}")
                return None
        except Exception as e:
            # CATCH-ALL: Unerwartete Fehler mit vollständiger Kontextinformation
            logger.error(f"? Error registering file {file_info.get('file_name', 'unknown')}: {e}")
            return None
    
    async def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Holt Datei-Informationen anhand des SHA256-Hash-Wertes.
        
        Diese Funktion implementiert die primäre Duplikat-Erkennungslogik
        für das RAG-System. SHA256-Hashes bieten:
        - Kryptographische Eindeutigkeit (2^256 mögliche Werte)
        - Deterministisches Verhalten (gleicher Inhalt = gleicher Hash)
        - Schutz vor Hash-Kollisionen durch 256-Bit-Länge
        
        Parameter:
            file_hash: SHA256-Hash als Hex-String (64 Zeichen)
            
        Rückgabe:
            Optional[Dict]: Datei-Informationen oder None wenn nicht gefunden
        """
        try:
            # CONNECTION POOL: Automatisches Connection-Management
            with db_manager.get_connection() as conn:
                # REAL DICT CURSOR: Für strukturierte Ergebnisse statt Tupel
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # OPTIMIERTE QUERY: Index-optimiert mit neuester Version zuerst
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE file_hash = %s
                    ORDER BY upload_date DESC
                    LIMIT 1
                """, (file_hash,))
                
                result = cur.fetchone()
                cur.close()
                
                if result:
                    # DATENKONVERTIERUNG: PostgreSQL ? Python Dictionary
                    result_dict = dict(result)
                    
                    # DATETIME-SERIALISIERUNG: PostgreSQL Timestamps ? ISO Strings
                    # Wichtig für JSON-Kompatibilität und Frontend-Konsumierung
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    return result_dict
                
                return None
                
        except Exception as e:
            # FEHLER-PROTOKOLLIERUNG: Wichtig für Database-Monitoring
            logger.error(f"? Error getting file by hash: {e}")
            return None
    
    async def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """
        Holt Datei-Informationen anhand der eindeutigen Database-ID.
        
        Diese Funktion wird für direkte Datei-Zugriffe verwendet,
        wenn die ID bereits bekannt ist (z.B. aus vorherigen Operationen).
        
        VERWENDUNG:
        - Status-Updates nach Verarbeitung
        - Chunk-Retrieval für bestimmte Dateien
        - Administrative Operationen
        
        Parameter:
            file_id: Eindeutige Database-ID (Primary Key)
            
        Rückgabe:
            Optional[Dict]: Datei-Informationen oder None wenn nicht gefunden
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # EINFACHE ID-QUERY: Primary Key Lookup (sehr schnell)
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE id = %s
                """, (file_id,))
                
                result = cur.fetchone()
                cur.close()
                
                if result:
                    result_dict = dict(result)
                    
                    # DATETIME-KONVERTIERUNG: Konsistent mit get_file_by_hash()
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    return result_dict
                
                return None
                
        except Exception as e:
            logger.error(f"? Error getting file by ID {file_id}: {e}")
            return None
    
    async def get_all_files(self) -> List[Dict[str, Any]]:
        """
        Holt alle Dateien für Dashboard und Verwaltungs-Interfaces.
        
        Diese Funktion implementiert eine Performance-optimierte Abfrage
        für System-Überblicke und Administrative Dashboards:
        
        VERWENDUNG:
        - System-Dashboard mit Datei-Übersicht
        - Administrative Verwaltungs-Interfaces
        - Bulk-Operationen und Batch-Processing
        
        Rückgabe:
            List[Dict]: Liste aller Datei-Informationen, sortiert nach Upload-Datum
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # VOLLSTÄNDIGE DATEI-LISTE: Sortiert nach Aktualität
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    ORDER BY upload_date DESC
                """)
                
                results = cur.fetchall()
                cur.close()
                
                # BATCH-KONVERTIERUNG: Effiziente Verarbeitung großer Ergebnis-Sets
                converted_results = []
                for result in results:
                    result_dict = dict(result)
                    
                    # KONSISTENTE DATETIME-BEHANDLUNG: Alle Timestamps ? ISO Format
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    converted_results.append(result_dict)
                
                return converted_results
                
        except Exception as e:
            logger.error(f"? Error getting all files: {e}")
            return []
    
    async def get_file_chunks(self, file_id: int, quality_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Holt Text-Chunks für eine spezifische Datei mit Qualitäts-Filterung.
        
        Diese Funktion ermöglicht den Zugriff auf die verarbeiteten Text-Segmente
        einer Datei mit optional anwendbaren Qualitäts-Schwellwerten:
        
        QUALITÄTS-FILTERUNG:
        Der quality_threshold Parameter filtert Chunks basierend auf dem
        chunk_quality_score, der von der ML-Pipeline berechnet wird:
        - 0.0 = Alle Chunks (Standard)
        - 0.5 = Mittlere Qualität und höher
        - 0.8 = Nur hochqualitative Chunks
        
        VERWENDUNG:
        - Debugging der Chunk-Qualität
        - Quality-Assurance für ML-Pipeline
        - Selective Retrieval für bessere Suchergebnisse
        
        Parameter:
            file_id: Database-ID der Ziel-Datei
            quality_threshold: Minimaler Qualitäts-Score (0.0-1.0)
            
        Rückgabe:
            List[Dict]: Gefilterte Liste der Text-Chunks mit Metadaten
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # QUALITÄTS-GEFILTERTE CHUNK-ABFRAGE
                # WICHTIG: Verwendet nur pdf_chunks table (chunk_metadata existiert nicht)
                cur.execute("""
                    SELECT * FROM pdf_chunks 
                    WHERE file_id = %s 
                    AND chunk_quality_score >= %s
                    ORDER BY chunk_index
                """, (file_id, quality_threshold))
                
                results = cur.fetchall()
                cur.close()
                
                # DIREKTE DICTIONARY-KONVERTIERUNG: Für JSON-Serialisierung
                return [dict(result) for result in results]
                
        except Exception as e:
            logger.error(f"? Error getting chunks for file {file_id}: {e}")
            return []
    
    async def _update_file_status_direct(self, file_id: int, status: str, 
                                       chunk_count: int = None, 
                                       processing_duration_ms: int = None,
                                       error_message: str = None):
        """
        Aktualisiert Datei-Status mit direktem SQL für maximale Performance.
        
        Diese interne Funktion implementiert atomische Status-Updates mit
        dynamischem Query-Building für optimale Database-Performance:    
        
        Parameter:
            file_id: Database-ID der zu aktualisierenden Datei
            status: Neuer Status (ENUM-Wert)
            chunk_count: Optionale Anzahl verarbeiteter Chunks
            processing_duration_ms: Optionale Verarbeitungszeit in Millisekunden
            error_message: Optionale Fehlermeldung bei Status "error"
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                # DYNAMISCHES QUERY-BUILDING: Nur geänderte Felder aktualisieren
                update_fields = ["status = %s"]
                params = [status]
                
                # OPTIONALE FELDER: Nur hinzufügen wenn Werte vorhanden
                if chunk_count is not None:
                    update_fields.append("chunk_count = %s")
                    params.append(chunk_count)
                
                if processing_duration_ms is not None:
                    update_fields.append("processing_duration_ms = %s")
                    params.append(processing_duration_ms)
                
                # ERROR-MESSAGE-HANDLING: Setzen oder löschen basierend auf Status
                if error_message is not None:
                    update_fields.append("error_message = %s")
                    params.append(error_message)
                elif status != "error":
                    # Lösche error_message bei erfolgreichen Status-Wechseln
                    update_fields.append("error_message = NULL")
                
                # STATUS-SPEZIFISCHE LOGIK: last_chunked für erfolgreiche Verarbeitung
                if status == "chunked":
                    update_fields.append("last_chunked = %s")
                    params.append(datetime.now())
                
                # WHERE-CLAUSE: Target-Datei
                params.append(file_id)
                
                # FINALE QUERY-KONSTRUKTION: Dynamisch gebuildet
                query = f"""
                    UPDATE uploaded_files 
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                """
                
                # ATOMARE AUSFÜHRUNG: Single Transaction
                cur.execute(query, params)
                affected_rows = cur.rowcount
                conn.commit()
                cur.close()
                
                # ERFOLGS-VALIDIERUNG: Prüfe ob Datei existiert
                if affected_rows > 0:
                    logger.info(f"? Updated file {file_id} status to {status}")
                    return True
                else:
                    logger.error(f"? No file found with ID {file_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"? Error updating file {file_id} status: {e}")
            return False
    
    async def update_file_status(self, file_id: int, status: str, 
                               chunk_count: int = None, 
                               processing_duration_ms: int = None,
                               error_message: str = None):
        """
        Öffentliche Methode für Status-Updates mit identischer Signatur.
        
        """
        return await self._update_file_status_direct(
            file_id, status, chunk_count, processing_duration_ms, error_message
        )
    
    async def delete_file(self, file_id: int, permanent: bool = False) -> bool:
        """
        Löscht Datei mit wählbarer Soft/Hard-Delete-Strategie.
        
        Diese Funktion implementiert zwei Lösch-Modi für verschiedene
        Compliance- und Operational-Anforderungen:
        
        SOFT DELETE (Standard):
        - Status ? "deleted" in Database
        - Physische Datei bleibt erhalten
        - Vector-Database-Einträge bleiben erhalten
        - Vollständige Recovery möglich
        
        HARD DELETE (permanent=True):
        - Komplette Entfernung aus PostgreSQL (CASCADE)
        - Physische Datei-Löschung im Filesystem
        - Vector-Database-Cleanup in ChromaDB
        - Unwiderruflicher Verlust aller Daten
        
        Parameter:
            file_id: Database-ID der zu löschenden Datei
            permanent: False = Soft Delete, True = Hard Delete
            
        Rückgabe:
            bool: True bei erfolgreichem Löschen, False bei Fehlern
        """
        try:
            if permanent:
                # HARD DELETE: Vollständige Entfernung aller Daten-Artefakte
                
                # SCHRITT 1: Datei-Informationen für Cleanup-Operationen
                file_info = await self.get_file_by_id(file_id)
                if not file_info:
                    return False
                
                # SCHRITT 2: Vector-Database-Cleanup
                try:
                    file_hash = file_info["file_hash"]
                    remove_documents_from_chroma(file_hash)
                    logger.info(f"??? Removed file {file_id} from ChromaDB")
                except Exception as e:
                    # NON-CRITICAL: Vector-DB-Fehler stoppen nicht den Löschprozess
                    logger.warning(f"?? Failed to remove from ChromaDB: {e}")
                
                # SCHRITT 3: Physische Datei-Entfernung
                try:
                    if file_info.get("file_path") and os.path.exists(file_info["file_path"]):
                        os.remove(file_info["file_path"])
                        logger.info(f"??? Removed physical file: {file_info['file_path']}")
                except Exception as e:
                    # NON-CRITICAL: Filesystem-Fehler sind nicht fatal
                    logger.warning(f"?? Failed to remove physical file: {e}")
                
                # SCHRITT 4: Database-Deletion mit CASCADE
                with db_manager.get_connection() as conn:
                    cur = conn.cursor()
                    
                    # Lösche Chunks zuerst (nur pdf_chunks table existiert)
                    cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                    
                    # Lösche Hauptdatei-Eintrag
                    cur.execute("DELETE FROM uploaded_files WHERE id = %s", (file_id,))
                    
                    conn.commit()
                    cur.close()
                
                logger.info(f"??? Permanently deleted file {file_id}")
            else:
                # SOFT DELETE: Status-Update auf "deleted"
                success = await self._update_file_status_direct(file_id, "deleted")
                if success:
                    logger.info(f"??? Soft deleted file {file_id}")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"? Error deleting file {file_id}: {e}")
            return False
    
    async def get_processing_queue(self) -> Dict[str, Any]:
        """
        Holt aktuellen Status der Verarbeitungs-Queue für Monitoring.
        
        Diese Funktion liefert detaillierte Informationen über den
        aktuellen Zustand der Dokumenten-Verarbeitungs-Pipeline:
        
        QUEUE-KATEGORIEN:
        - "uploaded": Dateien warten auf Verarbeitung
        - "processing": Dateien werden gerade verarbeitet
        - "error": Dateien mit Verarbeitungsfehlern
        
        Rückgabe:
            Dict: Strukturierte Queue-Informationen mit Counts und Details
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # STATUS-COUNTS: Aggregierte Zahlen für Dashboard
                status_counts = {}
                for status in ["uploaded", "processing", "error"]:  # Korrekte ENUM-Werte verwenden
                    cur.execute("""
                        SELECT COUNT(*) as count FROM uploaded_files 
                        WHERE status = %s
                    """, (status,))
                    
                    count = cur.fetchone()["count"]
                    status_counts[status] = count
                
                # DETAILLIERTE INFORMATIONEN: Für erweiterte Monitoring-Views
                cur.execute("""
                    SELECT id, file_name, upload_date, status, error_message
                    FROM uploaded_files 
                    WHERE status IN ('uploaded', 'processing', 'error')  -- Korrekte ENUM-Werte verwenden
                    ORDER BY upload_date
                """)
                
                files = cur.fetchall()
                cur.close()
                
                # KATEGORISIERUNG: Gruppierung nach Status für strukturierte Ausgabe
                details = {"uploaded_files": [], "processing_files": [], "error_files": []}
                
                for file_info in files:
                    # BASIS-INFORMATIONEN: Für alle Status-Kategorien
                    file_data = {
                        "id": file_info["id"],
                        "filename": file_info["file_name"],
                        "upload_date": file_info["upload_date"].isoformat() if file_info["upload_date"] else None
                    }
                    
                    # STATUS-SPEZIFISCHE ZUORDNUNG: Verteilung auf Kategorien
                    if file_info["status"] == "uploaded":  # Korrekte ENUM-Werte verwenden
                        details["uploaded_files"].append(file_data)
                    elif file_info["status"] == "processing":
                        details["processing_files"].append(file_data)
                    elif file_info["status"] == "error":
                        # ERROR-SPEZIFISCH: Zusätzliche Fehler-Information
                        file_data["error"] = file_info.get("error_message", "Unknown error")
                        details["error_files"].append(file_data)
                
                # STRUKTURIERTE RÜCKGABE: Für API-Konsumierung optimiert
                return {
                    "status": "success",
                    "queue": status_counts,
                    "details": details
                }
                
        except Exception as e:
            logger.error(f"? Error getting processing queue: {e}")
            return {"status": "error", "message": str(e)}
    
    async def rechunk_all_files(self) -> Dict[str, Any]:
        """
        Comprehensive Rechunking-Prozess für alle Dateien mit Enhanced Parser.
        
        Diese Funktion implementiert eine vollständige Neuverarbeitung aller
        Dateien im System mit der aktuellen ML-Pipeline. Verwendung:
        
        ANWENDUNGSFÄLLE:
        - Update auf neue Parser-Version oder ML-Modelle
        - Qualitäts-Verbesserung durch verbesserte Algorithmen
        - Recovery nach System-Updates oder Konfigurations-Änderungen
        - Batch-Neuverarbeitung für Performance-Optimierungen
        
        VERARBEITUNGS-PIPELINE:
        1. Datei-Discovery: Alle verarbeitbaren Dateien identifizieren
        2. Cleanup: Bestehende Chunks und Vektoren entfernen
        3. Reprocessing: Enhanced Parser mit Embedding-Integration
        4. Quality-Assessment: Bewertung der Chunk-Qualität
        5. Dual-Storage: PostgreSQL + ChromaDB Synchronisation
        6. Status-Update: Aktualisierung der Datei-Status
        
        Rückgabe:
            Dict: Detaillierte Verarbeitungs-Statistiken und Fehler-Reports
        """
        try:
            logger.info("?? Starting comprehensive rechunking process")
            
            # VORAUSSETZUNG: Format-Initialisierung für alle Dateitypen
            await self._ensure_formats_loaded()
            
            # DATEI-DISCOVERY: Alle rechunkbaren Dateien identifizieren
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # RECHUNKBARE DATEIEN: Verschiedene Status-Kategorien einschließen
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE status IN ('uploaded', 'chunked', 'error') 
                    AND status != 'deleted'
                    ORDER BY upload_date
                """)  # 'uploaded' Dateien für initiale Verarbeitung einschließen
                
                rechunkable_files = cur.fetchall()
                cur.close()
            
            # ERGEBNIS-TRACKING: Strukturierte Statistik-Sammlung
            results = {
                "total_files": len(rechunkable_files),
                "processed": 0,
                "errors": 0,
                "skipped": 0,
                "details": []
            }
            
            # BATCH-VERARBEITUNG: Sequentielle Abarbeitung aller Dateien
            for file_info in rechunkable_files:
                file_id = file_info["id"]
                filename = file_info["file_name"]
                file_path = file_info.get("file_path")
                file_hash = file_info["file_hash"]
                
                try:
                    logger.info(f"?? Rechunking file {file_id}: {filename}")
                    
                    # PHYSISCHE DATEI-VALIDIERUNG: Existenz prüfen vor Verarbeitung
                    if not file_path or not os.path.exists(file_path):
                        logger.warning(f"?? Physical file not found for {filename}")
                        await self._update_file_status_direct(
                            file_id, 
                            "error",
                            error_message="Physical file not found"
                        )
                        results["errors"] += 1
                        results["details"].append({
                            "file_id": file_id,
                            "filename": filename,
                            "status": "error",
                            "error": "Physical file not found"
                        })
                        continue
                    
                    # STATUS-UPDATE: Markiere als "in Verarbeitung"
                    await self._update_file_status_direct(file_id, "processing")
                    
                    # CLEANUP-PHASE: Entferne bestehende Chunks und Vektoren
                    try:
                        # POSTGRESQL-CLEANUP: Entferne Chunks
                        with db_manager.get_connection() as conn:
                            cur = conn.cursor()
                            # Nur pdf_chunks table verwenden
                            cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                            conn.commit()
                            cur.close()
                        
                        # CHROMADB-CLEANUP: Entferne Vektoren
                        remove_documents_from_chroma(file_hash)
                        
                        logger.info(f"?? Cleared existing chunks for {filename}")
                    except Exception as e:
                        # NON-CRITICAL: Cleanup-Fehler sind nicht fatal
                        logger.warning(f"?? Error clearing existing chunks for {filename}: {e}")
                    
                    # ML-PIPELINE: Enhanced Parser mit Embedding-Integration
                    processing_start = time.time()
                    
                    # PARSING-KONFIGURATION: Optimale Parameter für Produktion
                    parse_result = await parse_document_with_embeddings(
                        file_path=file_path,
                        strategy="hi_res",              # Höchste Qualität
                        chunking_strategy="title",       # Semantische Grenzen
                        max_characters=500,             # Optimale Chunk-Größe
                        generate_embeddings=True        # Atomische Verarbeitung
                    )
                    
                    # PARSING-VALIDIERUNG: Erfolg prüfen vor weiterer Verarbeitung
                    if not parse_result["success"]:
                        error_msg = f"Parsing failed: {parse_result.get('error', 'Unknown error')}"
                        await self._update_file_status_direct(file_id, "error", error_message=error_msg)
                        results["errors"] += 1
                        results["details"].append({
                            "file_id": file_id,
                            "filename": filename,
                            "status": "error",
                            "error": error_msg
                        })
                        continue
                    
                    # CHUNK-VERARBEITUNG: Advanced Chunking mit Quality-Assessment
                    processed_chunks = process_parsed_chunks(parse_result["chunks"])
                    embeddings = parse_result["embeddings"]
                    
                    # KRITISCHE VALIDIERUNG: Chunk-Embedding Synchronisation
                    if not processed_chunks or len(processed_chunks) != len(embeddings):
                        error_msg = f"Chunk-embedding mismatch: {len(processed_chunks)} chunks vs {len(embeddings)} embeddings"
                        await self._update_file_status_direct(file_id, "error", error_message=error_msg)
                        results["errors"] += 1
                        results["details"].append({
                            "file_id": file_id,
                            "filename": filename,
                            "status": "error",
                            "error": error_msg
                        })
                        continue
                    
                    # STORAGE-VORBEREITUNG: Daten für Dual-Storage strukturieren
                    documents = []
                    metadatas = []
                    ids = []
                    chunk_embeddings = []
                    
                    # CHUNK-ITERATION: Jeder Chunk für beide Storage-Systeme vorbereiten
                    for idx, (chunk, embedding) in enumerate(zip(processed_chunks, embeddings)):
                        text = chunk["text"]
                        metadata = chunk["metadata"]
                        page = metadata.get("page_number", 1)
                        
                        # CHROMADB-DATEN: Für Vector-Database
                        documents.append(text)
                        metadatas.append({
                            "source": filename,
                            "page": page,
                            "chunk_index": idx,
                            "file_extension": file_info.get("file_extension", ""),
                            "document_type": file_info.get("document_type", ""),
                            "file_id": file_id,
                            **metadata
                        })
                        
                        chunk_id = f"{filename}_{file_hash}_{idx}"
                        ids.append(chunk_id)
                        chunk_embeddings.append(embedding)
                        
                        # POSTGRESQL-CHUNK-DATEN: Für Metadata-Database
                        chunk_data = {
                            'file_name': filename,
                            'file_hash': file_hash,
                            'page_number': page,
                            'chunk_index': idx,
                            'text': text,
                            'word_count': len(text.split()),
                            'element_type': metadata.get("element_type", "Text"),
                            'contains_table': metadata.get("contains_table", False),
                            'contains_list': metadata.get("contains_list", False),
                            'contains_image_reference': metadata.get("contains_image_reference", False),
                            'chunk_quality_score': metadata.get("text_quality_score", 0.5),
                            'readability_score': metadata.get("readability_score"),
                            'processing_method': 'hybrid',  # Korrekter ENUM-Wert verwenden
                            'ocr_confidence': metadata.get("ocr_confidence"),
                            'metadata': metadata,
                            'language_detected': 'de',
                            'section_title': metadata.get("section_title")
                        }
                        
                        # POSTGRESQL-SPEICHERUNG: Transaktionale Chunk-Insertion
                        insert_chunk_metadata(file_id, chunk_data)
                    
                    # CHROMADB-SPEICHERUNG: Batch-Upload aller Vektoren
                    success = add_documents_to_chroma(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids,
                        embeddings=chunk_embeddings
                    )
                    
                    # VECTOR-STORAGE-VALIDIERUNG: Erfolg prüfen
                    if not success:
                        error_msg = "Failed to save to vector database"
                        await self._update_file_status_direct(file_id, "error", error_message=error_msg)
                        results["errors"] += 1
                        results["details"].append({
                            "file_id": file_id,
                            "filename": filename,
                            "status": "error",
                            "error": error_msg
                        })
                        continue
                    
                    # ERFOLGS-FINALISIERUNG: Status-Update mit Timing-Informationen
                    processing_time = int((time.time() - processing_start) * 1000)
                    await self._update_file_status_direct(
                        file_id=file_id,
                        status="chunked",
                        chunk_count=len(processed_chunks),
                        processing_duration_ms=processing_time
                    )
                    
                    # ERFOLGS-TRACKING: Statistiken aktualisieren
                    results["processed"] += 1
                    results["details"].append({
                        "file_id": file_id,
                        "filename": filename,
                        "status": "success",
                        "chunks": len(processed_chunks),
                        "processing_time_ms": processing_time
                    })
                    
                    logger.info(f"? Successfully rechunked {filename}: {len(processed_chunks)} chunks")
                    
                except Exception as e:
                    # DATEI-SPEZIFISCHE FEHLERBEHANDLUNG: Isolierte Fehler-Recovery
                    logger.error(f"? Error rechunking file {file_id} ({filename}): {e}")
                    
                    # FEHLER-STATUS: Markiere Datei als fehlerhaft
                    await self._update_file_status_direct(
                        file_id, 
                        "error",
                        error_message=str(e)
                    )
                    
                    # FEHLER-TRACKING: Detaillierte Fehler-Dokumentation
                    results["errors"] += 1
                    results["details"].append({
                        "file_id": file_id,
                        "filename": filename,
                        "status": "error",
                        "error": str(e)
                    })
            
            # ZUSAMMENFASSUNG: Finale Statistiken für Monitoring
            logger.info(f"?? Rechunking completed: {results['processed']} processed, {results['errors']} errors")
            return results
            
        except Exception as e:
            # KRITISCHER FEHLER: Kompletter Prozess-Failure
            logger.error(f"? Rechunking process failed: {e}")
            raise Exception(f"Rechunking failed: {str(e)}")

    async def retry_failed_files(self, max_retries: int = 3) -> Dict[str, Any]:
        """
        Retry-Mechanismus für fehlgeschlagene Datei-Verarbeitungen.
        
        Diese Funktion implementiert automatische Wiederholung für Dateien
        die während der Verarbeitung Fehler hatten. Retry-Logic berücksichtigt:
        
        VERWENDUNG:
        - Automatische Recovery nach System-Problemen
        - Network-Timeout-Recovery für Service-Aufrufe
        - Resource-Constraint-Recovery (Memory, Disk)
        - Manual Admin-Intervention nach Fixes
        
        Parameter:
            max_retries: Maximale Anzahl Wiederholungen pro Datei
            
        Rückgabe:
            Dict: Retry-Statistiken und detaillierte Ergebnisse
        """
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                # RETRY-KANDIDATEN: Fehlgeschlagene Dateien unter Retry-Limit
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE status = 'error' 
                    AND COALESCE(retry_count, 0) < %s
                    ORDER BY upload_date
                """, (max_retries,))
                
                failed_files = cur.fetchall()
                cur.close()
            
            # ERGEBNIS-TRACKING: Strukturierte Retry-Statistiken
            results = {
                "total_files": len(failed_files),
                "queued": 0,
                "skipped": 0,
                "details": []
            }
            
            # RETRY-VERARBEITUNG: Jede qualifizierte Datei für Wiederholung vorbereiten
            for file_info in failed_files:
                try:
                    # STATUS-RESET: Zurück zu "uploaded" für Neuverarbeitung
                    await self._update_file_status_direct(
                        file_info["id"], 
                        "uploaded",  # Korrekter ENUM-Wert verwenden
                        error_message=None  # Lösche vorherige Fehler-Nachricht
                    )
                    
                    # ERFOLGREICHE WARTESCHLANGEN-EINREIHUNG
                    results["queued"] += 1
                    results["details"].append({
                        "file_id": file_info["id"],
                        "filename": file_info["file_name"],
                        "status": "queued_for_retry",
                        "previous_error": file_info.get("error_message", "Unknown error")
                    })
                    
                    logger.info(f"?? Queued file {file_info['id']} for retry")
                    
                except Exception as e:
                    # RETRY-FEHLER: Kann Datei nicht für Wiederholung vorbereiten
                    results["skipped"] += 1
                    results["details"].append({
                        "file_id": file_info["id"],
                        "filename": file_info["file_name"],
                        "status": "retry_failed",
                        "error": str(e)
                    })
                    logger.error(f"? Failed to queue file {file_info['id']} for retry: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"? Error retrying failed files: {e}")
            return {"status": "error", "message": str(e)}

    async def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """
        System-Bereinigung für verwaiste Dateien und Database-Optimierung.
        
        Diese Funktion implementiert umfassende System-Wartung für
        optimale Performance und Datenintegrität:
        
        CLEANUP-KATEGORIEN:
        1. Orphaned Chunks: Chunks ohne zugehörige Datei-Records
        2. Missing Files: Database-Einträge ohne physische Dateien
        3. Database Optimization: ANALYZE für Query-Planner-Updates
        
        Rückgabe:
            Dict: Detaillierte Cleanup-Statistiken und Optimierungs-Ergebnisse
        """
        try:
            # ERGEBNIS-TRACKING: Strukturierte Cleanup-Statistiken
            results = {
                "orphaned_chunks_removed": 0,
                "missing_files_cleaned": 0,
                "database_optimized": False
            }
            
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                # CLEANUP 1: Orphaned Chunks entfernen
                # Chunks ohne zugehörige uploaded_files Records
                cur.execute("""
                    DELETE FROM pdf_chunks 
                    WHERE file_id NOT IN (SELECT id FROM uploaded_files)
                """)
                results["orphaned_chunks_removed"] = cur.rowcount
                
                # CLEANUP 2: Missing Physical Files identifizieren
                cur.execute("""
                    SELECT id, file_name, file_path FROM uploaded_files 
                    WHERE status != 'deleted'
                """)
                
                files_to_check = cur.fetchall()
                missing_count = 0
                
                # PHYSISCHE DATEI-VALIDIERUNG: Existenz im Filesystem prüfen
                for file_id, file_name, file_path in files_to_check:
                    if file_path and not os.path.exists(file_path):
                        # MISSING FILE: Status auf error setzen mit descriptive message
                        cur.execute("""
                            UPDATE uploaded_files 
                            SET status = 'error', 
                                error_message = 'Physical file missing'
                            WHERE id = %s
                        """, (file_id,))
                        missing_count += 1
                        logger.warning(f"?? Marked file as error (missing): {file_name}")
                
                results["missing_files_cleaned"] = missing_count
                
                # OPTIMIZATION: Database-Statistiken aktualisieren
                # ANALYZE aktualisiert Query-Planner-Statistiken für bessere Performance
                cur.execute("ANALYZE uploaded_files")
                cur.execute("ANALYZE pdf_chunks")  # Nur pdf_chunks table existiert
                
                results["database_optimized"] = True
                
                # COMMIT: Alle Änderungen atomisch bestätigen
                conn.commit()
                cur.close()
            
            # ERFOLGS-LOGGING: Detaillierte Cleanup-Zusammenfassung
            logger.info(f"?? Cleanup completed: {results}")
            return results
            
        except Exception as e:
            # CLEANUP-FEHLER: Kritisch für System-Wartung
            logger.error(f"? Cleanup failed: {e}")
            return {"status": "error", "message": str(e)}

    async def bulk_upload_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Bulk-Upload für Enterprise-Szenarien mit Batch-Verarbeitung.
        
        Diese Funktion ermöglicht Massen-Uploads für:
        - Migration bestehender Dokument-Archive
        - Batch-Import aus Content-Management-Systemen
        - Automatisierte Datei-Synchronisation
        - Administrative Bulk-Operationen
        
        Parameter:
            file_paths: Liste absoluter Dateipfade für Bulk-Upload
            
        Rückgabe:
            Dict: Detaillierte Batch-Verarbeitungs-Statistiken
        """
        # VORAUSSETZUNG: Format-Validierung für alle Dateien
        await self._ensure_formats_loaded()
        
        # ERGEBNIS-TRACKING: Strukturierte Batch-Statistiken
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "duplicates": 0,
            "details": []
        }
        
        # BATCH-ITERATION: Sequentielle Verarbeitung für Speicher-Effizienz
        for file_path in file_paths:
            try:
                # EXISTENZ-VALIDIERUNG: Physische Datei prüfen
                if not os.path.exists(file_path):
                    results["failed"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "failed",
                        "error": "File not found"
                    })
                    continue
                
                filename = os.path.basename(file_path)
                
                # FORMAT-VALIDIERUNG: Unterstützte Dateitypen prüfen
                if not self.is_supported_file_type(filename):
                    results["failed"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "failed",
                        "error": "Unsupported file type"
                    })
                    continue
                
                # HASH-BERECHNUNG: SHA256 für Eindeutigkeit
                file_hash = calculate_file_hash(file_path)
                
                # DUPLIKAT-PRÜFUNG: Bestehende Dateien identifizieren
                existing_file = await self.get_file_by_hash(file_hash)
                if existing_file and existing_file.get("status") != "deleted":
                    results["duplicates"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "duplicate",
                        "existing_file": existing_file["file_name"]
                    })
                    continue
                
                # DATEI-INFORMATIONEN: Strukturierte Metadaten vorbereiten
                file_info = {
                    "file_name": filename,
                    "file_hash": file_hash,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "file_extension": os.path.splitext(filename.lower())[1]
                }
                
                # REGISTRIERUNG: Standard-Registrierungs-Pipeline
                file_id = await self.register_file(file_info)
                
                # ERFOLGS-TRACKING: Basierend auf Registrierungs-Ergebnis
                if file_id:
                    results["successful"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "uploaded",
                        "file_id": file_id,
                        "filename": filename
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "failed",
                        "error": "Failed to register in database"
                    })
                    
            except Exception as e:
                # DATEI-SPEZIFISCHE FEHLERBEHANDLUNG: Isolierte Fehler-Recovery
                results["failed"] += 1
                results["details"].append({
                    "file_path": file_path,
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"? Error bulk uploading {file_path}: {e}")
        
        return results

file_manager = FileManager()