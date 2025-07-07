# =====================================
# RAG-SYSTEM API BACKEND
# =====================================

import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import time
import asyncio
from typing import List, Dict, Any, Optional
import shutil
import logging
from datetime import datetime
from services.db import (
    FileStatus, DocumentType, ProcessingMethod,          
    FileInfo, ChunkInfo, SystemMetrics,                  
    insert_chunk_metadata,                               
    log_query, get_system_performance,                   
    search_chunks_fulltext, get_query_analytics, get_database_health,  
    database_maintenance, test_db_connection, get_connection_params,    
    db_manager                                          
)
from services.llm import generate_answer                 
from services.retrieval import query_chroma, add_documents_to_chroma, check_chroma_connection  
from services.document_parser import (                   
    parse_document_with_embeddings,                     
    generate_embeddings_only,                           
    check_parser_connection,                            
    get_supported_formats as parser_get_formats         
)
from utils.advanced_chunking import process_parsed_chunks, get_chunk_statistics  
from services.file_manager import file_manager          
from services.hashing import calculate_file_hash        

# Konfiguration des Logging-Systems f�r Debugging und Monitoring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Erstelle FastAPI-Anwendung mit erweiterten Metadaten und API-Dokumentation

app = FastAPI(
    title="Enhanced RAG API Backend",
    version="2",
    description="RAG system",
    docs_url="/docs",      
    redoc_url="/redoc"     
)

# Konfiguration f�r Cross-Origin Resource Sharing (CORS)
# Erm�glicht Frontend-Zugriff von verschiedenen Domains

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,     
    allow_methods=["*"],        
    allow_headers=["*"],        
)

# Upload-Verzeichnis f�r hochgeladene Dateien

UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Verzeichnis erstellen, falls es nicht existiert

# Klassen definieren die Struktur und Validierung f�r API-Requests und -Responses

class QueryRequest(BaseModel):
    """
    Request-Model f�r Suchanfragen
    Definiert Parameter f�r semantische Suche mit Qualit�tsfilterung
    """
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")
    quality_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum chunk quality score")
    file_ids: Optional[List[int]] = Field(None, description="Limit search to specific files")

class QueryResponse(BaseModel):
    """
    Response-Model f�r Suchergebnisse
    Enth�lt generierte Antwort, Quellen und Performance-Metriken
    """
    answer: str                          # Generierte Antwort vom LLM
    sources: List[Dict[str, Any]]        # Quell-Dokumente mit Metadaten
    query: str                           # Original-Suchanfrage
    response_time_ms: int                # Antwortzeit in Millisekunden
    results_count: int                   # Anzahl gefundener Ergebnisse
    quality_metrics: Dict[str, float]    # Qualit�ts- und Relevanz-Metriken

class UploadResponse(BaseModel):
    """
    Response-Model f�r Datei-Upload
    Enth�lt Verarbeitungsergebnisse und Qualit�tsmetriken
    """
    status: str                          # Verarbeitungsstatus
    message: str                         # Statusnachricht
    chunks: int                          # Anzahl erstellter Chunks
    filename: str                        # Name der verarbeiteten Datei
    file_id: int                         # Eindeutige Datei-ID
    processing_time_ms: int              # Verarbeitungszeit in Millisekunden
    quality_metrics: Dict[str, Any]      # Detaillierte Qualit�tsanalyse

class MultiUploadResponse(BaseModel):
    """
    Response-Model f�r Mehrfach-Upload
    Zusammenfassung der Batch-Verarbeitung
    """
    status: str                          # Gesamtstatus der Batch-Operation
    total_files: int                     # Gesamtanzahl der Dateien
    successful: int                      # Erfolgreich verarbeitete Dateien
    failed: int                          # Fehlgeschlagene Verarbeitungen
    duplicates: int                      # Erkannte Duplikate
    details: List[Dict[str, Any]]        # Detaillierte Ergebnisse pro Datei

class FileInfoResponse(BaseModel):
    """
    Response-Model f�r Datei-Informationen
    Vollst�ndige Metadaten einer Datei im System
    """
    id: int                              # Eindeutige Datei-ID
    file_name: str                       # Urspr�nglicher Dateiname
    file_hash: str                       # SHA256-Hash der Datei
    file_size: int                       # Dateigr��e in Bytes
    file_extension: str                  # Dateierweiterung
    document_type: str                   # Dokumenttyp (PDF, Word, etc.)
    status: str                          # Aktueller Verarbeitungsstatus
    chunk_count: int                     # Anzahl der erstellten Chunks
    upload_date: str                     # Upload-Zeitstempel
    last_chunked: Optional[str]          # Letzter Verarbeitungszeitpunkt
    error_message: Optional[str]         # Fehlermeldung bei Problemen
    processing_duration_ms: Optional[int] # Verarbeitungsdauer
    quality_metrics: Dict[str, Any]      # Qualit�ts- und Inhaltsmetriken

class SystemHealthResponse(BaseModel):
    """
    Response-Model f�r System-Gesundheitscheck
    �bersicht �ber alle Service-Komponenten
    """
    status: str                          # Gesamtsystemstatus
    services: Dict[str, Dict[str, str]]  # Status aller Microservices
    database: Dict[str, Any]             # Datenbank-Gesundheitsmetriken
    performance_metrics: Dict[str, Any]  # System-Performance-Indikatoren
    version: str                         # API-Version
    fixes_applied: List[str]             # Liste der angewendeten kritischen Fixes

class FullTextSearchResponse(BaseModel):
    """
    Response-Model f�r Volltextsuche
    Ergebnisse der datenbankbasierten Textsuche
    """
    query: str                           # Suchbegriff
    results_count: int                   # Anzahl gefundener Ergebnisse
    response_time_ms: int                # Suchzeit in Millisekunden
    results: List[Dict[str, Any]]        # Suchergebnisse mit Metadaten

class RechunkResponse(BaseModel):
    """
    Response-Model f�r Neuverarbeitung aller Dokumente
    Detaillierte Ergebnisse der Batch-Neuverarbeitung
    """
    status: str                          # Gesamtstatus der Operation
    total_files: int                     # Anzahl zu verarbeitender Dateien
    processed: int                       # Erfolgreich verarbeitete Dateien
    errors: int                          # Fehlgeschlagene Verarbeitungen
    skipped: int                         # �bersprungene Dateien
    details: List[Dict[str, Any]]        # Detaillierte Ergebnisse pro Datei

# Hilfsfunktionen f�r Request-Verarbeitung und Analytics

def get_client_ip(request: Request) -> str:
    """
    Extrahiert die Client-IP-Adresse aus dem Request
    Ber�cksichtigt Proxy-Header f�r korrekte IP-Ermittlung
    """
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # Erste IP aus der Proxy-Kette verwenden
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def generate_session_id(request: Request) -> str:
    """
    Generiert eine Session-ID f�r Analytics-Zwecke
    Basiert auf IP, User-Agent und Zeitstempel (stundenbasiert)
    """
    import hashlib
    user_agent = request.headers.get("User-Agent", "")
    client_ip = get_client_ip(request)
    timestamp = str(int(time.time() // 3600))  # Stundenbasierte Session
    
    # Erstelle eindeutigen Session-Identifier
    session_data = f"{client_ip}:{user_agent}:{timestamp}"
    return hashlib.md5(session_data.encode()).hexdigest()

async def log_request_analytics(request: Request, query: str, results_count: int, 
                               response_time_ms: int, embedding_time_ms: int = None):
    """
    Protokolliert Request-Daten f�r Analytics und Performance-Monitoring
    Speichert Abfragen, Antwortzeiten und Benutzerkontext
    """
    try:
        client_ip = get_client_ip(request)
        session_id = generate_session_id(request)
        
        # Logge Abfrage in die Analytics-Datenbank
        log_query(
            query_text=query,
            results_count=results_count,
            response_time_ms=response_time_ms,
            embedding_time_ms=embedding_time_ms,
            collection_name="rag_docs",
            user_session=session_id,
            source_ip=client_ip
        )
    except Exception as e:
        logger.error(f"? Error logging analytics: {e}")

async def process_parsed_document_safely(file_id: int, file_info: dict, parse_result: dict, filename: str):
    """
    Sichere Verarbeitung geparster Dokumente
    
    Args:
        file_id: Eindeutige Datei-ID in der Datenbank
        file_info: Metadaten der Datei
        parse_result: Ergebnis der Dokumentenverarbeitung
        filename: Name der zu verarbeitenden Datei
    
    Returns:
        Tuple[bool, Union[str, Dict]]: (Erfolg, Ergebnis oder Fehlermeldung)
    """
    try:
        logger.info(f"?? Processing parsed document for file {file_id}: {filename}")
        
        # Schritt 1: Validiere Parse-Ergebnis
        if not parse_result["success"]:
            error_msg = f"Document parsing failed: {parse_result.get('error', 'Unknown error')}"
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Schritt 2: Extrahiere Chunks und Embeddings
        raw_chunks = parse_result["chunks"]
        embeddings = parse_result["embeddings"]
        
        # Schritt 3: KRITISCHE VALIDIERUNG - Chunk-Embedding-Synchronisation
        if not raw_chunks:
            error_msg = "No chunks extracted from document"
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Stelle sicher, dass Chunk- und Embedding-Anzahl �bereinstimmt
        if len(raw_chunks) != len(embeddings):
            error_msg = f"CRITICAL: Chunk-embedding mismatch! Chunks: {len(raw_chunks)}, Embeddings: {len(embeddings)}"
            logger.error(f"? {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        logger.info(f"? Validation passed: {len(raw_chunks)} chunks, {len(embeddings)} embeddings")
        
        # Schritt 4: Sichere Chunk-Verarbeitung
        processed_chunks = []
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                processed_chunks.append(chunk)
            else:
                # Konvertiere Objekt sicher zu Dictionary
                safe_chunk = {
                    "text": str(chunk.get("text", "")) if hasattr(chunk, "get") else str(chunk),
                    "metadata": chunk.get("metadata", {}) if hasattr(chunk, "get") else {}
                }
                processed_chunks.append(safe_chunk)
        
        # Schritt 5: Erweiterte Chunk-Verarbeitung anwenden
        enhanced_chunks = process_parsed_chunks(processed_chunks)
        
        if not enhanced_chunks:
            error_msg = "No valid chunks after processing"
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Schritt 6: KRITISCHE FINAL-VALIDIERUNG
        if len(enhanced_chunks) != len(embeddings):
            error_msg = f"Chunk processing altered count: {len(enhanced_chunks)} vs {len(embeddings)} embeddings"
            logger.error(f"? {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Schritt 7: Berechne Qualit�tsmetriken
        quality_stats = get_chunk_statistics(enhanced_chunks)
        
        # Schritt 8: Bereite Daten f�r Speicherung vor
        documents = []          # Texte f�r ChromaDB
        metadatas = []          # Metadaten f�r ChromaDB
        ids = []               # IDs f�r ChromaDB
        chunk_embeddings = []   # Embeddings f�r ChromaDB
        
        # Iteriere �ber alle Chunk-Embedding-Paare
        for idx, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings)):
            text = chunk["text"]
            metadata = chunk["metadata"]
            page = metadata.get("page_number", 1)
            
            # Bereite ChromaDB-Daten vor
            documents.append(text)
            metadatas.append({
                "source": filename,
                "page": page,
                "chunk_index": idx,
                "file_extension": file_info["file_extension"],
                "document_type": file_info.get("document_type", "pdf"),
                "file_id": file_id,
                **metadata  # Merge zus�tzliche Metadaten
            })
            
            # Erstelle eindeutige Chunk-ID
            chunk_id = f"{filename}_{file_info['file_hash']}_{idx}"
            ids.append(chunk_id)
            chunk_embeddings.append(embedding)
            
            # Schritt 9: Verwende korrekte Funktionssignatur
            chunk_data = {
                'file_name': filename,
                'file_hash': file_info["file_hash"],
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
                'processing_method': 'hybrid',
                'ocr_confidence': metadata.get("ocr_confidence"),
                'metadata': metadata,
                'language_detected': 'de',
                'section_title': metadata.get("section_title")
            }
            
            # Verwende korrigierte Funktionssignatur - file_id zuerst, dann chunk_data dict
            success = insert_chunk_metadata(file_id, chunk_data)
            if not success:
                error_msg = f"Failed to insert chunk {idx} for file {file_id}"
                logger.error(f"? {error_msg}")
                await file_manager.update_file_status(file_id, "error", error_message=error_msg)
                return False, error_msg
        
        # Schritt 10: Speichere in ChromaDB (Vektordatenbank)
        success = add_documents_to_chroma(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=chunk_embeddings
        )
        
        if not success:
            error_msg = "Failed to save to vector database"
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        logger.info(f"? Successfully processed {filename}: {len(enhanced_chunks)} chunks")
        return True, {
            "chunks": len(enhanced_chunks),
            "quality_metrics": quality_stats
        }
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        logger.error(f"? Error processing file {file_id}: {e}")
        await file_manager.update_file_status(file_id, "error", error_message=error_msg)
        return False, error_msg

# Systemhealth und -status Endpunkte

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """
    Umfassender System-Gesundheitscheck
    
    Pr�ft alle kritischen Systemkomponenten:
    - Datenbankverbindung
    - ChromaDB (Vektordatenbank)
    - Enhanced Parser Service
    - Performance-Metriken
    
    Returns:
        SystemHealthResponse: Detaillierte Systemgesundheit mit Fix-Indikatoren
    """
    try:
        # Pr�fe alle Services
        chroma_ok, chroma_msg = check_chroma_connection()
        parser_ok, parser_msg = await check_parser_connection()
        db_ok, db_msg = test_db_connection()
        
        # Sammle System-Metriken
        performance_metrics = {}
        try:
            metrics = get_system_performance()
            if metrics:
                performance_metrics = {
                    "total_files": metrics.total_files,
                    "total_chunks": metrics.total_chunks,
                    "files_ready": metrics.files_ready,
                    "avg_processing_time": metrics.avg_processing_time,
                    "storage_used": metrics.total_storage
                }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
        
        # Sammle Datenbank-Gesundheitsdaten
        db_health = {}
        try:
            db_health = get_database_health()
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
        
        # Bestimme Gesamtstatus
        overall_status = "healthy" if all([chroma_ok, parser_ok, db_ok]) else "degraded"
        
        return SystemHealthResponse(
            status=overall_status,
            services={
                "database": {"status": "ok" if db_ok else "error", "message": db_msg},
                "chroma": {"status": "ok" if chroma_ok else "error", "message": chroma_msg},
                "enhanced_parser": {"status": "ok" if parser_ok else "error", "message": parser_msg},
                "api": {"status": "ok", "message": "Enhanced API with critical fixes"}
            },
            database=db_health,
            performance_metrics=performance_metrics,
            version="2.1.0-FIXED",
            fixes_applied=[
                "insert_chunk_metadata_signature_fix",     # Funktionssignatur korrigiert
                "chunk_embedding_sync_validation",         # Synchronisation validiert
                "safe_processing_function",                # Sichere Verarbeitung implementiert
                "empty_chunk_filtering",                   # Leere Chunks gefiltert
                "dimension_validation",                    # Dimensionsvalidierung hinzugef�gt
                "function_signature_compatibility"        # Funktionskompatibilit�t sichergestellt
            ]
        )
        
    except Exception as e:
        logger.error(f"? Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/system/performance")
async def get_performance_metrics():
    """
    Detaillierte System-Performance-Metriken abrufen
    
    Liefert umfassende Leistungsdaten:
    - Datei-Verarbeitungsstatistiken
    - Chunk-Anzahl und -Qualit�t
    - Speicherverbrauch
    - Verarbeitungszeiten
    - Benutzeraktivit�t
    """
    try:
        metrics = get_system_performance()
        if not metrics:
            raise HTTPException(status_code=500, detail="Could not retrieve performance metrics")
        
        return {
            "status": "success",
            "metrics": {
                "files": {
                    "pending": metrics.files_pending,        # Dateien in Warteschlange
                    "processing": metrics.files_processing,  # Aktuell in Verarbeitung
                    "ready": metrics.files_ready,           # Bereit f�r Suche
                    "error": metrics.files_error,           # Fehlgeschlagene Verarbeitungen
                    "total": metrics.total_files             # Gesamtanzahl Dateien
                },
                "chunks": {
                    "total": metrics.total_chunks            # Gesamtanzahl Text-Chunks
                },
                "storage": {
                    "total_used": metrics.total_storage      # Verwendeter Speicherplatz
                },
                "performance": {
                    "avg_processing_time_ms": metrics.avg_processing_time,  # Durchschnittliche Verarbeitungszeit
                    "max_processing_time_ms": metrics.max_processing_time,  # Maximale Verarbeitungszeit
                    "avg_file_quality": metrics.avg_file_quality            # Durchschnittliche Datequalit�t
                },
                "activity": {
                    "files_uploaded_today": metrics.files_uploaded_today,   # Heute hochgeladene Dateien
                    "files_accessed_today": metrics.files_accessed_today    # Heute zugriff auf Dateien
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"? Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Dateiupload-Endpunkte 

@app.post("/upload_document", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Upload einzelnes Dokument mit erweiterter Verarbeitung und KRITISCHEN FIXES
    
    Verarbeitungsschritte:
    1. Dateityp-Validierung
    2. Tempor�re Speicherung
    3. Hash-Berechnung und Duplikat-Erkennung
    4. Datei-Registrierung in Datenbank
    5. Parsing mit Enhanced Parser
    6. Sichere Chunk-Verarbeitung (KRITISCHER FIX)
    7. Dual-Storage (PostgreSQL + ChromaDB)
    8. Status-Update und Qualit�tsmetriken
    
    Args:
        request: FastAPI Request-Objekt f�r Analytics
        file: Hochgeladene Datei
    
    Returns:
        UploadResponse: Detaillierte Verarbeitungsergebnisse
    """
    processing_start = time.time()
    file_id = None
    file_path = None
    
    try:
        # Schritt 1: Validiere Dateityp
        supported_formats = await parser_get_formats()
        file_ext = os.path.splitext(file.filename.lower())[1]
        
        if file_ext not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type '{file_ext}'. Supported: {', '.join(supported_formats.keys())}"
            )
        
        # Schritt 2: Speichere Datei tempor�r
        temp_filename = f"temp_{int(time.time())}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Schritt 3: Berechne Datei-Hash und pr�fe auf Duplikate
        file_hash = calculate_file_hash(file_path)
        existing_file = await file_manager.get_file_by_hash(file_hash)
        
        if existing_file:
            # Bereinige tempor�re Datei
            os.remove(file_path)
            
            if existing_file["status"] == "deleted":
                # Reaktiviere soft-gel�schte Datei
                logger.info(f"?? Reactivating deleted file: {existing_file['file_name']}")
                await file_manager.update_file_status(
                    existing_file["id"], 
                    "uploaded",
                    error_message=None
                )
                
                return UploadResponse(
                    status="reactivated",
                    message=f"Reactivated previously deleted file: {existing_file['file_name']}",
                    chunks=existing_file.get("chunk_count", 0),
                    filename=existing_file["file_name"],
                    file_id=existing_file["id"],
                    processing_time_ms=0,
                    quality_metrics={"status": "reactivated"}
                )
            else:
                # Datei existiert bereits und ist aktiv
                raise HTTPException(
                    status_code=409, 
                    detail=f"File already exists: {existing_file['file_name']} (status: {existing_file['status']})"
                )
        
        # Schritt 4: Benenne zu finalem Dateinamen um
        final_file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Behandle Dateinamen-Konflikte
        counter = 1
        while os.path.exists(final_file_path):
            name, ext = os.path.splitext(file.filename)
            final_filename = f"{name}_{counter}{ext}"
            final_file_path = os.path.join(UPLOAD_DIR, final_filename)
            counter += 1
        
        os.rename(file_path, final_file_path)
        final_filename = os.path.basename(final_file_path)
        
        # Schritt 5: Registriere Datei in Datenbank
        file_info = {
            "file_name": final_filename,
            "file_hash": file_hash,
            "file_path": final_file_path,
            "file_size": len(content),
            "file_extension": file_ext
        }
        
        file_id = await file_manager.register_file(file_info)
        
        if not file_id:
            # Bereinige Datei bei Registrierungsfehlern
            if os.path.exists(final_file_path):
                os.remove(final_file_path)
            raise HTTPException(status_code=500, detail="Failed to register file in database")
        
        # Schritt 6: Starte Verarbeitung
        await file_manager.update_file_status(file_id, "processing")
        
        try:
            # Parse Dokument mit integrierten Embeddings
            parse_start = time.time()
            parse_result = await parse_document_with_embeddings(
                file_path=final_file_path,
                strategy="hi_res",                # High-Resolution Parsing
                chunking_strategy="title",       # Title-basierte Chunk-Aufteilung
                max_characters=500,              # Maximale Chunk-Gr��e
                generate_embeddings=True         # Embeddings parallel generieren
            )
            
            # Schritt 7: KRITISCH - Verwende sichere Verarbeitungsfunktion
            success, result = await process_parsed_document_safely(
                file_id=file_id,
                file_info=file_info,
                parse_result=parse_result,
                filename=final_filename
            )
            
            if not success:
                raise HTTPException(status_code=400, detail=result)
            
            # Schritt 8: Berechne Verarbeitungszeit und aktualisiere Status
            processing_time = int((time.time() - processing_start) * 1000)
            
            await file_manager.update_file_status(
                file_id=file_id,
                status="chunked",
                chunk_count=result["chunks"],
                processing_duration_ms=processing_time
            )
            
            logger.info(f"? Successfully processed {final_filename}: {result['chunks']} chunks in {processing_time}ms")
            
            return UploadResponse(
                status="success",
                message=f"Document processed successfully with enhanced metadata and analytics",
                chunks=result["chunks"],
                filename=final_filename,
                file_id=file_id,
                processing_time_ms=processing_time,
                quality_metrics=result["quality_metrics"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            # Aktualisiere Dateistatus bei Verarbeitungsfehlern
            await file_manager.update_file_status(
                file_id=file_id,
                status="error",
                error_message=str(e)
            )
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        # Erweiterte Fehlerbehandlung
        if file_id:
            await file_manager.update_file_status(
                file_id=file_id,
                status="error",
                error_message=str(e)
            )
        
        # Bereinige tempor�re Dateien bei Fehlern
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        logger.error(f"? Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/upload_multiple", response_model=MultiUploadResponse)
async def upload_multiple_documents(request: Request, files: List[UploadFile] = File(...)):
    """
    Upload mehrerer Dokumente mit Batch-Verarbeitung und erweiterten Analytics
    
    Verarbeitet mehrere Dateien gleichzeitig und liefert detaillierte Statistiken:
    - Erfolgreiche Uploads
    - Fehlgeschlagene Verarbeitungen
    - Erkannte Duplikate
    - Unsupported Dateiformate
    
    Args:
        request: FastAPI Request-Objekt
        files: Liste hochgeladener Dateien
    
    Returns:
        MultiUploadResponse: Batch-Verarbeitungsstatistiken
    """
    try:
        supported_formats = await parser_get_formats()
        
        # Initialisiere Ergebnis-Tracking
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "duplicates": 0,
            "details": []
        }
        
        # Verarbeite jede Datei einzeln
        for file in files:
            file_ext = os.path.splitext(file.filename.lower())[1]
            
            # Validiere Dateityp
            if file_ext not in supported_formats:
                results["failed"] += 1
                results["details"].append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported file type '{file_ext}'"
                })
                continue
            
            try:
                # Speichere Datei
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                file_hash = calculate_file_hash(file_path)
                
                # Pr�fe auf Duplikate
                existing_file = await file_manager.get_file_by_hash(file_hash)
                if existing_file and existing_file["status"] != "deleted":
                    os.remove(file_path)  # Bereinige Duplikat
                    results["duplicates"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "duplicate",
                        "existing_file": existing_file["file_name"]
                    })
                    continue
                
                # Registriere Datei
                file_info = {
                    "file_name": file.filename,
                    "file_hash": file_hash,
                    "file_path": file_path,
                    "file_size": len(content),
                    "file_extension": file_ext
                }
                
                file_id = await file_manager.register_file(file_info)
                
                if file_id:
                    results["successful"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "uploaded",
                        "file_id": file_id,
                        "size": len(content)
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": "Failed to register in database"
                    })
                    
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"? Error processing {file.filename}: {e}")
        
        return MultiUploadResponse(
            status="completed",
            total_files=results["total_files"],
            successful=results["successful"],
            failed=results["failed"],
            duplicates=results["duplicates"],
            details=results["details"]
        )
        
    except Exception as e:
        logger.error(f"? Bulk upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

# Dateiverwaltungs-Endpunkte

@app.get("/files", response_model=List[FileInfoResponse])
async def get_uploaded_files():
    """
    Alle hochgeladenen Dateien mit erweiterten Metadaten abrufen
    
    Liefert vollst�ndige Datei-Informationen inkl.:
    - Basis-Metadaten (Name, Gr��e, Status)
    - Verarbeitungsstatistiken
    - Qualit�tsmetriken
    - Chunk-Informationen
    - Zeitstempel
    """
    try:
        files = await file_manager.get_all_files()
        
        # Konvertiere zu Response-Format mit Qualit�tsmetriken
        response_files = []
        for file_data in files:
            # Hole Chunks f�r Qualit�tsmetriken
            chunks = await file_manager.get_file_chunks(file_data["id"])
            
            # Berechne Qualit�tsmetriken
            quality_metrics = {
                "avg_chunk_quality": sum(c.get("chunk_quality_score", 0) for c in chunks) / len(chunks) if chunks else 0,
                "total_chunks": len(chunks),
                "chunks_with_tables": sum(1 for c in chunks if c.get("contains_table", False)),
                "chunks_with_lists": sum(1 for c in chunks if c.get("contains_list", False))
            }
            
            # DATETIME-KONVERTIERUNGS-FIX - Konvertiere datetime-Objekte zu ISO-Strings
            upload_date_str = None
            if file_data.get("upload_date"):
                if isinstance(file_data["upload_date"], str):
                    upload_date_str = file_data["upload_date"]
                else:
                    upload_date_str = file_data["upload_date"].isoformat()
            
            last_chunked_str = None
            if file_data.get("last_chunked"):
                if isinstance(file_data["last_chunked"], str):
                    last_chunked_str = file_data["last_chunked"]
                else:
                    last_chunked_str = file_data["last_chunked"].isoformat()
            
            response_files.append(FileInfoResponse(
                id=file_data["id"],
                file_name=file_data["file_name"],
                file_hash=file_data["file_hash"],
                file_size=file_data["file_size"],
                file_extension=file_data["file_extension"],
                document_type=file_data["document_type"],
                status=file_data["status"],
                chunk_count=file_data["chunk_count"],
                upload_date=upload_date_str,  
                last_chunked=last_chunked_str,  
                error_message=file_data["error_message"],
                processing_duration_ms=file_data.get("processing_duration_ms"),
                quality_metrics=quality_metrics
            ))
        
        return response_files
        
    except Exception as e:
        logger.error(f"? Error getting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
async def delete_file(file_id: int, permanent: bool = False):
    """
    L�sche eine Datei (standardm��ig Soft-Delete, optional permanent)
    
    Soft-Delete:
    - Setzt Status auf 'deleted'
    - Daten bleiben f�r Recovery verf�gbar
    - Schnelle Operation
    
    Permanent-Delete:
    - Entfernt alle Datenbank-Eintr�ge
    - L�scht physische Datei
    - Entfernt Chunks aus ChromaDB
    - Nicht r�ckg�ngig machbar
    
    Args:
        file_id: Eindeutige Datei-ID
        permanent: True f�r permanente L�schung
    """
    try:
        success = await file_manager.delete_file(file_id, permanent=permanent)
        
        if success:
            delete_type = "permanently deleted" if permanent else "soft deleted"
            return {
                "status": "success", 
                "message": f"File {delete_type} successfully",
                "permanent": permanent
            }
        else:
            raise HTTPException(status_code=404, detail="File not found or already deleted")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rechunk", response_model=RechunkResponse)
async def rechunk_all_documents():
    """
    KRITISCHER FIX: Neuverarbeitung aller Dokumente mit korrekten Funktionssignaturen
    
    Verarbeitet alle vorhandenen Dokumente neu:
    1. Findet alle verarbeitbaren Dateien
    2. L�scht existierende Chunks
    3. Verwendet Enhanced Parser mit Embeddings
    4. Wendet KRITISCHE FIXES an
    5. Speichert in PostgreSQL + ChromaDB
    6. Aktualisiert Status und Metriken
    
    Kritische Verbesserungen:
    - Korrekte Funktionssignaturen
    - Chunk-Embedding-Synchronisation
    - Sichere Verarbeitung
    - Umfassende Fehlerbehandlung
    """
    try:
        logger.info("?? Starting fixed rechunking process")
        
        # Hole alle Dateien, die neu verarbeitet werden k�nnen
        import psycopg2.extras
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM uploaded_files 
                WHERE status IN ('uploaded', 'chunked', 'error') 
                AND status != 'deleted'
                ORDER BY upload_date
            """)
            
            rechunkable_files = cur.fetchall()
            cur.close()
        
        # Initialisiere Ergebnis-Tracking
        results = {
            "total_files": len(rechunkable_files),
            "processed": 0,
            "errors": 0,
            "skipped": 0,
            "details": []
        }
        
        # Verarbeite jede Datei einzeln
        for file_info in rechunkable_files:
            file_id = file_info["id"]
            filename = file_info["file_name"]
            file_path = file_info.get("file_path")
            file_hash = file_info["file_hash"]
            
            try:
                logger.info(f"?? Rechunking file {file_id}: {filename}")
                
                # Pr�fe ob physische Datei existiert
                if not file_path or not os.path.exists(file_path):
                    logger.warning(f"?? Physical file not found for {filename}")
                    await file_manager.update_file_status(
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
                
                # Setze Status auf Verarbeitung
                await file_manager.update_file_status(file_id, "processing")
                
                # L�sche existierende Chunks
                try:
                    with db_manager.get_connection() as conn:
                        cur = conn.cursor()
                        # Verwende nur pdf_chunks Tabelle
                        cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                        conn.commit()
                        cur.close()
                    
                    # Entferne aus ChromaDB
                    from services.retrieval import remove_documents_from_chroma
                    remove_documents_from_chroma(file_hash)
                    
                    logger.info(f"?? Cleared existing chunks for {filename}")
                except Exception as e:
                    logger.warning(f"?? Error clearing existing chunks for {filename}: {e}")
                
                # Verarbeite neu mit Enhanced Parser
                processing_start = time.time()
                
                parse_result = await parse_document_with_embeddings(
                    file_path=file_path,
                    strategy="hi_res",
                    chunking_strategy="title",
                    max_characters=500,
                    generate_embeddings=True
                )
                
                # Verwende sichere Verarbeitungsfunktion
                file_info_dict = {
                    "file_extension": file_info.get("file_extension", ""),
                    "file_hash": file_hash,
                    "document_type": file_info.get("document_type", "pdf")
                }
                
                success, result = await process_parsed_document_safely(
                    file_id=file_id,
                    file_info=file_info_dict,
                    parse_result=parse_result,
                    filename=filename
                )
                
                if not success:
                    results["errors"] += 1
                    results["details"].append({
                        "file_id": file_id,
                        "filename": filename,
                        "status": "error",
                        "error": result
                    })
                    continue
                
                # Aktualisiere Status
                processing_time = int((time.time() - processing_start) * 1000)
                await file_manager.update_file_status(
                    file_id=file_id,
                    status="chunked",
                    chunk_count=result["chunks"],
                    processing_duration_ms=processing_time
                )
                
                results["processed"] += 1
                results["details"].append({
                    "file_id": file_id,
                    "filename": filename,
                    "status": "success",
                    "chunks": result["chunks"],
                    "processing_time_ms": processing_time
                })
                
                logger.info(f"? Successfully rechunked {filename}: {result['chunks']} chunks")
                
            except Exception as e:
                logger.error(f"? Error rechunking file {file_id} ({filename}): {e}")
                
                await file_manager.update_file_status(
                    file_id, 
                    "error",
                    error_message=str(e)
                )
                
                results["errors"] += 1
                results["details"].append({
                    "file_id": file_id,
                    "filename": filename,
                    "status": "error",
                    "error": str(e)
                })
        
        logger.info(f"?? Rechunking completed: {results['processed']} processed, {results['errors']} errors")
        
        return RechunkResponse(
            status="completed",
            total_files=results["total_files"],
            processed=results["processed"],
            errors=results["errors"],
            skipped=results.get("skipped", 0),
            details=results["details"]
        )
        
    except Exception as e:
        logger.error(f"? Rechunking process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rechunking failed: {str(e)}")

# Such- UND Abfrage-Endpunkte

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, query_request: QueryRequest):
    """
    Erweiterte Dokumentensuche mit Analytics und Qualit�tsfilterung
    
    Verarbeitungsschritte:
    1. Generiere Embedding f�r Suchanfrage
    2. Semantische Suche in ChromaDB
    3. Qualit�tsfilterung der Ergebnisse
    4. Datei-ID-Filterung (optional)
    5. LLM-basierte Antwortgenerierung
    6. Analytics-Logging
    7. Qualit�tsmetriken-Berechnung
    
    Args:
        request: FastAPI Request f�r Analytics
        query_request: Suchanfrage mit Parametern
    
    Returns:
        QueryResponse: Generierte Antwort mit Quellen und Metriken
    """
    query_start = time.time()
    embedding_start = None
    retrieval_start = None
    
    try:
        # Schritt 1: Generiere Query-Embedding
        embedding_start = time.time()
        query_embeddings = await generate_embeddings_only([query_request.query])
        query_embedding = query_embeddings[0]
        embedding_time = int((time.time() - embedding_start) * 1000)
        
        # Schritt 2: Suche �hnliche Dokumente mit Qualit�tsfilterung
        retrieval_start = time.time()
        similar_docs = query_chroma(
            embedding=query_embedding,
            n_results=query_request.max_results
        )
        
        # Schritt 3: Filtere nach Qualit�tsschwelle
        filtered_docs = [
            doc for doc in similar_docs 
            if doc.get("metadata", {}).get("chunk_quality_score", 0) >= query_request.quality_threshold
        ]
        
        # Schritt 4: Filtere nach Datei-IDs (falls angegeben)
        if query_request.file_ids:
            filtered_docs = [
                doc for doc in filtered_docs 
                if doc.get("metadata", {}).get("file_id") in query_request.file_ids
            ]
        
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        if not filtered_docs:
            # Keine relevanten Dokumente gefunden
            answer = "Sorry, I couldn't find relevant information matching your quality criteria and query."
            sources = []
            quality_metrics = {"avg_quality": 0, "avg_similarity": 0}
        else:
            # Schritt 5: Generiere Antwort mit LLM
            answer = await generate_answer(query_request.query, filtered_docs)
            
            # Schritt 6: Formatiere Quellen mit erweiterten Metadaten
            sources = []
            for doc in filtered_docs:
                metadata = doc.get("metadata", {})
                sources.append({
                    "text": doc["text"][:300] + "..." if len(doc["text"]) > 300 else doc["text"],
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", "N/A"),
                    "chunk_index": metadata.get("chunk_index", "N/A"),
                    "similarity": round(doc.get("similarity", 0.0), 3),
                    "quality_score": round(metadata.get("chunk_quality_score", 0), 3),
                    "contains_table": metadata.get("contains_table", False),
                    "contains_list": metadata.get("contains_list", False),
                    "element_type": metadata.get("element_type", "Text")
                })
            
            # Schritt 7: Berechne Qualit�tsmetriken
            quality_scores = [doc.get("metadata", {}).get("chunk_quality_score", 0) for doc in filtered_docs]
            similarities = [doc.get("similarity", 0) for doc in filtered_docs]
            
            quality_metrics = {
                "avg_quality": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
                "avg_similarity": round(sum(similarities) / len(similarities), 3) if similarities else 0,
                "min_quality": round(min(quality_scores), 3) if quality_scores else 0,
                "max_similarity": round(max(similarities), 3) if similarities else 0
            }
        
        # Schritt 8: Berechne Gesamtantwortzeit
        total_response_time = int((time.time() - query_start) * 1000)
        
        # Schritt 9: Logge Analytics
        await log_request_analytics(
            request=request,
            query=query_request.query,
            results_count=len(filtered_docs),
            response_time_ms=total_response_time,
            embedding_time_ms=embedding_time
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=query_request.query,
            response_time_ms=total_response_time,
            results_count=len(filtered_docs),
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"? Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/search/fulltext", response_model=FullTextSearchResponse)
async def fulltext_search(
    request: Request,
    q: str,
    file_ids: Optional[List[int]] = None,
    limit: int = 10,
    quality_threshold: float = 0.3
):
    """
    Volltextsuche mit deutscher Sprachunterst�tzung und Qualit�tsfilterung
    
    Verwendet PostgreSQL's integrierte Volltext-Suchfunktionen:
    - to_tsvector('german', text) f�r deutsche Wortanalyse
    - plainto_tsquery('german', query) f�r Suchanfragen
    - ts_rank() f�r Relevanz-Bewertung
    - Qualit�tsfilterung nach chunk_quality_score
    
    Args:
        request: FastAPI Request f�r Analytics
        q: Suchbegriff
        file_ids: Optional Datei-ID-Filter
        limit: Maximale Anzahl Ergebnisse
        quality_threshold: Mindestqualit�t der Chunks
    
    Returns:
        FullTextSearchResponse: Suchergebnisse mit Relevanz-Ranking
    """
    search_start = time.time()
    
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        # F�hre Volltextsuche aus
        results = search_chunks_fulltext(
            search_term=q,
            file_ids=file_ids,
            limit=limit,
            quality_threshold=quality_threshold
        )
        
        # Formatiere Ergebnisse
        formatted_results = []
        for chunk in results:
            formatted_results.append({
                "chunk_id": chunk.id,
                "file_id": chunk.file_id,
                "text": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                "full_text_length": len(chunk.text),
                "page": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "quality_score": chunk.chunk_quality_score,
                "contains_table": chunk.contains_table,
                "contains_list": chunk.contains_list,
                "element_type": chunk.element_type,
                "word_count": chunk.word_count
            })
        
        response_time = int((time.time() - search_start) * 1000)
        
        # Logge Analytics
        await log_request_analytics(
            request=request,
            query=f"fulltext: {q}",
            results_count=len(results),
            response_time_ms=response_time
        )
        
        return FullTextSearchResponse(
            query=q,
            results_count=len(results),
            response_time_ms=response_time,
            results=formatted_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Full-text search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Full-text search failed: {str(e)}")

@app.post("/process_uploaded_files")
async def process_uploaded_files():
    """
    Verarbeite alle hochgeladenen Dateien mit korrekten Funktionssignaturen
    
    Findet alle Dateien mit Status 'uploaded' und verarbeitet sie:
    1. Setzt Status auf 'processing'
    2. Verwendet Enhanced Parser mit Embeddings
    3. Wendet sichere Verarbeitungsfunktion an
    4. Speichert in PostgreSQL + ChromaDB
    5. Aktualisiert Status und Metriken
    """
    try:
        # Hole alle hochgeladenen Dateien
        files = await file_manager.get_all_files()
        
        results = {
            "total_files": 0,
            "processed": 0,
            "errors": 0,
            "details": []
        }
        
        # Filtere Dateien, die verarbeitet werden m�ssen
        for file_data in files:
            if file_data["status"] == "uploaded" and file_data["chunk_count"] == 0:
                results["total_files"] += 1
                file_id = file_data["id"]
                filename = file_data["file_name"]
                file_path = file_data["file_path"]
                
                logger.info(f"?? Processing uploaded file: {filename}")
                
                try:
                    # Setze Status auf Verarbeitung
                    await file_manager.update_file_status(file_id, "processing")
                    
                    # Verarbeite mit Parser
                    processing_start = time.time()
                    parse_result = await parse_document_with_embeddings(
                        file_path=file_path,
                        strategy="hi_res", 
                        chunking_strategy="title",
                        max_characters=500,
                        generate_embeddings=True
                    )
                    
                    # Verwende sichere Verarbeitungsfunktion
                    file_info_dict = {
                        "file_extension": file_data["file_extension"],
                        "file_hash": file_data["file_hash"],
                        "document_type": file_data.get("document_type", "pdf")
                    }
                    
                    success, result = await process_parsed_document_safely(
                        file_id=file_id,
                        file_info=file_info_dict,
                        parse_result=parse_result,
                        filename=filename
                    )
                    
                    if not success:
                        results["errors"] += 1
                        results["details"].append({
                            "file_id": file_id,
                            "filename": filename,
                            "status": "error",
                            "error": result
                        })
                        continue
                    
                    # Aktualisiere Status
                    processing_time = int((time.time() - processing_start) * 1000)
                    await file_manager.update_file_status(
                        file_id=file_id,
                        status="chunked",
                        chunk_count=result["chunks"],
                        processing_duration_ms=processing_time
                    )
                    
                    results["processed"] += 1
                    results["details"].append({
                        "file_id": file_id,
                        "filename": filename,
                        "status": "success",
                        "chunks": result["chunks"],
                        "processing_time_ms": processing_time
                    })
                    
                    logger.info(f"? Successfully processed {filename}: {result['chunks']} chunks")
                    
                except Exception as e:
                    logger.error(f"? Error processing file {file_id} ({filename}): {e}")
                    
                    await file_manager.update_file_status(
                        file_id,
                        "error", 
                        error_message=str(e)
                    )
                    
                    results["errors"] += 1
                    results["details"].append({
                        "file_id": file_id,
                        "filename": filename,
                        "status": "error",
                        "error": str(e)
                    })
        
        return {
            "status": "completed",
            "message": f"Processed {results['processed']}/{results['total_files']} files",
            **results
        }
        
    except Exception as e:
        logger.error(f"? Processing uploaded files failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

# Analytics und Debbugging Endpunkte

@app.get("/formats")
async def get_supported_formats():
    """
    Hole unterst�tzte Dateiformate vom Enhanced Parser Service
    """
    try:
        formats = await parser_get_formats()
        return {
            "status": "success",
            "supported_formats": formats,
            "total_formats": len(formats)
        }
    except Exception as e:
        logger.error(f"? Error getting formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/analytics")
async def get_system_analytics(days: int = 30):
    """
    Hole Abfrage-Analytics und Nutzungsstatistiken
    """
    try:
        query_analytics = get_query_analytics(days)
        db_health = get_database_health()
        
        return {
            "status": "success",
            "period_days": days,
            "query_analytics": query_analytics,
            "database_health": db_health,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"? Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/maintenance")
async def run_database_maintenance():
    """
    F�hre Datenbank-Wartungsaufgaben aus
    """
    try:
        results = database_maintenance()
        return {
            "status": "success",
            "maintenance_results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"? Database maintenance failed: {e}")
        raise HTTPException(status_code=500, detail=f"Maintenance failed: {str(e)}")

@app.post("/system/cleanup")
async def cleanup_system():
    """
    Bereinige verwaiste Dateien und optimiere System
    """
    try:
        cleanup_results = await file_manager.cleanup_orphaned_files()
        
        # F�hre Datenbank-Wartung aus
        maintenance_results = database_maintenance()
        
        return {
            "status": "success",
            "cleanup": cleanup_results,
            "maintenance": maintenance_results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"? System cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test_upload")
async def test_single_upload(file_id: int):
    """
    KRITISCHER FIX: Teste Verarbeitung einer einzelnen hochgeladenen Datei
    """
    try:
        file_info = await file_manager.get_file_by_id(file_id)
        if not file_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        logger.info(f"?? Testing upload processing for file {file_id}: {file_info['file_name']}")
        
        # Pr�fe ob Datei existiert
        if not os.path.exists(file_info["file_path"]):
            return {"status": "error", "message": "Physical file not found"}
        
        # Teste Parser-Service
        try:
            parse_result = await parse_document_with_embeddings(
                file_path=file_info["file_path"],
                strategy="hi_res",
                chunking_strategy="title", 
                max_characters=500,
                generate_embeddings=True
            )
            
            # Teste Verarbeitungsfunktion
            if parse_result["success"]:
                file_info_dict = {
                    "file_extension": file_info["file_extension"],
                    "file_hash": file_info["file_hash"],
                    "document_type": file_info.get("document_type", "pdf")
                }
                
                success, result = await process_parsed_document_safely(
                    file_id=file_id,
                    file_info=file_info_dict,
                    parse_result=parse_result,
                    filename=file_info["file_name"]
                )
                
                return {
                    "status": "success" if success else "error",
                    "file_info": {
                        "id": file_info["id"],
                        "name": file_info["file_name"],
                        "path": file_info["file_path"],
                        "status": file_info["status"]
                    },
                    "parse_result": {
                        "success": parse_result["success"],
                        "chunks": len(parse_result.get("chunks", [])),
                        "embeddings": len(parse_result.get("embeddings", [])),
                        "error": parse_result.get("error")
                    },
                    "processing_result": result if success else {"error": result}
                }
            else:
                return {
                    "status": "error",
                    "message": f"Parser error: {parse_result.get('error')}",
                    "file_info": {
                        "id": file_info["id"],
                        "name": file_info["file_name"],
                        "path": file_info["file_path"],
                        "status": file_info["status"]
                    }
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}",
                "file_info": {
                    "id": file_info["id"],
                    "name": file_info["file_name"],
                    "path": file_info["file_path"],
                    "status": file_info["status"]
                }
            }
            
    except Exception as e:
        logger.error(f"? Test upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Erweiterte HTTP-Exception-Behandlung mit Logging
    """
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.method} {request.url}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "detail": exc.detail,
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
                "fixes_available": "Check /health endpoint for applied fixes"
            }
        }
    )

@app.post("/retry_failed")
async def retry_failed_files():
    """
    Wiederhole Verarbeitung fehlgeschlagener Dateien
    """
    try:
        result = await file_manager.retry_failed_files(max_retries=3)
        return {
            "status": "success",
            "message": f"Retry completed: {result['queued']} files queued for retry",
            **result
        }
    except Exception as e:
        logger.error(f"? Error retrying failed files: {e}")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")

@app.get("/queue")
async def get_processing_queue():
    """
    Hole aktuellen Verarbeitungsqueue-Status
    """
    try:
        queue_info = await file_manager.get_processing_queue()
        
        if queue_info.get("status") == "error":
            raise HTTPException(status_code=500, detail=queue_info.get("message"))
        
        # Formatiere f�r Frontend-Kompatibilit�t
        summary = {
            "pending": queue_info.get("queue", {}).get("uploaded", 0),
            "processing": queue_info.get("queue", {}).get("processing", 0), 
            "completed": 0,  # W�rde separate Berechnung ben�tigen
            "errors": queue_info.get("queue", {}).get("error", 0)
        }
        
        return {
            "status": "success",
            "summary": summary,
            "details": queue_info.get("details", {}),
            "queue": queue_info.get("queue", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"? Error getting processing queue: {e}")
        raise HTTPException(status_code=500, detail=f"Queue check failed: {str(e)}")

@app.post("/cleanup")
async def cleanup_system_legacy():
    """
    Legacy-Bereinigungs-Endpunkt - leitet weiter zu /system/cleanup
    """
    try:
        # Rufe existierende System-Bereinigungs-Funktion auf
        cleanup_results = await file_manager.cleanup_orphaned_files()
        
        # F�hre Datenbank-Wartung aus
        maintenance_results = database_maintenance()
        
        return {
            "status": "success",
            "cleanup": cleanup_results,
            "maintenance": maintenance_results,
            "timestamp": datetime.now().isoformat(),
            "note": "Use /system/cleanup for future requests"
        }
    except Exception as e:
        logger.error(f"? Legacy cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue/details")
async def get_queue_details():
    """
    Hole detaillierte Queue-Informationen
    """
    try:
        # Hole Dateien nach Status f�r detaillierte Ansicht
        import psycopg2.extras
        with db_manager.get_connection() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Hole aktuelle Dateien in jedem Status
            cur.execute("""
                SELECT 
                    id, file_name, status, upload_date, error_message,
                    processing_duration_ms
                FROM uploaded_files 
                WHERE status IN ('uploaded', 'processing', 'error')
                ORDER BY upload_date DESC
                LIMIT 50
            """)
            
            recent_files = cur.fetchall()
            cur.close()
        
        # Gruppiere nach Status
        grouped = {
            "uploaded": [],
            "processing": [], 
            "error": []
        }
        
        for file_info in recent_files:
            status = file_info["status"]
            if status in grouped:
                file_data = {
                    "id": file_info["id"],
                    "filename": file_info["file_name"],
                    "upload_date": file_info["upload_date"].isoformat() if file_info["upload_date"] else None,
                    "error_message": file_info.get("error_message"),
                    "processing_time_ms": file_info.get("processing_duration_ms")
                }
                grouped[status].append(file_data)
        
        return {
            "status": "success",
            "queue_details": grouped,
            "counts": {status: len(files) for status, files in grouped.items()},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"? Error getting queue details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api-status")
async def api_endpoint_status():
    """
    Zeige verf�gbare API-Endpunkte f�r Debugging
    """
    endpoints = []
    
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            endpoints.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": getattr(route, 'name', 'unknown')
            })
    
    return {
        "total_endpoints": len(endpoints),
        "endpoints": sorted(endpoints, key=lambda x: x['path']),
        "critical_endpoints": [
            "/health", "/upload_document", "/files", "/query", 
            "/retry_failed", "/queue", "/cleanup", "/system/cleanup"
        ],
        "timestamp": datetime.now().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Allgemeine Exception-Behandlung f�r unerwartete Fehler
    """
    logger.error(f"Unexpected error: {str(exc)} - {request.method} {request.url}")
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "status_code": 500,
                "detail": "Internal server error - Check logs for details",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url),
                "suggestion": "Verify all services are running and check /health endpoint"
            }
        }
    )

# Startup- UND Shutdown-Events

@app.on_event("startup")
async def startup_event():
    """
    Initialisiere Anwendung beim Start
    """
    logger.info("?? Enhanced RAG API Backend v2.1.0-FIXED starting up...")
    
    # Teste Datenbankverbindung
    db_ok, db_msg = test_db_connection()
    if db_ok:
        logger.info(f"? Database: {db_msg}")
    else:
        logger.error(f"? Database: {db_msg}")
    
    # Teste andere Services
    chroma_ok, chroma_msg = check_chroma_connection()
    logger.info(f"{'?' if chroma_ok else '?'} ChromaDB: {chroma_msg}")
    
    parser_ok, parser_msg = await check_parser_connection()
    logger.info(f"{'?' if parser_ok else '?'} Parser: {parser_msg}")
    
    logger.info("?? CRITICAL FIXES APPLIED:")
    logger.info("   ? insert_chunk_metadata signature fixed")
    logger.info("   ? chunk-embedding synchronization validated")
    logger.info("   ? safe processing functions implemented")
    logger.info("   ? error handling enhanced")
    logger.info("?? Enhanced RAG API Backend ready with fixes!")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Bereinige beim Herunterfahren
    """
    logger.info("?? Enhanced RAG API Backend shutting down...")
    
    # Schlie�e Datenbankverbindungen
    try:
        db_manager.close_all_connections()
        logger.info("? Database connections closed")
    except Exception as e:
        logger.error(f"? Error closing database connections: {e}")
    
    logger.info("?? Enhanced RAG API Backend shutdown complete")


# Startpunkt

if __name__ == "__main__":
    import uvicorn
    
    # Production-ready Konfiguration
    uvicorn.run(
        app, 
        host="0.0.0.0",       # Alle Netzwerk-Interfaces binden
        port=80,              # Standard HTTP-Port
        log_level="info",     # Info-Level Logging
        access_log=True,      # Access-Logs aktivieren
        workers=1             # Single Worker f�r Docker-Deployment
    )