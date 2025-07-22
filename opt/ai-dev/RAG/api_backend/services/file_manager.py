# =====================================
# FILE MANAGER SERVICE 
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
from services.document_processor import process_parsed_document_safely
import psycopg2
import psycopg2.extras

# Logger für strukturierte Protokollierung
logger = logging.getLogger(__name__)

class FileManager:
    """
    ZENTRALE DATEIVERWALTUNG für RAG-Systeme (REFACTORED VERSION)
    
    Diese Klasse implementiert eine saubere FileManager-Version ohne
    Code-Duplizierung. Alle komplexe Verarbeitungslogik wurde in den
    zentralen document_processor ausgelagert.
    
    ARCHITEKTUR-VERBESSERUNGEN:
    - Keine duplizierte Verarbeitungslogik mehr
    - Single Source of Truth für Dokument-Processing
    - Konsistente Quality-Score-Behandlung
    - Einfachere Wartung und Testing
    """
    
    def __init__(self):
        """Initialisiert FileManager mit Lazy-Loading für Service-Dependencies"""
        self.supported_formats = None
        self._formats_initialized = False
    
    async def _initialize_formats(self):
        """Initialisiert unterstützte Dateiformate vom Parser-Service"""
        if self._formats_initialized:
            return
            
        try:
            self.supported_formats = await parser_get_formats()
            self._formats_initialized = True
            logger.info(f"✅ Loaded {len(self.supported_formats)} supported formats")
        except Exception as e:
            logger.error(f"❌ Failed to load supported formats: {e}")
            
            self.supported_formats = {
                '.pdf': 'PDF Document',
                '.txt': 'Text Document',
                '.docx': 'Word Document',
                '.doc': 'Word Document (Legacy)'
            }
            self._formats_initialized = True
    
    async def _ensure_formats_loaded(self):
        """Stellt sicher dass Dateiformate vor Verwendung geladen sind"""
        if not self._formats_initialized:
            await self._initialize_formats()
    
    def is_supported_file_type(self, filename: str) -> bool:
        """Prüft ob Dateityp unterstützt wird"""
        if not self.supported_formats:
            fallback_formats = {'.pdf', '.txt', '.docx', '.doc', '.pptx', '.xlsx'}
            file_ext = os.path.splitext(filename.lower())[1]
            return file_ext in fallback_formats
        
        file_ext = os.path.splitext(filename.lower())[1]
        return file_ext in self.supported_formats
    
    async def register_file(self, file_info: Dict[str, Any]) -> Optional[int]:
        """Registriert neue Datei mit bestehender Database-Struktur"""
        try:
            await self._ensure_formats_loaded()
            
            # Duplikat-Prüfung
            existing_file = await self.get_file_by_hash(file_info["file_hash"])
            
            if existing_file:
                if existing_file.get("status") == "deleted":
                    logger.info(f"������ Reactivating soft-deleted file: {existing_file['file_name']}")
                    await self._update_file_status_direct(
                        existing_file["id"], 
                        "uploaded",
                        error_message=None
                    )
                    return existing_file["id"]
                else:
                    logger.warning(f"⚠️ File already exists: {existing_file['file_name']} (status: {existing_file.get('status')})")
                    return None
            
            # Document-Type-Mapping
            file_ext = file_info.get("file_extension", "").lower()
            
            document_type_mapping = {
                '.pdf': 'pdf',
                '.docx': 'word',
                '.doc': 'word',
                '.pptx': 'powerpoint',
                '.ppt': 'powerpoint',
                '.xlsx': 'excel',
                '.xls': 'excel',
                '.txt': 'text',
                '.md': 'markdown',
                '.html': 'html',
                '.csv': 'csv'
            }
            
            document_type = document_type_mapping.get(file_ext, 'text')
            
            # Transaktionale Database-Insertion
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO uploaded_files 
                    (file_name, file_hash, file_path, file_size, file_extension, 
                     document_type, status, chunk_count, processing_method, upload_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    file_info["file_name"],
                    file_info["file_hash"],
                    file_info["file_path"],
                    file_info["file_size"],
                    file_ext,
                    document_type,
                    "uploaded",
                    0,
                    "hybrid",
                    datetime.now()
                ))
                
                file_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
            
            if file_id:
                logger.info(f"✅ Registered file: {file_info['file_name']} (ID: {file_id})")
                return file_id
            else:
                logger.error(f"❌ Failed to register file: {file_info['file_name']}")
                return None
                
        except psycopg2.IntegrityError as e:
            if "duplicate key" in str(e):
                logger.warning(f"⚠️ File with hash already exists: {file_info['file_name']}")
                return None
            else:
                logger.error(f"❌ Database integrity error: {e}")
                return None
        except Exception as e:
            logger.error(f"❌ Error registering file {file_info.get('file_name', 'unknown')}: {e}")
            return None
    
    async def get_file_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Holt Datei-Informationen anhand des SHA256-Hash-Wertes"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE file_hash = %s
                    ORDER BY upload_date DESC
                    LIMIT 1
                """, (file_hash,))
                
                result = cur.fetchone()
                cur.close()
                
                if result:
                    result_dict = dict(result)
                    
                    # DateTime-Serialisierung
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    return result_dict
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting file by hash: {e}")
            return None
    
    async def get_file_by_id(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Holt Datei-Informationen anhand der eindeutigen Database-ID"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE id = %s
                """, (file_id,))
                
                result = cur.fetchone()
                cur.close()
                
                if result:
                    result_dict = dict(result)
                    
                    # DateTime-Konvertierung
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    return result_dict
                
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting file by ID {file_id}: {e}")
            return None
    
    async def get_all_files(self) -> List[Dict[str, Any]]:
        """Holt alle Dateien für Dashboard und Verwaltungs-Interfaces"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    ORDER BY upload_date DESC
                """)
                
                results = cur.fetchall()
                cur.close()
                
                converted_results = []
                for result in results:
                    result_dict = dict(result)
                    
                    # Konsistente DateTime-Behandlung
                    for field in ['upload_date', 'last_chunked', 'last_accessed', 'created_at', 'updated_at']:
                        if result_dict.get(field):
                            if hasattr(result_dict[field], 'isoformat'):
                                result_dict[field] = result_dict[field].isoformat()
                    
                    converted_results.append(result_dict)
                
                return converted_results
                
        except Exception as e:
            logger.error(f"❌ Error getting all files: {e}")
            return []
    
    async def get_file_chunks(self, file_id: int, quality_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Holt Text-Chunks für eine spezifische Datei mit Qualitäts-Filterung"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM pdf_chunks 
                    WHERE file_id = %s 
                    AND chunk_quality_score >= %s
                    ORDER BY chunk_index
                """, (file_id, quality_threshold))
                
                results = cur.fetchall()
                cur.close()
                
                return [dict(result) for result in results]
                
        except Exception as e:
            logger.error(f"❌ Error getting chunks for file {file_id}: {e}")
            return []
    
    async def _update_file_status_direct(self, file_id: int, status: str, 
                                       chunk_count: int = None, 
                                       processing_duration_ms: int = None,
                                       error_message: str = None):
        """Aktualisiert Datei-Status mit direktem SQL für maximale Performance"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                # Dynamisches Query-Building
                update_fields = ["status = %s"]
                params = [status]
                
                if chunk_count is not None:
                    update_fields.append("chunk_count = %s")
                    params.append(chunk_count)
                
                if processing_duration_ms is not None:
                    update_fields.append("processing_duration_ms = %s")
                    params.append(processing_duration_ms)
                
                if error_message is not None:
                    update_fields.append("error_message = %s")
                    params.append(error_message)
                elif status != "error":
                    update_fields.append("error_message = NULL")
                
                if status == "chunked":
                    update_fields.append("last_chunked = %s")
                    params.append(datetime.now())
                
                params.append(file_id)
                
                query = f"""
                    UPDATE uploaded_files 
                    SET {', '.join(update_fields)}
                    WHERE id = %s
                """
                
                cur.execute(query, params)
                affected_rows = cur.rowcount
                conn.commit()
                cur.close()
                
                if affected_rows > 0:
                    logger.info(f"✅ Updated file {file_id} status to {status}")
                    return True
                else:
                    logger.error(f"❌ No file found with ID {file_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error updating file {file_id} status: {e}")
            return False
    
    async def update_file_status(self, file_id: int, status: str, 
                               chunk_count: int = None, 
                               processing_duration_ms: int = None,
                               error_message: str = None):
        """Öffentliche Methode für Status-Updates"""
        return await self._update_file_status_direct(
            file_id, status, chunk_count, processing_duration_ms, error_message
        )
    
    async def delete_file(self, file_id: int, permanent: bool = False) -> bool:
        """Löscht Datei mit wählbarer Soft/Hard-Delete-Strategie"""
        try:
            if permanent:
                # Hard Delete: Vollständige Entfernung aller Daten-Artefakte
                file_info = await self.get_file_by_id(file_id)
                if not file_info:
                    return False
                
                # Vector-Database-Cleanup
                try:
                    file_hash = file_info["file_hash"]
                    remove_documents_from_chroma(file_hash)
                    logger.info(f"������️ Removed file {file_id} from ChromaDB")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove from ChromaDB: {e}")
                
                # Physische Datei-Entfernung
                try:
                    if file_info.get("file_path") and os.path.exists(file_info["file_path"]):
                        os.remove(file_info["file_path"])
                        logger.info(f"������️ Removed physical file: {file_info['file_path']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove physical file: {e}")
                
                # Database-Deletion mit CASCADE
                with db_manager.get_connection() as conn:
                    cur = conn.cursor()
                    
                    cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                    cur.execute("DELETE FROM uploaded_files WHERE id = %s", (file_id,))
                    
                    conn.commit()
                    cur.close()
                
                logger.info(f"������️ Permanently deleted file {file_id}")
            else:
                # Soft Delete: Status-Update auf "deleted"
                success = await self._update_file_status_direct(file_id, "deleted")
                if success:
                    logger.info(f"������️ Soft deleted file {file_id}")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting file {file_id}: {e}")
            return False
    
    async def process_uploaded_queue(self) -> Dict[str, Any]:
        try:
            logger.info("������ Starting uploaded files queue processing")
            await self._ensure_formats_loaded()
            
            # Finde alle wartenden Dateien
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute("SELECT * FROM uploaded_files WHERE status = 'uploaded' ORDER BY upload_date")
                uploaded_files = cur.fetchall()
                cur.close()
            
            results = {"total_files": len(uploaded_files), "processed": 0, "errors": 0, "details": []}
            
            if not uploaded_files:
                return {"status": "success", "message": "No files in upload queue", **results}
            
            # Verarbeite jede Datei
            for file_info in uploaded_files:
                file_id = file_info["id"]
                filename = file_info["file_name"]
                file_path = file_info.get("file_path")
                
                try:
                    if not file_path or not os.path.exists(file_path):
                        await self._update_file_status_direct(file_id, "error", error_message="Physical file not found")
                        results["errors"] += 1
                        continue
                    
                    # Verarbeite mit geteilter Logik
                    await self._update_file_status_direct(file_id, "processing")
                    
                    parse_result = await parse_document_with_embeddings(
                        file_path=file_path, strategy="hi_res", chunking_strategy="title",
                        max_characters=500, generate_embeddings=True
                    )
                    
                    file_info_dict = {
                        "file_extension": file_info.get("file_extension", ""),
                        "file_hash": file_info["file_hash"],
                        "document_type": file_info.get("document_type", "pdf")
                    }
                    
                    success, result = await process_parsed_document_safely(
                        file_id=file_id, file_info=file_info_dict, 
                        parse_result=parse_result, filename=filename
                    )
                    
                    if success:
                        await self._update_file_status_direct(
                            file_id, "chunked", chunk_count=result["chunks"]
                        )
                        results["processed"] += 1
                        results["details"].append({
                            "file_id": file_id, "filename": filename, 
                            "status": "success", "chunks": result["chunks"]
                        })
                    else:
                        results["errors"] += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error processing {filename}: {e}")
                    await self._update_file_status_direct(file_id, "error", error_message=str(e))
                    results["errors"] += 1
            
            return {
                "status": "success",
                "message": f"Processed {results['processed']}/{results['total_files']} files",
                **results
            }
            
        except Exception as e:
            logger.error(f"❌ Queue processing failed: {e}")
            return {"status": "error", "message": str(e), "processed": 0, "errors": 1}
    
    async def get_processing_queue(self) -> Dict[str, Any]:
        """Holt aktuellen Status der Verarbeitungs-Queue für Monitoring"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                status_counts = {}
                for status in ["uploaded", "processing", "error"]:
                    cur.execute("""
                        SELECT COUNT(*) as count FROM uploaded_files 
                        WHERE status = %s
                    """, (status,))
                    
                    count = cur.fetchone()["count"]
                    status_counts[status] = count
                
                cur.execute("""
                    SELECT id, file_name, upload_date, status, error_message
                    FROM uploaded_files 
                    WHERE status IN ('uploaded', 'processing', 'error')
                    ORDER BY upload_date DESC
                    LIMIT 50
                """)
                
                files = cur.fetchall()
                cur.close()
                
                # Kategorisierung
                details = {"uploaded_files": [], "processing_files": [], "error_files": []}
                
                for file_info in files:
                    file_data = {
                        "id": file_info["id"],
                        "filename": file_info["file_name"],
                        "upload_date": file_info["upload_date"].isoformat() if file_info["upload_date"] else None
                    }
                    
                    if file_info["status"] == "uploaded":
                        details["uploaded_files"].append(file_data)
                    elif file_info["status"] == "processing":
                        details["processing_files"].append(file_data)
                    elif file_info["status"] == "error":
                        file_data["error"] = file_info.get("error_message", "Unknown error")
                        details["error_files"].append(file_data)
                
                return {
                    "status": "success",
                    "queue": status_counts,
                    "details": details
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting processing queue: {e}")
            return {"status": "error", "message": str(e)}
    
    async def rechunk_all_files(self) -> Dict[str, Any]:
        """
        REFACTORED: Rechunking mit zentraler Verarbeitungslogik
        
        Diese Funktion nutzt die zentrale process_parsed_document_safely()
        aus dem document_processor Service. KEINE Code-Duplizierung mehr!
        
        ARCHITEKTUR-VERBESSERUNGEN:
        - Single Source of Truth für Verarbeitung
        - Konsistente Quality-Score-Behandlung
        - Einfachere Wartung und Testing
        - Keine duplizierten Bugs möglich
        """
        try:
            logger.info("������ Starting comprehensive rechunking with centralized processing")
            
            # Voraussetzung: Format-Initialisierung
            await self._ensure_formats_loaded()
            
            # Datei-Discovery
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
            
            # Ergebnis-Tracking
            results = {
                "total_files": len(rechunkable_files),
                "processed": 0,
                "errors": 0,
                "skipped": 0,
                "details": []
            }
            
            # Batch-Verarbeitung
            for file_info in rechunkable_files:
                file_id = file_info["id"]
                filename = file_info["file_name"]
                file_path = file_info.get("file_path")
                file_hash = file_info["file_hash"]
                
                try:
                    logger.info(f"������ Rechunking file {file_id}: {filename}")
                    
                    # Physische Datei-Validierung
                    if not file_path or not os.path.exists(file_path):
                        logger.warning(f"⚠️ Physical file not found for {filename}")
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
                    
                    # Status-Update
                    await self._update_file_status_direct(file_id, "processing")
                    
                    # Cleanup-Phase
                    try:
                        with db_manager.get_connection() as conn:
                            cur = conn.cursor()
                            cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                            conn.commit()
                            cur.close()
                        
                        remove_documents_from_chroma(file_hash)
                        
                        logger.info(f"������ Cleared existing chunks for {filename}")
                    except Exception as e:
                        logger.warning(f"⚠️ Error clearing existing chunks for {filename}: {e}")
                    
                    # ML-Pipeline: Enhanced Parser
                    processing_start = time.time()
                    
                    parse_result = await parse_document_with_embeddings(
                        file_path=file_path,
                        strategy="hi_res",
                        chunking_strategy="title",
                        max_characters=500,
                        generate_embeddings=True
                    )
                    
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
                    
                    # Erfolgs-Finalisierung
                    processing_time = int((time.time() - processing_start) * 1000)
                    await self._update_file_status_direct(
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
                    
                    logger.info(f"✅ Successfully rechunked {filename}: {result['chunks']} chunks")
                    
                except Exception as e:
                    logger.error(f"❌ Error rechunking file {file_id} ({filename}): {e}")
                    
                    await self._update_file_status_direct(
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
            
            logger.info(f"������ Rechunking completed with centralized processing: {results['processed']} processed, {results['errors']} errors")
            return results
            
        except Exception as e:
            logger.error(f"❌ Rechunking process failed: {e}")
            raise Exception(f"Rechunking failed: {str(e)}")

    async def retry_failed_files(self, max_retries: int = 3) -> Dict[str, Any]:
        """Retry-Mechanismus für fehlgeschlagene Datei-Verarbeitungen"""
        try:
            with db_manager.get_connection() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                
                cur.execute("""
                    SELECT * FROM uploaded_files 
                    WHERE status = 'error' 
                    AND COALESCE(retry_count, 0) < %s
                    ORDER BY upload_date
                """, (max_retries,))
                
                failed_files = cur.fetchall()
                cur.close()
            
            results = {
                "total_files": len(failed_files),
                "queued": 0,
                "skipped": 0,
                "details": []
            }
            
            for file_info in failed_files:
                try:
                    await self._update_file_status_direct(
                        file_info["id"], 
                        "uploaded",
                        error_message=None
                    )
                    
                    results["queued"] += 1
                    results["details"].append({
                        "file_id": file_info["id"],
                        "filename": file_info["file_name"],
                        "status": "queued_for_retry",
                        "previous_error": file_info.get("error_message", "Unknown error")
                    })
                    
                    logger.info(f"������ Queued file {file_info['id']} for retry")
                    
                except Exception as e:
                    results["skipped"] += 1
                    results["details"].append({
                        "file_id": file_info["id"],
                        "filename": file_info["file_name"],
                        "status": "retry_failed",
                        "error": str(e)
                    })
                    logger.error(f"❌ Failed to queue file {file_info['id']} for retry: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrying failed files: {e}")
            return {"status": "error", "message": str(e)}

    async def cleanup_orphaned_files(self) -> Dict[str, Any]:
        """System-Bereinigung für verwaiste Dateien und Database-Optimierung"""
        try:
            results = {
                "orphaned_chunks_removed": 0,
                "missing_files_cleaned": 0,
                "database_optimized": False
            }
            
            with db_manager.get_connection() as conn:
                cur = conn.cursor()
                
                # Cleanup 1: Orphaned Chunks entfernen
                cur.execute("""
                    DELETE FROM pdf_chunks 
                    WHERE file_id NOT IN (SELECT id FROM uploaded_files)
                """)
                results["orphaned_chunks_removed"] = cur.rowcount
                
                # Cleanup 2: Missing Physical Files
                cur.execute("""
                    SELECT id, file_name, file_path FROM uploaded_files 
                    WHERE status != 'deleted'
                """)
                
                files_to_check = cur.fetchall()
                missing_count = 0
                
                for file_id, file_name, file_path in files_to_check:
                    if file_path and not os.path.exists(file_path):
                        cur.execute("""
                            UPDATE uploaded_files 
                            SET status = 'error', 
                                error_message = 'Physical file missing'
                            WHERE id = %s
                        """, (file_id,))
                        missing_count += 1
                        logger.warning(f"⚠️ Marked file as error (missing): {file_name}")
                
                results["missing_files_cleaned"] = missing_count
                
                # Database-Statistiken aktualisieren
                cur.execute("ANALYZE uploaded_files")
                cur.execute("ANALYZE pdf_chunks")
                
                results["database_optimized"] = True
                
                conn.commit()
                cur.close()
            
            logger.info(f"������ Cleanup completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return {"status": "error", "message": str(e)}

    async def bulk_upload_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Bulk-Upload für Enterprise-Szenarien mit Batch-Verarbeitung"""
        await self._ensure_formats_loaded()
        
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "duplicates": 0,
            "details": []
        }
        
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    results["failed"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "failed",
                        "error": "File not found"
                    })
                    continue
                
                filename = os.path.basename(file_path)
                
                if not self.is_supported_file_type(filename):
                    results["failed"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "failed",
                        "error": "Unsupported file type"
                    })
                    continue
                
                file_hash = calculate_file_hash(file_path)
                
                existing_file = await self.get_file_by_hash(file_hash)
                if existing_file and existing_file.get("status") != "deleted":
                    results["duplicates"] += 1
                    results["details"].append({
                        "file_path": file_path,
                        "status": "duplicate",
                        "existing_file": existing_file["file_name"]
                    })
                    continue
                
                file_info = {
                    "file_name": filename,
                    "file_hash": file_hash,
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path),
                    "file_extension": os.path.splitext(filename.lower())[1]
                }
                
                file_id = await self.register_file(file_info)
                
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
                results["failed"] += 1
                results["details"].append({
                    "file_path": file_path,
                    "status": "failed",
                    "error": str(e)
                })
                logger.error(f"❌ Error bulk uploading {file_path}: {e}")
        
        return results

# Globale FileManager-Instanz
file_manager = FileManager()