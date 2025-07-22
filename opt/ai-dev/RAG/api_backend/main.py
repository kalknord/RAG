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

# Database und Core Services
from services.db import (
    FileStatus, DocumentType, ProcessingMethod,          
    FileInfo, ChunkInfo, SystemMetrics,                  
    log_query, get_system_performance,                   
    search_chunks_fulltext, get_query_analytics, get_database_health,  
    database_maintenance, test_db_connection, get_connection_params,    
    db_manager                                          
)

# LLM und Retrieval Services
from services.llm import generate_answer                 
from services.retrieval import query_chroma, check_chroma_connection  

# Document Processing Services
from services.document_parser import (                   
    parse_document_with_embeddings,                     
    generate_embeddings_only,                           
    check_parser_connection,                            
    get_supported_formats as parser_get_formats         
)

# Utilities
from services.file_manager import file_manager          
from services.hashing import calculate_file_hash 

# Import der zentralen Verarbeitungslogik
from services.document_processor import process_parsed_document_safely

# Konfiguration des Logging-Systems für Debugging und Monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Erstelle FastAPI-Anwendung mit erweiterten Metadaten und API-Dokumentation
app = FastAPI(
    title="Enhanced RAG API Backend",
    version="2.1.0",
    description="RAG system",
    docs_url="/docs",      
    redoc_url="/redoc"     
)

# Konfiguration für Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        
    allow_credentials=True,     
    allow_methods=["*"],        
    allow_headers=["*"],        
)

# Upload-Verzeichnis für hochgeladene Dateien
UPLOAD_DIR = "/data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic Models für API Request/Response Validation
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results")
    quality_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum chunk quality score")
    file_ids: Optional[List[int]] = Field(None, description="Limit search to specific files")

class QueryResponse(BaseModel):
    answer: str                          
    sources: List[Dict[str, Any]]        
    query: str                           
    response_time_ms: int                
    results_count: int                   
    quality_metrics: Dict[str, float]    

class UploadResponse(BaseModel):
    status: str                          
    message: str                         
    chunks: int                          
    filename: str                        
    file_id: int                         
    processing_time_ms: int              
    quality_metrics: Dict[str, Any]      

class MultiUploadResponse(BaseModel):
    status: str                          
    total_files: int                     
    successful: int                      
    failed: int                          
    duplicates: int                      
    details: List[Dict[str, Any]]        

class SystemHealthResponse(BaseModel):
    status: str                          
    services: Dict[str, Dict[str, str]]  
    database: Dict[str, Any]             
    performance_metrics: Dict[str, Any]  
    version: str                         
    fixes_applied: List[str]             

class FullTextSearchResponse(BaseModel):
    query: str                           
    results_count: int                   
    response_time_ms: int                
    results: List[Dict[str, Any]]        

class RechunkResponse(BaseModel):
    status: str                          
    total_files: int                     
    processed: int                       
    errors: int                          
    skipped: int                         
    details: List[Dict[str, Any]]        

# Hilfsfunktionen für Request-Verarbeitung und Analytics
def get_client_ip(request: Request) -> str:
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def generate_session_id(request: Request) -> str:
    import hashlib
    user_agent = request.headers.get("User-Agent", "")
    client_ip = get_client_ip(request)
    timestamp = str(int(time.time() // 3600))
    session_data = f"{client_ip}:{user_agent}:{timestamp}"
    return hashlib.md5(session_data.encode()).hexdigest()

async def log_request_analytics(request: Request, query: str, results_count: int, 
                               response_time_ms: int, embedding_time_ms: int = None):
    try:
        client_ip = get_client_ip(request)
        session_id = generate_session_id(request)
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
        logger.error(f"❌ Error logging analytics: {e}")

# =====================================
# SYSTEMHEALTH UND STATUS ENDPUNKTE
# =====================================

@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    try:
        chroma_ok, chroma_msg = check_chroma_connection()
        parser_ok, parser_msg = await check_parser_connection()
        db_ok, db_msg = test_db_connection()
        
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
        
        db_health = {}
        try:
            db_health = get_database_health()
        except Exception as e:
            logger.error(f"Error getting database health: {e}")
        
        overall_status = "healthy" if all([chroma_ok, parser_ok, db_ok]) else "degraded"
        
        return SystemHealthResponse(
            status=overall_status,
            services={
                "database": {"status": "ok" if db_ok else "error", "message": db_msg},
                "chroma": {"status": "ok" if chroma_ok else "error", "message": chroma_msg},
                "enhanced_parser": {"status": "ok" if parser_ok else "error", "message": parser_msg},
                "api": {"status": "ok", "message": "API OK"}
            },
            database=db_health,
            performance_metrics=performance_metrics,
            version="2",
            fixes_applied=[
                "quality_scoring_bug_fix",             
                "shared_processing_logic",             
                "eliminated_code_duplication",         
                "single_source_of_truth_processing",   
                "chunk_embedding_sync_validation",     
                "dimension_validation",                
                "function_signature_compatibility"     
            ]
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/system/performance")
async def get_performance_metrics():
    try:
        metrics = get_system_performance()
        if not metrics:
            raise HTTPException(status_code=500, detail="Could not retrieve performance metrics")
        
        return {
            "status": "success",
            "metrics": {
                "files": {
                    "pending": metrics.files_pending,
                    "processing": metrics.files_processing,
                    "ready": metrics.files_ready,
                    "error": metrics.files_error,
                    "total": metrics.total_files
                },
                "chunks": {"total": metrics.total_chunks},
                "storage": {"total_used": metrics.total_storage},
                "performance": {
                    "avg_processing_time_ms": metrics.avg_processing_time,
                    "max_processing_time_ms": metrics.max_processing_time,
                    "avg_file_quality": metrics.avg_file_quality
                },
                "activity": {
                    "files_uploaded_today": metrics.files_uploaded_today,
                    "files_accessed_today": metrics.files_accessed_today
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================
# DATEIUPLOAD-ENDPUNKTE
# =====================================

@app.post("/upload_document", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Verwendet jetzt process_parsed_document_safely() aus document_processor.py
    anstatt duplizierte Logik zu haben.
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
        
        # Schritt 2: Speichere Datei temporär
        temp_filename = f"temp_{int(time.time())}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, temp_filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Schritt 3: Berechne Datei-Hash und prüfe auf Duplikate
        file_hash = calculate_file_hash(file_path)
        existing_file = await file_manager.get_file_by_hash(file_hash)
        
        if existing_file:
            os.remove(file_path)
            
            if existing_file["status"] == "deleted":
                logger.info(f"������ Reactivating deleted file: {existing_file['file_name']}")
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
                raise HTTPException(
                    status_code=409, 
                    detail=f"File already exists: {existing_file['file_name']} (status: {existing_file['status']})"
                )
        
        # Schritt 4: Benenne zu finalem Dateinamen um
        final_file_path = os.path.join(UPLOAD_DIR, file.filename)
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
                strategy="hi_res",
                chunking_strategy="title",
                max_characters=500,
                generate_embeddings=True
            )
            
            # Verwende geteilte Verarbeitungslogik
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
            
            logger.info(f"✅ Successfully processed {final_filename}: {result['chunks']} chunks in {processing_time}ms")
            
            return UploadResponse(
                status="success",
                message=f"Document processed successfully",
                chunks=result["chunks"],
                filename=final_filename,
                file_id=file_id,
                processing_time_ms=processing_time,
                quality_metrics=result["quality_metrics"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            await file_manager.update_file_status(
                file_id=file_id,
                status="error",
                error_message=str(e)
            )
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        if file_id:
            await file_manager.update_file_status(
                file_id=file_id,
                status="error",
                error_message=str(e)
            )
        
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        
        logger.error(f"❌ Error processing document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/upload_multiple", response_model=MultiUploadResponse)
async def upload_multiple_documents(request: Request, files: List[UploadFile] = File(...)):
    """Upload mehrerer Dokumente mit Batch-Verarbeitung"""
    try:
        supported_formats = await parser_get_formats()
        
        results = {
            "total_files": len(files),
            "successful": 0,
            "failed": 0,
            "duplicates": 0,
            "details": []
        }
        
        for file in files:
            file_ext = os.path.splitext(file.filename.lower())[1]
            
            if file_ext not in supported_formats:
                results["failed"] += 1
                results["details"].append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": f"Unsupported file type '{file_ext}'"
                })
                continue
            
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                file_hash = calculate_file_hash(file_path)
                existing_file = await file_manager.get_file_by_hash(file_hash)
                
                if existing_file and existing_file["status"] != "deleted":
                    os.remove(file_path)
                    results["duplicates"] += 1
                    results["details"].append({
                        "filename": file.filename,
                        "status": "duplicate",
                        "existing_file": existing_file["file_name"]
                    })
                    continue
                
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
                logger.error(f"❌ Error processing {file.filename}: {e}")
        
        return MultiUploadResponse(
            status="completed",
            total_files=results["total_files"],
            successful=results["successful"],
            failed=results["failed"],
            duplicates=results["duplicates"],
            details=results["details"]
        )
        
    except Exception as e:
        logger.error(f"❌ Bulk upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")

# =====================================
# DATEIVERWALTUNGS-ENDPUNKTE
# =====================================

@app.get("/files")
async def get_uploaded_files():
    try:
        files = await file_manager.get_all_files()
        
        response_files = []
        for file_data in files:
            chunks = await file_manager.get_file_chunks(file_data["id"])
            
            quality_metrics = {
                "avg_chunk_quality": sum(c.get("chunk_quality_score", 0) for c in chunks) / len(chunks) if chunks else 0,
                "total_chunks": len(chunks),
                "chunks_with_tables": sum(1 for c in chunks if c.get("contains_table", False)),
                "chunks_with_lists": sum(1 for c in chunks if c.get("contains_list", False))
            }
            
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
            
            response_files.append({
                "id": file_data["id"],
                "file_name": file_data["file_name"],
                "file_hash": file_data["file_hash"],
                "file_size": file_data["file_size"],
                "file_extension": file_data["file_extension"],
                "document_type": file_data["document_type"],
                "status": file_data["status"],
                "chunk_count": file_data["chunk_count"],
                "upload_date": upload_date_str,  
                "last_chunked": last_chunked_str,  
                "error_message": file_data["error_message"],
                "processing_duration_ms": file_data.get("processing_duration_ms"),
                "quality_metrics": quality_metrics
            })
        
        return response_files
        
    except Exception as e:
        logger.error(f"❌ Error getting files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/files/{file_id}")
async def delete_file(file_id: int, permanent: bool = False):
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
        logger.error(f"❌ Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Rechunk nutzt geteilte Logik aus file_manager
@app.post("/rechunk", response_model=RechunkResponse)
async def rechunk_all_documents():
    """
    Delegiert an file_manager.rechunk_all_files() welches
    die geteilte process_parsed_document_safely() Logik verwendet.
    """
    try:
        logger.info("������ Starting rechunking process")
        
        # Delegiere an FileManager mit geteilter Verarbeitungslogik
        results = await file_manager.rechunk_all_files()
        
        logger.info(f"✅ Rechunking completed: {results['processed']} processed, {results['errors']} errors")
        
        return RechunkResponse(
            status="completed",
            total_files=results["total_files"],
            processed=results["processed"],
            errors=results["errors"],
            skipped=results.get("skipped", 0),
            details=results["details"]
        )
        
    except Exception as e:
        logger.error(f"❌ Rechunking process failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rechunking failed: {str(e)}")

# =====================================
# SUCH- UND ABFRAGE-ENDPUNKTE
# =====================================

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: Request, query_request: QueryRequest):
    query_start = time.time()
    
    try:
        embedding_start = time.time()
        query_embeddings = await generate_embeddings_only([query_request.query])
        query_embedding = query_embeddings[0]
        embedding_time = int((time.time() - embedding_start) * 1000)
        
        retrieval_start = time.time()
        similar_docs = query_chroma(
            embedding=query_embedding,
            n_results=query_request.max_results
        )
        
        filtered_docs = [
            doc for doc in similar_docs 
            if doc.get("metadata", {}).get("chunk_quality_score", 0) >= query_request.quality_threshold
        ]
        
        if query_request.file_ids:
            filtered_docs = [
                doc for doc in filtered_docs 
                if doc.get("metadata", {}).get("file_id") in query_request.file_ids
            ]
        
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        if not filtered_docs:
            answer = "Sorry, I couldn't find relevant information matching your quality criteria and query."
            sources = []
            quality_metrics = {"avg_quality": 0, "avg_similarity": 0}
        else:
            answer = await generate_answer(query_request.query, filtered_docs)
            
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
            
            quality_scores = [doc.get("metadata", {}).get("chunk_quality_score", 0) for doc in filtered_docs]
            similarities = [doc.get("similarity", 0) for doc in filtered_docs]
            
            quality_metrics = {
                "avg_quality": round(sum(quality_scores) / len(quality_scores), 3) if quality_scores else 0,
                "avg_relevance": round(sum(similarities) / len(similarities), 3) if similarities else 0, 
                "min_quality": round(min(quality_scores), 3) if quality_scores else 0,
                "max_relevance": round(max(similarities), 3) if similarities else 0 
            }
        
        total_response_time = int((time.time() - query_start) * 1000)
        
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
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/search/fulltext", response_model=FullTextSearchResponse)
async def fulltext_search(
    request: Request,
    q: str,
    file_ids: Optional[List[int]] = None,
    limit: int = 10,
    quality_threshold: float = 0.3
):
    search_start = time.time()
    
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")
        
        results = search_chunks_fulltext(
            search_term=q,
            file_ids=file_ids,
            limit=limit,
            quality_threshold=quality_threshold
        )
        
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
        logger.error(f"❌ Full-text search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Full-text search failed: {str(e)}")

# =====================================
# WEITERE ENDPUNKTE 
# =====================================

@app.post("/process_uploaded_files")
async def process_uploaded_files():
    try:
        # Delegiere an FileManager
        result = await file_manager.process_uploaded_queue()
        return result
    except Exception as e:
        logger.error(f"❌ Processing uploaded files failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics und weitere Endpunkte
@app.get("/formats")
async def get_supported_formats():
    try:
        formats = await parser_get_formats()
        return {
            "status": "success",
            "supported_formats": formats,
            "total_formats": len(formats)
        }
    except Exception as e:
        logger.error(f"❌ Error getting formats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/analytics")
async def get_system_analytics(days: int = 30):
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
        logger.error(f"❌ Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
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


# Exception Handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
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
                "architecture": "RAG System"
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


# Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("������ RAG API Backend starting up...")
    
    db_ok, db_msg = test_db_connection()
    if db_ok:
        logger.info(f"✅ Database: {db_msg}")
    else:
        logger.error(f"❌ Database: {db_msg}")
    
    chroma_ok, chroma_msg = check_chroma_connection()
    logger.info(f"{'✅' if chroma_ok else '❌'} ChromaDB: {chroma_msg}")
    
    parser_ok, parser_msg = await check_parser_connection()
    logger.info(f"{'✅' if parser_ok else '❌'} Parser: {parser_msg}")
    

    logger.info("������ Enhanced RAG API Backend ready!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("������ Enhanced RAG API Backend shutting down...")
    try:
        db_manager.close_all_connections()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing database connections: {e}")
    logger.info("������ Enhanced RAG API Backend shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=80,
        log_level="info",
        access_log=True,
        workers=1
    )