# =====================================
# DOCUMENT PROCESSOR SERVICE
# =====================================

import logging
import time
from typing import Dict, Any, Tuple, List
from datetime import datetime

# Core Services
from services.db import insert_chunk_metadata, db_manager
from services.retrieval import add_documents_to_chroma
from utils.advanced_chunking import process_parsed_chunks, get_chunk_statistics

# Logger für strukturierte Protokollierung
logger = logging.getLogger(__name__)

def validate_quality_scores(chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validiert Quality-Scores auf Konsistenz"""
    if not chunks:
        return True, "No chunks to validate"
    
    for i, chunk in enumerate(chunks):
        quality = chunk.get("metadata", {}).get("chunk_quality_score", 0)
        if not isinstance(quality, (int, float)):
            return False, f"Chunk {i}: Quality score not numeric: {type(quality)}"
        if not (0.1 <= quality <= 1.0):
            return False, f"Chunk {i}: Quality score out of range: {quality}"
    
    avg_quality = sum(chunk.get("metadata", {}).get("chunk_quality_score", 0) for chunk in chunks) / len(chunks)
    if avg_quality < 0.3:
        logger.warning(f"⚠️ Low average quality: {avg_quality:.3f}")
    
    return True, f"Validated {len(chunks)} chunks, avg quality: {avg_quality:.3f}"

async def process_parsed_document_safely(
    file_id: int, 
    file_info: dict, 
    parse_result: dict, 
    filename: str,
    file_manager=None  
) -> Tuple[bool, Any]:
    """
    SINGLE SOURCE OF TRUTH für Dokumenten-Verarbeitung
    
    Diese zentrale Funktion wird von allen Document-Processing-Workflows verwendet:
    - main.py → upload_document() 
    - file_manager.py → rechunk_all_files()
    - Zukünftige Batch-Processing-Funktionen
    
    
    Args:
        file_id: Eindeutige Datei-ID in der Datenbank
        file_info: Dictionary mit Datei-Metadaten (hash, extension, etc.)
        parse_result: Ergebnis vom Enhanced Parser (chunks + embeddings)
        filename: Name der verarbeiteten Datei
        file_manager: Optional für Status-Updates (wird automatisch importiert falls None)
    
    Returns:
        Tuple[bool, Any]: (Erfolg, Ergebnis-Dict oder Fehlermeldung)
    """
    
    # Lazy import um zirkuläre Abhängigkeiten zu vermeiden
    if file_manager is None:
        from services.file_manager import file_manager as fm
        file_manager = fm
    
    processing_start = time.time()
    
    try:
        logger.info(f"������ Processing document for file {file_id}: {filename}")
        
        # =====================================
        # PHASE 1: PARSE-RESULT-VALIDIERUNG
        # =====================================
        
        if not parse_result.get("success", False):
            error_msg = f"Document parsing failed: {parse_result.get('error', 'Unknown parsing error')}"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Extrahiere Chunks und Embeddings
        raw_chunks = parse_result.get("chunks", [])
        embeddings = parse_result.get("embeddings", [])
        
        # KRITISCHE VALIDIERUNG: Chunk-Count
        if not raw_chunks:
            error_msg = "No chunks extracted from document"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # KRITISCHE VALIDIERUNG: Chunk-Embedding-Synchronisation
        if len(raw_chunks) != len(embeddings):
            error_msg = f"CRITICAL: Chunk-embedding count mismatch! Chunks: {len(raw_chunks)}, Embeddings: {len(embeddings)}"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        logger.info(f"✅ Validation passed: {len(raw_chunks)} chunks, {len(embeddings)} embeddings")
        
        # =====================================
        # PHASE 2: ENHANCED CHUNK-VERARBEITUNG
        # =====================================
        
        # Sichere Chunk-Verarbeitung mit Quality-Enhancement
        processed_chunks = []
        for chunk in raw_chunks:
            if isinstance(chunk, dict):
                processed_chunks.append(chunk)
            else:
                # Konvertiere nicht-dict Objekte zu Standard-Format
                safe_chunk = {
                    "text": str(chunk.get("text", "")) if hasattr(chunk, "get") else str(chunk),
                    "metadata": chunk.get("metadata", {}) if hasattr(chunk, "get") else {}
                }
                processed_chunks.append(safe_chunk)
        
        # Erweiterte Chunk-Verarbeitung mit Quality-Berechnung
        enhanced_chunks = process_parsed_chunks(processed_chunks)
        
        if not enhanced_chunks:
            error_msg = "No valid chunks after enhanced processing"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # FINAL-VALIDIERUNG: Count nach Processing
        if len(enhanced_chunks) != len(embeddings):
            error_msg = f"CRITICAL: Processing altered chunk count: {len(enhanced_chunks)} vs {len(embeddings)} embeddings"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        # Berechne Qualitätsmetriken für Reporting
        quality_stats = get_chunk_statistics(enhanced_chunks)
        logger.info(f"������ Quality metrics: avg_quality={quality_stats.get('average_quality_score', 0):.3f}")
        
        # =====================================
        # PHASE 3: DUAL-STORAGE-VORBEREITUNG
        # =====================================
        
        # ChromaDB-Daten (Vektor-Datenbank)
        documents = []          # Text-Inhalte
        metadatas = []          # Strukturierte Metadaten
        ids = []               # Eindeutige Chunk-IDs
        chunk_embeddings = []   # 384-dimensionale Vektoren
        
        # PostgreSQL-Daten (Metadata-Datenbank)
        chunk_metadata_list = []
        
        # =====================================
        # PHASE 4: CHUNK-ITERATION UND DATEN-AUFBEREITUNG
        # =====================================
        
        for idx, (chunk, embedding) in enumerate(zip(enhanced_chunks, embeddings)):
            text = chunk["text"]
            metadata = chunk["metadata"]
            page = metadata.get("page_number", 1)
            
            # ChromaDB-Metadaten (für Vektor-Suche)
            chroma_metadata = {
                "source": filename,
                "page": page,
                "chunk_index": idx,
                "file_extension": file_info.get("file_extension", ""),
                "document_type": file_info.get("document_type", "pdf"),
                "file_id": file_id,
                "chunk_quality_score": metadata.get("chunk_quality_score", 0.5),  # ✅ Quality für Filtering
                **metadata  # Merge alle zusätzlichen Metadaten
            }
            
            # ChromaDB-Daten sammeln
            documents.append(text)
            metadatas.append(chroma_metadata)
            
            # Eindeutige Chunk-ID für ChromaDB
            chunk_id = f"{filename}_{file_info['file_hash']}_{idx}"
            ids.append(chunk_id)
            chunk_embeddings.append(embedding)
            
            # PostgreSQL-Chunk-Daten mit KORREKTEM Quality-Key
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
                'contains_code': metadata.get("contains_code", False),
                'chunk_quality_score': metadata.get("chunk_quality_score", 0.5),  # ✅ KRITISCHER FIX!
                'readability_score': metadata.get("readability_score"),
                'processing_method': 'hybrid',
                'ocr_confidence': metadata.get("ocr_confidence"),
                'metadata': metadata,
                'language_detected': metadata.get("language_detected", 'de'),
                'section_title': metadata.get("section_title")
            }
            
            chunk_metadata_list.append((idx, chunk_data))
        
        # =====================================
        # PHASE 5: POSTGRESQL-SPEICHERUNG (ATOMISCH)
        # =====================================
        
        logger.info(f"������ Saving {len(chunk_metadata_list)} chunks to PostgreSQL...")
        
        success_count = 0
        for idx, chunk_data in chunk_metadata_list:
            try:
                # Verwende korrekte Funktionssignatur: file_id zuerst, dann chunk_data
                success = insert_chunk_metadata(file_id, chunk_data)
                if success:
                    success_count += 1
                else:
                    error_msg = f"Failed to insert chunk {idx} for file {file_id}"
                    logger.error(f"❌ {error_msg}")
                    await file_manager.update_file_status(file_id, "error", error_message=error_msg)
                    return False, error_msg
                    
            except Exception as e:
                error_msg = f"Exception inserting chunk {idx}: {str(e)}"
                logger.error(f"❌ {error_msg}")
                await file_manager.update_file_status(file_id, "error", error_message=error_msg)
                return False, error_msg
        
        logger.info(f"✅ Successfully saved {success_count}/{len(chunk_metadata_list)} chunks to PostgreSQL")
        
        # =====================================
        # PHASE 6: CHROMADB-SPEICHERUNG (BATCH)
        # =====================================
        
        logger.info(f"������ Saving {len(chunk_embeddings)} embeddings to ChromaDB...")
        
        try:
            chroma_success = add_documents_to_chroma(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=chunk_embeddings
            )
            
            if not chroma_success:
                error_msg = "Failed to save embeddings to ChromaDB vector database"
                logger.error(f"❌ {error_msg}")
                
                # Rollback: Entferne PostgreSQL-Chunks bei ChromaDB-Fehler
                try:
                    with db_manager.get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("DELETE FROM pdf_chunks WHERE file_id = %s", (file_id,))
                        conn.commit()
                        cur.close()
                    logger.info("������ Rolled back PostgreSQL chunks due to ChromaDB failure")
                except Exception as rollback_error:
                    logger.error(f"❌ Rollback failed: {rollback_error}")
                
                await file_manager.update_file_status(file_id, "error", error_message=error_msg)
                return False, error_msg
                
        except Exception as e:
            error_msg = f"ChromaDB storage exception: {str(e)}"
            logger.error(f"❌ {error_msg}")
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
            return False, error_msg
        
        logger.info(f"✅ Successfully saved embeddings to ChromaDB")
        
        # =====================================
        # PHASE 7: SUCCESS-FINALISIERUNG
        # =====================================
        
        processing_time = int((time.time() - processing_start) * 1000)
        
        # Debug-Logging für Quality-Scores
        quality_scores = [chunk["metadata"].get("chunk_quality_score", 0) for chunk in enhanced_chunks]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        logger.info(f"������ Average quality score: {avg_quality:.3f}")
        logger.info(f"������ Quality distribution: min={min(quality_scores):.3f}, max={max(quality_scores):.3f}")
        
        # Finale Erfolgs-Metadaten
        result_metrics = {
            "chunks": len(enhanced_chunks),
            "processing_time_ms": processing_time,
            "quality_metrics": quality_stats,
            "storage": {
                "postgresql_chunks": success_count,
                "chromadb_embeddings": len(chunk_embeddings),
                "dual_storage_success": True
            },
            "quality_analysis": {
                "avg_quality": avg_quality,
                "min_quality": min(quality_scores),
                "max_quality": max(quality_scores),
                "high_quality_chunks": sum(1 for q in quality_scores if q > 0.7)
            }
        }
        
        logger.info(f"������ Successfully processed {filename}: {len(enhanced_chunks)} chunks in {processing_time}ms")
        return True, result_metrics
        
    except Exception as e:
        # =====================================
        # COMPREHENSIVE ERROR HANDLING
        # =====================================
        
        processing_time = int((time.time() - processing_start) * 1000)
        error_msg = f"Document processing failed after {processing_time}ms: {str(e)}"
        logger.error(f"������ Critical error processing file {file_id}: {e}")
        
        # Versuche Status-Update (best effort)
        try:
            await file_manager.update_file_status(file_id, "error", error_message=error_msg)
        except Exception as status_error:
            logger.error(f"❌ Could not update file status: {status_error}")
        
        return False, error_msg

# =====================================
# HELPER FUNCTIONS
# =====================================

def validate_chunk_quality_distribution(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validiert die Verteilung der Quality-Scores für Debugging
    
    Args:
        chunks: Liste der verarbeiteten Chunks mit Metadaten
    
    Returns:
        Dict mit Quality-Analyse für Logging und Debugging
    """
    if not chunks:
        return {"status": "empty", "message": "No chunks to analyze"}
    
    quality_scores = []
    for chunk in chunks:
        quality = chunk.get("metadata", {}).get("chunk_quality_score", 0)
        quality_scores.append(quality)
    
    if not quality_scores:
        return {"status": "no_quality", "message": "No quality scores found"}
    
    # Statistiken berechnen
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    # Qualitäts-Buckets
    high_quality = sum(1 for q in quality_scores if q >= 0.7)
    medium_quality = sum(1 for q in quality_scores if 0.4 <= q < 0.7)
    low_quality = sum(1 for q in quality_scores if q < 0.4)
    
    return {
        "status": "analyzed",
        "total_chunks": len(quality_scores),
        "average_quality": round(avg_quality, 3),
        "min_quality": round(min(quality_scores), 3),
        "max_quality": round(max(quality_scores), 3),
        "distribution": {
            "high_quality": high_quality,
            "medium_quality": medium_quality, 
            "low_quality": low_quality
        },
        "quality_scores_sample": quality_scores[:10]  # Erste 10 für Debugging
    }

def log_processing_performance(
    file_id: int,
    filename: str, 
    start_time: float,
    chunks_count: int,
    embeddings_count: int
) -> None:
    """
    Loggt Performance-Metriken für Monitoring und Optimierung
    
    Args:
        file_id: Datei-ID
        filename: Dateiname
        start_time: Start-Zeitstempel
        chunks_count: Anzahl verarbeiteter Chunks  
        embeddings_count: Anzahl generierter Embeddings
    """
    
    processing_time = time.time() - start_time
    
    # Performance-Klassifikation
    if processing_time < 5:
        performance_class = "������ Fast"
    elif processing_time < 30:
        performance_class = "⚡ Normal"
    elif processing_time < 120:
        performance_class = "������ Slow"
    else:
        performance_class = "������ Very Slow"
    
    # Chunks-per-Second-Metrik
    chunks_per_second = chunks_count / processing_time if processing_time > 0 else 0
    
    logger.info(f"������ Performance Report for {filename}:")
    logger.info(f"   ������ File ID: {file_id}")
    logger.info(f"   ⏱️  Total Time: {processing_time:.2f}s ({performance_class})")
    logger.info(f"   ������ Chunks: {chunks_count} ({chunks_per_second:.1f} chunks/sec)")
    logger.info(f"   ������ Embeddings: {embeddings_count}")
    logger.info(f"   ������ Storage: Dual (PostgreSQL + ChromaDB)")

# =====================================
# VALIDATION FUNCTIONS
# =====================================

def validate_file_info(file_info: dict) -> Tuple[bool, str]:
    """
    Validiert file_info Dictionary auf Vollständigkeit
    
    Args:
        file_info: Dictionary mit Datei-Metadaten
    
    Returns:
        Tuple[bool, str]: (Valid, Fehlermeldung falls invalid)
    """
    required_fields = ["file_hash", "file_extension"]
    optional_fields = ["document_type", "file_size", "file_name"]
    
    # Prüfe Required Fields
    for field in required_fields:
        if field not in file_info or not file_info[field]:
            return False, f"Missing required field: {field}"
    
    # Validiere Hash-Format (SHA256 = 64 Hex-Zeichen)
    file_hash = file_info["file_hash"]
    if len(file_hash) != 64 or not all(c in "0123456789abcdef" for c in file_hash.lower()):
        return False, f"Invalid file_hash format: {file_hash}"
    
    # Validiere Extension-Format
    file_ext = file_info["file_extension"]
    if not file_ext.startswith(".") or len(file_ext) < 2:
        return False, f"Invalid file_extension format: {file_ext}"
    
    return True, "Valid"

def validate_parse_result(parse_result: dict) -> Tuple[bool, str]:
    """
    Validiert parse_result Dictionary vom Enhanced Parser
    
    Args:
        parse_result: Ergebnis vom Document Parser
    
    Returns:
        Tuple[bool, str]: (Valid, Fehlermeldung falls invalid)
    """
    # Prüfe Success-Flag
    if not parse_result.get("success", False):
        error = parse_result.get("error", "Unknown parsing error")
        return False, f"Parser reported failure: {error}"
    
    # Prüfe Chunks
    chunks = parse_result.get("chunks", [])
    if not chunks:
        return False, "No chunks in parse result"
    
    # Prüfe Embeddings (falls generate_embeddings=True)
    embeddings = parse_result.get("embeddings", [])
    if embeddings and len(chunks) != len(embeddings):
        return False, f"Chunk-embedding mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
    
    # Prüfe Chunk-Format
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            return False, f"Chunk {i} is not a dictionary: {type(chunk)}"
        
        if "text" not in chunk:
            return False, f"Chunk {i} missing 'text' field"
        
        if not chunk["text"] or not chunk["text"].strip():
            logger.warning(f"⚠️ Chunk {i} has empty text")
    
    return True, "Valid"

# =====================================
# MODULE INITIALIZATION
# =====================================

# Logging-Setup für das Modul
logger.info("������ Document Processor Service initialized")
logger.info("✅ Features: Quality scoring, Dual storage, Error recovery")
logger.info("������ Architecture: Single Source of Truth pattern")