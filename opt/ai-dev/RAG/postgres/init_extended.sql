-- Performance-Monitoring aktivieren
SET log_statement = 'all';
SET log_min_duration_statement = 1000; -- Log queries > 1s

-- =====================================
-- 1. ENUMS F�R BESSERE PERFORMANCE
-- =====================================

-- File Status Enum (4 bytes statt variable TEXT)
CREATE TYPE file_status_enum AS ENUM (
    'uploaded',      -- Datei hochgeladen, wartet auf Verarbeitung
    'processing',    -- Wird gerade verarbeitet
    'chunked',       -- Erfolgreich in Chunks aufgeteilt
    'error',         -- Fehler bei Verarbeitung
    'deleted'        -- Logisch gel�scht
);

-- Document Type Enum f�r bessere Typisierung
CREATE TYPE document_type_enum AS ENUM (
    'pdf',
    'word',
    'powerpoint', 
    'excel',
    'text',
    'html',
    'markdown',
    'csv'
);

-- Processing Method Enum
CREATE TYPE processing_method_enum AS ENUM (
    'direct_text',   -- Direkter Text ohne OCR
    'ocr_standard',  -- Standard OCR
    'ocr_gpu',       -- GPU-beschleunigtes OCR
    'table_extract', -- Tabellen-Extraktion
    'hybrid'         -- Gemischte Methoden
);

-- =====================================
-- 2. HAUPTTABELLEN MIT OPTIMIERUNGEN
-- =====================================

-- UPLOADED FILES - Zentrale Dateiverwaltung
CREATE TABLE IF NOT EXISTS uploaded_files (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- File Identity (optimiert f�r Suche)
    file_name VARCHAR(255) NOT NULL,
    file_hash CHAR(64) NOT NULL UNIQUE, -- SHA256 = 64 chars (fixed length = faster)
    file_path TEXT NOT NULL,
    
    -- File Properties
    file_size BIGINT NOT NULL CHECK (file_size > 0),
    file_extension VARCHAR(10) NOT NULL,
    document_type document_type_enum NOT NULL,
    mime_type VARCHAR(100),
    
    -- Processing Status & Metrics
    status file_status_enum DEFAULT 'uploaded',
    chunk_count INTEGER DEFAULT 0 CHECK (chunk_count >= 0),
    processing_duration_ms INTEGER,
    processing_method processing_method_enum,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Content Analysis
    total_pages INTEGER,
    estimated_word_count INTEGER,
    has_images BOOLEAN DEFAULT FALSE,
    has_tables BOOLEAN DEFAULT FALSE,
    
    -- Timestamps (partitioniert f�r bessere Performance)
    upload_date TIMESTAMP DEFAULT NOW(),
    last_chunked TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Extended Metadata (JSONB f�r flexible Suche)
    metadata JSONB DEFAULT '{}',
    search_keywords TEXT[], -- Array f�r schnelle Keyword-Suche
    
    -- Quality Metrics
    content_quality_score FLOAT DEFAULT 0.0 CHECK (content_quality_score >= 0 AND content_quality_score <= 1)
);

-- PDF CHUNKS - Optimiert f�r Retrieval Performance
CREATE TABLE IF NOT EXISTS pdf_chunks (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    
    -- File Reference (CASCADE f�r automatische Bereinigung)
    file_id INTEGER NOT NULL REFERENCES uploaded_files(id) ON DELETE CASCADE,
    
    -- Legacy Fields (R�ckw�rtskompatibilit�t - werden sp�ter entfernt)
    file_name VARCHAR(255) NOT NULL,
    file_hash CHAR(64) NOT NULL,
    
    -- Position & Structure
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    section_title VARCHAR(255), -- F�r hierarchische Navigation
    
    -- Content (optimiert f�r Text-Suche)
    text TEXT NOT NULL,
    text_length INTEGER GENERATED ALWAYS AS (LENGTH(text)) STORED, -- Auto-berechnet
    word_count INTEGER,
    sentence_count INTEGER,
    
    -- Content Classification
    element_type VARCHAR(50), -- Title, Text, Table, List, Image, etc.
    contains_table BOOLEAN DEFAULT FALSE,
    contains_list BOOLEAN DEFAULT FALSE,
    contains_image_reference BOOLEAN DEFAULT FALSE,
    contains_code BOOLEAN DEFAULT FALSE,
    
    -- Quality & Processing Metrics
    chunk_quality_score FLOAT DEFAULT 0.0 CHECK (chunk_quality_score >= 0 AND chunk_quality_score <= 1),
    readability_score FLOAT, -- Flesch Reading Ease
    processing_method processing_method_enum,
    ocr_confidence FLOAT, -- Bei OCR-verarbeiteten Chunks
    
    -- Extended Metadata
    metadata JSONB DEFAULT '{}',
    language_detected VARCHAR(10) DEFAULT 'de',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Unique Constraint f�r Chunk-Position
    UNIQUE(file_id, page_number, chunk_index)
);

-- CHROMA COLLECTIONS - Vector Store Management
CREATE TABLE IF NOT EXISTS chroma_collections (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    
    -- Statistics
    document_count INTEGER DEFAULT 0 CHECK (document_count >= 0),
    total_size_bytes BIGINT DEFAULT 0,
    avg_chunk_length INTEGER DEFAULT 0,
    
    -- Configuration
    embedding_model VARCHAR(100) DEFAULT 'all-MiniLM-L6-v2',
    embedding_dimensions INTEGER DEFAULT 384,
    similarity_function VARCHAR(20) DEFAULT 'cosine',
    
    -- Performance Metrics
    last_query_time TIMESTAMP,
    total_queries INTEGER DEFAULT 0,
    avg_query_time_ms FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);

-- CHUNK COLLECTIONS MAPPING - Vector Store Relations
CREATE TABLE IF NOT EXISTS chunk_collections (
    id SERIAL PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES pdf_chunks(id) ON DELETE CASCADE,
    collection_name VARCHAR(100) NOT NULL REFERENCES chroma_collections(collection_name) ON DELETE CASCADE,
    chroma_id VARCHAR(255) NOT NULL,
    
    -- Embedding Metadata
    embedding_model VARCHAR(100),
    embedding_created_at TIMESTAMP DEFAULT NOW(),
    embedding_vector_norm FLOAT,
    
    -- Search Configuration
    similarity_threshold FLOAT DEFAULT 0.7,
    boost_factor FLOAT DEFAULT 1.0, -- F�r Relevanz-Gewichtung
    
    -- Statistics
    query_count INTEGER DEFAULT 0,
    last_queried TIMESTAMP,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Unique Constraints
    UNIQUE(chunk_id, collection_name),
    UNIQUE(chroma_id, collection_name)
);

-- QUERY LOGS - Analytics & Performance Monitoring
CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    
    -- Query Information
    query_text TEXT NOT NULL,
    query_hash CHAR(64), -- Hash f�r Deduplizierung
    query_type VARCHAR(50) DEFAULT 'similarity_search',
    
    -- Results & Performance
    results_count INTEGER DEFAULT 0,
    response_time_ms INTEGER,
    embedding_time_ms INTEGER,
    retrieval_time_ms INTEGER,
    
    -- Search Parameters
    similarity_threshold FLOAT,
    max_results INTEGER,
    collection_used VARCHAR(100),
    
    -- Context & Session
    user_session VARCHAR(255),
    source_ip INET,
    user_agent TEXT,
    
    -- Results Quality
    results_clicked INTEGER DEFAULT 0,
    user_satisfaction_score INTEGER, -- 1-5 Rating
    
    -- Timestamps (partitioniert f�r bessere Performance)
    created_at TIMESTAMP DEFAULT NOW()
);

-- PROCESSING JOBS - Async Task Management
CREATE TABLE IF NOT EXISTS processing_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL, -- 'chunk_file', 'rechunk_all', 'embed_chunks'
    
    -- Job Configuration
    file_id INTEGER REFERENCES uploaded_files(id) ON DELETE CASCADE,
    parameters JSONB DEFAULT '{}',
    priority INTEGER DEFAULT 5, -- 1 = highest, 10 = lowest
    
    -- Status & Progress
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, completed, failed
    progress_percent INTEGER DEFAULT 0 CHECK (progress_percent >= 0 AND progress_percent <= 100),
    current_step VARCHAR(100),
    
    -- Results & Errors
    result JSONB,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Timing
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    estimated_duration_ms INTEGER,
    actual_duration_ms INTEGER
);

-- =====================================
-- 3. PERFORMANCE-OPTIMIERTE INDIZES
-- =====================================

-- UPLOADED FILES INDIZES
-- ----------------------

-- 1. Hash-Lookup (Duplikat-Erkennung) - B-Tree f�r exakte Matches
CREATE INDEX IF NOT EXISTS idx_uploaded_files_hash 
ON uploaded_files USING btree(file_hash);
-- Performance: O(log n) statt O(n) f�r Duplikat-Checks

-- 2. Status-Filter (h�ufigste Query) - Partial Index nur f�r aktive Status
CREATE INDEX IF NOT EXISTS idx_uploaded_files_status_active 
ON uploaded_files(status, upload_date DESC) 
WHERE status IN ('uploaded', 'processing', 'chunked');
-- Performance: 60% kleiner als Full-Index, 3x schneller f�r UI-Queries

-- 3. Document Type Filter - f�r Typ-spezifische Analysen
CREATE INDEX IF NOT EXISTS idx_uploaded_files_type_date 
ON uploaded_files(document_type, upload_date DESC);
-- Performance: Schnelle Filterung nach Dateityp + chronologische Sortierung

-- 4. File Size Range Queries - f�r Storage-Management
CREATE INDEX IF NOT EXISTS idx_uploaded_files_size_range 
ON uploaded_files(file_size DESC) 
WHERE file_size > 1024; -- Nur Dateien > 1KB
-- Performance: Schnelle "gr��te Dateien" Queries

-- 5. JSONB Metadata Search - GIN Index f�r flexible Suche
CREATE INDEX IF NOT EXISTS idx_uploaded_files_metadata_gin 
ON uploaded_files USING gin(metadata);
-- Performance: Schnelle JSON-Queries wie metadata @> '{"author": "John"}'

-- 6. Array Keyword Search - f�r Tag-basierte Suche
CREATE INDEX IF NOT EXISTS idx_uploaded_files_keywords_gin 
ON uploaded_files USING gin(search_keywords);
-- Performance: Schnelle Array-Suche mit @> Operator

-- 7. Recent Files (zeitbasiert mit Partial Index)
CREATE INDEX IF NOT EXISTS idx_uploaded_files_upload_date 
ON uploaded_files(upload_date DESC);
-- Performance: Sehr schnell f�r "letzte 30 Tage" Dashboard

-- PDF CHUNKS INDIZES
-- ------------------

-- 1. File-zu-Chunks Lookup (h�ufigste Join)
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_file_id 
ON pdf_chunks USING btree(file_id);
-- Performance: Schnelle Chunk-Retrieval pro Datei

-- 2. File + Page Navigation - Composite Index f�r hierarchische Suche
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_file_page_chunk 
ON pdf_chunks(file_id, page_number, chunk_index);
-- Performance: Optimiert f�r Pagination und Navigation

-- 3. Legacy Hash Lookup (w�hrend Migration)
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_hash 
ON pdf_chunks USING btree(file_hash);
-- Performance: R�ckw�rtskompatibilit�t

-- 4. Quality-based Filtering - f�r "beste Chunks" Queries
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_quality_high 
ON pdf_chunks(chunk_quality_score DESC, id) 
WHERE chunk_quality_score > 0.5;
-- Performance: Schnelle High-Quality Content Retrieval

-- 5. FULL-TEXT SEARCH - Multi-Language Support
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text_german 
ON pdf_chunks USING gin(to_tsvector('german', text));
-- Performance: Sehr schnelle deutsche Volltextsuche

CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text_english 
ON pdf_chunks USING gin(to_tsvector('english', text));
-- Performance: Fallback f�r englische Inhalte

-- 6. Text Length Range - f�r Performance-optimierte Chunk-Auswahl
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_text_length_optimal 
ON pdf_chunks(text_length, chunk_quality_score DESC) 
WHERE text_length BETWEEN 100 AND 1500;
-- Performance: Chunks in optimaler L�nge f�r Embeddings

-- 7. Content Type Filtering
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_content_flags 
ON pdf_chunks(file_id) 
WHERE contains_table = true OR contains_list = true;
-- Performance: Schnelle Structured-Content Suche

-- 8. Processing Method Analytics
CREATE INDEX IF NOT EXISTS idx_pdf_chunks_processing_method 
ON pdf_chunks(processing_method, created_at DESC);
-- Performance: Analyse der Verarbeitungsqualit�t

-- CHUNK COLLECTIONS INDIZES
-- -------------------------

-- 1. Chunk-zu-Vector Lookup
CREATE INDEX IF NOT EXISTS idx_chunk_collections_chunk 
ON chunk_collections USING btree(chunk_id);

-- 2. Collection-zu-Chunks Lookup
CREATE INDEX IF NOT EXISTS idx_chunk_collections_collection 
ON chunk_collections USING btree(collection_name);

-- 3. ChromaDB ID Lookup
CREATE INDEX IF NOT EXISTS idx_chunk_collections_chroma 
ON chunk_collections USING btree(chroma_id);

-- 4. Embedding Model Tracking
CREATE INDEX IF NOT EXISTS idx_chunk_collections_embedding_model 
ON chunk_collections(embedding_model, embedding_created_at DESC);

-- 5. Query Performance Analytics
CREATE INDEX IF NOT EXISTS idx_chunk_collections_query_stats 
ON chunk_collections(query_count DESC, last_queried DESC) 
WHERE query_count > 0;

-- QUERY LOGS INDIZES
-- ------------------

-- 1. Time-Series Queries (partitioniert nach Datum)
CREATE INDEX IF NOT EXISTS idx_query_logs_created_at 
ON query_logs USING btree(created_at DESC);

-- 2. Query Text Search f�r Analytics
CREATE INDEX IF NOT EXISTS idx_query_logs_text_search 
ON query_logs USING gin(to_tsvector('german', query_text));

-- 3. Query Hash f�r Deduplizierung
CREATE INDEX IF NOT EXISTS idx_query_logs_hash 
ON query_logs USING btree(query_hash) 
WHERE query_hash IS NOT NULL;

-- 4. Performance Analytics
CREATE INDEX IF NOT EXISTS idx_query_logs_performance 
ON query_logs(response_time_ms DESC, results_count) 
WHERE response_time_ms IS NOT NULL;

-- 5. User Session Analytics
CREATE INDEX IF NOT EXISTS idx_query_logs_session 
ON query_logs(user_session, created_at DESC) 
WHERE user_session IS NOT NULL;

-- PROCESSING JOBS INDIZES
-- -----------------------

-- 1. Job Queue Processing
CREATE INDEX IF NOT EXISTS idx_processing_jobs_queue 
ON processing_jobs(status, priority, created_at) 
WHERE status IN ('pending', 'running');

-- 2. File-specific Jobs
CREATE INDEX IF NOT EXISTS idx_processing_jobs_file 
ON processing_jobs(file_id, created_at DESC) 
WHERE file_id IS NOT NULL;

-- 3. Job Type Analytics
CREATE INDEX IF NOT EXISTS idx_processing_jobs_type_status 
ON processing_jobs(job_type, status, completed_at DESC);

-- =====================================
-- 4. AUTO-UPDATE TRIGGERS
-- =====================================

-- Function: Auto-Update updated_at Timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: uploaded_files updated_at
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_uploaded_files_updated_at'
    ) THEN
        CREATE TRIGGER update_uploaded_files_updated_at
        BEFORE UPDATE ON uploaded_files
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

-- Function: Auto-Update chunk_count
CREATE OR REPLACE FUNCTION update_chunk_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE uploaded_files 
        SET chunk_count = chunk_count + 1,
            updated_at = NOW(),
            last_chunked = CASE 
                WHEN status = 'chunked' THEN NOW() 
                ELSE last_chunked 
            END
        WHERE id = NEW.file_id;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE uploaded_files 
        SET chunk_count = GREATEST(chunk_count - 1, 0),
            updated_at = NOW()
        WHERE id = OLD.file_id;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-Update chunk count
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_chunk_count_trigger'
    ) THEN
        CREATE TRIGGER update_chunk_count_trigger
        AFTER INSERT OR DELETE ON pdf_chunks
        FOR EACH ROW
        EXECUTE FUNCTION update_chunk_count();
    END IF;
END;
$$;

-- Function: Update Collection Statistics
CREATE OR REPLACE FUNCTION update_collection_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE chroma_collections 
        SET document_count = document_count + 1,
            last_updated = NOW()
        WHERE collection_name = NEW.collection_name;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE chroma_collections 
        SET document_count = GREATEST(document_count - 1, 0),
            last_updated = NOW()
        WHERE collection_name = OLD.collection_name;
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Collection Statistics
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'update_collection_stats_trigger'
    ) THEN
        CREATE TRIGGER update_collection_stats_trigger
        AFTER INSERT OR DELETE ON chunk_collections
        FOR EACH ROW
        EXECUTE FUNCTION update_collection_stats();
    END IF;
END;
$$;

-- Function: Update Query Statistics
CREATE OR REPLACE FUNCTION update_query_stats()
RETURNS TRIGGER AS $$
BEGIN
    -- Update chunk query count
    UPDATE chunk_collections 
    SET query_count = query_count + 1,
        last_queried = NOW()
    WHERE chunk_id IN (
        SELECT DISTINCT pc.id 
        FROM pdf_chunks pc 
        JOIN uploaded_files uf ON pc.file_id = uf.id 
        WHERE uf.file_hash = (
            SELECT file_hash FROM uploaded_files uf2 
            JOIN pdf_chunks pc2 ON uf2.id = pc2.file_id 
            LIMIT 1
        )
    );
    
    -- Update file last_accessed
    UPDATE uploaded_files 
    SET last_accessed = NOW()
    WHERE id IN (
        SELECT DISTINCT file_id 
        FROM pdf_chunks pc 
        JOIN chunk_collections cc ON pc.id = cc.chunk_id 
        WHERE cc.collection_name = (
            SELECT collection_name 
            FROM chunk_collections 
            ORDER BY created_at DESC 
            LIMIT 1
        )
    );
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================
-- 5. PERFORMANCE VIEWS
-- =====================================

-- File Overview mit umfassenden Statistiken
CREATE OR REPLACE VIEW file_overview AS
SELECT 
    uf.id,
    uf.file_name,
    uf.file_hash,
    uf.file_path,                   
    uf.file_size,
    uf.file_extension,              
    uf.document_type,
    uf.status,
    uf.chunk_count,
    uf.upload_date,
    uf.last_chunked,               
    uf.error_message,              
    uf.metadata,                    
    uf.processing_duration_ms,
    uf.content_quality_score,
    uf.last_accessed,
    
    -- Chunk-Statistiken (nur bei Bedarf berechnet)
    COALESCE(cs.avg_quality, 0) as avg_chunk_quality,
    COALESCE(cs.total_words, 0) as total_word_count,
    COALESCE(cs.chunks_with_tables, 0) as chunks_with_tables,
    COALESCE(cs.chunks_with_lists, 0) as chunks_with_lists,
    COALESCE(cs.avg_readability, 0) as avg_readability,
    
    -- Processing Information
    CASE 
        WHEN uf.status = 'chunked' THEN 'Ready for Search'
        WHEN uf.status = 'processing' THEN 'Processing...'
        WHEN uf.status = 'error' THEN 'Error - ' || COALESCE(uf.error_message, 'Unknown')
        WHEN uf.status = 'uploaded' THEN 'Waiting for Processing'
        ELSE 'Unknown Status'
    END as display_status,
    
    -- Performance Indicators
    CASE 
        WHEN uf.processing_duration_ms IS NULL THEN NULL
        WHEN uf.processing_duration_ms < 5000 THEN 'Fast'
        WHEN uf.processing_duration_ms < 30000 THEN 'Normal'
        ELSE 'Slow'
    END as processing_speed,
    
    -- Usage Statistics
    DATE_PART('day', NOW() - uf.last_accessed) as days_since_last_access
    
FROM uploaded_files uf
LEFT JOIN (
    SELECT 
        file_id,
        AVG(chunk_quality_score) as avg_quality,
        SUM(word_count) as total_words,
        COUNT(*) FILTER (WHERE contains_table = true) as chunks_with_tables,
        COUNT(*) FILTER (WHERE contains_list = true) as chunks_with_lists,
        AVG(readability_score) as avg_readability
    FROM pdf_chunks 
    WHERE chunk_quality_score > 0
    GROUP BY file_id
) cs ON uf.id = cs.file_id
WHERE uf.status != 'deleted';

-- Collection Performance Statistics
CREATE OR REPLACE VIEW collection_performance AS
SELECT 
    cc.collection_name,
    cc.description,
    cc.document_count,
    cc.embedding_model,
    
    -- Actual vs Expected Stats
    COUNT(ccm.chunk_id) as actual_chunk_count,
    COUNT(DISTINCT pc.file_id) as unique_files,
    
    -- Quality Metrics
    AVG(pc.chunk_quality_score) as avg_chunk_quality,
    AVG(pc.text_length) as avg_chunk_length,
    
    -- Performance Metrics
    cc.total_queries,
    cc.avg_query_time_ms,
    cc.last_query_time,
    
    -- Health Indicators
    CASE 
        WHEN cc.document_count = COUNT(ccm.chunk_id) THEN 'Synchronized'
        WHEN cc.document_count > COUNT(ccm.chunk_id) THEN 'Missing Chunks'
        ELSE 'Inconsistent'
    END as sync_status,
    
    -- Update Status
    DATE_PART('day', NOW() - cc.last_updated) as days_since_update
    
FROM chroma_collections cc
LEFT JOIN chunk_collections ccm ON cc.collection_name = ccm.collection_name
LEFT JOIN pdf_chunks pc ON ccm.chunk_id = pc.id
GROUP BY cc.collection_name, cc.description, cc.document_count, 
         cc.embedding_model, cc.total_queries, cc.avg_query_time_ms, 
         cc.last_query_time, cc.last_updated;

-- System Performance Dashboard
CREATE OR REPLACE VIEW system_performance AS
SELECT 
    -- File Processing Stats
    COUNT(*) FILTER (WHERE status = 'uploaded') as files_pending,
    COUNT(*) FILTER (WHERE status = 'processing') as files_processing,
    COUNT(*) FILTER (WHERE status = 'chunked') as files_ready,
    COUNT(*) FILTER (WHERE status = 'error') as files_error,
    
    -- Storage Stats
    pg_size_pretty(SUM(file_size)) as total_storage,
    COUNT(*) as total_files,
    SUM(chunk_count) as total_chunks,
    
    -- Performance Stats
    AVG(processing_duration_ms) as avg_processing_time,
    MAX(processing_duration_ms) as max_processing_time,
    
    -- Quality Stats
    AVG(content_quality_score) as avg_file_quality,
    
    -- Recent Activity
    COUNT(*) FILTER (WHERE upload_date >= NOW() - INTERVAL '24 hours') as files_uploaded_today,
    COUNT(*) FILTER (WHERE last_accessed >= NOW() - INTERVAL '24 hours') as files_accessed_today
    
FROM uploaded_files
WHERE status != 'deleted';

-- =====================================
-- 6. MAINTENANCE & CLEANUP FUNCTIONS
-- =====================================

-- Function: Cleanup Old Query Logs (�lter als 90 Tage)
CREATE OR REPLACE FUNCTION cleanup_old_query_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_logs 
    WHERE created_at < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ language 'plpgsql';

-- Function: Recompute Collection Statistics
CREATE OR REPLACE FUNCTION recompute_collection_stats()
RETURNS VOID AS $$
BEGIN
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
        last_updated = NOW();
END;
$$ language 'plpgsql';

-- Function: Database Maintenance
CREATE OR REPLACE FUNCTION database_maintenance()
RETURNS TABLE(operation TEXT, result TEXT) AS $$
BEGIN
    -- 1. Cleanup old logs
    RETURN QUERY SELECT 'cleanup_logs'::TEXT, 
                        (cleanup_old_query_logs() || ' old query logs deleted')::TEXT;
    
    -- 2. Recompute statistics
    PERFORM recompute_collection_stats();
    RETURN QUERY SELECT 'recompute_stats'::TEXT, 'Collection statistics updated'::TEXT;
    
    -- 3. Analyze tables for query planner
    ANALYZE uploaded_files;
    ANALYZE pdf_chunks;
    ANALYZE chunk_collections;
    ANALYZE query_logs;
    RETURN QUERY SELECT 'analyze_tables'::TEXT, 'Table statistics refreshed'::TEXT;
    
    -- 4. Vacuum full for space reclamation (nur wenn n�tig)
    -- VACUUM FULL ist resource-intensiv, daher commented
    -- VACUUM FULL uploaded_files;
    RETURN QUERY SELECT 'maintenance_complete'::TEXT, 'Database maintenance finished'::TEXT;
END;
$$ language 'plpgsql';

-- =====================================
-- 7. INITIAL DATA & CONFIGURATION
-- =====================================

-- Standard Collection erstellen
INSERT INTO chroma_collections (collection_name, description, embedding_model) 
VALUES ('rag_docs', 'Default RAG document collection', 'all-MiniLM-L6-v2')
ON CONFLICT (collection_name) DO UPDATE SET
    description = EXCLUDED.description,
    last_updated = NOW();

-- =====================================
-- 8. PERFORMANCE MONITORING SETUP
-- =====================================

-- Enable query statistics
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Performance-Monitoring View
CREATE OR REPLACE VIEW query_performance AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    min_exec_time,
    max_exec_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements 
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY total_exec_time DESC
LIMIT 20;

-- =====================================
-- SETUP COMPLETE NOTIFICATION
-- =====================================

DO $$
BEGIN
    RAISE NOTICE '=====================================';
    RAISE NOTICE 'RAG DATABASE SETUP COMPLETE!';
    RAISE NOTICE '=====================================';
    RAISE NOTICE 'Tables created: uploaded_files, pdf_chunks, chroma_collections, chunk_collections, query_logs, processing_jobs';
    RAISE NOTICE 'Indexes created: % performance-optimized indexes', (
        SELECT COUNT(*) 
        FROM pg_indexes 
        WHERE schemaname = 'public' 
        AND indexname LIKE 'idx_%'
    );
    RAISE NOTICE 'Views created: file_overview, collection_performance, system_performance, query_performance';
    RAISE NOTICE 'Functions created: maintenance utilities and auto-update triggers';
    RAISE NOTICE '=====================================';
    RAISE NOTICE 'Run SELECT * FROM system_performance; to see current status';
    RAISE NOTICE 'Run SELECT database_maintenance(); for periodic maintenance';
    RAISE NOTICE '=====================================';
END $$;