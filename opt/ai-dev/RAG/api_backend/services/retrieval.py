# =====================================
# CHROMADB RETRIEVAL SERVICE - VEKTOR-DATENBANK-INTERFACE
# =====================================

import chromadb                     
from chromadb.config import Settings  
import os                          


# Connection-Management

def get_chroma_client():
    """
    Erstellt ChromaDB-Client mit einheitlicher Service-Discovery-Konfiguration.
    
    Diese Funktion implementiert das zentrale Connection-Management für alle
    ChromaDB-Operationen im RAG-System mit folgenden Design-Prinzipien:
    
    Rückgabe:
        chromadb.HttpClient: Konfigurierter Client für alle Vektor-Operationen
    """
    chroma_host = os.getenv("CHROMA_HOST", "rag_chromadb")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    return chromadb.HttpClient(
        host=chroma_host,           # Hostname oder IP für ChromaDB-Service
        port=int(chroma_port)       # Port-Konvertierung für Type-Safety
    )

def get_collection(collection_name: str = "rag_docs"):
    """
    Holt bestehende Collection oder erstellt neue mit Exception-sicherer Logik.
    
    Diese Funktion implementiert das Collection-Management-Pattern mit
    automatischer Collection-Erstellung bei Bedarf. Design-Prinzipien:
    
    Parameter:
        collection_name (str): Logischer Name für Dokument-Gruppierung
                              Sollte aussagekräftig und eindeutig sein
    
    Rückgabe:
        chromadb.Collection: Collection-Objekt für Dokument-Operationen
    """
    # CLIENT-VERBINDUNG: Zentralisierte Connection-Logik wiederverwenden
    client = get_chroma_client()
    
    try:
        # COLLECTION-RETRIEVAL: Versuche bestehende Collection zu laden
        collection = client.get_collection(collection_name)
    except Exception:
        # AUTOMATIC-CREATION: Collection existiert nicht, erstelle neuen
        collection = client.create_collection(collection_name)
    
    return collection

# Retrieval-Funktionen

def query_chroma(embedding: list[float], n_results: int = 5, collection_name: str = "rag_docs"):
    """
    Führt semantische Similarity-Search in ChromaDB für RAG-Retrieval durch.

    Parameter:
        embedding (list[float]): 384-dimensionaler Query-Vektor aus Embedding-Model
                                Muss kompatibel mit gespeicherten Dokument-Embeddings sein
        
        n_results (int): Anzahl zurückzugebender ähnlichster Dokumente
                        5 = Optimaler Default für LLM-Context-Länge
                        Höhere Werte für umfassendere Kontexte
        
        collection_name (str): Ziel-Collection für Suche
                              Ermöglicht Multi-Collection-Deployments
    
    Rückgabe:
        list[dict]: Sortierte Liste der ähnlichsten Dokumente mit Struktur:
                   [{"text": str, "metadata": dict, "distance": float, "similarity": float}]
                   Sortiert nach absteigender Similarity (höchste zuerst)
    """
    try:
        # COLLECTION-ACCESS: Sichere Collection-Referenz mit Auto-Creation
        collection = get_collection(collection_name)
        
        # VEKTOR-SUCHE: ChromaDB Query API für Similarity Search
        # query_embeddings als Liste ermöglicht Batch-Queries in zukünftigen Versionen
        results = collection.query(
            query_embeddings=[embedding],                               # Single-Query-Embedding als Liste
            n_results=n_results,                                       # Top-K Limitierung
            include=["metadatas", "documents", "distances"]            # Vollständige Daten für Kontext-Assembly
        )
        
        # ERGEBNIS-NORMALISIERUNG: ChromaDB-Format → Standard-Format-Transformation
        documents = []
        
        # SAFETY-CHECK: Validierung der Response-Struktur gegen leere Ergebnisse
        if results["documents"] and len(results["documents"]) > 0:
            
            # RESULT-ITERATION: Jedes gefundene Dokument in Standard-Format konvertieren
            for i, doc in enumerate(results["documents"][0]):
                
                # METADATEN-EXTRAKTION: Sichere Navigation durch nested structures
                # Fallback auf leeres dict verhindert KeyError-Exceptions
                metadata = results["metadatas"][0][i] if results["metadatas"] and len(results["metadatas"][0]) > i else {}
                
                # DISTANCE-EXTRAKTION: Cosine Distance von ChromaDB
                # 0.0 = identisch, 2.0 = maximal verschiedene (für normalisierte Vektoren)
                distance = results["distances"][0][i] if results["distances"] and len(results["distances"][0]) > i else 0.0
                
                # DOKUMENT-STRUKTURIERUNG: Einheitliches Format für RAG-Pipeline
                documents.append({
                    "text": doc,                                       # Original-Dokument-Text
                    "metadata": metadata,                              # Source, Page, etc. für Attribution
                    "distance": distance,                              # Raw Cosine Distance
                    "similarity": 1.0 - distance                      # Intuitive Similarity Score (1.0 = perfekt, 0.0 = unähnlich)
                })
        
        return documents
        
    except Exception as e:
        # ERROR-HANDLING: Graceful Degradation bei Vektor-DB-Fehlern
        # Leere Liste ermöglicht downstream-Processing ohne Crash
        print(f"Error querying ChromaDB: {e}")
        return []

# Doklumentenmanagement

def add_documents_to_chroma(documents: list[str], metadatas: list[dict], ids: list[str], 
                           embeddings: list[list[float]], collection_name: str = "rag_docs"):
    """
    Fügt Batch von Dokumenten mit Embeddings zu ChromaDB hinzu.
    
    Diese Funktion implementiert effizientes Bulk-Loading für die initiale
    Datenbank-Bevölkerung und kontinuierliche Dokument-Aufnahme:
    
    Parameter:
        documents (list[str]): Text-Inhalte der Dokument-Chunks
                              Sollten optimal für LLM-Kontext-Länge segmentiert sein
        
        metadatas (list[dict]): Strukturierte Metadaten pro Dokument
                               Standard-Keys: source, page, chunk_index, file_type
        
        ids (list[str]): Eindeutige Identifikatoren pro Dokument
                        Essentiell für Updates, Deletions und Referenzen
        
        embeddings (list[list[float]]): Vektor-Repräsentationen der Dokumente
                                       384-dimensional für all-MiniLM-L6-v2
        
        collection_name (str): Ziel-Collection für Batch-Insert
    
    Rückgabe:
        bool: True bei erfolgreichem Batch-Insert, False bei Fehlern
    """
    try:
        # COLLECTION-ACCESS: Thread-safe Collection-Referenz
        collection = get_collection(collection_name)
        
        # BULK-INSERT: Atomische Batch-Operation für alle Dokumente
        # ChromaDB optimiert interne Index-Updates für Batch-Operationen
        collection.add(
            documents=documents,        # Text-Content für Full-Text-Fallbacks
            metadatas=metadatas,       # Strukturierte Attribution und Filtering
            ids=ids,                   # Eindeutige Referenzen für CRUD-Operationen
            embeddings=embeddings      # Hochdimensionale Vektor-Repräsentationen
        )
        return True
        
    except Exception as e:
        # BATCH-ERROR-HANDLING: Detaillierte Fehler-Protokollierung
        # Wichtig für Debugging von Datenformat- oder Kapazitäts-Problemen
        print(f"Error adding documents to ChromaDB: {e}")
        return False

def remove_documents_from_chroma(file_hash: str, collection_name: str = "rag_docs"):
    """
    Entfernt alle Dokument-Chunks einer Datei basierend auf File-Hash.
    
    Diese Funktion implementiert Hash-basierte Bulk-Deletion für komplette
    Datei-Entfernung aus der Vektor-Datenbank.
    
    Parameter:
        file_hash (str): SHA256-Hash der zu entfernenden Datei
                        Muss in Chunk-IDs enthalten sein für Pattern-Matching
        
        collection_name (str): Ziel-Collection für Deletion-Operation
    
    Rückgabe:
        bool: True bei erfolgreichem Entfernen oder wenn keine Dokumente gefunden
              False nur bei tatsächlichen Fehlern (Netzwerk, Permissions, etc.)
    """
    try:
        # COLLECTION-ACCESS: Sichere Collection-Referenz
        collection = get_collection(collection_name)
        
        # ID-DISCOVERY: Alle IDs laden für Pattern-Matching
        # PERFORMANCE-NOTE: Bei großen Collections kann dies Memory-intensiv sein
        # Alternative: Server-side Filtering wenn ChromaDB-API dies unterstützt
        all_results = collection.get()
        
        # PATTERN-MATCHING: IDs identifizieren die File-Hash enthalten
        ids_to_delete = []
        if all_results["ids"]:
            for doc_id in all_results["ids"]:
                # HASH-INCLUSION-CHECK: File-Hash als Substring in ID
                if file_hash in doc_id:
                    ids_to_delete.append(doc_id)
        
        # CONDITIONAL-DELETION: Nur ausführen wenn tatsächlich IDs gefunden
        if ids_to_delete:
            # BULK-DELETION: Atomische Entfernung aller passenden Dokumente
            collection.delete(ids=ids_to_delete)
            print(f"Removed {len(ids_to_delete)} documents for file hash {file_hash}")
            return True
        else:
            # NO-OP-SUCCESS: Keine Dokumente gefunden ist kein Fehler
            print(f"No documents found for file hash {file_hash}")
            return True
        
    except Exception as e:
        # DELETION-ERROR-HANDLING: Kritische Operationen benötigen ausführliches Logging
        print(f"Error removing documents from ChromaDB: {e}")
        return False

def remove_documents_by_ids(document_ids: list[str], collection_name: str = "rag_docs"):
    """
    Entfernt spezifische Dokumente anhand exakter ID-Liste.
    
    Diese Funktion bietet präzise Kontrolle über Dokument-Deletion.
    
    Parameter:
        document_ids (list[str]): Exakte IDs der zu entfernenden Dokumente
                                 Muss exakt mit gespeicherten IDs übereinstimmen
        
        collection_name (str): Ziel-Collection für Deletion-Operation
    
    Rückgabe:
        bool: True bei erfolgreichem Entfernen, False bei Fehlern
    """
    try:
        # COLLECTION-ACCESS: Standard Collection-Retrieval-Pattern
        collection = get_collection(collection_name)
        
        # EXACT-ID-DELETION: Bulk-Deletion mit exakter ID-Liste
        # ChromaDB ignoriert nicht-existierende IDs gracefully
        collection.delete(ids=document_ids)
        print(f"Removed {len(document_ids)} documents from ChromaDB")
        return True
        
    except Exception as e:
        # ID-DELETION-ERROR-HANDLING: Spezifische Fehler-Meldung für ID-basierte Operationen
        print(f"Error removing documents by IDs: {e}")
        return False

def update_documents_in_chroma(documents: list[str], metadatas: list[dict], ids: list[str], 
                              embeddings: list[list[float]], collection_name: str = "rag_docs"):
    """
    Aktualisiert bestehende Dokumente in ChromaDB mit neuen Inhalten/Embeddings.
    
    Diese Funktion ermöglicht In-Place-Updates von Dokumenten ohne
    Delete-Add-Zyklen für bessere Performance und Konsistenz.
    
    Parameter:
        documents, metadatas, ids, embeddings: Identisch zu add_documents_to_chroma
                                              Aber IDs müssen bereits existieren
        
        collection_name (str): Ziel-Collection für Update-Operation
    
    Rückgabe:
        bool: True bei erfolgreichem Update, False bei Fehlern
    """
    try:
        # COLLECTION-ACCESS: Standard Collection-Retrieval für Updates
        collection = get_collection(collection_name)
        
        # BULK-UPDATE: Atomische Update-Operation für alle spezifizierten Dokumente
        # Existierende Dokumente werden vollständig überschrieben
        collection.update(
            documents=documents,        # Neue Text-Inhalte
            metadatas=metadatas,       # Aktualisierte oder erweiterte Metadaten
            ids=ids,                   # Bestehende IDs für Update-Targeting
            embeddings=embeddings      # Neue oder re-berechnete Embeddings
        )
        return True
        
    except Exception as e:
        # UPDATE-ERROR-HANDLING: Updates können komplexere Fehler haben (ID not found, etc.)
        print(f"Error updating documents in ChromaDB: {e}")
        return False

# Monitoring

def get_collection_stats(collection_name: str = "rag_docs"):
    """
    Sammelt umfassende Statistiken einer Collection für Monitoring und Debugging.
    
    Diese Funktion bietet Einblicke in Collection-Gesundheit und Performance
    für Administrative Dashboards.
    
    Parameter:
        collection_name (str): Name der zu analysierenden Collection
    
    Rückgabe:
        dict: Strukturierte Statistiken oder None bei Fehlern
              Struktur: {"name": str, "document_count": int, "has_embeddings": bool, ...}
    """
    try:
        # COLLECTION-ACCESS: Standard-Pattern für Stats-Collection
        collection = get_collection(collection_name)
        
        # DOCUMENT-COUNT: Primäre Kapazitäts-Metrik
        # O(1) Operation in ChromaDB durch Index-basierte Counting
        count = collection.count()
        
        # SAMPLE-ANALYSIS: Strukturelle Daten-Validierung
        # Limitiert auf 10 Dokumente für Performance bei großen Collections
        sample = collection.get(limit=10)
        
        # STATISTIK-AGGREGATION: Strukturierte Metriken für Dashboard-Konsumierung
        stats = {
            "name": collection_name,                                          # Collection-Identifikation
            "document_count": count,                                          # Haupt-Kapazitäts-Indikator
            "sample_documents": len(sample["documents"]) if sample["documents"] else 0,  # Sample-Verfügbarkeit
            "has_embeddings": bool(sample.get("embeddings")),                # Embedding-Feature-Detection
            "has_metadata": bool(sample.get("metadatas"))                    # Metadata-Feature-Detection
        }
        
        return stats
        
    except Exception as e:
        # STATS-ERROR-HANDLING: Monitoring-Funktionen sollten nie die Hauptanwendung unterbrechen
        print(f"Error getting collection stats: {e}")
        return None

def clear_collection(collection_name: str = "rag_docs"):
    """
    Löscht alle Dokumente aus einer Collection.
    
    Rückgabe:
        bool: True bei erfolgreichem Clearing, False bei Fehlern
    """
    try:
        # CLIENT-ACCESS: Direct Client für Collection-Level-Operationen
        client = get_chroma_client()
        
        # COLLECTION-DELETION: Vollständige Entfernung einschließlich Index-Strukturen
        try:
            client.delete_collection(collection_name)
        except:
            # GRACEFUL-HANDLING: Collection existiert möglicherweise nicht
            # Fehler werden ignoriert da das Ziel (leere Collection) erreicht wird
            pass  # Collection might not exist
        
        # COLLECTION-RECREATION: Saubere neue Collection mit Default-Konfiguration
        # Stellt sicher dass Collection in konsistentem Zustand für zukünftige Operationen
        client.create_collection(collection_name)
        
        print(f"Cleared collection: {collection_name}")
        return True
        
    except Exception as e:
        # CLEAR-ERROR-HANDLING: Kritische Operation benötigt ausführliches Error-Logging
        print(f"Error clearing collection: {e}")
        return False

def check_chroma_connection():
    """
    Überprüft ChromaDB-Verbindung und Service-Verfügbarkeit für Health-Monitoring.
    
    Diese Funktion implementiert umfassende Health-Checks für ChromaDB-Service.
    
    Rückgabe:
        tuple[bool, str]: (Verbindungsstatus, Detaillierte Beschreibung)
                         - (True, "Connected. Collections: [...]") bei Erfolg
                         - (False, "Connection failed: error_details") bei Fehlern
    """
    try:
        # CLIENT-CONNECTION-TEST: Basis-Konnektivitäts-Prüfung
        client = get_chroma_client()
        
        # SERVICE-FUNCTIONALITY-TEST: API-Response-Fähigkeit testen
        # list_collections() ist leichtgewichtige Operation für Health-Checks
        collections = client.list_collections()
        
        # SUCCESS-RESPONSE: Verbindung OK mit diagnostischen Informationen
        # Collection-Namen bieten Einblick in Service-Verfügbarkeit und Daten-Status
        return True, f"Connected. Collections: {[c.name for c in collections]}"
        
    except Exception as e:
        # CONNECTION-FAILURE: Detaillierte Fehler-Information für Debugging
        # Error-String enthält spezifische Failure-Ursache für Administrative-Diagnose
        return False, f"Connection failed: {str(e)}"