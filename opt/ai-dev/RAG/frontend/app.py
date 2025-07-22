# ----------------------------------------------------------------------
# BWB PoC RAG System  Streamlit-Frontend
# ----------------------------------------------------------------------

import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List
import time
import textwrap

# API Base URL
API_BASE_URL = os.getenv("API_BASE_URL", "http://rag_api_backend:80")

# ðŸŽ¨ Seitenkonfiguration
st.set_page_config(
    page_title="ðŸ¢ BWB PoC RAG System",
    page_icon="ðŸ”",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ðŸŽ¨ CSS fÃ¼r eine ansprechendere UI-Gestaltung
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .queue-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .developer-tools {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ðŸ”§ Utility-Funktionen 
def get_mime_type(filename):
    """ðŸ” Bestimmt den MIME-Type anhand der Dateiendung."""
    file_ext = os.path.splitext(filename.lower())[1]
    mime_types = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.ppt': 'application/vnd.ms-powerpoint',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.txt': 'text/plain',
        '.html': 'text/html',
        '.md': 'text/markdown',
        '.csv': 'text/csv'
    }
    return mime_types.get(file_ext, 'application/octet-stream')

def format_file_size(size_bytes):
    """ðŸ“ Gibt DateigrÃ¶ÃŸen in menschenlesbarem Format zurÃ¼ck."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def format_duration(ms):
    """â±ï¸ Konvertiert Millisekunden in handliches Zeitformat."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"

def create_status_badge(status):
    """ðŸ·ï¸ Erzeugt farbiges Status-Badge."""
    colors = {
        "chunked": "âœ…",
        "processing": "âš™ï¸", 
        "uploaded": "ðŸ“¤",
        "error": "âŒ",
        "deleted": "ðŸ—‘ï¸"
    }
    return f"{colors.get(status, 'â“')} {status.title()}"

def make_api_request(endpoint, method="GET", **kwargs):
    """ðŸŒ Generischer API-Request"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        return False, str(e)

# ðŸŽ¯ Sidebar 
with st.sidebar:
    st.header("ðŸ¢ BWB PoC RAG System")
    st.markdown("**ðŸ“š PDF + OCR Version**")
    st.markdown("---")

    # ðŸ§­ Erweiterte Navigation
    selected_page = st.selectbox(
        "ðŸ§­ Navigation:",
        [
            "ðŸ“Š Dashboard",
            "ðŸ“ Dateien hochladen", 
            "ðŸ—‚ï¸ Dateiverwaltung",
            "ðŸ” Suche & Abfrage",
            "ðŸ“ˆ Analytik",
            "ðŸ“‹ Queue Management",    
            "âš™ï¸ Systemverwaltung"
        ],
        key="navigation"
    )

    st.markdown("---")

    # âš¡ Schnellaktionen
    st.subheader("âš¡ Schnellaktionen")

    # ðŸ”„ Seite neuladen
    if st.button("ðŸ”„ Daten neu laden"):
        st.rerun()

    # ðŸ©º Health-Check
    if st.button("ðŸ©º Systemstatus prÃ¼fen"):
        with st.spinner("ðŸ” Systemstatus wird geprÃ¼ft..."):
            success, health = make_api_request("/health")
            if success:
                status = health.get("status", "unbekannt")
                if status == "healthy":
                    st.success(f"âœ… System: {status}")
                else:
                    st.warning(f"âš ï¸ System: {status}")
            else:
                st.error("âŒ System nicht erreichbar")

# ---------------- ðŸ“Š Dashboard ----------------
if selected_page == "ðŸ“Š Dashboard":
    st.title("ðŸ“Š System Dashboard")

    # ðŸ©º System Health
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("ðŸ©º Systemstatus")
    with col2:
        if st.button("ðŸ”„ Aktualisieren", key="health_refresh"):
            st.rerun()

    success, health_data = make_api_request("/health")

    if success:
        overall_status = health_data.get("status", "unbekannt")
        if overall_status == "healthy":
            st.success("âœ… Alle Systeme funktionieren ordnungsgemÃ¤ÃŸ")
        else:
            st.warning(f"âš ï¸ Systemstatus: {overall_status}")

        # Einzelne Services
        services = health_data.get("services", {})
        if services:
            cols = st.columns(len(services))
            for i, (service, info) in enumerate(services.items()):
                with cols[i]:
                    if info["status"] == "ok":
                        st.success(f"âœ… **{service.title()}**")
                    else:
                        st.error(f"âŒ **{service.title()}**")
                    st.caption(info["message"][:40] + "..." if len(info["message"]) > 40 else info["message"])
    else:
        st.error(f"âŒ System nicht erreichbar: {health_data}")

    st.markdown("---")

    # ðŸ“ˆ Performance Metrics
    st.subheader("ðŸ“ˆ Systemleistung")
    success, perf_data = make_api_request("/system/performance")

    if success:
        metrics = perf_data.get("metrics", {})
        col1, col2, col3, col4, col5 = st.columns(5)

        files_data = metrics.get("files", {})
        chunks_data = metrics.get("chunks", {})
        performance_data = metrics.get("performance", {})

        with col1:
            st.metric("ðŸ“ Dateien gesamt", files_data.get("total", 0))
        with col2:
            st.metric("âœ… Verarbeitete Dateien", files_data.get("ready", 0))
        with col3:
            st.metric("ðŸ§© Chunks gesamt", chunks_data.get("total", 0))
        with col4:
            avg_time = performance_data.get("avg_processing_time_ms")
            st.metric("â±ï¸ âŒ€ Verarbeitungszeit", format_duration(avg_time))
        with col5:
            quality = performance_data.get("avg_file_quality")
            st.metric("ðŸŽ¯ âŒ€ QualitÃ¤t", f"{quality:.2f}" if quality else "N/A")

        # ðŸ“Š Dateistatus-Verteilung
        st.subheader("ðŸ“Š Dateistatusverleilung")
        if files_data.get("total", 0) > 0:
            status_labels = ["âœ… Fertig", "âš™ï¸ In Bearbeitung", "â³ Warteschlange", "âŒ Fehler"]
            status_values = [
                files_data.get("ready", 0),
                files_data.get("processing", 0),
                files_data.get("pending", 0),
                files_data.get("error", 0)
            ]
            status_colors = ["#28a745", "#ffc107", "#17a2b8", "#dc3545"]
            
            fig = px.pie(
                values=status_values,
                names=status_labels,
                color_discrete_sequence=status_colors,
                title="ðŸ“Š Bearbeitungsstatus der Dateien"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ðŸ“ Noch keine Dateien im System vorhanden")
    else:
        st.error(f"âŒ Leistungsdaten konnten nicht geladen werden: {perf_data}")

# ---------------- ðŸ“ Dateien hochladen ----------------
elif selected_page == "ðŸ“ Dateien hochladen":
    st.title("ðŸ“ Dateien hochladen")

    upload_mode = st.radio(
        "ðŸ“¤ Upload-Methode wÃ¤hlen:",
        ["ðŸ“„ Einzeldokument", "ðŸ“š Mehrere Dokumente"],
        horizontal=True
    )

    if upload_mode == "ðŸ“„ Einzeldokument":
        st.subheader("ðŸ“„ Einzeldokument hochladen")

        uploaded_file = st.file_uploader(
            "ðŸ“Ž Dokument auswÃ¤hlen:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'html', 'xlsx', 'csv'],
            help="ðŸ’¡ UnterstÃ¼tzte Formate: PDF, Word, PowerPoint, Excel, Text, HTML, CSV, Markdown"
        )

        if uploaded_file is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**ðŸ“ Dateiname:** {uploaded_file.name}")
            with col2:
                st.info(f"**ðŸ“ GrÃ¶ÃŸe:** {format_file_size(uploaded_file.size)}")
            with col3:
                st.info(f"**ðŸ”§ Format:** {uploaded_file.type}")

            if st.button("ðŸš€ Hochladen & Verarbeiten", type="primary"):
                with st.spinner("âš™ï¸ Dokument wird verarbeitet..."):
                    try:
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), get_mime_type(uploaded_file.name))
                        }
                        success, result = make_api_request("/upload_document", method="POST", files=files)
                        
                        if success:
                            st.success("âœ… Dokument erfolgreich verarbeitet!")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ðŸ§© Chunks", result.get("chunks", 0))
                            with col2:
                                st.metric("â±ï¸ Zeit", format_duration(result.get("processing_time_ms", 0)))
                            with col3:
                                st.metric("ðŸ†” Datei-ID", result.get("file_id", "N/A"))
                            
                            quality = result.get("quality_metrics", {})
                            if quality:
                                st.subheader("ðŸŽ¯ QualitÃ¤tsanalyse")
                                st.json(quality)
                        else:
                            st.error(f"âŒ Upload fehlgeschlagen: {result}")
                    except Exception as e:
                        st.error(f"âŒ Fehler: {str(e)}")

    else:  # Multiple Files
        st.subheader("ðŸ“š Mehrere Dokumente hochladen")
        uploaded_files = st.file_uploader(
            "ðŸ“Ž Mehrere Dokumente auswÃ¤hlen:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'html', 'xlsx', 'csv'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"ðŸ“ {len(uploaded_files)} Dateien ausgewÃ¤hlt")
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. ðŸ“„ {file.name} ({format_file_size(file.size)})")
            
            if st.button("ðŸš€ Alle Dateien hochladen", type="primary"):
                with st.spinner("âš™ï¸ Dateien werden hochgeladen..."):
                    try:
                        files_data = []
                        for file in uploaded_files:
                            files_data.append(("files", (file.name, file.getvalue(), get_mime_type(file.name))))
                        
                        success, result = make_api_request("/upload_multiple", method="POST", files=files_data)
                        
                        if success:
                            st.success("âœ… Batch-Upload abgeschlossen!")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("ðŸ“ Gesamt", result.get("total_files", 0))
                            with col2:
                                st.metric("âœ… Erfolgreich", result.get("successful", 0))
                            with col3:
                                st.metric("âŒ Fehlgeschlagen", result.get("failed", 0))
                            with col4:
                                st.metric("ðŸ”„ Duplikate", result.get("duplicates", 0))
                            
                            details = result.get("details", [])
                            if details:
                                st.subheader("ðŸ“‹ Upload-Details")
                                df = pd.DataFrame(details)
                                st.dataframe(df, use_container_width=True)
                        else:
                            st.error(f"âŒ Batch-Upload fehlgeschlagen: {result}")
                    except Exception as e:
                        st.error(f"âŒ Fehler: {str(e)}")

# ---------------- ðŸ—‚ï¸ Dateiverwaltung ----------------
elif selected_page == "ðŸ—‚ï¸ Dateiverwaltung":
    st.title("ðŸ—‚ï¸ Dateiverwaltung")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("ðŸ”„ Dateien aktualisieren"):
            st.rerun()
    with col2:
        if st.button("ðŸ§¹ System bereinigen"):
            with st.spinner("ðŸ§¹ Bereinigung lÃ¤uft..."):
                # âœ… FIXED: Korrekter Endpunkt
                success, result = make_api_request("/system/cleanup", method="POST")
                if success:
                    st.success("âœ… Bereinigung abgeschlossen!")
                    st.json(result)
                else:
                    st.error(f"âŒ Bereinigung fehlgeschlagen: {result}")
    with col3:
        if st.button("ðŸ”„ Alle neu verarbeiten"):
            with st.spinner("âš™ï¸ Alle Dateien werden neu verarbeitet..."):
                success, result = make_api_request("/rechunk", method="POST")
                if success:
                    st.success(f"âœ… Neuverarbeitung abgeschlossen: {result.get('processed', 0)} verarbeitet")
                    st.json(result)
                else:
                    st.error(f"âŒ Neuverarbeitung fehlgeschlagen: {result}")

    success, files = make_api_request("/files")
    if success and files:
        st.subheader(f"ðŸ“‚ Dateien ({len(files)} gesamt)")
        
        for file in files:
            with st.expander(f"ðŸ“„ {file['file_name']} - {create_status_badge(file['status'])}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**ðŸ†” ID:** {file['id']}")
                    st.write(f"**ðŸ“‹ Typ:** {file['document_type']}")
                    st.write(f"**ðŸ“ GrÃ¶ÃŸe:** {format_file_size(file['file_size'])}")
                
                with col2:
                    st.write(f"**ðŸ“… Hochgeladen:** {file['upload_date'][:10] if file.get('upload_date') else 'N/A'}")
                    st.write(f"**ðŸ§© Chunks:** {file['chunk_count']}")
                    if file.get('processing_duration_ms'):
                        st.write(f"**â±ï¸ Dauer:** {format_duration(file['processing_duration_ms'])}")
                
                with col3:
                    quality = file.get('quality_metrics', {})
                    if quality:
                        st.write(f"**ðŸŽ¯ QualitÃ¤t:** {quality.get('avg_chunk_quality', 0):.2f}")
                
                with col4:
                    if st.button(f"ðŸ—‘ï¸ LÃ¶schen", key=f"del_{file['id']}"):
                        success, result = make_api_request(f"/files/{file['id']}", method="DELETE")
                        if success:
                            st.success("âœ… Datei gelÃ¶scht!")
                            st.rerun()
                        else:
                            st.error(f"âŒ LÃ¶schen fehlgeschlagen: {result}")
                
                if file.get('error_message'):
                    st.error(f"âŒ {file['error_message']}")
    elif success:
        st.info("ðŸ“ Noch keine Dateien hochgeladen")
    else:
        st.error(f"âŒ Dateien konnten nicht geladen werden: {files}")

# ---------------- ðŸ” Suche & Abfrage ----------------
elif selected_page == "ðŸ” Suche & Abfrage":
    st.title("ðŸ” Suche & Abfrage")

    tab1, tab2 = st.tabs(["ðŸ¤– KI-Suche", "ðŸ“ Textsuche"])

    with tab1:
        st.subheader("ðŸ¤– KI-gestÃ¼tzte semantische Suche")
        query = st.text_area(
            "â“ Ihre Frage:",
            value="Was muss ich beim mobilen Arbeiten beachten?",
            height=100,
            help="ðŸ’¡ Stellen Sie mir alle Fragen rund um die IT der Berliner Wasserbetriebe"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("ðŸ“Š Max. Ergebnisse:", 1, 20, 10)
        with col2:
            quality_threshold = st.slider("ðŸŽ¯ QualitÃ¤tsschwelle:", 0.0, 1.0, 0.7, 0.1)
        
        if st.button("ðŸ” Suchen", type="primary") and query.strip():
            with st.spinner("ðŸ¤– KI verarbeitet Ihre Frage..."):
                try:
                    payload = {
                        "query": query.strip(),
                        "max_results": max_results,
                        "quality_threshold": quality_threshold
                    }
                    success, result = make_api_request("/query", method="POST", json=payload)
                    
                    if success:
                        st.subheader("ðŸ’¬ Antwort")
                        raw_answer = result.get("answer", "Keine Antwort generiert")
                        answer = textwrap.dedent(raw_answer).strip()
                        st.markdown(answer)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("â±ï¸ Antwortzeit", f"{result.get('response_time_ms', 0)}ms")
                        with col2:
                            st.metric("ðŸ“‹ Ergebnisse", result.get("results_count", 0))
                        with col3:
                            quality = result.get("quality_metrics", {})
                            st.metric("ðŸŽ¯ Relevanz", f"{quality.get('avg_relevance', 0):.3f}")
                        
                        sources = result.get("sources", [])
                        if sources:
                            st.subheader("ðŸ“š Quellen")
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"ðŸ“„ Quelle {i}: {source.get('source', 'Unbekannt')} (Seite {source.get('page', 'N/A')})"):
                                    st.markdown(source.get('text', ''))
                                    st.caption(f"ðŸŽ¯ Ã„hnlichkeit: {source.get('similarity', 0):.3f} | ðŸ“Š QualitÃ¤t: {source.get('quality_score', 0):.3f}")
                    else:
                        st.error(f"âŒ Suche fehlgeschlagen: {result}")
                except Exception as e:
                    st.error(f"âŒ Fehler: {str(e)}")

    with tab2:
        st.subheader("ðŸ“ Volltextsuche")
        search_term = st.text_input(
            "ðŸ”¤ Suchbegriffe:",
            placeholder="Geben Sie spezifische WÃ¶rter oder Phrasen ein...",
            help="ðŸ’¡ Direkte Textsuche im Dokumentinhalt"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("ðŸ“Š Max. Ergebnisse:", 1, 50, 10)
        with col2:
            quality_threshold = st.slider("ðŸŽ¯ QualitÃ¤tsfilter:", 0.0, 1.0, 0.3, 0.1)
        
        if st.button("ðŸ” Text durchsuchen", type="primary") and search_term:
            with st.spinner("ðŸ“ Textinhalt wird durchsucht..."):
                try:
                    params = {
                        "q": search_term,
                        "limit": limit,
                        "quality_threshold": quality_threshold
                    }
                    success, result = make_api_request("/search/fulltext", params=params)
                    
                    if success:
                        st.success(f"âœ… {result.get('results_count', 0)} Ergebnisse in {result.get('response_time_ms', 0)}ms gefunden")
                        results = result.get("results", [])
                        
                        for i, res in enumerate(results, 1):
                            with st.expander(f"ðŸ“„ Ergebnis {i} - QualitÃ¤t: {res.get('quality_score', 0):.3f}"):
                                st.markdown(res.get('text', ''))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"ðŸ“– Seite: {res.get('page', 'N/A')}")
                                with col2:
                                    st.caption(f"ðŸ”¤ WÃ¶rter: {res.get('word_count', 0)}")
                    else:
                        st.error(f"âŒ Suche fehlgeschlagen: {result}")
                except Exception as e:
                    st.error(f"âŒ Fehler: {str(e)}")

# ---------------- ðŸ“ˆ Analytik ----------------
elif selected_page == "ðŸ“ˆ Analytik":
    st.title("ðŸ“ˆ Systemanalytik")

    days = st.selectbox(
        "ðŸ“… Analysezeitraum:",
        [7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"Letzte {x} Tage"
    )

    success, analytics = make_api_request(f"/system/analytics?days={days}")
    
    if success:
        query_stats = analytics.get("query_analytics", {})
        
        if query_stats:
            st.subheader("ðŸ“Š Abfragestatistiken")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ðŸ” Abfragen gesamt", query_stats.get("total_queries", 0))
            with col2:
                st.metric("ðŸŽ¯ Einzigartige Abfragen", query_stats.get("unique_queries", 0))
            with col3:
                avg_time = query_stats.get("avg_response_time", 0)
                st.metric("â±ï¸ âŒ€ Antwortzeit", format_duration(avg_time))
            with col4:
                st.metric("ðŸ“‹ âŒ€ Ergebnisse", f"{query_stats.get('avg_results_count', 0):.1f}")

            st.subheader("ðŸš€ Leistungsanalyse")
            col1, col2 = st.columns(2)
            
            with col1:
                slow_queries = query_stats.get("slow_queries", 0)
                total_queries = query_stats.get("total_queries", 1)
                fast_queries = total_queries - slow_queries
                
                fig = px.pie(
                    values=[fast_queries, slow_queries],
                    names=["ðŸš€ Schnell (<5s)", "ðŸŒ Langsam (>5s)"],
                    title="âš¡ðŸ• Abfrageleistung",
                    color_discrete_sequence=["#28a745", "#dc3545"]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                zero_results = query_stats.get("zero_result_queries", 0)
                successful = total_queries - zero_results
                
                fig = px.pie(
                    values=[successful, zero_results],
                    names=["âœ… Ergebnisse gefunden", "âŒ Keine Ergebnisse"],
                    title="ðŸŽ¯ Erfolgsquote der Abfragen",
                    color_discrete_sequence=["#17a2b8", "#ffc107"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ðŸ“Š Keine Abfragedaten fÃ¼r den gewÃ¤hlten Zeitraum verfÃ¼gbar")

        db_health = analytics.get("database_health", {})
        if db_health:
            st.subheader("ðŸ—„ï¸ Datenbankzustand")
            table_sizes = db_health.get("table_sizes", [])
            if table_sizes:
                df = pd.DataFrame(table_sizes)
                st.dataframe(df[['tablename', 'size']], use_container_width=True)
    else:
        st.error(f"âŒ Analytikdaten konnten nicht geladen werden: {analytics}")

# ---------------- ðŸ“‹ Queue Management (NEU) ----------------
elif selected_page == "ðŸ“‹ Queue Management":
    st.title("ðŸ“‹ Verarbeitungsqueue-Management")

    # Quick Actions
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("ðŸ”„ Queue aktualisieren"):
            st.rerun()
    with col2:
        if st.button("âš™ï¸ Queue verarbeiten"):
            with st.spinner("âš™ï¸ Queue wird verarbeitet..."):
                success, result = make_api_request("/process_uploaded_files", method="POST")
                if success:
                    st.success("âœ… Queue-Verarbeitung abgeschlossen!")
                    st.json(result)
                else:
                    st.error(f"âŒ Queue-Verarbeitung fehlgeschlagen: {result}")
    with col3:
        if st.button("ðŸ”„ Fehlgeschlagene wiederholen"):
            with st.spinner("âš™ï¸ Fehlgeschlagene Dateien werden wiederholt..."):
                success, result = make_api_request("/retry_failed", method="POST")
                if success:
                    st.success(f"âœ… Wiederholung abgeschlossen: {result.get('queued', 0)} Dateien in Warteschlange")
                    st.json(result)
                else:
                    st.error(f"âŒ Wiederholung fehlgeschlagen: {result}")

    # Queue Overview
    st.subheader("ðŸ“Š Queue-Ãœbersicht")
    success, queue_info = make_api_request("/queue")
    
    if success:
        summary = queue_info.get("summary", {})
        
        # Queue Status Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="queue-card">
                <h3>â³ Wartend</h3>
                <h2>{summary.get('pending', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="queue-card">
                <h3>âš™ï¸ In Bearbeitung</h3>
                <h2>{summary.get('processing', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="queue-card">
                <h3>âœ… Abgeschlossen</h3>
                <h2>{summary.get('completed', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="queue-card">
                <h3>âŒ Fehler</h3>
                <h2>{summary.get('errors', 0)}</h2>
            </div>
            """, unsafe_allow_html=True)

        # Detailed Queue Information
        st.subheader("ðŸ“‹ Detaillierte Queue-Informationen")
        success_details, queue_details = make_api_request("/queue/details")
        
        if success_details:
            queue_data = queue_details.get("queue_details", {})
            
            # Tabs fÃ¼r verschiedene Queue-Status
            tab1, tab2, tab3 = st.tabs(["â³ Warteschlange", "âš™ï¸ In Bearbeitung", "âŒ Fehler"])
            
            with tab1:
                uploaded_files = queue_data.get("uploaded_files", [])
                if uploaded_files:
                    df = pd.DataFrame(uploaded_files)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Keine Dateien in der Warteschlange")
            
            with tab2:
                processing_files = queue_data.get("processing_files", [])
                if processing_files:
                    df = pd.DataFrame(processing_files)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Keine Dateien werden aktuell verarbeitet")
            
            with tab3:
                error_files = queue_data.get("error_files", [])
                if error_files:
                    df = pd.DataFrame(error_files)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("Keine fehlgeschlagenen Dateien")
        else:
            st.warning("âš ï¸ Detaillierte Queue-Informationen nicht verfÃ¼gbar")
    else:
        st.error(f"âŒ Queue-Informationen konnten nicht geladen werden: {queue_info}")

# ---------------- âš™ï¸ Systemverwaltung (Erweitert) ----------------
elif selected_page == "âš™ï¸ Systemverwaltung":
    st.title("âš™ï¸ Systemverwaltung")

    # System Health Status
    st.subheader("ðŸ©º Detaillierter Systemstatus")
    success, health = make_api_request("/health")
    
    if success:
        services = health.get("services", {})
        for service, info in services.items():
            if info["status"] == "ok":
                st.success(f"âœ… **{service.title()}**: {info['message']}")
            else:
                st.error(f"âŒ **{service.title()}**: {info['message']}")
    else:
        st.error(f"âŒ SystemintegritÃ¤tsprÃ¼fung fehlgeschlagen: {health}")

    st.markdown("---")

    # Wartungsoperationen
    st.subheader("ðŸ”§ Wartungsoperationen")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("ðŸ”§ Datenbankwartung"):
            with st.spinner("ðŸ”§ Wartung lÃ¤uft..."):
                success, result = make_api_request("/system/maintenance", method="POST")
                if success:
                    st.success("âœ… Wartung abgeschlossen!")
                    results = result.get("maintenance_results", {})
                    for op, msg in results.items():
                        st.info(f"**{op}**: {msg}")
                else:
                    st.error(f"âŒ Wartung fehlgeschlagen: {result}")
    
    with col2:
        if st.button("ðŸ”„ Fehlgeschlagene wiederholen"):
            with st.spinner("âš™ï¸ Fehlgeschlagene Dateien werden wiederholt..."):
                success, result = make_api_request("/retry_failed", method="POST")
                if success:
                    st.success(f"âœ… Wiederholung abgeschlossen: {result.get('queued', 0)} Dateien in Warteschlange")
                    st.json(result)
                else:
                    st.error(f"âŒ Wiederholung fehlgeschlagen: {result}")
    
    with col3:
        if st.button("ðŸ“‹ Verarbeitungsqueue"):
            success, queue = make_api_request("/queue")
            if success:
                summary = queue.get("summary", {})
                st.info(f"â³ Wartend: {summary.get('pending', 0)}")
                st.info(f"âš™ï¸ In Bearbeitung: {summary.get('processing', 0)}")
                st.info(f"âœ… Abgeschlossen: {summary.get('completed', 0)}")
                st.info(f"âŒ Fehler: {summary.get('errors', 0)}")
            else:
                st.error(f"âŒ Queue-ÃœberprÃ¼fung fehlgeschlagen: {queue}")

    st.markdown("---")

    # Konfiguration
    st.subheader("âš™ï¸ Systemkonfiguration")
    success, formats = make_api_request("/formats")
    
    if success:
        supported = formats.get("supported_formats", {})
        st.write(f"ðŸ“‹ UnterstÃ¼tzte Formate: {len(supported)}")
        
        format_list = list(supported.items())
        cols = st.columns(3)
        for i, (ext, desc) in enumerate(format_list):
            col_idx = i % 3
            with cols[col_idx]:
                st.caption(f"ðŸ“„ {ext}: {desc}")

    st.markdown("---")

    # â­Developer Tools
    st.subheader("ðŸ”§ Developer Tools")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("ðŸ“‹ API-Status anzeigen"):
            success, api_status = make_api_request("/api-status")
            if success:
                st.subheader("ðŸ” VerfÃ¼gbare API-Endpunkte")
                endpoints = api_status.get("endpoints", [])
                
                # Gruppiere Endpunkte nach Kategorie
                categories = {
                    "System": [],
                    "Dateien": [],
                    "Suche": [],
                    "Queue": [],
                    "Andere": []
                }
                
                for endpoint in endpoints:
                    path = endpoint.get("path", "")
                    if any(word in path for word in ["/health", "/system", "/formats"]):
                        categories["System"].append(endpoint)
                    elif any(word in path for word in ["/files", "/upload"]):
                        categories["Dateien"].append(endpoint)
                    elif any(word in path for word in ["/query", "/search"]):
                        categories["Suche"].append(endpoint)
                    elif any(word in path for word in ["/queue", "/process", "/retry"]):
                        categories["Queue"].append(endpoint)
                    else:
                        categories["Andere"].append(endpoint)
                
                for category, endpoints in categories.items():
                    if endpoints:
                        st.write(f"**{category}-Endpunkte:**")
                        for endpoint in endpoints:
                            methods = ", ".join(endpoint.get("methods", []))
                            st.code(f"{methods} {endpoint.get('path', '')}")
            else:
                st.error("âŒ API-Status nicht verfÃ¼gbar")
    
    with col2:
        if st.button("ðŸ¥ Erweiterte Systemdiagnose"):
            with st.spinner("ðŸ” Systemdiagnose lÃ¤uft..."):
                # Kombiniere mehrere Health-Checks
                health_success, health_data = make_api_request("/health")
                perf_success, perf_data = make_api_request("/system/performance")
                analytics_success, analytics_data = make_api_request("/system/analytics?days=7")
                
                st.markdown("### ðŸ¥ Diagnoseergebnisse")
                
                if health_success:
                    fixes = health_data.get("fixes_applied", [])
                    if fixes:
                        st.success(f"âœ… Angewandte Fixes: {len(fixes)}")
                        for fix in fixes:
                            st.info(f"ðŸ”§ {fix}")
                
                if perf_success:
                    metrics = perf_data.get("metrics", {})
                    files_data = metrics.get("files", {})
                    total_files = files_data.get("total", 0)
                    ready_files = files_data.get("ready", 0)
                    
                    if total_files > 0:
                        success_rate = (ready_files / total_files) * 100
                        if success_rate > 90:
                            st.success(f"âœ… Hohe Erfolgsquote: {success_rate:.1f}%")
                        elif success_rate > 70:
                            st.warning(f"âš ï¸ Moderate Erfolgsquote: {success_rate:.1f}%")
                        else:
                            st.error(f"âŒ Niedrige Erfolgsquote: {success_rate:.1f}%")

    # Debug-Informationen
    with st.expander("ðŸ› Debug-Informationen & System-Internals"):
        st.markdown("""
        <div class="developer-tools">
        <h4>ðŸ” System-Internals</h4>
        """, unsafe_allow_html=True)
        
        debug_info = {
            "API_URL": API_BASE_URL,
            "Streamlit_Version": st.__version__,
            "Current_Time": datetime.now().isoformat(),
            "Features": [
                "Queue Management",
                "Developer Tools", 
                "Extended Analytics",
                "API Status Monitoring"
            ]
        }
        
        st.json(debug_info)
        st.markdown("</div>", unsafe_allow_html=True)

# ðŸ“ Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        ðŸ¢ <strong>BWB PoC RAG System</strong><br>
        ðŸš€ FastAPI â€¢ ðŸ—„ï¸ ChromaDB â€¢ ðŸ˜ PostgreSQL â€¢ ðŸ“Š Streamlit<br>
    </div>
    """,
    unsafe_allow_html=True
)

# ðŸ”„ Auto-Refresh fÃ¼r Dashboard
if selected_page == "ðŸ“Š Dashboard":
    if st.sidebar.checkbox("ðŸ”„ Auto-Aktualisierung (30s)"):
        time.sleep(30)
        st.rerun()