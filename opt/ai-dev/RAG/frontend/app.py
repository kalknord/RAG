 -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# BWB PoC RAG System – Streamlit-Frontend
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


API_BASE_URL = os.getenv("API_BASE_URL", "http://rag_api_backend:80")

# Seitenkonfiguration
st.set_page_config(
    page_title="?? BWB PoC RAG System",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Zusätzliche CSS-Klassen für eine ansprechendere UI-Gestaltung.
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Diese kleineren Utilities unterstützen z. B. Formatierungen oder
# Fehlerbehandlung und halten den Hauptcode schlanker.

def get_mime_type(filename):
    """Bestimmt den MIME-Type anhand der Dateiendung."""
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
    """Gibt Dateigrößen in menschenlesbarem Format zurück (KB/MB/GB)."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"


def format_duration(ms):
    """Konvertiert Millisekunden in ein handliches Zeitformat."""
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.1f}s"
    else:
        return f"{ms/60000:.1f}m"


def create_status_badge(status):
    """Erzeugt ein farbiges Status-Badge für die Dateiverwaltung."""
    colors = {
        "chunked": "?",
        "processing": "??", 
        "uploaded": "??",
        "error": "?",
        "deleted": "???"
    }
    return f"{colors.get(status, '?')} {status.title()}"


def make_api_request(endpoint, method="GET", **kwargs):
    """Generischer Wrapper für Aufrufe an das FastAPI-Backend."""
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

# Sidebar

with st.sidebar:
    st.header("?? BWB PoC RAG System")
    st.markdown("**?? PDF + OCR Version**")
    st.markdown("---")

    # -------------- Navigation --------------
    selected_page = st.selectbox(
        "??? Navigation:",
        [
            "?? Dashboard",
            "?? Dateien hochladen", 
            "?? Dateiverwaltung",
            "?? Suche & Abfrage",
            "?? Analytik",
            "?? Systemverwaltung"
        ],
        key="navigation"
    )

    st.markdown("---")

    # -------------- Schnellaktionen --------------
    st.subheader("? Schnellaktionen")

    # Seite neuladen
    if st.button("?? Daten neu laden"):
        st.rerun()

    # Health-Check des Backends
    if st.button("?? Systemstatus prüfen"):
        with st.spinner("?? Systemstatus wird geprüft..."):
            success, health = make_api_request("/health")
            if success:
                status = health.get("status", "unbekannt")
                if status == "healthy":
                    st.success(f"? System: {status}")
                else:
                    st.warning(f"?? System: {status}")
            else:
                st.error("? System nicht erreichbar")

# --------------------------Dashboard--------------------------------------------
if selected_page == "?? Dashboard":
    st.title("?? System Dashboard")

    # ---------------- System Health ----------------
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("?? Systemstatus")
    with col2:
        if st.button("?? Aktualisieren", key="health_refresh"):
            st.rerun()

    # Anfrage an /health
    success, health_data = make_api_request("/health")

    if success:
        overall_status = health_data.get("status", "unbekannt")
        # Globaler Health-Badge
        if overall_status == "healthy":
            st.success("? Alle Systeme funktionieren ordnungsgemäß")
        else:
            st.warning(f"?? Systemstatus: {overall_status}")

        # Einzelne Services (Vector-DB, OCR-Pipeline, ...)
        services = health_data.get("services", {})
        cols = st.columns(len(services))
        for i, (service, info) in enumerate(services.items()):
            with cols[i]:
                if info["status"] == "ok":
                    st.success(f"? **{service.title()}**")
                else:
                    st.error(f"? **{service.title()}**")
                st.caption(info["message"][:40] + "..." if len(info["message"]) > 40 else info["message"])
    else:
        st.error(f"? System nicht erreichbar: {health_data}")

    st.markdown("---")

    # ---------------- Performance Metrics ----------------
    st.subheader("?? Systemleistung")
    success, perf_data = make_api_request("/system/performance")

    if success:
        metrics = perf_data.get("metrics", {})
        # Metriken (Dateien, Chunks, Performance-Dauer, ...)
        col1, col2, col3, col4, col5 = st.columns(5)

        files_data = metrics.get("files", {})
        chunks_data = metrics.get("chunks", {})
        performance_data = metrics.get("performance", {})

        with col1:
            st.metric("?? Dateien gesamt", files_data.get("total", 0))
        with col2:
            st.metric("? Verarbeitete Dateien", files_data.get("ready", 0))
        with col3:
            st.metric("?? Chunks gesamt", chunks_data.get("total", 0))
        with col4:
            avg_time = performance_data.get("avg_processing_time_ms")
            st.metric("?? Ø Verarbeitungszeit", format_duration(avg_time))
        with col5:
            quality = performance_data.get("avg_file_quality")
            st.metric("? Ø Qualität", f"{quality:.2f}" if quality else "N/A")

        # Visualisierung des Dateistatus als Torten-Diagramm
        st.subheader("?? Dateistatusverleilung")
        if files_data.get("total", 0) > 0:
            status_labels = ["Fertig", "In Bearbeitung", "Warteschlange", "Fehler"]
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
                title="?? Bearbeitungsstatus der Dateien"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("?? Noch keine Dateien im System vorhanden")
    else:
        st.error(f"? Leistungsdaten konnten nicht geladen werden: {perf_data}")


# Dateien hochladen

elif selected_page == "?? Dateien hochladen":
    st.title("?? Dateien hochladen")

    # Benutzer kann Einzel- oder Mehrfach-Upload wählen
    upload_mode = st.radio(
        "Upload-Methode wählen:",
        ["?? Einzeldokument", "?? Mehrere Dokumente"],
        horizontal=True
    )

    # ---------- Einzeldokument-Upload ----------
    if upload_mode == "?? Einzeldokument":
        st.subheader("?? Einzeldokument hochladen")

        uploaded_file = st.file_uploader(
            "Dokument auswählen:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'html', 'xlsx', 'csv'],
            help="?? Unterstützte Formate: PDF, Word, PowerPoint, Excel, Text, HTML, CSV, Markdown"
        )

        if uploaded_file is not None:
            # Kurzinformationen zur Datei
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**?? Dateiname:** {uploaded_file.name}")
            with col2:
                st.info(f"**?? Größe:** {format_file_size(uploaded_file.size)}")
            with col3:
                st.info(f"**?? Format:** {uploaded_file.type}")

            # Datei zum Backend senden
            if st.button("?? Hochladen & Verarbeiten", type="primary"):
                with st.spinner("?? Dokument wird verarbeitet..."):
                    try:
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), get_mime_type(uploaded_file.name))
                        }
                        success, result = make_api_request("/upload_document", method="POST", files=files)
                        if success:
                            st.success("? Dokument erfolgreich verarbeitet!")
                            # Ergebnis-Metriken anzeigen
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("?? Chunks", result.get("chunks", 0))
                            with col2:
                                st.metric("?? Zeit", format_duration(result.get("processing_time_ms", 0)))
                            with col3:
                                st.metric("?? Datei-ID", result.get("file_id", "N/A"))
                            # Qualitätsdaten, falls vorhanden
                            quality = result.get("quality_metrics", {})
                            if quality:
                                st.subheader("? Qualitätsanalyse")
                                st.json(quality)
                        else:
                            st.error(f"? Upload fehlgeschlagen: {result}")
                    except Exception as e:
                        st.error(f"? Fehler: {str(e)}")

    # ---------- Multi-Upload ----------
    else:  # Multiple Files
        st.subheader("?? Mehrere Dokumente hochladen")
        uploaded_files = st.file_uploader(
            "Mehrere Dokumente auswählen:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md', 'html', 'xlsx', 'csv'],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.info(f"?? {len(uploaded_files)} Dateien ausgewählt")
            # Dateiliste ausgeben
            for i, file in enumerate(uploaded_files):
                st.write(f"{i+1}. ?? {file.name} ({format_file_size(file.size)})")
            # Sammel-Upload starten
            if st.button("?? Alle Dateien hochladen", type="primary"):
                with st.spinner("? Dateien werden hochgeladen..."):
                    try:
                        files_data = []
                        for file in uploaded_files:
                            files_data.append(("files", (file.name, file.getvalue(), get_mime_type(file.name))))
                        success, result = make_api_request("/upload_multiple", method="POST", files=files_data)
                        if success:
                            st.success("? Batch-Upload abgeschlossen!")
                            # Zusammenfassung
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("?? Gesamt", result.get("total_files", 0))
                            with col2:
                                st.metric("? Erfolgreich", result.get("successful", 0))
                            with col3:
                                st.metric("? Fehlgeschlagen", result.get("failed", 0))
                            with col4:
                                st.metric("?? Duplikate", result.get("duplicates", 0))
                            # Detailtabelle
                            details = result.get("details", [])
                            if details:
                                st.subheader("?? Upload-Details")
                                df = pd.DataFrame(details)
                                st.dataframe(df, use_container_width=True)
                        else:
                            st.error(f"? Batch-Upload fehlgeschlagen: {result}")
                    except Exception as e:
                        st.error(f"? Fehler: {str(e)}")


# Dateiverwaltung

elif selected_page == "?? Dateiverwaltung":
    st.title("?? Dateiverwaltung")

    # Buttons für Refresh, Cleanup, Re-Chunking
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("?? Dateien aktualisieren"):
            st.rerun()
    with col2:
        if st.button("?? System bereinigen"):
            with st.spinner("?? Bereinigung läuft..."):
                success, result = make_api_request("/cleanup", method="POST")
                if success:
                    st.success("? Bereinigung abgeschlossen!")
                    st.json(result)
                else:
                    st.error(f"? Bereinigung fehlgeschlagen: {result}")
    with col3:
        if st.button("?? Alle neu verarbeiten"):
            with st.spinner("?? Alle Dateien werden neu verarbeitet..."):
                success, result = make_api_request("/rechunk", method="POST")
                if success:
                    st.success(f"? Neuverarbeitung abgeschlossen: {result.get('processed', 0)} verarbeitet")
                    st.json(result)
                else:
                    st.error(f"? Neuverarbeitung fehlgeschlagen: {result}")

    # Dateiliste vom Backend holen
    success, files = make_api_request("/files")
    if success and files:
        st.subheader(f"?? Dateien ({len(files)} gesamt)")
        for file in files:
            with st.expander(f"?? {file['file_name']} - {create_status_badge(file['status'])}"):
                # Metadaten in Spalten
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**?? ID:** {file['id']}")
                    st.write(f"**?? Typ:** {file['document_type']}")
                    st.write(f"**?? Größe:** {format_file_size(file['file_size'])}")
                with col2:
                    st.write(f"**?? Hochgeladen:** {file['upload_date'][:10]}")
                    st.write(f"**?? Chunks:** {file['chunk_count']}")
                    if file.get('processing_duration_ms'):
                        st.write(f"**?? Dauer:** {format_duration(file['processing_duration_ms'])}")
                with col3:
                    quality = file.get('quality_metrics', {})
                    if quality:
                        st.write(f"**? Qualität:** {quality.get('content_quality_score', 0):.2f}")
                with col4:
                    # Lösch-Button
                    if st.button(f"??? Löschen", key=f"del_{file['id']}"):
                        success, result = make_api_request(f"/files/{file['id']}", method="DELETE")
                        if success:
                            st.success("? Datei gelöscht!")
                            st.rerun()
                        else:
                            st.error(f"? Löschen fehlgeschlagen: {result}")
                # Fehlermeldung anzeigen, falls vorhanden
                if file.get('error_message'):
                    st.error(f"? {file['error_message']}")
    elif success:
        st.info("?? Noch keine Dateien hochgeladen")
    else:
        st.error(f"? Dateien konnten nicht geladen werden: {files}")

# Suche & Abfrage

elif selected_page == "?? Suche & Abfrage":
    st.title("?? Suche & Abfrage")

    # Zwei Tabs: KI-gestützte Suche und klassische Volltextsuche
    tab1, tab2 = st.tabs(["?? KI-Suche", "?? Textsuche"])

    # ---------- KI-Suche ----------
    with tab1:
        st.subheader("?? KI-gestützte semantische Suche")
        query = st.text_area(
            "Ihre Frage:",
            value="Was muss ich beim mobilen Arbeiten beachten?",
            height=100,
            help="?? Stellen Sie mir alle Fragen rund um die IT der Berliner Wasserbetriebe"
        )
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.slider("?? Max. Ergebnisse:", 1, 20, 5)
        with col2:
            quality_threshold = st.slider("? Qualitätsschwelle:", 0.0, 1.0, 0.3, 0.1)
        if st.button("?? Suchen", type="primary") and query.strip():
            with st.spinner("?? KI verarbeitet Ihre Frage..."):
                try:
                    payload = {
                        "query": query.strip(),
                        "max_results": max_results,
                        "quality_threshold": quality_threshold
                    }
                    success, result = make_api_request("/query", method="POST", json=payload)
                    if success:
                        st.subheader("?? Antwort")
                        st.markdown(result.get("answer", "Keine Antwort generiert"))
                        # Performance-Metriken
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("?? Antwortzeit", f"{result.get('response_time_ms', 0)}ms")
                        with col2:
                            st.metric("?? Ergebnisse", result.get("results_count", 0))
                        with col3:
                            quality = result.get("quality_metrics", {})
                            st.metric("?? Relevanz", f"{quality.get('avg_relevance', 0):.3f}")
                        # Quellenauflistung
                        sources = result.get("sources", [])
                        if sources:
                            st.subheader("?? Quellen")
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"?? Quelle {i}: {source.get('source', 'Unbekannt')} (Seite {source.get('page', 'N/A')})"):
                                    st.markdown(source.get('text', ''))
                                    st.caption(f"?? Ähnlichkeit: {source.get('similarity', 0):.3f} | ? Qualität: {source.get('quality_score', 0):.3f}")
                    else:
                        st.error(f"? Suche fehlgeschlagen: {result}")
                except Exception as e:
                    st.error(f"? Fehler: {str(e)}")

    # ---------- Volltextsuche ----------
    with tab2:
        st.subheader("?? Volltextsuche")
        search_term = st.text_input(
            "Suchbegriffe:",
            placeholder="Geben Sie spezifische Wörter oder Phrasen ein...",
            help="?? Direkte Textsuche im Dokumentinhalt"
        )
        col1, col2 = st.columns(2)
        with col1:
            limit = st.slider("?? Max. Ergebnisse:", 1, 50, 10)
        with col2:
            quality_threshold = st.slider("? Qualitätsfilter:", 0.0, 1.0, 0.3, 0.1)
        if st.button("?? Text durchsuchen", type="primary") and search_term:
            with st.spinner("?? Textinhalt wird durchsucht..."):
                try:
                    params = {
                        "q": search_term,
                        "limit": limit,
                        "quality_threshold": quality_threshold
                    }
                    success, result = make_api_request("/search/fulltext", params=params)
                    if success:
                        st.success(f"? {result.get('results_count', 0)} Ergebnisse in {result.get('response_time_ms', 0)}ms gefunden")
                        results = result.get("results", [])
                        for i, res in enumerate(results, 1):
                            with st.expander(f"?? Ergebnis {i} - Qualität: {res.get('quality_score', 0):.3f}"):
                                st.markdown(res.get('text', ''))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"?? Seite: {res.get('page', 'N/A')}")
                                with col2:
                                    st.caption(f"?? Wörter: {res.get('word_count', 0)}")
                    else:
                        st.error(f"? Suche fehlgeschlagen: {result}")
                except Exception as e:
                    st.error(f"? Fehler: {str(e)}")


# Analytics

elif selected_page == "?? Analytik":
    st.title("?? Systemanalytik")

    # Zeitraum wählen
    days = st.selectbox(
        "?? Analysezeitraum:",
        [7, 14, 30, 60, 90],
        index=2,
        format_func=lambda x: f"Letzte {x} Tage"
    )

    # Analytics vom Backend abrufen
    success, analytics = make_api_request(f"/system/analytics?days={days}")
    if success:
        query_stats = analytics.get("query_analytics", {})
        if query_stats:
            st.subheader("?? Abfragestatistiken")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("?? Abfragen gesamt", query_stats.get("total_queries", 0))
            with col2:
                st.metric("?? Einzige Abfragen", query_stats.get("unique_queries", 0))
            with col3:
                avg_time = query_stats.get("avg_response_time", 0)
                st.metric("?? Ø Antwortzeit", format_duration(avg_time))
            with col4:
                st.metric("?? Ø Ergebnisse", f"{query_stats.get('avg_results_count', 0):.1f}")
            # Performancediagramme (langsame vs. schnelle Abfragen, Erfolgsquote)
            st.subheader("?? Leistungsanalyse")
            col1, col2 = st.columns(2)
            with col1:
                slow_queries = query_stats.get("slow_queries", 0)
                total_queries = query_stats.get("total_queries", 1)
                fast_queries = total_queries - slow_queries
                fig = px.pie(
                    values=[fast_queries, slow_queries],
                    names=["? Schnell (<5s)", "?? Langsam (>5s)"],
                    title="????? Abfrageleistung",
                    color_discrete_sequence=["#28a745", "#dc3545"]
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                zero_results = query_stats.get("zero_result_queries", 0)
                successful = total_queries - zero_results
                fig = px.pie(
                    values=[successful, zero_results],
                    names=["? Ergebnisse gefunden", "? Keine Ergebnisse"],
                    title="?? Erfolgsquote der Abfragen",
                    color_discrete_sequence=["#17a2b8", "#ffc107"]
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("?? Keine Abfragedaten für den gewählten Zeitraum verfügbar")
        # Datenbankgesundheit
        db_health = analytics.get("database_health", {})
        if db_health:
            st.subheader("?? Datenbankzustand")
            table_sizes = db_health.get("table_sizes", [])
            if table_sizes:
                df = pd.DataFrame(table_sizes)
                st.dataframe(df[['tablename', 'size']], use_container_width=True)
    else:
        st.error(f"? Analytikdaten konnten nicht geladen werden: {analytics}")


# Systemverwaltung

elif selected_page == "?? Systemverwaltung":
    st.title("?? Systemverwaltung")

    # Health-Status anzeigen
    st.subheader("?? Systemstatus")
    success, health = make_api_request("/health")
    if success:
        services = health.get("services", {})
        for service, info in services.items():
            if info["status"] == "ok":
                st.success(f"? **{service.title()}**: {info['message']}")
            else:
                st.error(f"? **{service.title()}**: {info['message']}")
    else:
        st.error(f"? Systemintegritätsprüfung fehlgeschlagen: {health}")

    st.markdown("---")

    # ---------------- Wartungsoperationen ----------------
    st.subheader("??? Wartungsoperationen")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("??? Datenbankwartung"):
            with st.spinner("?? Wartung läuft..."):
                success, result = make_api_request("/system/maintenance", method="POST")
                if success:
                    st.success("? Wartung abgeschlossen!")
                    results = result.get("maintenance_results", {})
                    for op, msg in results.items():
                        st.info(f"**{op}**: {msg}")
                else:
                    st.error(f"? Wartung fehlgeschlagen: {result}")
    with col2:
        if st.button("?? Fehlgeschlagene wiederholen"):
            with st.spinner("? Fehlgeschlagene Dateien werden wiederholt..."):
                success, result = make_api_request("/retry_failed", method="POST")
                if success:
                    st.success(f"? Wiederholung abgeschlossen: {result.get('retried', 0)} Dateien in Warteschlange")
                else:
                    st.error(f"? Wiederholung fehlgeschlagen: {result}")
    with col3:
        if st.button("?? Verarbeitungsqueue"):
            success, queue = make_api_request("/queue")
            if success:
                summary = queue.get("summary", {})
                st.info(f"? Wartend: {summary.get('pending', 0)}")
                st.info(f"?? In Bearbeitung: {summary.get('processing', 0)}")
                st.info(f"? Abgeschlossen: {summary.get('completed', 0)}")
                st.info(f"? Fehler: {summary.get('errors', 0)}")
            else:
                st.error(f"? Queue-Überprüfung fehlgeschlagen: {queue}")

    # ---------------- Konfiguration ----------------
    st.subheader("?? Konfiguration")
    success, formats = make_api_request("/formats")
    if success:
        supported = formats.get("supported_formats", {})
        st.write(f"?? Unterstützte Formate: {len(supported)}")
        format_list = list(supported.items())
        cols = st.columns(3)
        for i, (ext, desc) in enumerate(format_list):
            col_idx = i % 3
            with cols[col_idx]:
                st.caption(f"?? {ext}: {desc}")
    # ---------------- Debug-Infos ----------------
    with st.expander("?? Debug-Informationen"):
        debug_info = {
            "API_URL": API_BASE_URL,
            "Streamlit_Version": st.__version__,
            "Current_Time": datetime.now().isoformat()
        }
        st.json(debug_info)


# Fußzeile

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        ?? <strong>BWB PoC RAG System</strong><br>
        ? FastAPI • ?? ChromaDB • ?? PostgreSQL • ?? Streamlit<br>
        <em>?? Erweiterte Dokumentenverarbeitung • ?? Vektorsuche • ?? Echtzeitanalytik</em>
    </div>
    """,
    unsafe_allow_html=True
)


# ?? Auto-Refresh

if selected_page == "?? Dashboard":
    if st.sidebar.checkbox("?? Auto-Aktualisierung (30s)"):
        time.sleep(30)
        st.rerun()
