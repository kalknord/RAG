# =====================================
# LLM SERVICE CLIENT - ANTWORT-GENERIERUNG
# =====================================

import httpx                        
import os                          
import asyncio                     

LLM_SERVER_URL = os.getenv("LLM_SERVER_URL", "http://rag_llm_server:8000/completion")

async def generate_answer(question: str, docs: list[dict], max_tokens: int = 10000):
    """
    Generiert kontextbezogene Antworten basierend auf Retrieval-Ergebnissen.
    
    Diese Funktion implementiert den finalen Schritt der RAG-Pipeline, wo
    semantisch relevante Dokumente in eine kohärente, natürlichsprachliche
    Antwort transformiert werden.
    
    Parameter:
        question (str): Natürlichsprachliche Benutzer-Frage
                       Sollte spezifisch und im Kontext der verfügbaren Dokumente sein
        
        docs (list[dict]): Retrieval-Ergebnisse aus Vektor-Suche
                          Jedes dict enthält 'text', 'metadata' mit 'source', 'page'
                          Sortiert nach Relevanz (höchste Similarity zuerst)
        
        max_tokens (int): Maximale Länge der generierten Antwort
                         512 = Balance zwischen Detailliertheit und Performance
                         Höhere Werte für komplexere Antworten, niedrigere für Effizienz
    
    Rückgabe:
        str: Deutschsprachige, kontextbezogene Antwort basierend auf bereitgestellten Dokumenten
             Enthält Quellenverweise wenn möglich, ehrliche "Weiß nicht"-Antworten bei Wissenslücken
    """
    
    # KONTEXT-ASSEMBLY: Transformation von Retrieval-Chunks zu strukturiertem Kontext
    context_parts = []
    for doc in docs:
        # METADATEN-EXTRAKTION: Sichere Navigation durch nested dictionaries
        source = doc.get('metadata', {}).get('source', 'Unbekannt')
        page = doc.get('metadata', {}).get('page', 'N/A')
        text = doc.get('text', '')
        
        # QUELLENANGABEN-FORMAT: Strukturierte Zuordnung von Content zu Origin
        context_parts.append(f"{text} (Quelle: {source}, Seite {page})")
    
    # KONTEXT-AGGREGATION: Zusammenfügung aller relevanten Chunks
    context = "\n\n".join(context_parts)
    
    # PROMPT ENGINEERING: Optimierter Prompt für deutsches SauerkrautLM-Modell
    prompt = f"""
    
Bitte beantworte die folgende Frage ausschließlich auf Grundlage des bereitgestellten Kontexts.
    
Die Informationen stammen aus offiziellen Dokumenten der Berliner Wasserbetriebe AöR, einem kommunalen Wasserver- und Entsorgungsunternehmen mit Sitz in Berlin. Diese Dokumente umfassen insbesondere:
    
    Arbeitsanweisungen
      
    Dienstvereinbarungen
      
    Betriebsvereinbarungen
      
    Interne Richtlinien
    
Die Inhalte dieser Unterlagen betreffen unter anderem die folgenden Themen:
    
    Arbeitsorganisation und Zuständigkeiten
      
    Sicherheitsvorschriften und Arbeitsschutz
      
    Schichtdienstregelungen
      
    Umweltschutzmaßnahmen
      
    Technisch-betriebliche Abläufe und Verfahrensanweisungen
      
Bitte beachte bei der Beantwortung der Frage die folgenden Vorgaben:
      
Antworte ausschließlich auf Deutsch.
    
    Verwende nur Informationen, die explizit im bereitgestellten Kontext enthalten sind.
      
    Ziehe keine Rückschlüsse aus allgemeinem Wissen oder Erfahrungswerten. Nur der konkrete Wortlaut des Kontexts ist maßgeblich.
      
    Formatiere die Antwort nicht in Markdown, sondern verwende einfachen, klaren Fließtext.

Nutze bei Fragen zu ABläufen gerne Stickpunkte oder Handlungsabläufe.
      
Wenn sich die gestellte Frage nicht eindeutig anhand des Kontexts beantworten lässt, gib dies bitte offen und ehrlich an. Verwende dafür folgende Formulierung:
„Im gegebenen Kontext liegen dazu keine Informationen vor.“

Gebe keine Quellen im Text an.
      
Gib die Antwort strukturiert in Markdown aus. Entscheide über Überschriften und Unterpunkte selbst. Es darf keine Einrückungen geben. Achte darauf, das der unterschied der Textgrößen (z.B. H1 und normaler Text) nicht zu groß sind.
      
Achte darauf, dass die Antwort korrektes Markdown ist – z. B.:
    
    Absätze: \n\n
          
    Listen: - Punkt 1
          
    Fett: **Text**
          
    Überschriften: # H1, ## H2, usw.

Kontext:
{context}

Frage: {question}

Antwort:"""

    # HTTP-CLIENT: Asynchroner Request mit optimierten Timeout-Einstellungen
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # LLM-API-REQUEST: llama.cpp Server-kompatible Parameter
            response = await client.post(
                LLM_SERVER_URL,
                json={
                    "prompt": prompt,                   # Vollständig konstruierter Prompt
                    "n_predict": max_tokens,            # Maximale Response-Länge
                    "temperature": 0.0,                 # Kreativitäts-Balance (0.0=deterministisch, 1.0=kreativ)
                    "top_k": 40,                        # Vocabulary-Sampling für Diversität
                    "top_p": 0.9,                       # Nucleus-Sampling für Qualität
                    "stop": ["Frage:", "Kontext:", "\n\nFrage:", "\n\nKontext:"]  # Stop-Sequences für saubere Terminierung
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Logik normalisiert Antworten verschiedener LLM-Service-Implementierungen
            if "content" in result:
                # llama.cpp Server Standard-Format
                return result["content"].strip()
            elif "text" in result:
                # Alternative direkte Text-Einbettung
                return result["text"].strip()
            elif "choices" in result and len(result["choices"]) > 0:
                # OpenAI-API-kompatibles Format mit Choices-Array
                return result["choices"][0].get("text", "").strip()
            else:
                # FALLBACK: Unbekanntes Response-Format
                return "Keine Antwort vom Sprachmodell erhalten."
                
        except httpx.HTTPError as e:
            # HTTP-TRANSPORT-FEHLER: Netzwerk, Timeout, HTTP-Status-Probleme
            print(f"HTTP error when calling LLM service: {e}")
            return f"Fehler beim Abrufen der Antwort: {str(e)}"
        except Exception as e:
            # ALLGEMEINE FEHLERBEHANDLUNG: JSON-Parsing, unexpected Exceptions
            print(f"Error calling LLM service: {e}")
            return f"Unerwarteter Fehler: {str(e)}"

def generate_answer_sync(question: str, docs: list[dict], max_tokens: int = 512):
    """
    Synchrone Wrapper-Funktion für Rückwärtskompatibilität mit sync-Code.
    
    Diese Funktion ermöglicht die Verwendung der async LLM-Pipeline in
    synchronen Kontexten ohne explizite async/await-Behandlung.
    
    Parameter:
        question, docs, max_tokens: Identisch zur async-Version
    
    Rückgabe:
        str: Synchron zurückgegebene LLM-Response (identisch zur async-Version)
    """
    # Erstellt temporäre Event-Loop, führt async-Function aus, bereinigt Loop
    return asyncio.run(generate_answer(question, docs, max_tokens))