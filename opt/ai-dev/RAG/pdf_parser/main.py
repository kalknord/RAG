-*- coding: utf-8 -*-
# ---------------------------------------------------------------------
# Enhanced Document Parser
# ---------------------------------------------------------------------

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from PIL import Image
import pytesseract
import io
import fitz  # PyMuPDF
import re

# ---------- Optionale XLSX-Unterstützung ----------
try:
    import pandas as pd
    import openpyxl
    PANDAS_XLSX_AVAILABLE = True
except ImportError:
    PANDAS_XLSX_AVAILABLE = False

# ---------- Optionale Sentence-Transformer-Embeddings ----------
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Logger-Setup für einheitliche Service-Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI-Instanz mit Metadaten
app = FastAPI(title="Enhanced Document Parser - QUALITY FIX v2.8.0", version="2.8.0")

# Globale Objekte (werden beim Start geladen/konfiguriert)
embedding_model = None  # SentenceTransformer-Instanz oder None
tesseract_available = False  # Flag nach erfolgreicher OCR-Konfiguration

# Hilfsroutinen zur Bewertung & Klassifizierung von Textabschnitten.

def calculate_text_quality(text: str) -> float:
    """Einfacher, heuristischer Qualitäts-Score von 0.1 bis 1.0."""
    if not text or len(text.strip()) < 5:
        return 0.1  # Minimal bei sehr wenigen Zeichen

    score = 1.0
    text = text.strip()

    # --Textlänge ----------------------------------------------
    char_count = len(text)
    if char_count < 20:
        score *= 0.3  # sehr kurz
    elif char_count < 50:
        score *= 0.6  # kurz
    elif char_count > 2000:
        score *= 0.8  # sehr lang
    elif char_count > 1000:
        score *= 0.9  # lang

    # --Durchschnittliche Wortlänge ----------------------------
    words = text.split()
    word_count = len(words)
    if word_count > 0:
        avg_word_length = char_count / word_count
        if avg_word_length < 2:
            score *= 0.7  # evtl. OCR-Artefakte
        elif avg_word_length > 15:
            score *= 0.8  # Fachchinesisch / Fehler
    else:
        score *= 0.2

    # --Sonderzeichen-Rate -------------------------------------
    special_chars = len(re.findall(r'[^\w\s\.,!?\-()]', text))
    special_ratio = special_chars / char_count if char_count else 0
    if special_ratio > 0.2:
        score *= 0.6
    elif special_ratio > 0.1:
        score *= 0.8

    # --Großbuchstaben-Rate ------------------------------------
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / char_count if char_count else 0
    if upper_ratio > 0.5:
        score *= 0.7
    elif upper_ratio > 0.3:
        score *= 0.9

    # --Wiederholende Zeichen (z. B. "====") ------------------
    repeated_chars = len(re.findall(r'(.)\1{3,}', text))
    if repeated_chars:
        score *= 0.8

    # --Satzzeichen-Indikatoren --------------------------------
    sentence_indicators = len(re.findall(r'[.!?]', text))
    if char_count > 100 and sentence_indicators == 0:
        score *= 0.7

    # --Ziffern-Quote (Tabellen) --------------------------------
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / char_count if char_count else 0
    if digit_ratio > 0.3:
        score *= 1.1  # Tabelleninfos als wertvoll erachten

    # --Whitespace-Quote ---------------------------------------
    whitespace_count = sum(1 for c in text if c.isspace())
    whitespace_ratio = whitespace_count / char_count if char_count else 0
    if whitespace_ratio > 0.4:
        score *= 0.8

    # Normalisierung: mindestens 0.1, maximal 1.0
    return round(max(0.1, min(1.0, score)), 2)


def detect_content_features(text: str) -> dict:
    """Erkennt Tabellen, Listen, Zahlen, E-Mails, URLs & Sprache."""
    features = {
        'contains_table': False,
        'contains_list': False,
        'contains_numbers': False,
        'contains_email': False,
        'contains_url': False,
        'language_indicators': 'de'  # Default: deutsch
    }

    # -- Tabellen-Muster -------------------------------------------
    table_patterns = [
        r'\|.*\|',                # Pipe-Tabellen
        r'\t.*\t',               # Tab-separiert
        r'^\s*[-+|=\s]+$',       # horizontale Linien
        r'\d+\s*[%â‚¬$]',         # Zahlen mit Einheiten
        r'[A-Za-z]+\s*:\s*\d+', # Key-Value-Paare
    ]
    for pattern in table_patterns:
        if re.search(pattern, text, re.MULTILINE):
            features['contains_table'] = True
            break

    # -- Listen-Muster ---------------------------------------------
    list_patterns = [
        r'^\s*[-•*]\s+',     # Bulletpoints
        r'^\s*\d+\.\s+',  # Nummerierte Listen
        r'^\s*[a-zA-Z]\)\s+',  # a) b) c) …
    ]
    for pattern in list_patterns:
        if re.search(pattern, text, re.MULTILINE):
            features['contains_list'] = True
            break

    # -- Zahlen, Mails, URLs ---------------------------------------
    if re.search(r'\d{3,}', text):
        features['contains_numbers'] = True
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        features['contains_email'] = True
    if re.search(r'https?://[^\s]+', text):
        features['contains_url'] = True

    # -- Grobe Spracherkennung (de/en) ------------------------------
    german_indicators = ['der', 'die', 'das', 'und', 'oder', 'mit', 'für', 'von', 'ist', 'sind']
    english_indicators = ['the', 'and', 'or', 'with', 'for', 'from', 'is', 'are', 'this', 'that']
    text_lower = text.lower()
    german_count = sum(1 for w in german_indicators if w in text_lower)
    english_count = sum(1 for w in english_indicators if w in text_lower)
    if english_count > german_count:
        features['language_indicators'] = 'en'

    return features

# Model-/OCR-Setup
def load_embedding_model():
    """Lädt Sentence-Transformer (lokal oder aus HF-Hub)."""
    global embedding_model
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        logger.warning("SentenceTransformers not available")
        return False
    try:
        model_path = "/models/all-MiniLM-L6-v2"
        if os.path.exists(model_path):
            logger.info(f"Loading embedding model from {model_path}")
            embedding_model = SentenceTransformer(model_path)
        else:
            logger.info("Loading embedding model from HuggingFace Hub")
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("? Embedding model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"? Failed to load embedding model: {e}")
        return False


def setup_tesseract():
    """Versucht, Tesseract-Binary zu finden und einen Kurz-OCR-Test auszuführen."""
    global tesseract_available
    try:
        possible_paths = ['/usr/bin/tesseract', '/usr/local/bin/tesseract', 'tesseract']
        tesseract_cmd = next((p for p in possible_paths if os.path.exists(p) or p == 'tesseract'), None)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            # Mini-Smoke-Test
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            tesseract_available = True
            logger.info(f"? Tesseract OCR configured at {tesseract_cmd}")
            return True
        else:
            logger.error("? Tesseract not found")
            return False
    except Exception as e:
        logger.error(f"? Failed to setup Tesseract: {e}")
        return False

#OCR-Scanning
def extract_text_with_tesseract(image_path: str, languages: List[str] = ["deu", "eng"]) -> str:
    """Führt eine Basis-Bildvorverarbeitung durch und liest Text via OCR."""
    if not tesseract_available:
        return ""
    try:
        image = Image.open(image_path).convert('RGB')
        # -- Kontrast & Rauschentfernung mittels OpenCV --------------
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced)
        enhanced_image = Image.fromarray(denoised)
        # OCR-Call
        lang_string = "+".join(languages)
        config = f'--oem 3 --psm 6 -l {lang_string}'
        text = pytesseract.image_to_string(enhanced_image, config=config)
        return text.strip()
    except Exception as e:
        logger.error(f"Tesseract OCR error: {e}")
        return ""

def simple_text_element(text: str, page_num: int = 1, filename: str = ""):
    """Erstellt ein einfaches Dict im gleichen Stil wie "unstructured"-Elemente."""
    return {
        "text": text,
        "metadata": {
            "page_number": page_num,
            "filename": filename,
            "element_type": "Text",
            "source": "pymupdf"
        }
    }



# PDF-Parsing mit PYMUPDF (+ OCR)
def parse_pdf_with_pymupdf(file_path: str, extract_images: bool = True, use_ocr: bool = True, ocr_languages: List[str] = ["deu", "eng"]) -> tuple:
    """Liest Text & (optional) Bilder aus PDF, führt ggf. OCR durch."""
    try:
        doc = fitz.open(file_path)
        elements = []
        ocr_stats = {"images_processed": 0, "ocr_text_length": 0}
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # -- Direkt extrahierter Text ---------------------------
            text = page.get_text()
            if text.strip():
                elements.append(simple_text_element(text, page_num + 1, os.path.basename(file_path)))
            # -- Bilder + OCR ---------------------------------------
            if extract_images and use_ocr and tesseract_available:
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        # Nur RGB/Grau (keine CMYK/Masken) ----------------
                        if pix.n - pix.alpha < 4:
                            os.makedirs("/tmp/ocr_temp", exist_ok=True)
                            img_path = f"/tmp/ocr_temp/page_{page_num}_img_{img_index}.png"
                            pix.save(img_path)
                            ocr_text = extract_text_with_tesseract(img_path, ocr_languages)
                            if ocr_text.strip():
                                elements.append(simple_text_element(ocr_text, page_num + 1, os.path.basename(file_path)))
                                ocr_stats["images_processed"] += 1
                                ocr_stats["ocr_text_length"] += len(ocr_text)
                            if os.path.exists(img_path):
                                os.remove(img_path)
                        pix = None  # Speicher freigeben
                    except Exception as img_error:
                        logger.warning(f"Failed to process image {img_index} on page {page_num}: {img_error}")
                        continue
        doc.close()
        logger.info(f"PDF parsed with PyMuPDF: {len(elements)} elements, OCR stats: {ocr_stats}")
        return elements, ocr_stats
    except Exception as e:
        logger.error(f"PyMuPDF parsing error: {e}")
        return [], {}


# Chungking mit Qualitätsbewertung
def enhanced_chunk_splitting(text: str, max_chars: int = 500) -> List[dict]:
    """Teilt Text intelligent in Chunks und versieht sie mit Quality & Features."""
    if len(text) <= max_chars:
        quality = calculate_text_quality(text)
        features = detect_content_features(text)
        return [{
            'text': text,
            'quality_score': quality,
            'features': features
        }]
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence != sentences[-1]:
            sentence += "."
        # Chunk zusammenbauen --------------------------------------
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                quality = calculate_text_quality(current_chunk)
                features = detect_content_features(current_chunk)
                chunks.append({'text': current_chunk, 'quality_score': quality, 'features': features})
            current_chunk = sentence
    if current_chunk:
        quality = calculate_text_quality(current_chunk)
        features = detect_content_features(current_chunk)
        chunks.append({'text': current_chunk, 'quality_score': quality, 'features': features})
    return chunks

def parse_document_simple(file_path: str) -> tuple:
    """Delegiert auf passende Parser-Funktion je nach Dateiendung."""
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.pdf':
        return parse_pdf_with_pymupdf(file_path)
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            return [simple_text_element(text, 1, os.path.basename(file_path))], {"parser": "simple_text"}
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

class ParseRequest(BaseModel):
    file_path: str
    strategy: str = "hi_res"
    chunking_strategy: str = "enhanced"
    max_characters: int = 500
    generate_embeddings: bool = True
    use_ocr: bool = True
    ocr_languages: List[str] = ["deu", "eng"]
    extract_images: bool = True

class ParseResponse(BaseModel):
    success: bool
    pages: List[str]
    chunks: List[Dict[str, Any]]
    embeddings: List[List[float]]
    metadata: Dict[str, Any]
    ocr_info: Dict[str, Any] = {}
    error: Optional[str] = None

class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    success: bool
    error: Optional[str] = None

# Starup-Event
@app.on_event("startup")
async def startup_event():
    """Lädt ML-Modelle & OCR beim Service-Start."""
    embedding_success = load_embedding_model()
    tesseract_success = setup_tesseract()
    logger.info("?? QUALITY-ENHANCED DOCUMENT PARSER STARTUP:")
    logger.info(f"   - Embedding model: {embedding_success}")
    logger.info(f"   - Tesseract OCR: {tesseract_success}")
    logger.info("   - Quality calculation: ENABLED")
    logger.info("   - Content detection: ENABLED")
    logger.info("   ? No internet downloads required")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Encode-Wrapper für Sentence-Transformers."""
    global embedding_model
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    try:
        embeddings = embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {e}")

#API-Endpunkte
@app.post("/parse", response_model=ParseResponse)
async def parse_document(request: ParseRequest):
    """Hauptroute: Dokument einlesen ? chunks ? embeddings (optional)."""
    try:
        file_path = request.file_path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file
