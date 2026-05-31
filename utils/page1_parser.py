"""
Lecture complète de la page 1 d'un formulaire d'examen.

Assemble les lectures de : grid_decoder, ocr_utils, signature_utils
pour produire le dictionnaire PAGE-01 attendu par autoReadForm.
"""

import cv2
import numpy as np
from datetime import datetime

from utils.grid_decoder import (
    normalize_page,
    read_student_id,
    read_group,
    read_conditions,
    read_note_maximale,
    read_note_pour_valider,
    extract_signature_roi,
    extract_cryptogram,
    get_roi,
    ROI_CODES_EXAM,
    ROI_FIRSTNAME,
    ROI_NAME,
)
from utils.ocr_utils import ocr_codes_exam, ocr_text, ocr_handwritten_unite
from utils.signature_utils import match_signature_to_id


# ---------------------------------------------------------------------------
# Lecture du prénom et du nom manuscrits
# ---------------------------------------------------------------------------

_CELL_W = 27   # largeur approximative d'une cellule de lettre (pixels normalisés)
_MAX_CELLS = 15  # nombre max de cellules à lire


def _ocr_cells(roi: np.ndarray) -> str:
    """
    Lit une rangée de cellules de lettres manuscrites.

    Méthode : segmentation par colonnes séparatrices (projection verticale),
    puis OCR cellule par cellule via easyocr.

    Algorithme bas niveau :
    1. Binariser l'image
    2. Projection verticale → identifier les colonnes séparatrices (forte densité)
    3. Segmenter les cellules entre ces colonnes
    4. OCR chaque cellule agrandie
    5. Fusionner les lettres reconnues
    """
    from utils.ocr_utils import _get_reader, _to_gray
    import re as _re

    if roi is None or roi.size == 0:
        return ""

    gray = _to_gray(roi)
    h, w = gray.shape

    # Binariser pour projection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    col_proj = np.sum(binary.astype(np.float32), axis=0) / (h * 255)

    # Trouver les colonnes "vides" = séparatrices inter-cellules
    # Entre chaque lettre, il y a une fine ligne verticale (densité élevée)
    # ET des zones vides (densité faible)
    # Détecter les transitions bas → haut de la projection
    # pour trouver le début de chaque cellule

    # Alternative plus simple : détection des cellules par leur espacement régulier
    # En cherchant le premier séparateur et en estimant la largeur
    # Trouver les colonnes avec projection > 0.3 (probables séparateurs)
    is_sep = col_proj > 0.30

    # Trouver les runs de séparateurs
    sep_starts = []
    in_sep = False
    for x, s in enumerate(is_sep):
        if s and not in_sep:
            in_sep = True
            sep_starts.append(x)
        elif not s and in_sep:
            in_sep = False

    if not sep_starts:
        return ""

    # Construire les ROIs des cellules (entre les séparateurs)
    boundaries = [0] + sep_starts + [w]
    cells = []
    for i in range(len(boundaries) - 1):
        x0, x1 = boundaries[i], boundaries[i + 1]
        if x1 - x0 > 5:  # cellule de largeur minimum
            cells.append((x0, x1))

    if not cells:
        return ""

    reader = _get_reader()
    letters = []
    for x0, x1 in cells[:_MAX_CELLS]:
        cell_img = roi[:, x0:x1]
        if cell_img.shape[1] < 3:
            continue
        # Agrandir la cellule pour meilleure OCR
        cell_big = cv2.resize(cell_img,
                              (cell_img.shape[1] * 6, cell_img.shape[0] * 6),
                              interpolation=cv2.INTER_CUBIC)
        results = reader.readtext(cell_big, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        if results:
            letter = _re.sub(r"[^A-Za-z]", "", "".join(results))
            if letter:
                letters.append(letter[0].upper())

    return "".join(letters)


def _clean_name_text(raw: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-zÀ-ÿ\- ]", "", raw)
    return " ".join(cleaned.split()).strip()


def read_firstname(form_img: np.ndarray) -> str:
    """
    Lit le prénom manuscrit.
    Essai 1 : OCR direct sur la ROI entière avec upscale ×6 (plus robuste
              pour les ROIs de petite hauteur ~24 px).
    Essai 2 : segmentation cellule par cellule (fallback).
    """
    roi = get_roi(form_img, ROI_FIRSTNAME)
    raw = ocr_text(roi, scale=6)
    cleaned = _clean_name_text(raw)
    if cleaned:
        return cleaned.capitalize()
    result = _ocr_cells(roi)
    return result.capitalize() if result else ""


def read_name(form_img: np.ndarray) -> str:
    """
    Lit le nom manuscrit.
    Même stratégie que read_firstname mais retourne en majuscules.
    """
    roi = get_roi(form_img, ROI_NAME)
    raw = ocr_text(roi, scale=6)
    cleaned = _clean_name_text(raw)
    if cleaned:
        return cleaned.upper()
    result = _ocr_cells(roi)
    return result.upper() if result else ""


# ---------------------------------------------------------------------------
# Lecture de la bande CODES EXAM
# ---------------------------------------------------------------------------

def read_codes_exam(form_img: np.ndarray) -> dict:
    """
    Lit la bande colorée CODES EXAM.
    Retourne { 'module', 'professor', 'date', 'code' }.
    """
    roi = get_roi(form_img, ROI_CODES_EXAM)
    return ocr_codes_exam(roi)


# ---------------------------------------------------------------------------
# Validation du cryptogramme
# ---------------------------------------------------------------------------

def compare_cryptograms(crypto_refs: list[np.ndarray],
                        crypto_query: np.ndarray,
                        threshold: float = 0.75) -> bool:
    """
    Vérifie que tous les cryptogrammes d'une liste sont identiques au
    cryptogramme de référence (page 1).

    Méthode : corrélation normalisée (NCC) entre images binarisées.
    """
    if not crypto_refs:
        return True

    def binarize(img: np.ndarray) -> np.ndarray:
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return b

    ref_size = (80, 35)
    ref_b = cv2.resize(binarize(crypto_query), ref_size).astype(np.float32) / 255.0

    for crypto in crypto_refs:
        if crypto.size == 0:
            continue
        q_b = cv2.resize(binarize(crypto), ref_size).astype(np.float32) / 255.0
        # NCC (Normalized Cross-Correlation)
        numer = float(np.sum(ref_b * q_b))
        denom = np.sqrt(float(np.sum(ref_b ** 2)) * float(np.sum(q_b ** 2)))
        ncc = numer / (denom + 1e-9)
        if ncc < threshold:
            return False
    return True


# ---------------------------------------------------------------------------
# Parser principal PAGE-01
# ---------------------------------------------------------------------------

def parse_page1(
    form_img: np.ndarray,
    desc_db: dict | None = None,
    expected_student_id: int | None = None,
    crypto_pages: list[np.ndarray] | None = None,
) -> dict:
    """
    Lit tous les champs de la page 1 et retourne un dictionnaire
    dont la structure correspond à l'onglet PAGE-01 du xlsx :

    {
        'module': str,
        'professor': str,
        'date': str,
        'code': str,
        'notes_cours': int,         # 0 ou 1
        'notes_manuscrites': int,   # 0 ou N (max pages)
        'ordinateur': int,          # 0 ou 1
        'calculatrice': int,        # 0 ou 1
        'brouillon': int,           # 0 ou N (max feuilles)
        'note_max': int,
        'note_valid': int,
        'prenom': str,
        'nom': str,
        'validation_signature': int,  # 1 = OK, 0 = non reconnu
        'group': str,
        'student_id': int | None,
        'validation_cryptogramme': int,  # 1 = OK, 0 = incohérent
    }
    """
    norm = normalize_page(form_img)

    # ---- Textes imprimés (CODES EXAM) ------------------------------------
    codes = read_codes_exam(norm)

    # ---- Conditions d'examen ---------------------------------------------
    conds = read_conditions(norm)

    # ---- Notes -----------------------------------------------------------
    note_max   = read_note_maximale(norm)
    note_valid = read_note_pour_valider(norm)

    # ---- Identifiants graphiques (grilles) -------------------------------
    student_id = read_student_id(norm)
    group      = read_group(norm)

    # ---- Prénom / Nom manuscrits -----------------------------------------
    prenom = read_firstname(norm)
    nom    = read_name(norm)

    # ---- Validation de la signature --------------------------------------
    sig_img  = extract_signature_roi(norm)
    sig_valid = 0
    if desc_db:
        sid_str = str(expected_student_id) if expected_student_id else str(student_id)
        _, validated = match_signature_to_id(
            sig_img, desc_db, expected_id=sid_str, threshold=0.18
        )
        sig_valid = 1 if validated else 0

    # ---- Validation du cryptogramme --------------------------------------
    crypto_ref  = extract_cryptogram(norm)
    crypto_ok   = 1
    if crypto_pages:
        crypto_ok = 1 if compare_cryptograms(crypto_pages, crypto_ref) else 0

    return {
        "module":                 codes.get("module", ""),
        "professor":              codes.get("professor", ""),
        "date":                   codes.get("date", ""),
        "code":                   codes.get("code", ""),
        "notes_cours":            conds["notes_cours"],
        "notes_manuscrites":      conds["notes_manuscrites"],
        "ordinateur":             conds["ordinateur"],
        "calculatrice":           conds["calculatrice"],
        "brouillon":              conds["brouillon"],
        "note_max":               note_max,
        "note_valid":             note_valid,
        "prenom":                 prenom,
        "nom":                    nom,
        "validation_signature":   sig_valid,
        "group":                  group or "",
        "student_id":             student_id,
        "validation_cryptogramme": crypto_ok,
    }
