"""
Lecture automatique des pages d'examen (page 5 → fin).

Approche bas niveau :
  - Détection des blocs de questions par lignes horizontales longues
    (morphologie mathématique + transformée de Hough)
  - Détection des cases à cocher MCQ par composantes connexes
  - Vérification X via analyse de contraste d'encre
  - OCR pour les réponses numériques (mantisse, exposant, unité)
"""

import cv2
import numpy as np

from utils.grid_decoder import normalize_page
from utils.checkbox_reader import (
    preprocess_for_checkbox,
    ink_ratio,
    has_x_pattern,
)
from utils.ocr_utils import (
    ocr_handwritten_mantisse,
    ocr_handwritten_exposant,
    ocr_handwritten_unite,
    ocr_text,
)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

HEADER_BAND_H  = 82     # hauteur de la bande Module/Code/Date en haut de page
MIN_LINE_WIDTH = 480    # largeur min d'une ligne horizontale détectée
LINE_MERGE_TOL = 40     # tolérance de fusion pour les lignes proches (px)

# Colonne gauche où chercher les checkboxes MCQ
MCQ_X_START = 12
MCQ_X_END   = 90
MCQ_MIN_SZ  = 12        # taille min d'un côté de checkbox
MCQ_MAX_SZ  = 35        # taille max d'un côté de checkbox
MCQ_MIN_AREA = 80

# Seuil de détection pour une case cochée
CHECKED_INK_THRESHOLD = 0.10

# Choix MCQ en ordre alphabétique
MCQ_CHOICES = "ABCDEFGH"

# Zones des réponses numériques (en proportion de la hauteur du bloc)
# Mantisse : grande case à gauche de ".10"
MANTISSE_X_FRAC = (0.05, 0.28)
MANTISSE_Y_FRAC = (0.50, 0.88)
# Exposant : petite case au-dessus de ".10"
EXPOSANT_X_FRAC = (0.28, 0.42)
EXPOSANT_Y_FRAC = (0.38, 0.66)
# Unité : case à droite de ".10"
UNITE_X_FRAC    = (0.48, 0.72)
UNITE_Y_FRAC    = (0.55, 0.90)


# ---------------------------------------------------------------------------
# Détection des blocs de questions
# ---------------------------------------------------------------------------

def _find_horizontal_lines(page_bin: np.ndarray,
                            min_width: int = MIN_LINE_WIDTH) -> list[int]:
    """
    Retourne les y-positions des lignes horizontales longues.
    Méthode : ouverture morphologique horizontale (bas niveau).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 2))
    horiz = cv2.morphologyEx(page_bin, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ys = []
    for cnt in contours:
        _, y, w, _ = cv2.boundingRect(cnt)
        if w >= min_width:
            ys.append(y)
    return sorted(ys)


def _merge_close_lines(ys: list[int], tol: int = LINE_MERGE_TOL) -> list[int]:
    """Fusionne les lignes horizontales proches (artefacts de numérisation)."""
    if not ys:
        return []
    merged = [ys[0]]
    for y in ys[1:]:
        if y - merged[-1] > tol:
            merged.append(y)
    return merged


def detect_question_blocks(exam_page: np.ndarray) -> list[tuple[int, int]]:
    """
    Retourne la liste des (y_start, y_end) pour chaque bloc de question.
    Le premier bloc commence après la bande d'en-tête de page.
    """
    gray = cv2.cvtColor(exam_page, cv2.COLOR_BGR2GRAY) \
           if len(exam_page.shape) == 3 else exam_page
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    ys = _find_horizontal_lines(binary)
    ys = _merge_close_lines(ys)

    H = exam_page.shape[0]
    # On inclut les bords haut/bas
    boundaries = [y for y in ys if HEADER_BAND_H <= y <= H - 30]
    if not boundaries:
        return [(HEADER_BAND_H, H)]

    blocks = []
    prev = HEADER_BAND_H
    for y in boundaries:
        if y - prev > 50:  # bloc de hauteur minimale
            blocks.append((prev, y))
        prev = y
    # Dernier bloc jusqu'au bas
    if H - prev > 50:
        blocks.append((prev, H))

    return blocks


# ---------------------------------------------------------------------------
# Détection des checkboxes MCQ dans un bloc
# ---------------------------------------------------------------------------

def _find_mcq_checkboxes(block_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Retourne la liste des bounding boxes (x,y,w,h) des cases à cocher MCQ
    trouvées dans la colonne gauche du bloc.
    Utilise l'analyse de composantes connexes.
    """
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY) \
           if len(block_img.shape) == 3 else block_img
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    strip = binary[:, MCQ_X_START:MCQ_X_END]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(strip, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if (MCQ_MIN_SZ <= w <= MCQ_MAX_SZ and
                MCQ_MIN_SZ <= h <= MCQ_MAX_SZ and
                area >= MCQ_MIN_AREA):
            # Ratio W/H proche de 1 → case carrée
            if 0.5 < w / (h + 1e-6) < 2.0:
                boxes.append((x + MCQ_X_START, y, w, h))

    return sorted(boxes, key=lambda b: b[1])  # trier par y


def _is_checked_box(block_img: np.ndarray, x: int, y: int,
                    w: int, h: int) -> bool:
    """
    Détermine si une case est cochée.
    Utilise la densité d'encre et la détection du motif X.
    """
    roi = block_img[y:y + h, x:x + w]
    binary = preprocess_for_checkbox(roi)
    margin = max(2, int(min(binary.shape) * 0.12))
    if binary.shape[0] > 2 * margin and binary.shape[1] > 2 * margin:
        inner = binary[margin:-margin, margin:-margin]
    else:
        inner = binary
    ratio = ink_ratio(inner)
    if ratio < CHECKED_INK_THRESHOLD:
        return False
    return has_x_pattern(inner) or ratio > 0.30


def _parse_mcq_choices(block_img: np.ndarray) -> dict[str, int]:
    """
    Détecte les choix MCQ cochés dans le bloc.
    Retourne {'A': 1/None, 'B': 1/None, ...} pour les choix présents.
    """
    boxes = _find_mcq_checkboxes(block_img)
    result = {}
    for idx, (x, y, w, h) in enumerate(boxes):
        if idx >= len(MCQ_CHOICES):
            break
        letter = MCQ_CHOICES[idx]
        result[letter] = 1 if _is_checked_box(block_img, x, y, w, h) else None
    return result


# ---------------------------------------------------------------------------
# Détection des réponses numériques
# ---------------------------------------------------------------------------

def _has_numerical_answer(block_img: np.ndarray) -> bool:
    """
    Détecte si le bloc contient une zone de réponse numérique
    (texte '.10' ou 'Value/Valeur').
    Utilise une projection sur la colonne centrale pour repérer
    la structure 'mantisse × 10^exposant'.
    """
    # Heuristique : un bloc numérique a une colonne centrale avec peu de cases
    # mais des rectangles larges (les cases mantisse/exposant)
    h, w = block_img.shape[:2]
    # Chercher des contours larges dans la zone centrale du bloc
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY) \
           if len(block_img.shape) == 3 else block_img
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Un bloc numérique a une grande case mantisse (large et haute)
    # dans la partie inférieure (y > 50% de la hauteur du bloc)
    lower = binary[h // 2:, :]
    num, _, stats, _ = cv2.connectedComponentsWithStats(lower, connectivity=8)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if bw > w * 0.12 and bh > h * 0.08 and area > 300:
            return True

    # Alternative : chercher les checkboxes → si aucune, probablement numérique
    boxes = _find_mcq_checkboxes(block_img)
    return len(boxes) == 0


def _find_answer_boxes(block_img: np.ndarray
                       ) -> list[tuple[int, int, int, int]]:
    """
    Détecte les cadres bordurés des zones de réponse numérique
    (mantisse, exposant, unité) dans le bloc.

    Méthode bas niveau : dilatation morphologique + contours externes.
    Retourne une liste de (x, y, w, h) triée par position x.
    """
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY) \
           if len(block_img.shape) == 3 else block_img
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    h, w = gray.shape

    # Chercher les rectangles dans la moitié inférieure du bloc
    lower = binary[h // 2:, :]
    contours, _ = cv2.findContours(lower, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Filtrer : taille cohérente avec une case de réponse
        if (w * 0.04 < bw < w * 0.35 and
                h * 0.03 < bh < h * 0.25 and
                bw * bh > 200):
            boxes.append((bx, by + h // 2, bw, bh))

    return sorted(boxes, key=lambda b: b[0])


def _parse_numerical_answer(block_img: np.ndarray) -> dict:
    """
    Extrait mantisse, exposant et unité via détection de cadres bordurés.
    Fallback sur fractions fixes si la détection échoue.
    """
    h, w = block_img.shape[:2]

    boxes = _find_answer_boxes(block_img)

    mantisse_img = exposant_img = unite_img = None

    if len(boxes) >= 2:
        # Trier par taille : le plus grand = mantisse, le plus petit = exposant
        by_area = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        # Mantisse : grande boîte la plus à gauche parmi les grandes
        large = [b for b in by_area if b[2] * b[3] >= by_area[0][2] * by_area[0][3] * 0.4]
        large_sorted_x = sorted(large, key=lambda b: b[0])

        if large_sorted_x:
            bx, by, bw, bh = large_sorted_x[0]
            mantisse_img = block_img[by:by + bh, bx:bx + bw]

        # Exposant : petite boîte au-dessus du ".10" = le plus haut (y le plus petit)
        small = [b for b in boxes if b[2] * b[3] < by_area[0][2] * by_area[0][3] * 0.5]
        if small:
            small_sorted_y = sorted(small, key=lambda b: b[1])
            bx, by, bw, bh = small_sorted_y[0]
            exposant_img = block_img[by:by + bh, bx:bx + bw]

        # Unité : boîte la plus à droite parmi les grandes
        if len(large_sorted_x) >= 2:
            bx, by, bw, bh = large_sorted_x[-1]
            unite_img = block_img[by:by + bh, bx:bx + bw]

    # Fallback : fractions fixes
    if mantisse_img is None:
        mantisse_img = block_img[int(h*0.68):int(h*0.92), int(w*0.03):int(w*0.22)]
    if exposant_img is None:
        exposant_img = block_img[int(h*0.58):int(h*0.76), int(w*0.24):int(w*0.36)]
    if unite_img is None:
        unite_img = block_img[int(h*0.68):int(h*0.92), int(w*0.42):int(w*0.60)]

    mantisse = ocr_handwritten_mantisse(mantisse_img)
    exposant = ocr_handwritten_exposant(exposant_img)
    unite    = ocr_handwritten_unite(unite_img)

    return {"mantisse": mantisse, "exposant": exposant, "unite": unite}


# ---------------------------------------------------------------------------
# Parser de bloc unique
# ---------------------------------------------------------------------------

def parse_question_block(block_img: np.ndarray,
                         question_num: int) -> dict:
    """
    Parse un bloc de question et retourne un dictionnaire :
    {
        'question': int,
        'choix': {'A': 1 or None, 'B': 1 or None, ...},   # MCQ
        'mantisse': float or None,
        'exposant': int or None,
        'unite': str or None,
    }
    """
    result = {
        "question": question_num,
        "choix": {},
        "mantisse": None,
        "exposant": None,
        "unite": None,
    }

    # Ignorer la barre d'en-tête du bloc (environ 22% du haut)
    header_cut = max(20, int(block_img.shape[0] * 0.22))
    content = block_img[header_cut:, :]

    try:
        boxes = _find_mcq_checkboxes(content)
        if boxes:
            result["choix"] = _parse_mcq_choices(content)
        elif _has_numerical_answer(content):
            num_data = _parse_numerical_answer(content)
            result.update(num_data)
        else:
            result["choix"] = {}
    except Exception:
        pass  # bloc illisible → résultat vide

    return result


# ---------------------------------------------------------------------------
# Parser principal : toutes les pages d'examen
# ---------------------------------------------------------------------------

def parse_exam_pages(pdf_images: list[np.ndarray],
                     exam_start_page: int = 4) -> list[dict]:
    """
    Parse les pages d'examen (index exam_start_page → fin).

    Args:
        pdf_images      : liste d'images BGR (issues de pdf_to_images)
        exam_start_page : index de la première page d'examen (0-indexé)

    Returns:
        Liste de dictionnaires, un par question.
    """
    all_questions = []
    q_num = 1

    for page_idx in range(exam_start_page, len(pdf_images)):
        page_img = normalize_page(pdf_images[page_idx])
        blocks   = detect_question_blocks(page_img)

        for (y_start, y_end) in blocks:
            block = page_img[y_start:y_end, :]
            if block.shape[0] < 150:
                continue  # bloc de pied de page (numéro, cryptogramme) → ignorer
            q_data = parse_question_block(block, q_num)
            all_questions.append(q_data)
            q_num += 1

    return all_questions


# ---------------------------------------------------------------------------
# Conversion vers le format xlsx
# ---------------------------------------------------------------------------

CHOICE_COLS = ["CHOIX A", "CHOIX B", "CHOIX C", "CHOIX D",
               "CHOIX E", "CHOIX F", "CHOIX G", "CHOIX H"]


def questions_to_exam_rows(questions: list[dict]) -> list[dict]:
    """
    Convertit la liste de questions parsées en liste de lignes
    compatibles avec l'onglet EXAM du xlsx.

    Format d'une ligne :
    { 'QUESTION': int, 'CHOIX A': 1|None, ..., 'MANTISSE': float|None,
      'EXPOSANT': int|None, 'UNITE': str|None }
    """
    rows = []
    for q in questions:
        row = {"QUESTION": q["question"]}
        for col in CHOICE_COLS:
            letter = col.split(" ")[1]  # 'A', 'B', ...
            row[col] = q.get("choix", {}).get(letter, None)
        row["MANTISSE"] = q.get("mantisse")
        row["EXPOSANT"] = q.get("exposant")
        row["UNITE"]    = q.get("unite")
        rows.append(row)
    return rows
