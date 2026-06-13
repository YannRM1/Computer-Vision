"""
Lecture automatique des pages d'examen (page 5 -> fin).

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

# Paramètres de réglage centralisés dans utils/config.py (§8 : éviter les
# valeurs codées en dur dispersées). Ajuster les valeurs là-bas, pas ici.
from utils.config import (
    MCQ_FILLED_CANCEL,
    HEADER_BAND_H, MIN_LINE_WIDTH, LINE_MERGE_TOL,
    MCQ_X_START, MCQ_X_END, MCQ_MIN_SZ, MCQ_MAX_SZ, MCQ_MIN_AREA,
    CHECKED_INK_THRESHOLD, MCQ_CHOICES,
    MANTISSE_X_FRAC, MANTISSE_Y_FRAC,
    EXPOSANT_X_FRAC, EXPOSANT_Y_FRAC,
    UNITE_X_FRAC, UNITE_Y_FRAC,
)


# ---------------------------------------------------------------------------
# Détection des blocs de questions
# ---------------------------------------------------------------------------

def _find_horizontal_lines(page_bin: np.ndarray,
                            min_width: int = MIN_LINE_WIDTH) -> list[int]:
    """
    Retourne les y-positions des lignes horizontales longues.
    Méthode : ouverture morphologique horizontale (bas niveau).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width, 1))
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


def _has_giant_gap(ys: list[int], page_h: int, giant: int = 750) -> bool:
    """True si l'espacement entre séparateurs successifs (bords de page
    inclus) dépasse `giant` px : aucune question légitime n'est aussi haute,
    c'est le signe de séparateurs non détectés."""
    pts = [HEADER_BAND_H] + sorted(ys) + [page_h]
    return any(b - a > giant for a, b in zip(pts, pts[1:]))


def detect_question_blocks(exam_page: np.ndarray) -> list[tuple[int, int]]:
    """
    Retourne la liste des (y_start, y_end) pour chaque bloc de question.
    Le premier bloc commence après la bande d'en-tête de page.
    """
    gray = cv2.cvtColor(exam_page, cv2.COLOR_BGR2GRAY) \
           if len(exam_page.shape) == 3 else exam_page
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    ys = _find_horizontal_lines(binary)

    # Repli pour les scans aux séparateurs pointillés / pâlis : si la page
    # produit un bloc géant (fusion manifeste de plusieurs questions), on
    # recommence avec un seuil plus permissif et un pontage horizontal des
    # traits. Ce mode n'est PAS utilisé par défaut : sur les pages saines, le
    # pontage coupe des questions en deux via leurs lignes internes (tableaux).
    if _has_giant_gap(ys, exam_page.shape[0]):
        _, b2 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        b2 = cv2.morphologyEx(b2, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
        ys2 = _find_horizontal_lines(b2)
        if len(ys2) > len(ys):
            ys = ys2

    ys = _merge_close_lines(ys)

    H = exam_page.shape[0]
    boundaries = [y for y in ys if HEADER_BAND_H <= y <= H - 30]
    if not boundaries:
        return [(HEADER_BAND_H, H)]

    blocks = []
    prev = HEADER_BAND_H
    for y in boundaries:
        if y - prev > 50:
            blocks.append((prev, y))
        prev = y
    if H - prev > 50:
        blocks.append((prev, H))

    # Supprimer les blocs parasites : toute fausse ligne horizontale détectée
    # à l'intérieur d'une question produit un bloc anormalement court.
    # On filtre les blocs dont la hauteur est < 35 % de la médiane des autres.
    if len(blocks) > 2:
        heights = sorted([y1 - y0 for y0, y1 in blocks])
        median_h = heights[len(heights) // 2]
        min_h = max(50, int(median_h * 0.35))
        blocks = [(y0, y1) for y0, y1 in blocks if (y1 - y0) >= min_h]

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
            # Ratio W/H proche de 1 -> case carrée
            if 0.5 < w / (h + 1e-6) < 2.0:
                boxes.append((x + MCQ_X_START, y, w, h))

    boxes = _keep_aligned_column(boxes)
    return sorted(boxes, key=lambda b: b[1])  # trier par y


def _keep_aligned_column(boxes: list[tuple[int, int, int, int]],
                         x_tol: int = 10
                         ) -> list[tuple[int, int, int, int]]:
    """
    Ne conserve que la plus grande COLONNE de cases verticalement alignées
    (même bord gauche, à x_tol près).

    Les vraies cases à cocher MCQ sont empilées en colonne dans la marge gauche,
    toutes au même x. Des fragments de l'énoncé (lettres, morceaux d'équation)
    tombent parfois dans la bande de recherche et étaient comptés comme des
    cases, décalant l'assignation des lettres (A,B,…). En gardant le groupe le
    plus nombreux partageant le même x, on élimine ces intrus.
    """
    if len(boxes) <= 1:
        return boxes
    best: list = []
    for ref in boxes:
        grp = [b for b in boxes if abs(b[0] - ref[0]) <= x_tol]
        if len(grp) > len(best):
            best = grp
    return best


def _box_ink_ratio(block_img: np.ndarray, x: int, y: int,
                   w: int, h: int) -> tuple[float, bool]:
    """Retourne (ratio d'encre intérieur, présence d'un motif X) pour une case."""
    roi = block_img[y:y + h, x:x + w]
    binary = preprocess_for_checkbox(roi)
    margin = max(2, int(min(binary.shape) * 0.12))
    if binary.shape[0] > 2 * margin and binary.shape[1] > 2 * margin:
        inner = binary[margin:-margin, margin:-margin]
    else:
        inner = binary
    return ink_ratio(inner), has_x_pattern(inner)


def _parse_mcq_choices(block_img: np.ndarray) -> dict[str, int]:
    """
    Détecte les choix MCQ cochés dans le bloc.

    Critère RELATIF (par question) plutôt qu'un seuil absolu : une case vide ne
    contient que le bord du carré (ratio d'encre ~0.13), une case cochée ressort
    nettement au-dessus de cette ligne de base — que la marque soit un grand X
    (~0.5) ou une simple coche légère (~0.29). Comparer chaque case aux autres
    cases de LA MÊME question rend la décision robuste à l'épaisseur du trait et
    au niveau de gris du scan (là où un seuil fixe ratait les coches légères).

    Retourne {'A': 1/None, 'B': 1/None, ...} pour les choix présents.
    """
    boxes = _find_mcq_checkboxes(block_img)
    result = {}
    feats = []   # (ratio, has_x) par case, dans l'ordre des lettres
    for idx, (x, y, w, h) in enumerate(boxes):
        if idx >= len(MCQ_CHOICES):
            break
        feats.append(_box_ink_ratio(block_img, x, y, w, h))

    if not feats:
        return result

    ratios = [f[0] for f in feats]
    # Ligne de base = médiane des cases (la plupart sont vides).
    baseline = float(np.median(ratios))

    # Convention imprimee sur le formulaire : une case entierement noircie
    # (ratio >= MCQ_FILLED_CANCEL, contre ~0.3-0.55 pour une croix) est un
    # choix corrige -> interpretee non cochee, conformement aux instructions.
    for idx, (ratio, is_x) in enumerate(feats):
        letter = MCQ_CHOICES[idx]
        if ratio >= MCQ_FILLED_CANCEL:
            result[letter] = None
            continue
        strong   = is_x or ratio > 0.32
        relative = (ratio > baseline + 0.07 and ratio > 0.16)
        x_light  = is_x and ratio > baseline + 0.03
        result[letter] = 1 if (strong or relative or x_light) else None
    return result


# ---------------------------------------------------------------------------
# Détection des réponses numériques
# ---------------------------------------------------------------------------

def _has_numerical_answer(block_img: np.ndarray) -> bool:
    """
    Détecte si le bloc contient une zone de réponse numérique
    (structure 'mantisse x 10^exposant').

    Critères robustes (ordre d'évaluation) :
      1. Grand rectangle dans le tiers inférieur du bloc (case mantisse).
         Seuil abaissé + binarisation Otsu pour les blocs clairs.
      2. Si pas de rectangle mais aucune checkbox MCQ -> probablement numérique.
    """
    h, w = block_img.shape[:2]
    gray = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY) \
           if len(block_img.shape) == 3 else block_img

    # Binarisation Otsu (plus robuste que seuil fixe 200)
    _, binary_otsu = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Binarisation fixe (fallback pour scans clairs où Otsu sur-seuille)
    _, binary_fixed = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    for binary in (binary_fixed, binary_otsu):
        # Chercher dans le tiers inférieur du bloc (pas seulement la moitié)
        lower = binary[h * 2 // 3:, :]
        num, _, stats, _ = cv2.connectedComponentsWithStats(lower,
                                                             connectivity=8)
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            # Critères assouplis : petite case exposant aussi acceptable
            if bw > w * 0.08 and bh > h * 0.05 and area > 150:
                return True

    # Fallback : si pas de vraie rangée de checkboxes MCQ (>= 2), supposer numérique
    boxes = _find_mcq_checkboxes(block_img)
    return len(boxes) < 2


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


def numeric_answer_crops(block_img: np.ndarray):
    """
    Localise les zones de réponse numérique d'un bloc et renvoie les crops
    (mantisse_img, exposant_img, unite_img) via détection de cadres bordurés,
    avec repli sur fractions fixes. Partagée entre la lecture
    (_parse_numerical_answer) et la construction du jeu de chiffres annotés.
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

    # Fallback : fractions fixes calibrées sur la structure du formulaire
    # La zone de réponse numérique est toujours dans le tiers inférieur du bloc.
    if mantisse_img is None:
        mantisse_img = block_img[int(h*0.72):int(h*0.95), int(w*0.02):int(w*0.25)]
    if exposant_img is None:
        exposant_img = block_img[int(h*0.65):int(h*0.85), int(w*0.25):int(w*0.38)]
    if unite_img is None:
        unite_img = block_img[int(h*0.72):int(h*0.95), int(w*0.42):int(w*0.68)]
    return mantisse_img, exposant_img, unite_img


def _parse_numerical_answer(block_img: np.ndarray) -> dict:
    """
    Extrait mantisse, exposant et unité via détection de cadres bordurés.
    Fallback sur fractions fixes si la détection échoue.
    """
    mantisse_img, exposant_img, unite_img = numeric_answer_crops(block_img)
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
        # Un vrai bloc MCQ comporte au moins 2 cases de choix alignées ; une
        # seule « case » détectée est presque toujours un chiffre manuscrit de
        # la mantisse qui déborde dans la bande gauche (bloc numérique).
        if len(boxes) >= 2:
            result["choix"] = _parse_mcq_choices(content)
        elif _has_numerical_answer(content):
            num_data = _parse_numerical_answer(content)
            result.update(num_data)
        else:
            result["choix"] = {}
    except Exception:
        pass  # bloc illisible -> résultat vide

    return result


# ---------------------------------------------------------------------------
# Parser principal : toutes les pages d'examen
# ---------------------------------------------------------------------------

def parse_exam_pages(pdf_images: list[np.ndarray],
                     exam_start_page: int = 4) -> list[dict]:
    """
    Parse les pages d'examen (index exam_start_page -> fin).

    Args:
        pdf_images      : liste d'images BGR (issues de pdf_to_images)
        exam_start_page : index de la première page d'examen (0-indexé)

    Returns:
        Liste de dictionnaires, un par question.
    """
    all_questions = []
    for q_num, block in enumerate(iter_question_blocks(pdf_images,
                                                       exam_start_page), 1):
        all_questions.append(parse_question_block(block, q_num))
    return all_questions


def read_page_number(page_img: np.ndarray):
    """
    Lit le numero de page imprime (bas-centre du repere canonique).

    Texte imprime -> OCR autorise (§4.1). Bas niveau pour isoler le chiffre :
    Otsu + composantes connexes -> plus grosse composante de taille « chiffre »
    dans la bande de pied de page, recadree et agrandie avant OCR.
    Renvoie l'entier lu ou None.
    """
    from utils.ocr_utils import _get_reader
    strip = page_img[1205:1270, 340:560]
    if strip.size == 0:
        return None
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    binv = cv2.threshold(g, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if 10 < h < 55 and 4 < w < 45 and a > 30 \
                and (best is None or a > best[0]):
            best = (a, x, y, w, h)
    if best is None:
        return None
    _, x, y, w, h = best
    d = binv[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
    d = cv2.copyMakeBorder(cv2.bitwise_not(d), 12, 12, 12, 12,
                           cv2.BORDER_CONSTANT, value=255)
    d = cv2.resize(d, (d.shape[1] * 4, d.shape[0] * 4),
                   interpolation=cv2.INTER_CUBIC)
    for t in _get_reader().readtext(d, detail=0, allowlist="0123456789"):
        s = "".join(c for c in str(t) if c.isdigit())
        if s:
            return int(s[0])
    return None


def iter_question_blocks(pdf_images: list[np.ndarray],
                         exam_start_page: int = 4):
    """
    Itère sur les blocs de question (images), dans l'ordre des questions,
    en appliquant les mêmes filtres anti-pied-de-page que la lecture.
    Partagé entre parse_exam_pages et la construction du jeu de chiffres
    annotés (l'appariement question <-> vérité terrain exige le même ordre).
    """
    # Réordonner les pages d'examen selon le numero imprime : un scan
    # recto-verso peut les inverser par paires, ce qui placerait les reponses
    # dans le desordre. On ne reordonne que si TOUS les numeros sont lus et
    # forment une permutation strictement desordonnee (sinon on garde l'ordre
    # du fichier : conservateur, jamais de degradation).
    pages = [normalize_page(pdf_images[i])
             for i in range(exam_start_page, len(pdf_images))]
    nums = [read_page_number(p) for p in pages]
    if (len(pages) > 1 and all(v is not None for v in nums)
            and len(set(nums)) == len(nums) and nums != sorted(nums)):
        pages = [p for _, p in sorted(zip(nums, pages), key=lambda t: t[0])]

    for page_img in pages:
        blocks   = detect_question_blocks(page_img)

        page_h = page_img.shape[0]
        for (y_start, y_end) in blocks:
            block = page_img[y_start:y_end, :]
            if block.shape[0] < 150:
                continue  # bloc de pied de page (numéro, cryptogramme) -> ignorer
            if y_start > page_h - 180:
                continue  # bande de pied de page, même si le bloc dépasse 150 px
            # Bloc touchant le bas de page sans en-tete « QUESTION N » juste
            # sous sa ligne superieure = pied de page (numero + cryptogramme +
            # equerres de coin), pas une question : le compter decalerait la
            # numerotation. La bande d'en-tete est un discriminant plus sur que
            # l'encre totale, contaminee par la ligne separatrice et le
            # mobilier de pied de page (mesure : questions hdr>=0.050, pieds
            # <=0.022).
            if y_end >= page_h - 5:
                hdr = block[12:72]
                gh = (cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)
                      if hdr.ndim == 3 else hdr)
                hdr_ink = float((cv2.threshold(gh, 200, 255,
                                 cv2.THRESH_BINARY_INV)[1] > 0).mean()) \
                          if hdr.size else 0.0
                if hdr_ink < 0.035:
                    continue
            yield block


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
