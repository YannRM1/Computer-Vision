"""
Décodage des grilles graphiques de la page 1.

Zones lues (formulaire normalisé 900×1270 px) :
  - STUDENT ID grid : 5 colonnes × 10 lignes
  - GROUP grid      : 3 colonnes × 10 lignes (2 chiffres + 1 lettre)
  - Conditions d'examen : 5 catégories YES/NO + champs Max number

Toutes les méthodes utilisent uniquement des opérations bas niveau :
filtrage, morphologie, seuillage, analyse de composantes connexes.
"""

import cv2
import numpy as np

from utils.checkbox_reader import (
    read_grid_one_per_col,
    read_grid_checked,
    preprocess_for_checkbox,
    ink_ratio,
    is_filled_square,
)

# ---------------------------------------------------------------------------
# Constantes de calibration – coordonnées dans le formulaire 900 × 1270 px
# Calibrées empiriquement sur FORM2_62445 (Student ID 62445, Group G02B)
# ---------------------------------------------------------------------------

# CODES EXAM (bande colorée : Module, Professeur, Date, Code)
ROI_CODES_EXAM    = (0, 65, 900, 65)

# Grille Student ID
# 5 colonnes (une par chiffre), 10 lignes (digits 0-9)
# Colonnes à x ≈ 733, 761, 790, 818, 847 ; lignes à y ≈ 251, 286, …, 554
ROI_STUDENT_ID    = (725, 247, 155, 330)
# Variante photo : la zone active détectée par get_active_area sur une photo
# fournit un mapping légèrement décalé par rapport au rendu PDF de référence.
# Ces coordonnées ont été calibrées en mesurant la position réelle de la
# grille sur des images normalisées issues de photos (FORM1/2/3).
ROI_STUDENT_ID_PHOTO = (692, 217, 131, 328)
STUDENT_ID_ROWS   = 10
STUDENT_ID_COLS   = 5

# Grille Group (10 lignes × 3 colonnes : chiffre1, chiffre2, lettre)
# Colonnes à x ≈ 524, 553, 609 ; mêmes lignes que Student ID
ROI_GROUP_GRID    = (516, 247, 105, 330)
ROI_GROUP_GRID_PHOTO = (455, 260, 135, 360)
GROUP_ROWS        = 10
# Proportions relatives des 3 colonnes dans la ROI (somme = 1)
GROUP_COL_WIDTHS  = [0.27, 0.27, 0.46]

# Case signature
ROI_SIGNATURE     = (30, 272, 372, 288)

# Cellules prénom manuscrit (grille de lettres individuelles)
# y=211 = ligne supérieure des cellules, h=24 = hauteur intérieure
ROI_FIRSTNAME     = (3, 211, 415, 24)

# Cellules nom manuscrit (après la ligne "NAME / NOM")
ROI_NAME          = (3, 270, 415, 24)

# Section CONDITIONS D'EXAMEN
# Cases YES/NO à y ≈ 784 (taille 26×26)
# Ordre : Lecture notes, Double-sided, Laptop, Calculator, Scratch paper
COND_Y_YESNO      = (784, 810)
COND_CHECKBOX_W   = 26

CONDITIONS = [
    # (x_YES, x_NO,  has_max, x_max0, x_max1, y_max0, y_max1)
    (101, 173, False, 0,   0,   0,   0  ),   # Lecture notes
    (269, 342, True,  286, 346, 816, 842),   # Double-sided sheets
    (438, 511, False, 0,   0,   0,   0  ),   # Laptop
    (608, 681, False, 0,   0,   0,   0  ),   # Calculator
    (778, 850, True,  812, 858, 816, 842),   # Scratch paper
]

# Cases Note maximale / Note pour valider
# Lues ensemble (zone englobante) pour améliorer la robustesse OCR
ROI_NOTES_COMBINED = (505, 900, 128, 108)   # contient les deux valeurs empilées
ROI_NOTE_MAX       = (510, 903, 118, 48)     # conservé pour fallback
ROI_NOTE_VALID     = (510, 951, 118, 48)     # conservé pour fallback

# Cryptogramme (petit graphique bas de page)
ROI_CRYPTO     = (180, 1228, 94, 42)


# ---------------------------------------------------------------------------
# Normalisation du formulaire (recadrage + redimensionnement)
# ---------------------------------------------------------------------------

FORM_W = 900
FORM_H = 1270


def get_active_area(img: np.ndarray,
                    is_photo: bool = False) -> tuple[int, int, int, int]:
    """
    Retourne (x0, y0, x1, y1) de la zone active du formulaire.

    - PDF  (is_photo=False) : cherche les pixels sombres (encre) sur fond blanc
      → THRESH_BINARY_INV, seuil fixe 200.
    - Photo (is_photo=True) : cherche le papier blanc sur fond sombre (bureau)
      → THRESH_BINARY, seuil fixe 200.
      Avec un fond sombre, THRESH_BINARY_INV marquerait aussi le bureau comme
      "contenu" et renverrait la bounding-box de toute l'image.
    """
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_photo:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    H, W = gray.shape
    row_has = np.any(binary > 0, axis=1)
    col_has = np.any(binary > 0, axis=0)
    r0 = int(np.argmax(row_has))
    r1 = H - int(np.argmax(row_has[::-1])) - 1
    c0 = int(np.argmax(col_has))
    c1 = W - int(np.argmax(col_has[::-1])) - 1
    return c0, r0, c1, r1


def _looks_like_photo(img: np.ndarray) -> bool:
    """Heuristique : photo si bords irréguliers / faible blanc périphérique.
    Permet d'enclencher le deskew automatiquement."""
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    # bord moyen (échantillon des 4 bandes périphériques)
    band = max(5, min(H, W) // 50)
    edges = np.concatenate([
        gray[:band, :].ravel(), gray[-band:, :].ravel(),
        gray[:, :band].ravel(), gray[:, -band:].ravel()
    ])
    return float(edges.mean()) < 230.0  # un scan PDF a des bords ~ blancs


def normalize_page(img: np.ndarray, is_photo: bool | None = None) -> np.ndarray:
    """
    Normalise un formulaire vers le repère (FORM_W, FORM_H).

    Étape 1 (nouvelle) : si on détecte les 4 L-brackets de coin, applique
    une correction perspective qui ramène l'image dans un rectangle
    légèrement plus grand que le formulaire utile (pour conserver
    compatibilité avec les ROIs calibrées). Sinon on continue avec
    l'image originale.

    Étape 2 : deskew (si photo) + crop bbox des pixels actifs + resize
    final vers (FORM_W, FORM_H). C'est cette étape qui produit le
    repère cohérent avec les coordonnées de ROI calibrées.
    """
    if is_photo is None:
        is_photo = _looks_like_photo(img)

    from utils.form_aligner import find_corner_brackets, deskew

    # Étape 1 : pour les photos uniquement, tenter une correction perspective
    # via les 4 L-brackets de coin (les PDFs sont déjà alignés et tout warp
    # perturberait la calibration des ROIs).
    if is_photo:
        # Seuil abaissé à 0.45 pour détecter les brackets même sur des photos
        # légèrement floues ou moins contrastées (était 0.55).
        brackets = find_corner_brackets(img, min_score=0.45)
        if brackets is not None:
            # Reproduit la géométrie d'un PDF : brackets aux mêmes ratios
            # de marge (~7.3% horizontal, ~4.3% vertical).
            inter_w, inter_h = 1655, 2340
            src = np.array([brackets["TL"], brackets["TR"],
                            brackets["BR"], brackets["BL"]], dtype=np.float32)
            bx, by = int(inter_w * 0.073), int(inter_h * 0.043)
            dst = np.array([[bx, by], [inter_w - bx, by],
                            [inter_w - bx, inter_h - by], [bx, inter_h - by]],
                           dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (inter_w, inter_h))
        else:
            img = deskew(img)

    # Étape 2 : crop bbox + resize vers le repère final
    # Pour les photos, chercher le papier blanc (is_photo=True) ;
    # pour les PDFs, chercher le contenu sombre sur fond blanc (is_photo=False).
    x0, y0, x1, y1 = get_active_area(img, is_photo=is_photo)
    H, W = img.shape[:2]
    mx = max(1, int(W * 0.005)); my = max(1, int(H * 0.005))
    x0 = min(W - 2, x0 + mx); y0 = min(H - 2, y0 + my)
    x1 = max(x0 + 1, x1 - mx); y1 = max(y0 + 1, y1 - my)
    return cv2.resize(img[y0:y1, x0:x1], (FORM_W, FORM_H))


# ---------------------------------------------------------------------------
# Extraction des ROIs
# ---------------------------------------------------------------------------

def get_roi(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = roi
    return img[y:y + h, x:x + w]


# ---------------------------------------------------------------------------
# Lecture du Student ID
# ---------------------------------------------------------------------------

def _score_grid_read(digits: list) -> float:
    """Score de confiance d'une lecture : pénalise les None et les valeurs 0/9
    (souvent du bruit de bord), favorise les colonnes avec une coche claire."""
    if digits is None:
        return -1.0
    score = 0.0
    for d in digits:
        if d is None:
            score -= 2.0
        elif d in (0, 9):
            score += 0.3   # valeurs fréquentes mais ambiguës
        else:
            score += 1.0
    return score


def read_student_id(form_img: np.ndarray, is_photo: bool = False) -> int | None:
    """
    Lit l'identifiant étudiant depuis la grille graphique.
    Retourne un entier (ex: 62445) ou None si lecture impossible.

    Pour les photos, un scan adaptatif teste plusieurs décalages horizontaux
    du ROI pour compenser les variations d'alignement après normalisation.
    """
    if not is_photo:
        roi = get_roi(form_img, ROI_STUDENT_ID)
        digits = read_grid_one_per_col(roi, rows=STUDENT_ID_ROWS, cols=STUDENT_ID_COLS)
        if None in digits:
            return None
        try:
            return int("".join(str(d) for d in digits))
        except Exception:
            return None

    # Pour les photos : tester plusieurs décalages horizontaux (-50 → +30 px)
    # et garder la lecture la plus confiante.
    x0, y0, w0, h0 = ROI_STUDENT_ID_PHOTO
    best_digits = None
    best_score  = -999.0
    for dx in range(-50, 31, 10):
        x = max(0, x0 + dx)
        if x + w0 > form_img.shape[1]:
            continue
        roi = form_img[y0:y0 + h0, x:x + w0]
        digits = read_grid_one_per_col(roi, rows=STUDENT_ID_ROWS, cols=STUDENT_ID_COLS)
        sc = _score_grid_read(digits)
        if sc > best_score:
            best_score  = sc
            best_digits = digits

    if best_digits is None or None in best_digits:
        return None
    try:
        return int("".join(str(d) for d in best_digits))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lecture du Groupe
# ---------------------------------------------------------------------------

def _split_group_cols(roi: np.ndarray) -> list[np.ndarray]:
    """
    Découpe la ROI group en 3 colonnes non uniformes
    (chiffre1, chiffre2, lettre) selon GROUP_COL_WIDTHS.
    """
    h, w = roi.shape[:2]
    cols = []
    x = 0
    for prop in GROUP_COL_WIDTHS:
        w_col = max(1, int(w * prop))
        cols.append(roi[:, x:x + w_col])
        x += w_col
    return cols


def read_group(form_img: np.ndarray) -> str | None:
    """
    Lit le code groupe depuis la grille graphique.
    Retourne une chaîne de type 'G02B' ou None.

    Structure de la grille (10 lignes × 3 colonnes) :
      col 0 → 1er chiffre du numéro de groupe (0-9)
      col 1 → 2ème chiffre du numéro de groupe (0-9)
      col 2 → lettre du groupe (A=0, B=1, ..., J=9)
    """
    roi = get_roi(form_img, ROI_GROUP_GRID)
    cols = _split_group_cols(roi)

    results = []
    for c_img in cols:
        best_row, best_ratio = None, 0.05
        row_h = c_img.shape[0] // GROUP_ROWS
        for r in range(GROUP_ROWS):
            cell = c_img[r * row_h:(r + 1) * row_h, :]
            binary = preprocess_for_checkbox(cell)
            margin = max(1, int(min(binary.shape) * 0.1))
            inner = binary[margin:-margin, margin:-margin]
            ratio = ink_ratio(inner)
            if ratio > best_ratio:
                best_ratio = ratio
                best_row = r
        results.append(best_row)

    if None in results:
        return None

    digit1 = str(results[0])
    digit2 = str(results[1])
    letter  = chr(ord('A') + results[2])
    return f"G{digit1}{digit2}{letter}"


# ---------------------------------------------------------------------------
# Lecture des conditions d'examen
# ---------------------------------------------------------------------------

def _read_condition(form_img: np.ndarray, cond: tuple) -> int:
    """
    Lit une condition d'examen.
    Retourne :
      0           si NO est coché
      1           si YES est coché sans champ 'Max number'
      max_number  si YES est coché avec champ 'Max number' (≥ 1)
    """
    x_yes, x_no, has_max, x_max0, x_max1, y_max0, y_max1 = cond
    y0, y1 = COND_Y_YESNO
    h, w = y1 - y0, COND_CHECKBOX_W

    roi_yes = form_img[y0:y1, x_yes:x_yes + w]
    roi_no  = form_img[y0:y1, x_no:x_no + w]

    # Seuil abaissé : 0.15 (était 0.25) — les cases YES/NO sont parfois
    # cochées légèrement (peu d'encre) et passaient en faux négatif.
    yes_filled = is_filled_square(roi_yes, threshold=0.15)
    no_filled  = is_filled_square(roi_no,  threshold=0.15)

    # Si les deux sont remplis (artefact), choisir le plus sombre
    if yes_filled and no_filled:
        y_ratio = ink_ratio(preprocess_for_checkbox(roi_yes))
        n_ratio = ink_ratio(preprocess_for_checkbox(roi_no))
        yes_filled = y_ratio >= n_ratio
        no_filled  = not yes_filled

    if not yes_filled:
        return 0

    # YES coché
    if not has_max:
        return 1

    # Lire la valeur max number (2 chiffres imprimés dans des cases)
    max_roi = form_img[y_max0:y_max1, x_max0:x_max1]
    return _read_two_digit_box(max_roi)


def _read_two_digit_box(roi: np.ndarray) -> int:
    """
    Lit un entier sur 1-2 chiffres dans une case imprimée ('| 0 | 1 |' → 1).
    Délègue à ocr_number qui filtre les bordures du cadre.
    """
    if roi is None or roi.size == 0:
        return 1
    try:
        from utils.ocr_utils import ocr_number
        val = ocr_number(roi)
        return val if val is not None else 1
    except Exception:
        return 1


def read_conditions(form_img: np.ndarray) -> dict:
    """
    Retourne un dictionnaire avec les 5 conditions d'examen :
      {
        'notes_cours': int,
        'notes_manuscrites': int,
        'ordinateur': int,
        'calculatrice': int,
        'brouillon': int,
      }
    """
    keys = ['notes_cours', 'notes_manuscrites', 'ordinateur',
            'calculatrice', 'brouillon']
    values = [_read_condition(form_img, c) for c in CONDITIONS]
    return dict(zip(keys, values))


# ---------------------------------------------------------------------------
# Lecture de la signature (extraction de la sous-image)
# ---------------------------------------------------------------------------

def extract_signature_roi(form_img: np.ndarray) -> np.ndarray:
    """
    Extrait la sous-image **intérieure** de la boîte de signature.

    Le ROI nominal (ROI_SIGNATURE) inclut le cadre rectangulaire et le label
    "SIGNATURE". Cette fonction :
      1. Découpe le ROI nominal.
      2. Détecte le cadre rectangulaire par morphologie (filtres horizontaux
         et verticaux pour isoler les longues lignes du cadre).
      3. Retourne l'intérieur du cadre avec une petite marge négative pour
         garantir qu'aucun pixel du trait du cadre ne soit conservé.

    Si la détection échoue, repli sur un crop fixe correspondant à la position
    typique du rectangle dans le ROI nominal.
    """
    roi = get_roi(form_img, ROI_SIGNATURE)
    if roi is None or roi.size == 0:
        return roi

    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Binarisation Otsu inverse : 255 = encre (cadre + signature)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Isoler les longues lignes du cadre par morphologie
    # (kernels horizontal et vertical proportionnels à la taille du ROI)
    kh = max(15, W // 8)
    kv = max(15, H // 8)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kh, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kv))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    vert  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    frame = cv2.bitwise_or(horiz, vert)

    # Bounding box du cadre détecté
    coords = cv2.findNonZero(frame)
    if coords is not None and len(coords) > 50:
        x, y, w, h = cv2.boundingRect(coords)
        # Marge intérieure (proportionnelle) pour éliminer le trait du cadre
        m = max(4, int(min(w, h) * 0.08))
        x0 = max(0, x + m)
        y0 = max(0, y + m)
        x1 = min(W, x + w - m)
        y1 = min(H, y + h - m)
        if (x1 - x0) > 20 and (y1 - y0) > 20:
            return roi[y0:y1, x0:x1]

    # Repli : crop fixe interne approximatif
    return roi[60:260, 50:340]


# ---------------------------------------------------------------------------
# Lecture de la zone Note (OCR de chiffres imprimés)
# ---------------------------------------------------------------------------

def read_note_box(form_img: np.ndarray, roi: tuple,
                  top_crop_frac: float = 0.15):
    x, y, w, h = roi
    crop_y = y + int(h * top_crop_frac)
    region = form_img[crop_y:y + h, x:x + w]
    try:
        from utils.ocr_utils import ocr_number
        return ocr_number(region)
    except Exception:
        return None


def _read_both_notes(form_img: np.ndarray):
    import re as _re
    x, y, w, h = ROI_NOTES_COMBINED
    roi = form_img[y:y + h, x:x + w]
    try:
        from utils.ocr_utils import _to_gray, _ocr_raw
        gray = _to_gray(roi)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        eq = clahe.apply(gray)
        big = cv2.resize(eq, (eq.shape[1] * 6, eq.shape[0] * 6),
                         interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(big, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = _ocr_raw(thresh, allowlist="0123456789")
        nums = [int(n) for n in _re.findall(r"\d+", text)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        if len(nums) == 1:
            return nums[0], None
    except Exception:
        pass
    return None, None


def read_note_maximale(form_img: np.ndarray):
    nm, _ = _read_both_notes(form_img)
    if nm is None:
        nm = read_note_box(form_img, ROI_NOTE_MAX)
    return nm


def read_note_pour_valider(form_img: np.ndarray):
    _, nv = _read_both_notes(form_img)
    if nv is None:
        nv = read_note_box(form_img, ROI_NOTE_VALID)
    return nv


# ---------------------------------------------------------------------------
# Cryptogramme
# ---------------------------------------------------------------------------

def extract_cryptogram(form_img: np.ndarray) -> np.ndarray:
    return get_roi(form_img, ROI_CRYPTO)

