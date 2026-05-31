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
# Depuis le recalage par template (cf. normalize_page), les photos sont
# redressées dans le MÊME repère canonique que les PDFs. On utilise donc les
# mêmes coordonnées de ROI pour les photos et pour les PDFs.
ROI_STUDENT_ID_PHOTO = ROI_STUDENT_ID
STUDENT_ID_ROWS   = 10
STUDENT_ID_COLS   = 5

# Grille Group (10 lignes × 3 colonnes : chiffre1, chiffre2, lettre)
# Colonnes à x ≈ 524, 553, 609 ; mêmes lignes que Student ID
ROI_GROUP_GRID    = (516, 247, 105, 330)
ROI_GROUP_GRID_PHOTO = ROI_GROUP_GRID
GROUP_ROWS        = 10
# Proportions relatives des 3 colonnes dans la ROI (somme = 1)
GROUP_COL_WIDTHS  = [0.27, 0.27, 0.46]

# Case signature
ROI_SIGNATURE     = (30, 272, 372, 288)
# Intérieur strict de la boîte de signature dans le repère canonique
# (exclut le label "SIGNATURE" au-dessus et le trait du cadre). Mesuré sur le
# template de référence. Le recalage rendant le repère stable, ce crop fixe est
# fiable pour les photos comme pour les PDFs.
ROI_SIGNATURE_INNER = (120, 358, 264, 150)

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

# Template de recalage pour les photos (objet FormTemplate), positionné par le
# pipeline via set_photo_template(). Tant qu'il est None, on retombe sur
# l'ancien chemin (deskew + bounding box).
_PHOTO_TEMPLATE = None


def set_photo_template(template) -> None:
    """
    Enregistre le template de recalage des photos (objet
    utils.template_register.FormTemplate, ou None pour désactiver).
    """
    global _PHOTO_TEMPLATE
    _PHOTO_TEMPLATE = template


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


def normalize_page(img: np.ndarray, is_photo: bool | None = None,
                   use_template: bool = False) -> np.ndarray:
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

    from utils.form_aligner import deskew

    # Étape 1 : recaler sur le template de référence par homographie (ORB).
    # On le fait pour les photos (toujours) et pour les PDFs de page 1 quand
    # use_template=True. En cas de succès, l'image est déjà dans le repère
    # canonique (FORM_W × FORM_H) -> on renvoie directement et toutes les ROIs
    # s'appliquent. Les pages d'examen (use_template=False) ne sont pas
    # concernées : elles ne matcheraient pas le template de page 1.
    want_template = _PHOTO_TEMPLATE is not None and (is_photo or use_template)
    if want_template:
        from utils.template_register import register_to_template
        reg = register_to_template(img, _PHOTO_TEMPLATE)
        if reg is not None:
            return reg
        # Repli si le recalage échoue.
        if is_photo:
            img = deskew(img)
    elif is_photo:
        # Pas de template fourni : ancien comportement (deskew + bbox).
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

def read_student_id(form_img: np.ndarray, is_photo: bool = False) -> int | None:
    """
    Lit l'identifiant étudiant depuis la grille graphique.
    Retourne un entier (ex: 62445) ou None si lecture impossible.
    Le ROI utilisé dépend de la source (PDF vs photo).
    """
    roi_coords = ROI_STUDENT_ID_PHOTO if is_photo else ROI_STUDENT_ID
    roi = get_roi(form_img, roi_coords)
    digits = read_grid_one_per_col(roi, rows=STUDENT_ID_ROWS, cols=STUDENT_ID_COLS)
    if None in digits:
        return None
    try:
        return int("".join(str(d) for d in digits))
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
    Extrait la sous-image intérieure de la boîte de signature.

    Le repère étant stabilisé par le recalage (photos) ou déjà aligné (PDFs),
    on découpe directement l'intérieur calibré de la boîte (ROI_SIGNATURE_INNER),
    ce qui exclut le label "SIGNATURE" et le trait du cadre — sources de bruit
    qui faisaient échouer l'identification auparavant.
    """
    inner = get_roi(form_img, ROI_SIGNATURE_INNER)
    if inner is None or inner.size == 0:
        # Repli : intérieur approximatif de l'ancien ROI nominal.
        return get_roi(form_img, ROI_SIGNATURE)
    return inner
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

