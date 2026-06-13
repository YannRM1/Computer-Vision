"""
Décodage des grilles graphiques de la page 1.

Zones lues (formulaire normalisé 900x1270 px) :
  - STUDENT ID grid : 5 colonnes x 10 lignes
  - GROUP grid      : 3 colonnes x 10 lignes (2 chiffres + 1 lettre)
  - Conditions d'examen : 5 catégories YES/NO + champs Max number

Toutes les méthodes utilisent uniquement des opérations bas niveau :
filtrage, morphologie, seuillage, analyse de composantes connexes.
"""

import cv2
import numpy as np

from utils.checkbox_reader import (
    read_grid_one_per_col,
    preprocess_for_checkbox,
    ink_ratio,
    is_filled_square,
)

# ---------------------------------------------------------------------------
# Constantes de calibration – coordonnées dans le repère canonique 900 x 1270 px
# (mesurées sur le gabarit de référence du formulaire).
# ---------------------------------------------------------------------------

# CODES EXAM (bande colorée : Module, Professeur, Date, Code)
ROI_CODES_EXAM    = (0, 65, 900, 65)

# Grille Student ID
# 5 colonnes (une par chiffre), 10 lignes (digits 0-9)
# Colonnes à x ~ 733, 761, 790, 818, 847 ; lignes à y ~ 251, 286, …, 554
ROI_STUDENT_ID    = (725, 247, 155, 330)
# Depuis le recalage par template (cf. normalize_page), les photos sont
# redressées dans le MÊME repère canonique que les PDFs. On utilise donc les
# mêmes coordonnées de ROI pour les photos et pour les PDFs.
ROI_STUDENT_ID_PHOTO = ROI_STUDENT_ID
STUDENT_ID_ROWS   = 10
STUDENT_ID_COLS   = 5

# Grille Group (10 lignes x 3 colonnes : chiffre1, chiffre2, lettre).
# Centres des colonnes de cases mesures sur le gabarit : x ~ 540, 569 et 625.
ROI_GROUP_GRID    = (526, 247, 58, 330)    # les 2 colonnes chiffres (serrees)
ROI_GROUP_GRID_PHOTO = ROI_GROUP_GRID
GROUP_ROWS        = 10
# Proportions relatives des 2 colonnes chiffres dans ROI_GROUP_GRID
GROUP_COL_WIDTHS  = [0.5, 0.5, 0.0]
# ROI separee pour la colonne lettre (cases uniquement, labels exclus).
ROI_GROUP_LETTER  = (610, 247, 32, 330)

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
# Cases YES/NO à y ~ 784 (taille 26x26)
# Ordre : Lecture notes, Double-sided, Laptop, Calculator, Scratch paper
COND_Y_YESNO      = (784, 810)
COND_CHECKBOX_W   = 26

CONDITIONS = [
    # (x_YES, x_NO,  has_max, x_max0, x_max1, y_max0, y_max1)
    # Boites "Max number" mesurees sur les bordures verticales du gabarit
    # (positions identiques sur les 3 formulaires) : dizaines puis unites.
    (101, 173, False, 0,   0,   0,   0  ),   # Lecture notes
    (269, 342, True,  325, 370, 817, 843),   # Double-sided sheets
    (438, 511, False, 0,   0,   0,   0  ),   # Laptop
    (608, 681, False, 0,   0,   0,   0  ),   # Calculator
    (778, 850, True,  833, 878, 817, 843),   # Scratch paper
]

# Cases Note maximale / Note pour valider
# Lues ensemble (zone englobante) pour améliorer la robustesse OCR
ROI_NOTES_COMBINED = (505, 900, 128, 108)   # contient les deux valeurs empilées
ROI_NOTE_MAX       = (510, 903, 118, 48)     # conservé pour fallback
ROI_NOTE_VALID     = (510, 951, 118, 48)     # conservé pour fallback

# Cryptogramme (petit graphique bas de page). Le glyphe occupe x~[234,274],
# y~[1239,1269] dans le repere canonique ; marge incluse de chaque cote.
ROI_CRYPTO     = (180, 1225, 130, 45)


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


def get_active_template():
    """Template de recalage actuellement actif (ou None)."""
    return _PHOTO_TEMPLATE


def get_active_area(img: np.ndarray,
                    is_photo: bool = False) -> tuple[int, int, int, int]:
    """
    Retourne (x0, y0, x1, y1) de la zone active du formulaire.

    - PDF  (is_photo=False) : cherche les pixels sombres (encre) sur fond blanc
      -> THRESH_BINARY_INV, seuil fixe 200.
    - Photo (is_photo=True) : cherche le papier blanc sur fond sombre (bureau)
      -> THRESH_BINARY, seuil fixe 200.
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

    Étape 1 : si on détecte les 4 L-brackets de coin, applique une correction
    perspective qui ramène l'image dans un rectangle légèrement plus grand que
    le formulaire utile (pour rester compatible avec les ROIs calibrées). Sinon
    on continue avec l'image originale.

    Étape 2 : deskew (si photo) + crop bbox des pixels actifs + resize
    final vers (FORM_W, FORM_H). C'est cette étape qui produit le
    repère cohérent avec les coordonnées de ROI calibrées.
    """
    if is_photo is None:
        is_photo = _looks_like_photo(img)

    # Photo anormalement haute = plusieurs pages photographiees d'un coup
    # (ex. FORM3_62766, ratio h/w ~6.7 au lieu de ~1.4 pour une page). On ne
    # garde que la page du haut (la page 1 d'identification) avant recalage.
    if is_photo:
        h0, w0 = img.shape[:2]
        if h0 > 2.0 * w0:
            img = img[:int(round(w0 * 1.45)), :]

    from utils.form_aligner import deskew

    # Étape 1 : recaler sur le template de référence par homographie (ORB).
    # On le fait pour les photos (toujours) et pour les PDFs de page 1 quand
    # use_template=True. En cas de succès, l'image est déjà dans le repère
    # canonique (FORM_W x FORM_H) -> on renvoie directement et toutes les ROIs
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
        # Pas de template fourni : deskew + bbox.
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

def _read_id_by_boxes(form_img: np.ndarray, expand: int = 16) -> int | None:
    """
    Lit la grille Student ID en LOCALISANT d'abord chacune des 50 cases
    (composantes connexes carrees de ~20 px), puis en mesurant l'encre de
    chaque interieur. Insensible aux derives de quelques pixels du recalage
    (quadrillage fixe coupant les cases en deux) et aux variations de
    contraste (seuillage adaptatif, pas d'Otsu par cellule).
    Renvoie None si la structure 5 x 10 n'est pas retrouvee ou si une
    colonne ne contient aucune coche nette.
    """
    x, y, w, h = ROI_STUDENT_ID
    g = form_img if form_img.ndim == 2 else cv2.cvtColor(form_img, cv2.COLOR_BGR2GRAY)
    H_img, W_img = g.shape
    x0, y0 = max(0, x - expand), max(0, y - expand)
    sub = g[y0:min(H_img, y + h + expand), x0:min(W_img, x + w + expand)]
    binary = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if 14 <= bw <= 34 and 14 <= bh <= 34 and 0.6 < bw / bh < 1.6 and area >= 30:
            boxes.append((cent[i][0], cent[i][1], bx, by, bw, bh))
    if len(boxes) < 35:
        return None

    def clusters(vals, gap=12):
        out, cur = [], [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] > gap:
                out.append(float(np.mean(cur)))
                cur = []
            cur.append(v)
        out.append(float(np.mean(cur)))
        return out

    cx = clusters(sorted(p[0] for p in boxes))
    cy = clusters(sorted(p[1] for p in boxes))
    if len(cx) != STUDENT_ID_COLS or len(cy) != STUDENT_ID_ROWS:
        return None
    digits = []
    for c in range(STUDENT_ID_COLS):
        vals = []
        for r in range(STUDENT_ID_ROWS):
            cand = min(boxes, key=lambda p: (p[0] - cx[c]) ** 2 + (p[1] - cy[r]) ** 2)
            bx, by, bw, bh = cand[2], cand[3], cand[4], cand[5]
            mx, my = max(3, bw // 5), max(3, bh // 5)
            inner = binary[by + my:by + bh - my, bx + mx:bx + bw - mx]
            vals.append(float((inner > 0).mean()) if inner.size else 0.0)
        arr = np.array(vals)
        best = int(arr.argmax())
        # coche nette exigee : assez d'encre ET nettement au-dessus du fond
        if arr[best] < 0.05 or arr[best] < 2.0 * max(float(np.median(arr)), 0.01):
            return -1        # structure trouvee mais colonne sans coche -> vide
        digits.append(str(best))
    return int("".join(digits))


def read_student_id(form_img: np.ndarray, is_photo: bool = False) -> int | None:
    """
    Lit l'identifiant étudiant depuis la grille graphique.
    Retourne un entier (ex: 62445) ou None si lecture impossible.

    Methode principale : localisation des cases par composantes connexes
    (_read_id_by_boxes). Repli : quadrillage fixe du ROI calibre.
    """
    by_boxes = _read_id_by_boxes(form_img)
    if by_boxes == -1:
        return None          # grille localisee et vide : pas de repli
    if by_boxes is not None:
        return by_boxes
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


def _read_grid_col_best_row(col_img: np.ndarray, rows: int) -> int | None:
    """
    Trouve la ligne cochée dans une colonne de grille via binarisation globale
    et maximum de ratio d'encre par ligne.
    Binarisation globale (vs. par cellule) = seuil cohérent sur toute la colonne.
    """
    gray = col_img if len(col_img.shape) == 2 \
           else cv2.cvtColor(col_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_h = binary.shape[0] // rows
    ratios = []
    for r in range(rows):
        cell = binary[r * row_h:(r + 1) * row_h, :]
        margin = 2
        inner = cell[margin:-margin, margin:-margin] \
                if cell.shape[0] > 2 * margin else cell
        ratios.append(float(np.mean(inner) / 255.0))
    if not ratios:
        return None
    best = int(np.argmax(ratios))
    # N'accepter le résultat que si le pic est clairement au-dessus du fond
    arr = np.array(ratios)
    if arr[best] < 0.04:
        return None
    return best


def _read_group_by_boxes(form_img: np.ndarray, expand: int = 16) -> str | None:
    """
    Lit le groupe en LOCALISANT les 30 cases (10 lignes x 3 colonnes) par
    composantes connexes, comme _read_id_by_boxes : insensible aux derives
    du recalage, et renvoie None sur grille vide (pas d'hallucination).
    """
    x, y, w, h = ROI_GROUP_GRID
    lx, _, lw, _ = ROI_GROUP_LETTER
    x0 = max(0, x - expand)
    x1 = lx + lw + expand
    g = form_img if form_img.ndim == 2 else cv2.cvtColor(form_img, cv2.COLOR_BGR2GRAY)
    H_img = g.shape[0]
    y0 = max(0, y - expand)
    sub = g[y0:min(H_img, y + h + expand), x0:x1]
    binary = cv2.adaptiveThreshold(sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 35, 10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if 14 <= bw <= 34 and 14 <= bh <= 34 and 0.6 < bw / bh < 1.6 and area >= 30:
            boxes.append((cent[i][0], cent[i][1], bx, by, bw, bh))
    if len(boxes) < 22:
        return None

    def clusters(vals, gap=12):
        out, cur = [], [vals[0]]
        for v in vals[1:]:
            if v - cur[-1] > gap:
                out.append(float(np.mean(cur)))
                cur = []
            cur.append(v)
        out.append(float(np.mean(cur)))
        return out

    cx = clusters(sorted(p[0] for p in boxes))
    cy = clusters(sorted(p[1] for p in boxes))
    if len(cx) != 3 or len(cy) != GROUP_ROWS:
        return None
    rows_found = []
    for c in range(3):
        vals = []
        for r in range(GROUP_ROWS):
            cand = min(boxes, key=lambda p: (p[0] - cx[c]) ** 2 + (p[1] - cy[r]) ** 2)
            bx, by, bw, bh = cand[2], cand[3], cand[4], cand[5]
            mx, my = max(3, bw // 5), max(3, bh // 5)
            inner = binary[by + my:by + bh - my, bx + mx:bx + bw - mx]
            vals.append(float((inner > 0).mean()) if inner.size else 0.0)
        arr = np.array(vals)
        best = int(arr.argmax())
        if arr[best] < 0.05 or arr[best] < 2.0 * max(float(np.median(arr)), 0.01):
            return ""        # structure trouvee mais colonne sans coche -> vide
        rows_found.append(best)
    return f"G{rows_found[0]}{rows_found[1]}{chr(ord('A') + rows_found[2])}"


def read_group(form_img: np.ndarray) -> str | None:
    """
    Lit le code groupe depuis la grille graphique.
    Retourne une chaîne de type 'G02B' ou None.

    Methode principale : localisation des 30 cases par composantes connexes
    (_read_group_by_boxes). Repli : quadrillage fixe des ROIs calibres.
    Structure (10 lignes x 3 colonnes) :
      col 0 -> 1er chiffre (0-9)  dans ROI_GROUP_GRID
      col 1 -> 2ème chiffre (0-9) dans ROI_GROUP_GRID
      col 2 -> lettre (A-J)       dans ROI_GROUP_LETTER
                                  (ROI séparée, exclut les labels imprimés)
    """
    by_boxes = _read_group_by_boxes(form_img)
    if by_boxes == "":
        return None          # grille localisee et vide : pas de repli (il hallucinerait)
    if by_boxes is not None:
        return by_boxes
    # ---- Colonnes chiffres (dans ROI_GROUP_GRID) -------------------------
    roi_digits = get_roi(form_img, ROI_GROUP_GRID)
    cols_all   = _split_group_cols(roi_digits)

    digit_results = []
    for c_img in cols_all[:2]:   # seulement les 2 premières colonnes (chiffres)
        best = _read_grid_col_best_row(c_img, GROUP_ROWS)
        digit_results.append(best)

    # ---- Colonne lettre (ROI dédiée, checkbox uniquement) ----------------
    letter_col = get_roi(form_img, ROI_GROUP_LETTER)
    letter_row = _read_grid_col_best_row(letter_col, GROUP_ROWS)

    if None in digit_results or letter_row is None:
        return None

    digit1 = str(digit_results[0])
    digit2 = str(digit_results[1])
    letter  = chr(ord('A') + letter_row)
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
      max_number  si YES est coché avec champ 'Max number' (>= 1)
    """
    x_yes, x_no, has_max, x_max0, x_max1, y_max0, y_max1 = cond
    y0, y1 = COND_Y_YESNO
    h, w = y1 - y0, COND_CHECKBOX_W

    roi_yes = form_img[y0:y1, x_yes:x_yes + w]
    roi_no  = form_img[y0:y1, x_no:x_no + w]

    # Seuil 0.15 : les cases YES/NO sont parfois cochées légèrement (peu
    # d'encre) ; un seuil plus haut les manquerait.
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
    Lit un entier sur 1-2 chiffres dans un box "| tens | units |".

    L'image contient deux cellules côte-à-côte : la gauche = dizaines,
    la droite = unités. On lit chaque cellule séparément pour éviter
    que l'OCR ne confonde l'ordre (lisant '01' comme '10').

    Si la valeur résultante est 0 mais le ROI a clairement de l'encre,
    on retourne 1 par sécurité.
    """
    if roi is None or roi.size == 0:
        return 1
    try:
        from utils.ocr_utils import ocr_number
        h, w = roi.shape[:2]
        mid = w // 2
        # Marge interieure : ecarte les barres verticales du box, que l'OCR
        # lisait comme des chiffres (« |0| » devenait 7).
        mx, my = max(3, w // 10), max(2, h // 8)
        tens_roi  = roi[my:h - my, mx:mid - 2]
        units_roi = roi[my:h - my, mid + 2:w - mx]

        tens  = ocr_number(tens_roi)
        units = ocr_number(units_roi)

        tens  = tens  if tens  is not None else 0
        units = units if units is not None else 0

        # Un demi-box ne contient qu'UN chiffre : une lecture multi-chiffres
        # signale un artefact -> ne garder que le dernier chiffre.
        tens, units = tens % 10, units % 10

        val = tens * 10 + units
        return val if val > 0 else 1
    except Exception:
        # Repli : lire tout le ROI d'un coup
        try:
            from utils.ocr_utils import ocr_number
            val = ocr_number(roi)
            return val if val is not None else 1
        except Exception:
            return 1


# NB : read_note_maximale / read_note_pour_valider sont définies plus bas
# (via _read_both_notes, lecture conjointe des deux notes).

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

