"""
OCR pour textes imprimés et manuscrits.

Backend principal : easyocr (pur Python, sans binaire externe).
Pour les petites cases numériques : classification bas niveau par
analyse de composantes connexes (sans OCR externe).
"""

import re
import cv2
import numpy as np

# -----------------------------------------------------------------------
# Initialisation easyocr (singleton, chargé une seule fois)
# -----------------------------------------------------------------------
_reader = None

def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(['en', 'fr'], verbose=False)
    return _reader


# -----------------------------------------------------------------------
# Prétraitement commun
# -----------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _upscale_binarize(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Agrandit l'image et binarise pour améliorer la lisibilité OCR."""
    gray = _to_gray(img)
    h, w = gray.shape
    big = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(big, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _extract_ink_channel(img: np.ndarray) -> np.ndarray:
    """
    Pour les encres colorées (rouge), retourne le canal offrant le
    meilleur contraste.  Sinon, retourne le niveau de gris.
    """
    if len(img.shape) == 2:
        return img
    b, g, r = cv2.split(img)
    if float(np.mean(r)) < float(np.mean(g)) - 15:
        return cv2.bitwise_not(g)   # stylo rouge → canal vert inversé
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -----------------------------------------------------------------------
# Lecture via easyocr
# -----------------------------------------------------------------------

def _ocr_raw(img: np.ndarray, allowlist: str | None = None) -> str:
    """Lance easyocr et retourne le texte brut concaténé."""
    reader = _get_reader()
    kwargs = {}
    if allowlist:
        kwargs["allowlist"] = allowlist
    try:
        results = reader.readtext(img, detail=0, **kwargs)
        return " ".join(str(r) for r in results).strip()
    except Exception:
        return ""


def ocr_text(img: np.ndarray, lang: str = "en", scale: int = 3) -> str:
    """Texte générique. scale=6 recommandé pour les ROIs très petites (< 30 px)."""
    processed = _upscale_binarize(img, scale=scale)
    return _ocr_raw(processed)


# -----------------------------------------------------------------------
# Lecture de la bande CODES EXAM
# -----------------------------------------------------------------------

_RE_MODULE  = re.compile(r"Module\s*[|:\s]\s*([\w.]+)", re.IGNORECASE)
_RE_PROF    = re.compile(r"Profess(?:or|eur)\s*[|:\s]\s*([\w-]+)", re.IGNORECASE)
_RE_DATE    = re.compile(r"Date\s*[|:\s]\s*(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})",
                         re.IGNORECASE)
_RE_CODE    = re.compile(r"Code\s*[|:\s]\s*([\w-]+)", re.IGNORECASE)


def ocr_codes_exam(img: np.ndarray) -> dict:
    """
    Lit la bande colorée CODES EXAM (Module, Professor, Date, Code).

    Améliorations :
      - Fond coloré -> on retire la composante chromatique en passant par
        l'égalisation locale (CLAHE) avant binarisation.
      - On lance l'OCR sur l'image binarisée ET sur l'image en niveaux de
        gris upscalée, puis on fusionne en gardant le meilleur match par
        regex (autorise les caractères ambigus comme O/0).
    """
    gray = _to_gray(img)
    # CLAHE pour neutraliser le fond coloré uniforme
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    H, W = eq.shape
    big = cv2.resize(eq, (W * 4, H * 4), interpolation=cv2.INTER_CUBIC)

    # Variante 1 : Otsu pur sur l'image upscalée
    _, b1 = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text1 = _ocr_raw(b1)
    # Variante 2 : OCR direct sur niveau de gris CLAHE (EasyOCR sait gérer)
    text2 = _ocr_raw(big)

    combined = text1 + "  " + text2
    result = {"module": "", "professor": "", "date": "", "code": ""}
    for pat, key in [(_RE_MODULE, "module"), (_RE_PROF, "professor"),
                     (_RE_DATE, "date"), (_RE_CODE, "code")]:
        m = pat.search(combined)
        if m:
            result[key] = m.group(1).strip()
    return result


# -----------------------------------------------------------------------
# Lecture bas niveau de chiffres dans de petites cases imprimées
# (sans OCR externe – approche composantes connexes)
# -----------------------------------------------------------------------

def _digit_features(cell_bin: np.ndarray) -> dict:
    """
    Calcule des caractéristiques d'une case binarisée contenant
    un chiffre imprimé.
    """
    h, w = cell_bin.shape
    total = h * w

    ink   = int(np.count_nonzero(cell_bin))
    ratio = ink / (total + 1e-6)

    # Projections
    col_proj = np.sum(cell_bin, axis=0) / 255.0
    row_proj = np.sum(cell_bin, axis=1) / 255.0

    # Centre de masse vertical
    col_center = float(np.sum(col_proj * np.arange(w)) /
                        (np.sum(col_proj) + 1e-6)) / w

    # Densité des bandes verticales gauche / centre / droite
    third = max(1, w // 3)
    d_left   = float(np.mean(cell_bin[:, :third]))       / 255.0
    d_mid    = float(np.mean(cell_bin[:, third:2*third])) / 255.0
    d_right  = float(np.mean(cell_bin[:, 2*third:]))      / 255.0

    # Densité haut / bas
    half = max(1, h // 2)
    d_top    = float(np.mean(cell_bin[:half, :])) / 255.0
    d_bottom = float(np.mean(cell_bin[half:, :])) / 255.0

    # Centre horizontal (pour détecter "0" creux vs "8" plein)
    inner_h = max(1, h // 4)
    inner_w = max(1, w // 4)
    inner   = cell_bin[inner_h:-inner_h, inner_w:-inner_w] \
              if h > 2*inner_h and w > 2*inner_w else cell_bin
    d_inner = float(np.mean(inner)) / 255.0

    return dict(ratio=ratio, col_center=col_center,
                d_left=d_left, d_mid=d_mid, d_right=d_right,
                d_top=d_top, d_bottom=d_bottom, d_inner=d_inner)


def _classify_digit(cell_gray: np.ndarray) -> int:
    """
    Classifie un chiffre 0-9 dans une petite cellule.
    Règles heuristiques bas niveau calibrées sur des chiffres
    imprimés et manuscrits.
    """
    h, w = cell_gray.shape[:2]
    if h < 2 or w < 2:
        return 0

    # Normaliser vers une taille standard
    TARGET_H, TARGET_W = 80, 60
    cell_big = cv2.resize(_to_gray(cell_gray), (TARGET_W, TARGET_H),
                          interpolation=cv2.INTER_CUBIC)
    _, bin_cell = cv2.threshold(cell_big, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Rogner les bords (bordure de la case si présente)
    margin = 4
    if bin_cell.shape[0] > 2 * margin and bin_cell.shape[1] > 2 * margin:
        bin_cell = bin_cell[margin:-margin, margin:-margin]

    f = _digit_features(bin_cell)

    bh, bw = bin_cell.shape
    ratio = f['ratio']

    # Cas extrême : image presque vide
    if ratio < 0.04:
        return 0

    # --- Détection de "1" ---
    # "1" : trait vertical fin (w_original << h_original) ou ratio encre faible
    aspect = w / (h + 1e-6)
    if aspect < 0.35 or (ratio < 0.18 and f['d_mid'] > f['d_left'] and
                          f['d_mid'] > f['d_right']):
        return 1

    # --- Détection de "0" ---
    # Anneau (centre creux, ratio modéré)
    if f['d_inner'] < 0.12 and ratio > 0.22:
        return 0

    # --- Détection de "7" ---
    if (f['d_top'] > 0.40 and f['d_left'] < 0.20 and
            f['d_bottom'] < f['d_top'] * 0.7):
        return 7

    # --- Détection de "1" (basse densité) ---
    if ratio < 0.18:
        return 1

    # --- "2" : dense en haut, diagonal vers le bas-gauche ---
    if f['d_top'] > f['d_bottom'] * 1.25 and f['d_left'] > f['d_right'] * 0.8:
        return 2

    # --- "3" : concentré à droite ---
    if f['d_right'] > f['d_left'] * 1.6:
        return 3

    # --- "8" : deux anneaux ---
    if ratio > 0.50 and f['d_inner'] > 0.25:
        return 8

    # --- "4" : ligne verticale + horizontale ---
    if 0.25 <= ratio <= 0.45 and abs(f['d_left'] - f['d_right']) < 0.10:
        return 4

    # --- "5" : arc supérieur gauche + ligne bas droite ---
    if f['d_bottom'] > f['d_top'] * 1.2 and f['d_left'] > 0.20:
        return 5

    # --- "6" : boucle en bas ---
    if f['d_bottom'] > 0.38 and f['d_inner'] < 0.20:
        return 6

    # --- "9" : boucle en haut ---
    if f['d_top'] > 0.38 and f['d_inner'] < 0.20:
        return 9

    # Fallback
    return int(round(ratio * 10)) % 10


def ocr_digit(img: np.ndarray) -> str:
    """Lit un chiffre unique imprimé dans une petite cellule."""
    gray = _to_gray(img)
    d = _classify_digit(gray)
    return str(d)


def _segment_digits(img_gray: np.ndarray,
                    min_width_frac: float = 0.06,
                    gap_frac: float = 0.04) -> list[np.ndarray]:
    """
    Segmente les chiffres d'une image via projection verticale.
    Filtre les lignes de cadre (très fines ou très larges).

    Returns liste de sous-images (une par chiffre détecté).
    """
    h, w = img_gray.shape[:2]

    # Binariser (encre = 255)
    _, binary = cv2.threshold(img_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    col_proj = np.sum(binary.astype(np.float32), axis=0)
    max_proj = col_proj.max()
    if max_proj < 1:
        return []

    # Normaliser
    norm_proj = col_proj / max_proj

    # Seuil de gap : colonne considérée vide si < gap_frac
    gap_th = gap_frac

    # Trouver les régions de contenu (encre détectée)
    regions = []
    in_region = False
    start = 0
    for x, v in enumerate(norm_proj):
        if v > gap_th and not in_region:
            in_region = True
            start = x
        elif v <= gap_th and in_region:
            in_region = False
            regions.append((start, x))
    if in_region:
        regions.append((start, w))

    # Filtrer : on garde les régions de largeur entre min_width_frac et 60%
    min_w = max(3, int(w * min_width_frac))
    max_w = int(w * 0.60)
    digit_imgs = []
    for s, e in regions:
        if min_w <= (e - s) <= max_w:
            digit_imgs.append(img_gray[:, max(0, s-1):min(w, e+1)])
    return digit_imgs


def ocr_number(img: np.ndarray) -> int | None:
    """
    Lit un entier imprimé dans une case (ex : Note maximale = 20).
    Essaie easyocr d'abord, puis segmentation + classification bas niveau.
    """
    if img is None or img.size == 0:
        return None

    # -- Tentative easyocr (upscale agressif) ----------------------------
    gray = _to_gray(img)
    big  = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6),
                      interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = _ocr_raw(thresh, allowlist="0123456789")
    text = re.sub(r"\D", "", text)
    if text:
        try:
            return int(text)
        except ValueError:
            pass

    # -- Fallback : segmentation bas niveau + classification ---------------
    digit_imgs = _segment_digits(gray)
    if digit_imgs:
        digits = [_classify_digit(d) for d in digit_imgs]
        try:
            return int("".join(str(d) for d in digits))
        except Exception:
            pass
    return None


# -----------------------------------------------------------------------
# Lecture des réponses numériques manuscrites
# -----------------------------------------------------------------------

def _ocr_handwritten(img: np.ndarray, allowlist: str) -> str:
    """
    OCR pour textes manuscrits (encre rouge ou noire).
    Extrait d'abord le canal le plus contrasté (canal vert inversé pour
    l'encre rouge), puis applique CLAHE avant l'OCR.
    """
    # Extraire le canal d'encre optimal (gère encre rouge ET noire)
    ink = _extract_ink_channel(img)
    gray = ink if len(ink.shape) == 2 else _to_gray(ink)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    eq = clahe.apply(gray)
    big = cv2.resize(eq, (eq.shape[1] * 6, eq.shape[0] * 6),
                     interpolation=cv2.INTER_CUBIC)
    text = _ocr_raw(big, allowlist=allowlist)
    return text.strip()


def _segment_mantisse(img_gray: np.ndarray) -> float | None:
    """
    Segmente et classifie les chiffres d'une mantisse manuscrite.

    Stratégie :
    1. Rogner les N premiers/derniers pixels sur chaque bord (= bordure du cadre).
    2. Seuillage fixe (pas Otsu, trop sensible aux bordures) pour les chiffres.
    3. Composantes connexes → classification chiffre par chiffre.
    4. Détection du séparateur décimal (virgule/point).
    """
    h, w = img_gray.shape[:2]

    # Marge à rogner pour éliminer les bordures du cadre
    margin = max(3, int(min(h, w) * 0.06))
    inner = img_gray[margin:h - margin, margin:w - margin]

    if inner.size == 0:
        return None

    # Seuil fixe : tout pixel plus sombre que 210 est de l'encre
    # (rouge manuscrit ≈ gray 120-160 selon l'image, blanc ≈ 230+)
    _, binary = cv2.threshold(inner, 210, 255, cv2.THRESH_BINARY_INV)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    char_info = []
    ih, iw = inner.shape[:2]
    for i in range(1, num):
        cx, cy, cw, ch_c, area = stats[i]
        # Ignorer artefacts ou composantes qui couvrent presque tout
        if area < 6 or cw > iw * 0.65 or ch_c > ih * 0.95:
            continue
        char_img = inner[max(0, cy - 1):cy + ch_c + 1,
                         max(0, cx - 1):cx + cw + 1]
        char_info.append((cx, cw, ch_c, area, char_img))

    char_info.sort(key=lambda c: c[0])

    if not char_info:
        return None

    digit_h_ref = max((c[2] for c in char_info), default=1)
    digits = []

    for cx, cw, ch_c, area, char_img in char_info:
        # Séparateur décimal : composante très petite en hauteur et largeur
        if ch_c < digit_h_ref * 0.45 and cw < digit_h_ref * 0.55:
            digits.append(".")
        else:
            d = _classify_digit(char_img)
            digits.append(str(d))

    if not digits:
        return None

    text = "".join(digits)
    try:
        return float(text)
    except ValueError:
        # Nettoyer : supprimer les points en double, ne garder que les chiffres + 1 point
        text2 = re.sub(r"[^0-9.]", "", text)
        # Si plusieurs points, garder uniquement le premier
        parts = text2.split(".")
        if len(parts) > 2:
            text2 = parts[0] + "." + "".join(parts[1:])
        try:
            return float(text2) if text2 else None
        except ValueError:
            return None


def ocr_handwritten_mantisse(img: np.ndarray) -> float | None:
    """
    Lit la mantisse manuscrite (nombre décimal, encre rouge ou noire).

    Stratégie en 3 étapes :
    1. easyocr sur image agrandie + CLAHE (meilleur pour textes lisibles)
    2. Segmentation bas niveau (fallback)
    3. easyocr sans allowlist restrictif (dernier recours)
    """
    if img is None or img.size == 0:
        return None

    def _try_parse(text: str) -> float | None:
        t = text.replace(",", ".").strip()
        t = re.sub(r"[^0-9.\-]", "", t)
        if not t:
            return None
        # Corriger les ambiguïtés courantes OCR
        # "4" parfois lu à la place de "1" dans l'encre fine
        try:
            return float(t)
        except ValueError:
            return None

    gray = _to_gray(img)

    # Étape 1 : easyocr avec CLAHE × 8 (meilleure résolution)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    eq = clahe.apply(gray)
    big = cv2.resize(eq, (eq.shape[1] * 8, eq.shape[0] * 8),
                     interpolation=cv2.INTER_CUBIC)
    for al in ["0123456789.,-", None]:
        kwargs = {"allowlist": al} if al else {}
        results = _ocr_raw(big, **kwargs)
        val = _try_parse(results)
        if val is not None:
            return val

    # Étape 2 : segmentation bas niveau
    eq2 = clahe.apply(gray)
    result = _segment_mantisse(eq2)
    if result is not None:
        return result

    return None


def ocr_handwritten_exposant(img: np.ndarray) -> int | None:
    """Lit l'exposant manuscrit (entier, éventuellement négatif)."""
    if img is None or img.size == 0:
        return None
    # Essai bas niveau d'abord
    gray = _to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    segs = _segment_digits(gray, min_width_frac=0.05)
    if segs:
        digits = [_classify_digit(s) for s in segs]
        try:
            return int("".join(str(d) for d in digits))
        except Exception:
            pass
    # Fallback easyocr
    text = _ocr_handwritten(img, allowlist="0123456789-")
    text = re.sub(r"[^0-9\-]", "", text)
    try:
        return int(text) if text else None
    except ValueError:
        return None


def ocr_handwritten_unite(img: np.ndarray) -> str | None:
    """Lit l'unité (texte imprimé ou manuscrit)."""
    if img is None or img.size == 0:
        return None
    processed = _upscale_binarize(img, scale=3)
    text = _ocr_raw(processed)
    return text.strip() if text else None
