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
# Date nue sans label (ex: "17/11/2025" dans le header du haut)
_RE_DATE_BARE = re.compile(r"\b(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})\b")


def ocr_top_header(header_img: np.ndarray) -> dict:
    """
    Lit la bande d'en-tête (y ≈ 10-55 de la page normalisée) qui contient,
    entre les brackets de coin, les champs :
        [module_code]   [section_code]   [date]

    Ces données sont plus fiables que la zone CODES_EXAM pour la date car
    elles correspondent aux valeurs réelles de l'examen (pas au template).

    Retourne {"module": str, "code": str, "date": str}.
    """
    gray = _to_gray(header_img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    big = cv2.resize(eq, (eq.shape[1] * 4, eq.shape[0] * 4),
                     interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text1 = _ocr_raw(thresh)
    text2 = _ocr_raw(big)
    combined = text1 + " " + text2

    result = {"module": "", "code": "", "date": ""}

    # Date : chercher un motif DD/MM/YYYY ou variantes
    m_date = _RE_DATE_BARE.search(combined)
    if m_date:
        result["date"] = m_date.group(1).strip()

    # Les tokens alphanumériques restants (sans la date) → module et code
    tokens = [t for t in re.split(r"\s+", combined)
              if re.match(r"^[\w.\-]+$", t)
              and not re.match(r"^\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}$", t)]
    # Heuristique : module = token contenant un point (ex: IG.2405),
    #               code   = token avec tirets (ex: S1-01-G1)
    for tok in tokens:
        if "." in tok and not result["module"]:
            result["module"] = tok
        elif "-" in tok and not result["code"]:
            result["code"] = tok

    return result


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
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    H, W = eq.shape
    big = cv2.resize(eq, (W * 4, H * 4), interpolation=cv2.INTER_CUBIC)

    _, b1 = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text1 = _ocr_raw(b1)
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

# Reconnaissance des chiffres : assurée par le CNN (utils/digit_cnn.py) pour le
# manuscrit (mantisse / exposant) et par easyocr pour l'imprimé (ocr_number).
# L'ancien classifieur heuristique (_classify_digit / _digit_features) a été retiré :
# redondant avec le CNN et difficile à justifier (sec. 4.1 : réseaux pour le texte).


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
    """Lit un entier imprimé dans une case (ex : Note maximale = 20) via easyocr."""
    if img is None or img.size == 0:
        return None
    gray = _to_gray(img)
    big  = cv2.resize(gray, (gray.shape[1] * 6, gray.shape[0] * 6),
                      interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(big, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = re.sub(r"\D", "", _ocr_raw(thresh, allowlist="0123456789"))
    if text:
        try:
            return int(text)
        except ValueError:
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


def _box_is_empty(img: np.ndarray, margin_frac: float = 0.08,
                  min_ink_ratio: float = 0.010) -> bool:
    """
    True si la case (cadre retiré) ne contient quasiment pas d'encre.

    Empêche le réseau de « halluciner » des chiffres dans une case vide
    (ex. exposant absent -> doit rester None, pas 100001). On compte l'encre
    en excluant les composantes assimilables au cadre.
    """
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    m = max(2, int(min(h, w) * margin_frac))
    inner = gray[m:h - m, m:w - m]
    if inner.size == 0:
        return True
    _, b = cv2.threshold(inner, 200, 255, cv2.THRESH_BINARY_INV)
    ih, iw = inner.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    ink = 0
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw > iw * 0.70 or ch > ih * 0.97:                       # bord de cadre
            continue
        if max(cw, ch) / max(1, min(cw, ch)) > 6 and area > 20:    # trait de cadre
            continue
        ink += int(area)
    return (ink / float(inner.size)) < min_ink_ratio


def _segment_mantisse(img_gray: np.ndarray) -> float | None:
    """
    Segmente et classifie les chiffres d'une mantisse manuscrite (bas niveau).

    Rognage du cadre -> seuillage -> composantes connexes. On ne conserve que :
      - les composantes « chiffre » : hauteur proche de la plus grande trouvée ;
      - le « séparateur décimal » : petite composante, étroite, dans la moitié
        basse de la case (un point/virgule se pose sur la ligne de base).
    Le bruit et les fragments de cadre sont rejetés (évite d'ajouter de faux
    chiffres comme 3.1 -> 13.1). Classification par le CNN (repli heuristique).
    Renvoie None si aucune composante chiffre (case vide / illisible).
    """
    h, w = img_gray.shape[:2]
    margin = max(3, int(min(h, w) * 0.06))
    inner = img_gray[margin:h - margin, margin:w - margin]
    if inner.size == 0:
        return None
    ih, iw = inner.shape[:2]

    # Seuil fixe (l'encre rouge/noire ≈ gris < 210, blanc ≈ 230+).
    _, binary = cv2.threshold(inner, 210, 255, cv2.THRESH_BINARY_INV)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 1) Pré-filtrage : retirer artefacts minuscules et fragments de cadre.
    cand = []
    for i in range(1, num):
        cx, cy, cw, ch_c, area = (int(v) for v in stats[i])
        if area < 6:
            continue
        if cw > iw * 0.65 or ch_c > ih * 0.95:                       # trop gros
            continue
        if max(cw, ch_c) / max(1, min(cw, ch_c)) > 6 and area > 20:  # trait/cadre
            continue
        cand.append((cx, cy, cw, ch_c, area))
    if not cand:
        return None

    # 2) Hauteur de référence = plus grande composante (= un chiffre).
    h_ref = max(c[3] for c in cand)

    comps = []   # (cx, cw, ch_c, is_sep, char_img)
    for (cx, cy, cw, ch_c, area) in cand:
        # Séparateur décimal : composante nettement plus basse que les chiffres,
        # étroite, dont le CENTRE est dans la moitié basse (un point/virgule se
        # pose sur la ligne de base ; une virgule manuscrite peut être assez
        # haute, d'où le seuil 0.70*h_ref au lieu de 0.55).
        center_y = cy + ch_c / 2.0
        is_sep = (ch_c <= 0.70 * h_ref
                  and cw <= iw * 0.22
                  and center_y >= ih * 0.55)
        is_digit = not is_sep and ch_c >= 0.55 * h_ref
        if not is_digit and not is_sep:
            continue                                                  # bruit
        char_img = inner[max(0, cy - 1):cy + ch_c + 1,
                         max(0, cx - 1):cx + cw + 1]
        comps.append((cx, cw, ch_c, is_sep, char_img))

    comps.sort(key=lambda c: c[0])
    digit_imgs = [c[4] for c in comps if not c[3]]
    if not digit_imgs:
        return None

    from utils import digit_cnn
    cnn_pred = digit_cnn.predict(digit_imgs)
    if cnn_pred is None:                  # pas de modèle CNN -> repli easyocr en amont
        return None

    digits, k = [], 0
    for (cx, cw, ch_c, is_sep, char_img) in comps:
        if is_sep:
            digits.append(".")
            continue
        digits.append(cnn_pred[k])        # '' si illisible -> ignoré par le join
        k += 1

    text = "".join(digits).strip(".")    # pas de point en tête/queue
    if not text:
        return None
    if text.count(".") > 1:              # garder un seul séparateur décimal
        first = text.index(".")
        text = text[:first + 1] + text[first + 1:].replace(".", "")
    try:
        return float(text)
    except ValueError:
        return None


def ocr_handwritten_mantisse(img: np.ndarray) -> float | None:
    """
    Lit la mantisse manuscrite (nombre décimal, encre rouge ou noire).

    Deux étapes : segmentation bas niveau + CNN de chiffres (méthode principale),
    puis repli easyocr (image agrandie + CLAHE) si le CNN échoue ou est absent.
    """
    if img is None or img.size == 0:
        return None
    if _box_is_empty(img):              # case vide -> pas de mantisse (anti-hallucination)
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
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))

    # Étape 1 (prioritaire si CNN dispo) : segmentation bas niveau + CNN chiffres.
    # Champ manuscrit -> on privilégie le réseau de neurones (axe 6.3).
    from utils import digit_cnn
    if digit_cnn.available():
        result = _segment_mantisse(clahe.apply(gray))
        if result is not None:
            return result

    # Étape 2 : easyocr avec CLAHE × 8 (repli, ou cas sans modèle CNN)
    eq = clahe.apply(gray)
    big = cv2.resize(eq, (eq.shape[1] * 8, eq.shape[0] * 8),
                     interpolation=cv2.INTER_CUBIC)
    for al in ["0123456789.,-", None]:
        kwargs = {"allowlist": al} if al else {}
        results = _ocr_raw(big, **kwargs)
        val = _try_parse(results)
        if val is not None:
            return val
    return None


def _segment_exposant(img_gray: np.ndarray) -> int | None:
    """
    Segmente et classifie l'exposant manuscrit (entier, signe « - » possible).

    Même approche bas niveau que _segment_mantisse : rognage du cadre,
    seuillage, composantes connexes, rejet des fragments de cadre (qui étaient
    lus comme des « 1 » par l'ancienne segmentation par projection : 2 -> 121).
    Le signe moins est une composante large et plate à gauche des chiffres.
    """
    # On travaille sur le crop ENTIER (pas de rognage par marge : les chiffres
    # écrits contre le cadre seraient amputés). Le cadre et ses fragments sont
    # éliminés par filtrage des composantes (bords, traits fins, élongation).
    inner = img_gray
    ih, iw = inner.shape[:2]
    if inner.size == 0:
        return None

    _, binary = cv2.threshold(inner, 210, 255, cv2.THRESH_BINARY_INV)
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cand = []
    for i in range(1, num):
        cx, cy, cw, ch_c, area = (int(v) for v in stats[i])
        if area < 6:
            continue
        if cw > iw * 0.80 or ch_c > ih * 0.92:                       # cadre
            continue
        touches_edge = (cx <= 1 or cy <= 1
                        or cx + cw >= iw - 1 or cy + ch_c >= ih - 1)
        if touches_edge and (cw <= 3 or ch_c <= 3):                  # résidu de bord
            continue
        ratio = max(cw, ch_c) / max(1, min(cw, ch_c))
        is_flat = cw > ch_c                       # candidat signe moins
        # Trait vertical parasite (cadre) : très fin (≤ 2 px) ou très allongé
        # contre un bord. Un « 1 » manuscrit fait ≥ 3 px de large et est
        # éloigné des bords -> conservé.
        if not is_flat and (cw <= 2 or (ratio > 6 and touches_edge)):
            continue
        cand.append((cx, cy, cw, ch_c, area))
    if not cand:
        return None

    h_ref = max(c[3] for c in cand)
    digits, neg = [], False
    for (cx, cy, cw, ch_c, area) in sorted(cand, key=lambda c: c[0]):
        # Signe moins : plat (plus large que haut), petit en hauteur,
        # centré verticalement, et situé avant tout chiffre.
        center_y = cy + ch_c / 2.0
        if (not digits and cw >= 1.5 * ch_c and ch_c <= 0.45 * h_ref
                and ih * 0.20 <= center_y <= ih * 0.80):
            neg = True
            continue
        if ch_c < 0.55 * h_ref:
            continue                                                 # bruit
        digits.append(inner[max(0, cy - 1):cy + ch_c + 1,
                            max(0, cx - 1):cx + cw + 1])
    if not digits:
        return None

    from utils import digit_cnn
    cnn_pred = digit_cnn.predict(digits)
    if cnn_pred is None:
        return None
    text = "".join(cnn_pred)
    if not text:
        return None
    try:
        return -int(text) if neg else int(text)
    except ValueError:
        return None


def ocr_handwritten_exposant(img: np.ndarray) -> int | None:
    """Lit l'exposant manuscrit (entier, éventuellement négatif)."""
    if img is None or img.size == 0:
        return None
    if _box_is_empty(img):              # case vide -> exposant None (anti-hallucination)
        return None
    gray = _to_gray(img)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    gray = clahe.apply(gray)
    from utils import digit_cnn
    if digit_cnn.available():
        result = _segment_exposant(gray)
        if result is not None:
            return result
    text = _ocr_handwritten(img, allowlist="0123456789-")   # repli easyocr
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
