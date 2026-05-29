"""
Détection des marques de coin (L-brackets) et correction de perspective.

Approche bas niveau :
  1. Binarisation adaptative
  2. Template-matching de 4 L synthétiques (un par coin)
  3. Correction de perspective via cv2.getPerspectiveTransform

Fallback : si la détection des brackets échoue (photo bruitée, brackets
faibles), retourne l'image inchangée — l'appelant utilisera l'ancien
chemin (deskew + bounding box).
"""

import cv2
import numpy as np


def to_gray(img):
    return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray, blur=5):
    b = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, binary = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Détection des L-brackets (template matching)
# ---------------------------------------------------------------------------

def _make_L(size, thickness, orient):
    pad = size // 2
    tpl = np.zeros((size + 2 * pad, size + 2 * pad), dtype=np.float32)
    t = thickness
    if orient == "TL":
        tpl[pad:pad + t, pad:pad + size] = 1
        tpl[pad:pad + size, pad:pad + t] = 1
    elif orient == "TR":
        tpl[pad:pad + t, pad:pad + size] = 1
        tpl[pad:pad + size, pad + size - t:pad + size] = 1
    elif orient == "BL":
        tpl[pad + size - t:pad + size, pad:pad + size] = 1
        tpl[pad:pad + size, pad:pad + t] = 1
    else:  # BR
        tpl[pad + size - t:pad + size, pad:pad + size] = 1
        tpl[pad:pad + size, pad + size - t:pad + size] = 1
    return tpl, pad


def _find_bracket(quad_binf, orient, sizes=(60, 80, 100, 130), ts=(4, 5, 6, 8)):
    """Cherche le meilleur L dans une région binarisée (float 0..1)."""
    best = None
    H, W = quad_binf.shape
    for s in sizes:
        for t in ts:
            tpl, pad = _make_L(s, t, orient)
            if tpl.shape[0] >= H or tpl.shape[1] >= W:
                continue
            r = cv2.matchTemplate(quad_binf, tpl, cv2.TM_CCOEFF_NORMED)
            _, sc, _, loc = cv2.minMaxLoc(r)
            if best is None or sc > best[2]:
                best = (loc[0], loc[1], sc, s, pad)
    return best


def find_corner_brackets(img, search_frac=0.20, min_score=0.55):
    """
    Détecte 4 L-brackets (un par coin). Retourne {'TL':(x,y),...} ou None.
    """
    gray = to_gray(img)
    H, W = gray.shape
    binary = binarize(gray)
    binf = binary.astype(np.float32) / 255.0

    sx = int(W * search_frac); sy = int(H * search_frac)
    quads = {
        "TL": (binf[:sy, :sx], 0, 0),
        "TR": (binf[:sy, W - sx:], W - sx, 0),
        "BL": (binf[H - sy:, :sx], 0, H - sy),
        "BR": (binf[H - sy:, W - sx:], W - sx, H - sy),
    }
    out = {}
    for k, (q, dx, dy) in quads.items():
        b = _find_bracket(q, k)
        if b is None or b[2] < min_score:
            return None
        x, y, sc, sz, pad = b
        if k == "TL":   cx, cy = x + pad, y + pad
        elif k == "TR": cx, cy = x + pad + sz - 1, y + pad
        elif k == "BL": cx, cy = x + pad, y + pad + sz - 1
        else:           cx, cy = x + pad + sz - 1, y + pad + sz - 1
        out[k] = (cx + dx, cy + dy)
    return out


# ---------------------------------------------------------------------------
# Correction de perspective via les 4 brackets
# ---------------------------------------------------------------------------

# Position attendue des 4 brackets dans le formulaire normalisé (900x1270),
# calibrée sur le rendu PDF (DPI 200) : brackets à env. 7.5%/4.3% des bords.
BRACKET_DST_NORM = {
    "TL": (66, 55),
    "TR": (834, 55),
    "BL": (66, 1215),
    "BR": (834, 1215),
}


def warp_from_brackets(img, brackets, fw=900, fh=1270):
    src = np.array([
        brackets["TL"], brackets["TR"],
        brackets["BR"], brackets["BL"],
    ], dtype=np.float32)
    dst = np.array([
        BRACKET_DST_NORM["TL"], BRACKET_DST_NORM["TR"],
        BRACKET_DST_NORM["BR"], BRACKET_DST_NORM["BL"],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (fw, fh))


def align_form(img, fw=900, fh=1270):
    """
    Renvoie l'image alignée vers le repère (fw, fh) si les 4 brackets sont
    détectés ; sinon None (l'appelant doit utiliser un fallback).
    """
    b = find_corner_brackets(img)
    if b is None:
        return None
    return warp_from_brackets(img, b, fw, fh)


# ---------------------------------------------------------------------------
# Deskew (rotation simple - utilisé en fallback ou en pré-traitement)
# ---------------------------------------------------------------------------

def _hough_dominant_angle(binary):
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        a = np.degrees(theta) - 90
        if abs(a) < 45:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def deskew(img, max_angle=15.0):
    gray = to_gray(img)
    binary = binarize(gray)
    angle = _hough_dominant_angle(binary)
    if abs(angle) > max_angle:
        return img
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Fonctions de compat (utilisées par grid_decoder)
# ---------------------------------------------------------------------------

def find_active_area(img):
    """Bounding box rapide via Otsu."""
    gray = to_gray(img)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    H, W = gray.shape
    row_has = np.any(binary > 0, axis=1)
    col_has = np.any(binary > 0, axis=0)
    r0 = int(np.argmax(row_has))
    r1 = H - int(np.argmax(row_has[::-1])) - 1
    c0 = int(np.argmax(col_has))
    c1 = W - int(np.argmax(col_has[::-1])) - 1
    return c0, r0, c1 - c0, r1 - r0


FORM_W = 900
FORM_H = 1270


def normalize_form_image(img, is_photo=False):
    """Pipeline complet : essaie l'alignement bracket-based, sinon
    deskew + bbox + resize."""
    aligned = align_form(img, FORM_W, FORM_H)
    if aligned is not None:
        return aligned
    if is_photo:
        img = deskew(img)
    x, y, w, h = find_active_area(img)
    return cv2.resize(img[y:y + h, x:x + w], (FORM_W, FORM_H))
