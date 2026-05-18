"""
Détection des marques de coin et alignement du formulaire.

Approche bas niveau :
  1. Binarisation (seuillage Otsu après flou gaussien)
  2. Détection des lignes horizontales/verticales (Transformée de Hough)
  3. Identification des 4 marques de coin (brackets L)
  4. Correction de perspective (indispensable pour les photos)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Prétraitement de base
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(gray: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    """Binarisation Otsu après flou gaussien. Retourne 0=fond, 255=encre."""
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Détection des marques de coin
# ---------------------------------------------------------------------------

def _find_corner_candidates(binary: np.ndarray, min_area: int = 200,
                             max_area: int = 8000) -> list[tuple[int, int, int, int]]:
    """
    Retourne les bounding boxes (x, y, w, h) de composantes connexes
    dont la forme ressemble à un L (rapport d'aspect proche de 1,
    aire remplie modérément).
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        # Le L a un ratio W/H proche de 1 et un taux de remplissage faible
        ratio = w / (h + 1e-6)
        fill = area / (w * h + 1e-6)
        if 0.5 < ratio < 2.0 and fill < 0.65:
            candidates.append((x, y, w, h))
    return candidates


def find_active_area(img: np.ndarray) -> tuple[int, int, int, int]:
    """
    Détecte la zone active du formulaire (intérieur des marques de coin).
    Retourne (x, y, w, h) de la zone active en pixels.

    Pour les PDFs propres, utilise les lignes de bord décelables.
    """
    gray = to_gray(img)
    H, W = gray.shape

    # Essai 1 : détection des brackets via composantes connexes
    binary = binarize(gray)
    candidates = _find_corner_candidates(binary)

    if len(candidates) >= 4:
        xs = [c[0] for c in candidates]
        ys = [c[1] for c in candidates]
        x0, y0 = int(np.percentile(xs, 10)), int(np.percentile(ys, 10))
        x1 = int(np.percentile([c[0] + c[2] for c in candidates], 90))
        y1 = int(np.percentile([c[1] + c[3] for c in candidates], 90))
        # Ajouter un léger retrait vers l'intérieur
        margin = int(min(W, H) * 0.01)
        x0 = max(0, x0 + margin)
        y0 = max(0, y0 + margin)
        x1 = min(W, x1 - margin)
        y1 = min(H, y1 - margin)
        return x0, y0, x1 - x0, y1 - y0

    # Fallback : marges fixes basées sur les proportions de la page
    mx, my = int(W * 0.04), int(H * 0.04)
    return mx, my, W - 2 * mx, H - 2 * my


def warp_to_form(img: np.ndarray,
                 src_corners: np.ndarray,
                 out_w: int = 900,
                 out_h: int = 1270) -> np.ndarray:
    """
    Applique une correction de perspective en mappant src_corners
    (4 points [TL, TR, BR, BL]) vers un rectangle standard.
    """
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_corners.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


# ---------------------------------------------------------------------------
# Correction d'inclinaison (pour les photos)
# ---------------------------------------------------------------------------

def _hough_dominant_angle(binary: np.ndarray) -> float:
    """
    Retourne l'angle d'inclinaison dominant (en degrés) via Hough.
    Valeur proche de 0 = déjà droit.
    """
    edges = cv2.Canny(binary, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return 0.0
    angles = []
    for rho, theta in lines[:, 0]:
        angle = np.degrees(theta) - 90  # ramener vers 0
        if abs(angle) < 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(img: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Corrige une légère inclinaison (rotation) de l'image.
    Ne fait rien si l'angle détecté dépasse max_angle (cas extrême).
    """
    gray = to_gray(img)
    binary = binarize(gray)
    angle = _hough_dominant_angle(binary)
    if abs(angle) > max_angle:
        return img
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (W, H),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Pipeline complet : normalise une image vers le repère du formulaire
# ---------------------------------------------------------------------------

FORM_W = 900   # largeur de référence (pixels)
FORM_H = 1270  # hauteur de référence (pixels)


def normalize_form_image(img: np.ndarray,
                         is_photo: bool = False) -> np.ndarray:
    """
    Prépare une image de formulaire pour l'extraction des champs.

    - Pour un PDF: recadre sur la zone active et redimensionne.
    - Pour une photo: deskew + correction perspective + recadrage.

    Retourne une image BGR de taille (FORM_H, FORM_W).
    """
    if is_photo:
        img = deskew(img)

    x, y, w, h = find_active_area(img)
    cropped = img[y:y + h, x:x + w]
    resized = cv2.resize(cropped, (FORM_W, FORM_H))
    return resized
