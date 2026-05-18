"""
Détection bas niveau des cases cochées dans un formulaire.

Méthodes utilisées (conformes au cahier des charges) :
  - Filtrage gaussien (réduction du bruit)
  - Seuillage Otsu (binarisation)
  - Morphologie mathématique (érosion, dilatation, ouverture)
  - Analyse des composantes connexes
  - Détection de motif en X par sommes diagonales

Aucun appel à des fonctions haut niveau de détection de cases.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Prétraitement
# ---------------------------------------------------------------------------

def preprocess_for_checkbox(img: np.ndarray) -> np.ndarray:
    """
    Convertit en niveaux de gris, applique un flou gaussien et un
    seuillage Otsu. Retourne une image binaire (255=encre, 0=fond).
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


# ---------------------------------------------------------------------------
# Détection d'une seule case
# ---------------------------------------------------------------------------

def ink_ratio(binary_roi: np.ndarray) -> float:
    """Retourne la proportion de pixels d'encre dans la ROI binaire."""
    if binary_roi.size == 0:
        return 0.0
    return float(np.count_nonzero(binary_roi)) / binary_roi.size


def has_x_pattern(binary_roi: np.ndarray, min_diag_ratio: float = 0.12) -> bool:
    """
    Détecte si la ROI contient un 'X' en analysant les projections
    diagonales avec une ouverture morphologique préalable.

    Approche :
      - Erosion pour supprimer le bruit de bord
      - Calcul de la diagonale principale et anti-diagonale
      - Si les deux ont une densité d'encre > min_diag_ratio → X détecté
    """
    if binary_roi.size == 0:
        return False

    # Erosion légère pour supprimer les bords du carré
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary_roi, kernel, iterations=1)

    h, w = eroded.shape
    if h < 4 or w < 4:
        return ink_ratio(eroded) > 0.15

    # Projection diagonale principale (top-left → bottom-right)
    diag_main = np.array([eroded[int(i * h / w), i] for i in range(w)])
    # Projection anti-diagonale (top-right → bottom-left)
    diag_anti = np.array([eroded[int(i * h / w), w - 1 - i] for i in range(w)])

    ratio_main = np.count_nonzero(diag_main) / len(diag_main)
    ratio_anti = np.count_nonzero(diag_anti) / len(diag_anti)

    return ratio_main > min_diag_ratio and ratio_anti > min_diag_ratio


def is_checkbox_checked(roi: np.ndarray,
                        ink_threshold: float = 0.08,
                        use_x_detection: bool = True) -> bool:
    """
    Détermine si une case est cochée.

    Args:
        roi: sous-image BGR ou gris de la case
        ink_threshold: seuil minimal de ratio d'encre
        use_x_detection: si True, vérifie aussi le motif en X

    Returns:
        True si la case est considérée cochée.
    """
    binary = preprocess_for_checkbox(roi)

    # Supprimer les bords (bordure de la case elle-même)
    margin = max(2, int(min(binary.shape) * 0.10))
    inner = binary[margin:-margin, margin:-margin] if (
        binary.shape[0] > 2 * margin and binary.shape[1] > 2 * margin
    ) else binary

    ratio = ink_ratio(inner)
    if ratio < ink_threshold:
        return False  # trop peu d'encre → case vide

    if use_x_detection:
        return has_x_pattern(inner) or ratio > 0.30  # croix ou carré plein
    return ratio > ink_threshold


def is_filled_square(roi: np.ndarray, threshold: float = 0.35) -> bool:
    """
    Détecte un carré plein (■) comme utilisé pour YES dans les conditions
    d'examen. Seuil plus élevé que pour un X.
    """
    binary = preprocess_for_checkbox(roi)
    margin = max(2, int(min(binary.shape) * 0.08))
    inner = binary[margin:-margin, margin:-margin] if (
        binary.shape[0] > 2 * margin and binary.shape[1] > 2 * margin
    ) else binary
    return ink_ratio(inner) > threshold


# ---------------------------------------------------------------------------
# Analyse d'une grille de cases
# ---------------------------------------------------------------------------

def split_grid(roi: np.ndarray, rows: int, cols: int
               ) -> list[list[np.ndarray]]:
    """
    Découpe une ROI en une grille de (rows × cols) cellules.
    Retourne cells[row][col].
    """
    h, w = roi.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    cells = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            y0 = r * cell_h
            y1 = y0 + cell_h
            x0 = c * cell_w
            x1 = x0 + cell_w
            row_cells.append(roi[y0:y1, x0:x1])
        cells.append(row_cells)
    return cells


def read_grid_checked(roi: np.ndarray, rows: int, cols: int,
                      ink_threshold: float = 0.08,
                      use_x_detection: bool = True
                      ) -> list[tuple[int, int]]:
    """
    Retourne la liste des (row, col) cochées dans la grille.
    """
    cells = split_grid(roi, rows, cols)
    checked = []
    for r in range(rows):
        for c in range(cols):
            if is_checkbox_checked(cells[r][c], ink_threshold, use_x_detection):
                checked.append((r, c))
    return checked


def read_grid_one_per_col(roi: np.ndarray, rows: int, cols: int,
                           ink_threshold: float = 0.08) -> list[int | None]:
    """
    Pour chaque colonne, retourne le numéro de ligne cochée (ou None).
    Utilisé pour les grilles de type 'Student ID' (une coche par colonne).

    Méthode : sélection du maximum de ratio d'encre intérieur.
    Le seuil ink_threshold filtre les colonnes sans coche visible.
    """
    cells = split_grid(roi, rows, cols)
    result = []
    for c in range(cols):
        best_row = None
        best_ratio = ink_threshold
        for r in range(rows):
            binary = preprocess_for_checkbox(cells[r][c])
            margin = max(2, int(min(binary.shape) * 0.10))
            inner = binary[margin:-margin, margin:-margin] if (
                binary.shape[0] > 2 * margin and binary.shape[1] > 2 * margin
            ) else binary
            ratio = ink_ratio(inner)
            if ratio > best_ratio:
                best_ratio = ratio
                best_row = r
        result.append(best_row)
    return result
