"""
Reconnaissance de chiffres manuscrits isoles (cases mantisse / exposant).

Conforme a la consigne (sec. 4.1) : les TEXTES manuscrits peuvent etre traites
par des methodes de haut niveau (reseaux de neurones). On entraine un petit CNN
sur EMNIST-digits (chiffres 0-9) via train_digit_cnn.py ; ce module fournit le
modele et l'inference par case.

Architecture identique a utils/letter_cnn.py (un seul schema a decrire dans le
rapport), seule la tete de classification change : 10 classes (0-9) au lieu de 26.

La segmentation des chiffres dans les cadres mantisse/exposant reste BAS NIVEAU
(projection / composantes connexes, cf. ocr_utils.py) ; ce module ne fait que la
CLASSIFICATION d'une case deja isolee.

Si torch ou le fichier de poids est absent, les fonctions renvoient None et
l'appelant retombe sur l'heuristique / easyocr existante.
"""

import os
import numpy as np
import cv2

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "digit_cnn.pt")
INPUT_SIZE = 28
N_CLASSES = 10   # 0-9

_model = None
_loaded = False


# ---------------------------------------------------------------------------
# Architecture (definie aussi cote entrainement, doit rester identique)
# ---------------------------------------------------------------------------

def build_model():
    """Construit le CNN. Importe torch paresseusement."""
    import torch.nn as nn

    class DigitCNN(nn.Module):
        def __init__(self, n_classes=N_CLASSES):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),                       # 28 -> 14
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.MaxPool2d(2),                       # 14 -> 7
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return DigitCNN()


def load_model():
    """Charge le modele entraine (singleton). Renvoie None si indisponible."""
    global _model, _loaded
    if _loaded:
        return _model
    _loaded = True
    try:
        import torch
        if not os.path.isfile(MODEL_PATH):
            _model = None
            return None
        model = build_model()
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model = model
    except Exception:
        _model = None
    return _model


def available():
    """True si le CNN est utilisable (torch + poids presents)."""
    return load_model() is not None


# ---------------------------------------------------------------------------
# Pretraitement d'une case -> image 28x28 type EMNIST (chiffre blanc / fond noir)
# ---------------------------------------------------------------------------

def prep_cell(cell):
    """
    Transforme la sous-image d'un chiffre isole en entree 28x28 (np.float32,
    0..1), chiffre blanc centre sur fond noir, comme EMNIST. Renvoie None si
    vide.

    Identique au pretraitement des lettres : binarisation Otsu inverse,
    suppression des fragments de cadre, recadrage sur la boite englobante,
    redimensionnement (ratio conserve) a ~20 px puis centrage dans 28x28.
    """
    if cell is None or cell.size == 0:
        return None
    gray = cell if cell.ndim == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    binv = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = binv.shape

    # Supprime les fragments de cadre : composantes tres allongees couvrant
    # >75% de la largeur (trait horizontal) ou de la hauteur (trait vertical).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    clean = np.zeros_like(binv)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 4:
            continue
        if bw > 0.75 * W and bh < 0.30 * H:      # ligne horizontale (bordure)
            continue
        if bh > 0.75 * H and bw < 0.30 * W:      # ligne verticale (bordure)
            continue
        clean[lab == i] = 255
    binv = clean

    ys, xs = np.nonzero(binv)
    if len(xs) < 6:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = binv[y0:y1 + 1, x0:x1 + 1]

    # Redimensionne (ratio conserve) vers ~20px puis centre dans 28x28.
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    _, resized = cv2.threshold(resized, 60, 255, cv2.THRESH_BINARY)
    # Epaissit le trait pour se rapprocher du style EMNIST (traits gras).
    resized = cv2.dilate(resized, np.ones((2, 2), np.uint8), iterations=1)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
    ox, oy = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    # Leger flou pour imiter l'anti-aliasing d'EMNIST (niveaux de gris).
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas.astype(np.float32) / 255.0


def predict(cells):
    """
    Predit le chiffre (0-9) de chaque case d'une liste. Renvoie une liste de
    chaines ('' si case illisible). Renvoie None si le modele est indisponible
    (l'appelant doit alors utiliser un repli).
    """
    model = load_model()
    if model is None:
        return None
    import torch
    batch, idx_map = [], []
    for i, c in enumerate(cells):
        arr = prep_cell(c)
        if arr is not None:
            batch.append(arr)
            idx_map.append(i)
    out = [""] * len(cells)
    if not batch:
        return out
    x = torch.from_numpy(np.stack(batch)[:, None, :, :])  # (N,1,28,28)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).tolist()
    for j, i in enumerate(idx_map):
        out[i] = str(int(pred[j]))
    return out


def predict_one(cell):
    """Predit un seul chiffre. Renvoie '' si illisible, None si modele absent."""
    res = predict([cell])
    return None if res is None else res[0]
