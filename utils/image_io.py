"""
Lecture robuste d'images, quel que soit le format réel du fichier.

Beaucoup de photos prises au smartphone sont enregistrées au format HEIF/HEIC
(iPhone) mais portent une extension .jpg / .png trompeuse. cv2.imread et
PIL échouent silencieusement dessus. Ce module tente, dans l'ordre :

  1. cv2.imread
  2. cv2.imdecode (gère les chemins non-ASCII sous Windows)
  3. PIL (avec support HEIF si pillow-heif est installé)

Retourne une image BGR (np.ndarray) ou None.
"""

import cv2
import numpy as np

# Active le décodage HEIF/HEIC dans PIL si la lib est disponible.
try:
    import pillow_heif  # type: ignore
    pillow_heif.register_heif_opener()
    _HEIF_OK = True
except Exception:
    _HEIF_OK = False


def imread_robust(path: str):
    """Lit une image en BGR. Retourne None si tous les décodeurs échouent."""
    # 1. Voie standard OpenCV
    img = cv2.imread(path)
    if img is not None:
        return img

    # 2. imdecode (chemins accentués / non-ASCII sous Windows)
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass

    # 3. PIL (gère le HEIF/HEIC si pillow-heif est présent)
    try:
        from PIL import Image
        pil = Image.open(path)
        pil.load()
        pil = pil.convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def heif_support_available() -> bool:
    """Indique si le décodage HEIF/HEIC est actif (pillow-heif installé)."""
    return _HEIF_OK
