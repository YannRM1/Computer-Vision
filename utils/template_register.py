"""
Recalage d'une photo de formulaire sur un template de référence.

Les photos de première page sont prises à main levée : rotation, perspective,
ombres. Un recadrage par bounding-box ne suffit pas pour qu'un ROI fixe tombe
sur la bonne grille. Comme le gabarit du formulaire est identique pour tous les
étudiants, on recale chaque photo sur un template propre par mise en
correspondance de points (ORB) + homographie RANSAC.

Le template est l'image normalisée d'une page 1 (repère canonique 900 × 1270).
Après recalage, la photo se trouve dans CE MÊME repère : on peut donc lire tous
les champs avec les ROIs calibrées pour les PDFs.
"""

import cv2
import numpy as np

# Détecteur partagé (coûteux à instancier).
_ORB = cv2.ORB_create(6000)
_BF = cv2.BFMatcher(cv2.NORM_HAMMING)

_WORK_MAX = 1500   # taille de travail pour la détection de features
_MIN_GOOD = 15     # nombre minimal d'appariements valides


class FormTemplate:
    """Template de référence pré-calculé (image grise + features ORB)."""

    def __init__(self, ref_bgr: np.ndarray):
        self.gray = (ref_bgr if ref_bgr.ndim == 2
                     else cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY))
        self.h, self.w = self.gray.shape[:2]
        self.kp, self.des = _ORB.detectAndCompute(self.gray, None)


def find_reference_pdf(*dirs: str):
    """Renvoie le chemin du 1er PDF trouvé dans les répertoires donnés."""
    import os
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".pdf"):
                return os.path.join(d, f)
    return None


def build_form_template(pdf_path: str):
    """
    Construit un FormTemplate à partir de la 1re page d'un PDF de référence.
    Le rendu est normalisé dans le repère canonique avant extraction des
    features. Retourne un FormTemplate ou None.
    """
    try:
        from utils.pdf_utils import pdf_to_images
        from utils.grid_decoder import normalize_page
        pages = pdf_to_images(pdf_path, dpi=150)
        if not pages:
            return None
        return FormTemplate(normalize_page(pages[0], is_photo=False))
    except Exception:
        return None


def load_bundled_template():
    """
    Charge le template de référence figé livré avec le projet
    (utils/assets/form_template_ref.png), rendu depuis la page calibrée.
    C'est le repère sur lequel les ROIs sont calibrées : on l'utilise en
    priorité pour garantir un recalage déterministe, indépendant du PDF
    présent dans le répertoire. Retourne un FormTemplate ou None.
    """
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "form_template_ref.png")
    if not os.path.isfile(path):
        return None
    ref = cv2.imread(path)
    return FormTemplate(ref) if ref is not None else None


def get_photo_template(*pdf_dirs: str):
    """
    Retourne le meilleur template disponible : le template figé livré avec le
    projet en priorité, sinon le rendu d'un PDF trouvé dans les répertoires
    donnés. Retourne None si rien n'aboutit.
    """
    t = load_bundled_template()
    if t is not None:
        return t
    ref_pdf = find_reference_pdf(*pdf_dirs)
    return build_form_template(ref_pdf) if ref_pdf else None


def register_to_template(img_bgr: np.ndarray, template):
    """
    Recale `img_bgr` sur `template` et renvoie l'image redressée dans le repère
    canonique (template.w × template.h), ou None si le recalage échoue.
    """
    if template is None or template.des is None:
        return None

    gray = (img_bgr if img_bgr.ndim == 2
            else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))

    # Sous-échantillonnage pour accélérer la détection sur les photos 4000px.
    s = _WORK_MAX / max(gray.shape)
    if s < 1.0:
        small = cv2.resize(gray, (int(gray.shape[1] * s),
                                  int(gray.shape[0] * s)))
    else:
        small, s = gray, 1.0

    kp2, des2 = _ORB.detectAndCompute(small, None)
    if des2 is None or len(kp2) < _MIN_GOOD:
        return None

    matches = _BF.knnMatch(template.des, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < _MIN_GOOD:
        return None

    src = np.float32([template.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / s

    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    if H is None:
        return None
    if mask is not None and int(mask.sum()) < _MIN_GOOD:
        return None

    return cv2.warpPerspective(img_bgr, H, (template.w, template.h))
