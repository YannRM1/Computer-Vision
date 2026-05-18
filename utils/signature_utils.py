"""
Gestion de la base de données de signatures et comparaison.

Méthodes bas niveau :
  - Binarisation et normalisation de la signature
  - Descripteur HOG (Histogram of Oriented Gradients) via scikit-image
  - Moments de Hu (invariants à la rotation et l'échelle)
  - Score de similarité cosinus entre descripteurs

La base est chargée depuis des fichiers ZIP ou un répertoire.
"""

import os
import io
import zipfile

import cv2
import numpy as np
from skimage.feature import hog


# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------

SIG_SIZE    = (128, 64)     # taille de normalisation (W, H)
HOG_PIXELS  = (8, 8)       # pixels par cellule HOG
HOG_CELLS   = (2, 2)       # cellules par bloc
HOG_ORIENT  = 9            # bins d'orientation HOG


# ---------------------------------------------------------------------------
# Chargement de la base de signatures
# ---------------------------------------------------------------------------

def _load_image_from_bytes(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def load_signatures_from_zip(zip_path: str) -> dict[str, list[np.ndarray]]:
    """
    Charge toutes les signatures depuis un fichier ZIP.

    Structure attendue :
      ZIP/
        student_id/
          student_id_000.png
          student_id_001.png
          ...

    Retourne { student_id (str) : [img_gray, ...] }
    """
    db: dict[str, list[np.ndarray]] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            # Extraire l'ID depuis le chemin
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            # Le dossier parent est le student_id
            student_id = parts[-2] if len(parts) >= 2 else None
            # Alternative : extraire du nom de fichier (ex : 19283_000.png)
            fname = os.path.splitext(parts[-1])[0]
            if "_" in fname:
                student_id = fname.rsplit("_", 1)[0]

            if student_id is None:
                continue
            data = z.read(name)
            img = _load_image_from_bytes(data)
            if img is not None:
                db.setdefault(student_id, []).append(img)
    return db


def load_signatures_from_dir(directory: str) -> dict[str, list[np.ndarray]]:
    """
    Charge les signatures depuis un répertoire structuré :
      directory/
        student_id/
          student_id_000.png
    """
    db: dict[str, list[np.ndarray]] = {}
    for student_id in os.listdir(directory):
        subdir = os.path.join(directory, student_id)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(subdir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                db.setdefault(student_id, []).append(img)
    return db


def load_signatures(sig_path: str) -> dict[str, list[np.ndarray]]:
    """
    Charge la base de signatures depuis un fichier ZIP ou un répertoire.
    Si plusieurs ZIPs existent dans sig_path, les fusionne.
    """
    db: dict[str, list[np.ndarray]] = {}

    if os.path.isfile(sig_path) and sig_path.endswith(".zip"):
        return load_signatures_from_zip(sig_path)

    if os.path.isdir(sig_path):
        # Chercher des ZIPs
        zips = [f for f in os.listdir(sig_path) if f.lower().endswith(".zip")]
        if zips:
            for z in zips:
                partial = load_signatures_from_zip(os.path.join(sig_path, z))
                for sid, imgs in partial.items():
                    db.setdefault(sid, []).extend(imgs)
            return db
        # Sinon, répertoire classique
        return load_signatures_from_dir(sig_path)

    return db


# ---------------------------------------------------------------------------
# Prétraitement des signatures
# ---------------------------------------------------------------------------

def preprocess_signature(img: np.ndarray) -> np.ndarray:
    """
    Normalise une signature brute :
      1. Conversion en niveaux de gris (si couleur)
      2. Détourage automatique (crop des bords blancs)
      3. Binarisation Otsu
      4. Redimensionnement vers SIG_SIZE
    """
    # Étape 1 : gris
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Étape 2 : rognage automatique
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None and len(coords) > 10:
        x, y, w, h = cv2.boundingRect(coords)
        # Ajouter une petite marge
        margin = 4
        x = max(0, x - margin); y = max(0, y - margin)
        w = min(gray.shape[1] - x, w + 2 * margin)
        h = min(gray.shape[0] - y, h + 2 * margin)
        gray = gray[y:y + h, x:x + w]
        binary = binary[y:y + h, x:x + w]

    # Étape 3 : redimensionner
    resized = cv2.resize(binary, SIG_SIZE)
    return resized


def compute_signature_descriptor(img_norm: np.ndarray) -> np.ndarray:
    """
    Calcule un vecteur de descripteurs combinant HOG + Moments de Hu.

    HOG : capture l'orientation des contours de la signature.
    Hu moments : invariants géométriques (forme globale).

    Retourne un vecteur numpy 1D normalisé (norme L2 = 1).
    """
    # HOG features
    hog_feat = hog(
        img_norm,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_PIXELS,
        cells_per_block=HOG_CELLS,
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # Moments de Hu (7 valeurs invariantes)
    moments = cv2.moments(img_norm)
    hu = cv2.HuMoments(moments).flatten()
    # Transformation logarithmique pour réduire l'échelle
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    feat = np.concatenate([hog_feat, hu])
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat.astype(np.float32)


# ---------------------------------------------------------------------------
# Base de données de descripteurs (pré-calculés)
# ---------------------------------------------------------------------------

def build_descriptor_db(raw_db: dict[str, list[np.ndarray]]
                        ) -> dict[str, np.ndarray]:
    """
    Pré-calcule le descripteur moyen de chaque étudiant.

    Retourne { student_id : descriptor_vector }
    """
    desc_db: dict[str, np.ndarray] = {}
    for sid, imgs in raw_db.items():
        descs = []
        for img in imgs:
            norm = preprocess_signature(img)
            desc = compute_signature_descriptor(norm)
            descs.append(desc)
        if descs:
            # Descripteur moyen (robuste aux variations inter-signatures)
            desc_db[sid] = np.mean(descs, axis=0)
    return desc_db


# ---------------------------------------------------------------------------
# Comparaison et identification
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identify_signature(
    query_img: np.ndarray,
    desc_db: dict[str, np.ndarray],
    threshold: float = 0.75,
) -> tuple[str | None, float]:
    """
    Identifie la signature query dans la base de descripteurs.

    Args:
        query_img : image de la signature (BGR ou gris)
        desc_db   : base de descripteurs {student_id: descriptor}
        threshold : seuil minimal de similarité pour valider

    Returns:
        (student_id, score) ou (None, score) si non reconnu.
    """
    if not desc_db:
        return None, 0.0

    norm = preprocess_signature(query_img)
    query_desc = compute_signature_descriptor(norm)

    best_id, best_score = None, -1.0
    for sid, ref_desc in desc_db.items():
        score = cosine_similarity(query_desc, ref_desc)
        if score > best_score:
            best_score = score
            best_id = sid

    if best_score < threshold:
        return None, best_score
    return best_id, best_score


def match_signature_to_id(
    query_img: np.ndarray,
    desc_db: dict[str, np.ndarray],
    expected_id: str | None = None,
    threshold: float = 0.75,
) -> tuple[str | None, bool]:
    """
    Compare la signature et retourne :
      (student_id_reconnu, validation_booléenne)

    Si expected_id est fourni, la validation est True seulement si
    l'ID reconnu correspond à l'ID attendu.
    """
    identified, score = identify_signature(query_img, desc_db, threshold)

    if expected_id is not None:
        validated = (identified == expected_id)
    else:
        validated = (identified is not None)

    return identified, validated
