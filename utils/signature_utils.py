"""
Reconnaissance de signature - méthodes bas niveau.

Pipeline :
  - Binarisation Otsu + nettoyage des résidus de cadre
  - Recadrage strict sur l'encre + resize sur canvas fixe (ratio préservé)
  - Score combiné : NCC (template binaire) + HOG (orientations) + Hu moments
  - Agrégation MAX sur les N signatures de référence par étudiant
"""

import os
import zipfile

import cv2
import numpy as np
from skimage.feature import hog


# ----------------------------- Paramètres ---------------------------------

SIG_SIZE       = (192, 96)    # (W, H) du canvas normalisé
HOG_PIXELS     = (16, 16)
HOG_CELLS      = (2, 2)
HOG_ORIENT     = 9
MIN_INK_PIXELS = 50

# Poids des composantes du score
W_NCC, W_HOG, W_HU = 0.6, 0.35, 0.05


# ---------------------- Chargement de la base ----------------------------

def _decode(data):
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def load_signatures_from_zip(zip_path):
    db = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            parts = [p for p in name.split("/") if p]
            if len(parts) < 2:
                continue
            sid = parts[-2]
            fname = os.path.splitext(parts[-1])[0]
            if "_" in fname:
                sid = fname.rsplit("_", 1)[0]
            img = _decode(z.read(name))
            if img is not None:
                db.setdefault(sid, []).append(img)
    return db


def load_signatures_from_dir(directory):
    db = {}
    for sid in os.listdir(directory):
        sub = os.path.join(directory, sid)
        if not os.path.isdir(sub):
            continue
        for fname in os.listdir(sub):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(sub, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                db.setdefault(sid, []).append(img)
    return db


def load_signatures(sig_path):
    db = {}
    if os.path.isfile(sig_path) and sig_path.endswith(".zip"):
        return load_signatures_from_zip(sig_path)
    if os.path.isdir(sig_path):
        zips = [f for f in os.listdir(sig_path) if f.lower().endswith(".zip")]
        if zips:
            for z in zips:
                p = load_signatures_from_zip(os.path.join(sig_path, z))
                for sid, imgs in p.items():
                    db.setdefault(sid, []).extend(imgs)
            return db
        return load_signatures_from_dir(sig_path)
    return db


# ---------------------- Prétraitement ------------------------------------

def _clean_frame_artifacts(binary):
    H, W = binary.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    border = max(2, int(min(H, W) * 0.03))
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 8:
            continue
        touches = (x <= border or y <= border or
                   x + w >= W - border or y + h >= H - border)
        # Composantes touchant les bords et "grosses" (cadre) -> rejetées
        if touches and area > (H * W) * 0.005:
            continue
        # Composantes très allongées (lignes du cadre) -> rejetées
        ratio = max(w, h) / max(1, min(w, h))
        if ratio > 10 and area > 30:
            continue
        keep[labels == i] = 255
    return keep


def preprocess_signature(img):
    """Gris -> Otsu -> nettoyage cadre -> bbox+marge -> resize avec ratio.
    Le résultat est ensuite re-centré sur son centre de masse pour
    réduire la sensibilité à un petit décalage du trait.
    Retourne canvas binaire (H, W) = (SIG_SIZE[1], SIG_SIZE[0])."""
    target_w, target_h = SIG_SIZE
    if img is None or img.size == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = _clean_frame_artifacts(binary)

    coords = cv2.findNonZero(binary)
    if coords is None or len(coords) < MIN_INK_PIXELS:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    m = max(3, int(min(w, h) * 0.10))
    x = max(0, x - m); y = max(0, y - m)
    w = min(binary.shape[1] - x, w + 2 * m)
    h = min(binary.shape[0] - y, h + 2 * m)
    crop = binary[y:y + h, x:x + w]

    scale = min(target_w / crop.shape[1], target_h / crop.shape[0])
    new_w = max(1, int(crop.shape[1] * scale))
    new_h = max(1, int(crop.shape[0] * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Re-binarise (cubic peut introduire des niveaux intermediaires)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    # Recentrage par centre de masse (vs centrage geometrique simple)
    M = cv2.moments(resized)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        target_cx = new_w // 2
        target_cy = new_h // 2
        # Decalage souhaite (pour amener le COM au centre)
        dx = target_cx - cx
        dy = target_cy - cy
        Mtr = np.float32([[1, 0, dx], [0, 1, dy]])
        resized = cv2.warpAffine(resized, Mtr, (new_w, new_h),
                                 borderValue=0)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


# ---------------------- Descripteurs -------------------------------------

def _hog_vec(img_bin):
    return hog(
        img_bin.astype(np.float32) / 255.0,
        orientations=HOG_ORIENT,
        pixels_per_cell=HOG_PIXELS,
        cells_per_block=HOG_CELLS,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def _hu_vec(img_bin):
    M = cv2.moments(img_bin)
    hu = cv2.HuMoments(M).flatten()
    return (-np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).astype(np.float32)


def _ncc_template(a_bin, b_bin):
    """Corrélation normalisée entre 2 images binaires de même taille
    (variant zero-mean, range [-1, 1])."""
    a = (a_bin > 0).astype(np.float32)
    b = (b_bin > 0).astype(np.float32)
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a * b).sum() / (na * nb))


# ------------------- Base de descripteurs ---------------------------------

def build_descriptor_db(raw_db):
    """Retourne { sid: { 'tpl': (N, H, W) uint8, 'hog': (N, D) f32, 'hu': (N, 7) f32 } }."""
    desc_db = {}
    for sid, imgs in raw_db.items():
        tpls, hogs, hus = [], [], []
        for img in imgs:
            pp = preprocess_signature(img)
            if (pp > 0).sum() < MIN_INK_PIXELS:
                continue
            tpls.append(pp)
            hogs.append(_hog_vec(pp))
            hus.append(_hu_vec(pp))
        if tpls:
            desc_db[sid] = {
                "tpl": np.stack(tpls, axis=0),
                "hog": np.stack(hogs, axis=0),
                "hu":  np.stack(hus,  axis=0),
            }
    return desc_db


# ---------------------- Comparaison --------------------------------------

def _cos(a, B):
    """cosine entre vecteur a (D,) et matrice B (N, D). Retourne (N,)."""
    Bn = np.linalg.norm(B, axis=1) + 1e-9
    an = np.linalg.norm(a) + 1e-9
    return (B @ a) / (Bn * an)


def cosine_similarity(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def identify_signature(query_img, desc_db, threshold=0.18, margin=0.0):
    """Identifie une signature.
    Score = W_NCC * NCC + W_HOG * cos(HOG) + W_HU * cos(Hu).
    Pour chaque étudiant, on prend le MAX sur ses N signatures.
    Retourne (student_id|None, score)."""
    if not desc_db:
        return None, 0.0
    pp_q = preprocess_signature(query_img)
    if (pp_q > 0).sum() < MIN_INK_PIXELS:
        return None, 0.0
    h_q  = _hog_vec(pp_q)
    hu_q = _hu_vec(pp_q)

    scores = []
    for sid, d in desc_db.items():
        # NCC per template
        nccs = np.array([_ncc_template(pp_q, t) for t in d["tpl"]], dtype=np.float32)
        hogs = _cos(h_q,  d["hog"])
        hus  = _cos(hu_q, d["hu"])
        combined = W_NCC * nccs + W_HOG * hogs + W_HU * hus
        scores.append((sid, float(combined.max())))

    scores.sort(key=lambda t: t[1], reverse=True)
    best_id, best_score = scores[0]
    second = scores[1][1] if len(scores) > 1 else -1.0
    if best_score < threshold:
        return None, best_score
    if (best_score - second) < margin:
        return None, best_score
    return best_id, best_score


def match_signature_to_id(query_img, desc_db, expected_id=None, threshold=0.30):
    identified, _ = identify_signature(query_img, desc_db, threshold)
    if expected_id is not None:
        return identified, (identified == expected_id)
    return identified, (identified is not None)
