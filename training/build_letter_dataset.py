"""
Construction d'un jeu de donnees de lettres annotees a partir des formulaires.

Pour fine-tuner le CNN (sec. 4.1/4.2 : reseau de neurones autorise pour le
texte, mise au point sur donnees annotees), on exploite les verites terrain :
les noms (Prenom ligne 13, Nom ligne 14) sont connus dans les xlsx fournis.

Pour chaque formulaire (PDF + photo) :
  - recalage -> repere canonique
  - segmentation des cases du Prenom et du Nom (collect_name_cells)
  - si le nombre de cases NON VIDES == nombre de lettres (A-Z) du nom verite,
    on apparie case <-> lettre dans l'ordre et on enregistre chaque case
    pretraitee (28x28) avec son label.

Sortie : letter_dataset.npz  (X: (N,28,28) float32, y: (N,) int 0..25)
+ dataset_preview.png pour verification visuelle.

Usage : python build_letter_dataset.py
"""
import os, re, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openpyxl
from utils.image_io import imread_robust
from utils.template_register import get_photo_template
from utils.grid_decoder import normalize_page, set_photo_template
from utils.page1_parser import collect_name_cells, FIRSTNAME_Y, NAME_Y
from utils import letter_cnn

try:
    import fitz
except ImportError:
    fitz = None

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "PROJECT 2026 -DATABASE-20260518")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG")


def render_pdf_p1(path, dpi=150):
    doc = fitz.open(path)
    pix = doc[0].get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    doc.close()
    return arr


def truth_names(form_dir, sid, form):
    p = os.path.join(form_dir, f"EXAM_{form}_{sid}.xlsx")
    if not os.path.isfile(p):
        return None, None
    try:
        ws = openpyxl.load_workbook(p, data_only=True)["PAGE-01"]
        return (str(ws.cell(13, 2).value or ""), str(ws.cell(14, 2).value or ""))
    except Exception:
        return None, None


def letters_only(name):
    return re.sub(r"[^A-Za-z]", "", name).upper()


def _first_n_cells(norm, y_range, n):
    """Renvoie les n premieres cases de la grille de lettres (ajustee par
    formulaire). Les noms sont ecrits a partir de la 1re case, contigus et
    alignes a gauche : on apparie donc directement case k <-> lettre k, sans
    dependre d'une detection vide/pleine (qui tronquait les lettres fines comme
    I/J et faisait chuter le rendement du jeu de donnees)."""
    from utils.page1_parser import _fit_name_grid, NAME_CELLS
    from utils.ocr_utils import _to_gray
    y0, y1 = y_range
    x0, pitch = _fit_name_grid(norm, y_range)
    cells = []
    for k in range(min(n, NAME_CELLS)):
        xa = int(round(x0 + k * pitch))
        xb = int(round(x0 + (k + 1) * pitch))
        cell = norm[y0:y1, xa:xb]
        if cell.size == 0:
            return None
        g = _to_gray(cell)
        ch, cw = g.shape
        cells.append(g[3:max(4, ch - 3), 4:max(5, cw - 4)])
    return cells


def add_samples(norm, name, y_range, X, y, dropped):
    """Extrait une case par lettre du nom (verite terrain) et l'ajoute au jeu.

    Appariement par POSITION (case k <-> lettre k) sur les n premieres cases,
    n = longueur du nom : robuste et a haut rendement (~1500 lettres sur les 3
    formulaires, contre ~120 avec l'ancien appariement par comptage exact)."""
    letters = letters_only(name)
    if not letters or len(letters) > 15:
        return
    cells = _first_n_cells(norm, y_range, len(letters))
    if cells is None or len(cells) != len(letters):
        dropped[0] += 1
        return
    for cell, ch in zip(cells, letters):
        arr = letter_cnn.prep_cell(cell)
        if arr is None:
            continue
        X.append(arr.astype(np.float32))
        y.append(ord(ch) - ord('A'))


def main():
    set_photo_template(get_photo_template(DATA))  # template figé (form identique)
    X, y = [], []
    dropped = [0]
    n_forms = 0

    for form in ("FORM1", "FORM2", "FORM3"):
        fdir = os.path.join(DATA, form)
        if not os.path.isdir(fdir):
            continue
        # PDFs
        for f in sorted(os.listdir(fdir)):
            mp = re.match(rf"EXAM_{form}_(\d+)\.pdf$", f)
            mi = re.match(rf"EXAM_{form}_(\d+)\.(jpg|jpeg|png|bmp)$", f, re.I)
            if mp and fitz is not None:
                sid = mp.group(1)
                if sid == "00000":
                    continue
                tf, tn = truth_names(fdir, sid, form)
                if tf is None:
                    continue
                try:
                    norm = normalize_page(render_pdf_p1(os.path.join(fdir, f)),
                                          is_photo=False, use_template=True)
                except Exception:
                    continue
                add_samples(norm, tf, FIRSTNAME_Y, X, y, dropped)
                add_samples(norm, tn, NAME_Y, X, y, dropped)
                n_forms += 1
            elif mi:
                sid = mi.group(1)
                if sid == "00000":
                    continue
                tf, tn = truth_names(fdir, sid, form)
                if tf is None:
                    continue
                img = imread_robust(os.path.join(fdir, f))
                if img is None:
                    continue
                try:
                    norm = normalize_page(img, is_photo=True)
                except Exception:
                    continue
                add_samples(norm, tf, FIRSTNAME_Y, X, y, dropped)
                add_samples(norm, tn, NAME_Y, X, y, dropped)
                n_forms += 1

    if not X:
        print("[dataset] aucun echantillon ! (verifier verites terrain)")
        return
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "letter_dataset.npz")
    np.savez_compressed(out, X=X, y=y)
    # repartition par lettre
    counts = {chr(ord('A') + i): int((y == i).sum()) for i in range(26)}
    print(f"[dataset] {len(X)} lettres depuis {n_forms} formulaires "
          f"(noms ignores car comptage incoherent : {dropped[0]})")
    print("[dataset] repartition:",
          " ".join(f"{k}:{v}" for k, v in counts.items() if v))
    print(f"[dataset] sauvegarde : {out}")

    # apercu visuel (50 premieres)
    prev = X[:50]
    tiles = [cv2.copyMakeBorder((t * 255).astype(np.uint8), 1, 10, 1, 1,
             cv2.BORDER_CONSTANT, value=0) for t in prev]
    for t, lab in zip(tiles, y[:50]):
        cv2.putText(t, chr(ord('A') + lab), (2, t.shape[0] - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255, 1)
    rows = [np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1],
            cv2.BORDER_CONSTANT, value=0) for r in rows]
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "dataset_preview.png"), np.vstack(rows))
    print("[dataset] apercu : dataset_preview.png")


if __name__ == "__main__":
    main()
