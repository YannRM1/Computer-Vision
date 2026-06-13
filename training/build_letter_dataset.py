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

# Crops bruts (niveaux de gris) collectes pour l'apercu lisible.
_PREVIEW_RAW = []


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


def is_single_token(name):
    """True si le nom est un seul mot contigu (sans espace ni tiret).

    Un nom compose (« AL BITAR », « LOU-ANN ») occupe une case VIDE a
    l'emplacement de l'espace/tiret : l'appariement par position decale alors
    TOUTES les etiquettes d'une case (la case vide recoit la lettre suivante).
    On exclut donc ces noms du jeu : mieux vaut ~5 % de donnees en moins que
    des labels faux."""
    return len(re.findall(r"[A-Za-z]+", name)) == 1


def _first_n_cells(norm, y_range, n):
    """Localise la suite contigue de n cases non vides correspondant au nom.

    Le fit de grille peut deriver d'une case a gauche (case de tete vide) ou
    a droite (premiere lettre hors cadre), et certaines photos donnent un fit
    faux. Plutot que de corriger ces derives au cas par cas, on evalue le
    vide de TOUTES les cases de la ligne et on exige une UNIQUE suite
    contigue de cases non vides de longueur exactement n : toute incoherence
    (ancrage decale, case parasite, lettre coupee) viole la condition et le
    formulaire est rejete -- aucun label corrompu ne peut entrer dans le jeu."""
    from utils.page1_parser import _fit_name_grid, _fit_name_band, NAME_CELLS
    from utils.ocr_utils import _to_gray
    y0, y1 = _fit_name_band(norm, y_range)
    x0, pitch = _fit_name_grid(norm, (y0, y1))

    def cell_at(k):
        xa = int(round(x0 + k * pitch))
        xb = int(round(x0 + (k + 1) * pitch))
        cell = norm[y0:y1, xa:xb]
        if cell.size == 0:
            return None
        g = _to_gray(cell)
        ch, cw = g.shape
        return g[3:max(4, ch - 3), 4:max(5, cw - 4)]

    cells = [cell_at(k) for k in range(NAME_CELLS)]
    # Garde-fou contraste : sur une case vide de photo, Otsu binarise le
    # bruit du capteur et prep_cell croit voir une lettre. Une vraie ecriture
    # contraste bien plus (mesure : vide std<=9, ecrit std>=46).
    filled = [c is not None and c.std() >= 15
              and letter_cnn.prep_cell(c) is not None
              for c in cells]

    runs, k = [], 0
    while k < NAME_CELLS:
        if filled[k]:
            j = k
            while j < NAME_CELLS and filled[j]:
                j += 1
            runs.append((k, j - k))
            k = j
        else:
            k += 1
    good = [s for s, length in runs if length == n]
    if len(good) != 1:
        return None
    return cells[good[0]:good[0] + n]


def add_samples(norm, name, y_range, X, y, groups, sid, dropped):
    """Extrait une case par lettre du nom (verite terrain) et l'ajoute au jeu.

    Appariement par POSITION (case k <-> lettre k) sur les n premieres cases,
    n = longueur du nom : robuste et a haut rendement (~1500 lettres sur les 3
    formulaires, contre ~120 avec l'ancien appariement par comptage exact).

    `groups` recoit l'identifiant etudiant de chaque lettre : indispensable
    pour la validation croisee par GROUPE (toutes les lettres d'un meme
    scripteur dans le meme fold, sinon le K-fold est optimiste : le reseau
    reconnait l'ecriture de la personne, pas la lettre)."""
    letters = letters_only(name)
    if not letters or len(letters) > 15:
        return
    if not is_single_token(name):       # nom compose -> etiquettes decalees
        dropped[0] += 1
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
        groups.append(int(sid))
        if len(_PREVIEW_RAW) < 70:          # crops bruts pour l'apercu lisible
            _PREVIEW_RAW.append((cell.copy(), ch))


def main():
    set_photo_template(get_photo_template(DATA))  # template figé (form identique)
    X, y, groups = [], [], []
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
                add_samples(norm, tf, FIRSTNAME_Y, X, y, groups, sid, dropped)
                add_samples(norm, tn, NAME_Y, X, y, groups, sid, dropped)
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
                add_samples(norm, tf, FIRSTNAME_Y, X, y, groups, sid, dropped)
                add_samples(norm, tn, NAME_Y, X, y, groups, sid, dropped)
                n_forms += 1

    if not X:
        print("[dataset] aucun echantillon ! (verifier verites terrain)")
        return
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups, dtype=np.int64)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "letter_dataset.npz")
    np.savez_compressed(out, X=X, y=y, groups=groups)
    # repartition par lettre
    counts = {chr(ord('A') + i): int((y == i).sum()) for i in range(26)}
    print(f"[dataset] {len(X)} lettres, {len(np.unique(groups))} etudiants, {n_forms} formulaires "
          f"(noms ignores car comptage incoherent : {dropped[0]})")
    print("[dataset] repartition:",
          " ".join(f"{k}:{v}" for k, v in counts.items() if v))
    print(f"[dataset] sauvegarde : {out}")

    # Apercu LISIBLE : crops bruts en niveaux de gris + label sous chaque case
    # (les entrees binarisees 28x28 du CNN sont peu parlantes pour un humain).
    TW, TH, LB, PER_ROW = 72, 84, 26, 14
    rows_img, row = [], []
    for cell, ch in _PREVIEW_RAW:
        g = cell if cell.ndim == 2 else cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        t = cv2.resize(g, (TW, TH), interpolation=cv2.INTER_CUBIC)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(t, (0, 0), (TW - 1, TH - 1), (180, 180, 180), 1)
        band = np.full((LB, TW, 3), 255, np.uint8)
        cv2.putText(band, ch, (TW // 2 - 8, LB - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 60, 140), 2)
        row.append(np.vstack([t, band]))
        if len(row) == PER_ROW:
            rows_img.append(np.hstack(row)); row = []
    if row:
        pad = [np.full((TH + LB, TW, 3), 255, np.uint8)] * (PER_ROW - len(row))
        rows_img.append(np.hstack(row + pad))
    if rows_img:
        canvas = cv2.copyMakeBorder(np.vstack(rows_img), 8, 8, 8, 8,
                                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "dataset_preview.png"), canvas)
    print("[dataset] apercu : dataset_preview.png")


if __name__ == "__main__":
    main()
