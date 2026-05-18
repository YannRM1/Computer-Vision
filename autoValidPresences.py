"""
PROGRAMME 1 : Validation des présences

Fonctions principales :
  autoValidPresences(presences_dir, signatures_dir, results_dir)
  autoValidID(img_path, desc_db, output_xlsx_path, results_dir)

Produit : EXAM_FORMXX_PRESENCES.xlsx
  Colonnes : imageName | studentID_grid | studentID_signature
"""

import os
import cv2
import numpy as np
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font

from utils.form_aligner import deskew
from utils.grid_decoder  import normalize_page, read_student_id, extract_signature_roi
from utils.signature_utils import (
    load_signatures,
    build_descriptor_db,
    identify_signature,
)

# Extensions d'images acceptées
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Initialisation du fichier Excel résultat (PRESENCES)
# ---------------------------------------------------------------------------

def _init_presence_xlsx(xlsx_path: str) -> Workbook:
    wb = Workbook()
    ws = wb.active
    ws.title = "PRESENCES"
    headers = ["imageName", "studentID_grid", "studentID_signature"]
    for col, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.font = Font(bold=True)
    wb.save(xlsx_path)
    return wb


def _append_presence_row(wb: Workbook, xlsx_path: str,
                          image_name: str,
                          id_grid: int | None,
                          id_sig: str | None) -> None:
    ws = wb.active
    ws.append([image_name,
               id_grid if id_grid is not None else "",
               int(id_sig) if id_sig and id_sig.isdigit() else (id_sig or "")])
    wb.save(xlsx_path)


# ---------------------------------------------------------------------------
# Traitement d'une image (sous-fonction)
# ---------------------------------------------------------------------------

def autoValidID(img_path: str,
                desc_db: dict,
                xlsx_path: str,
                results_dir: str,
                wb: Workbook | None = None) -> tuple[int | None, str | None]:
    """
    Traite une image de première page et met à jour le xlsx.

    Returns:
        (studentID_grid, studentID_signature)
        studentID_signature est l'ID reconnu par la signature, ou None.
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] Impossible de lire : {img_path}")
        return None, None

    # 1. Deskew (correction d'inclinaison pour les photos)
    img = deskew(img)

    # 2. Normaliser vers le repère du formulaire
    norm = normalize_page(img)

    # 3. Lire le StudentID depuis la grille
    student_id_grid = read_student_id(norm)

    # 4. Extraire la sous-image de signature
    sig_img = extract_signature_roi(norm)

    # 5. Comparer la signature à la base de données
    if sig_img is not None and sig_img.size > 100 and desc_db:
        id_sig, score = identify_signature(sig_img, desc_db, threshold=0.72)
    else:
        id_sig = None

    # 6. Écrire dans le xlsx
    image_name = os.path.basename(img_path)
    if wb is not None:
        _append_presence_row(wb, xlsx_path, image_name,
                             student_id_grid, id_sig)

    print(f"  {image_name}: grid={student_id_grid}, sig={id_sig}")
    return student_id_grid, id_sig


# ---------------------------------------------------------------------------
# Fonction principale Programme 1
# ---------------------------------------------------------------------------

def autoValidPresences(presences_dir: str,
                       signatures_dir: str,
                       results_dir: str) -> str:
    """
    Valide les présences pour tous les eleves.

    Args:
        presences_dir  : répertoire contenant les photos de première page
        signatures_dir : répertoire / ZIP avec la base de signatures
        results_dir    : répertoire de sortie

    Returns:
        Chemin vers le fichier XLSX généré.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Nom du fichier de sortie basé sur le répertoire source
    form_name = os.path.basename(presences_dir.rstrip("/\\"))
    xlsx_filename = form_name + "_PRESENCES.xlsx"
    xlsx_path = os.path.join(results_dir, xlsx_filename)

    print(f"[P1] Chargement des signatures depuis : {signatures_dir}")
    raw_db  = load_signatures(signatures_dir)
    print(f"  -> {len(raw_db)} eleves chargés ({sum(len(v) for v in raw_db.values())} signatures)")
    desc_db = build_descriptor_db(raw_db)

    # Créer le fichier xlsx
    wb = _init_presence_xlsx(xlsx_path)

    # Lister les images de présence
    images = sorted([
        f for f in os.listdir(presences_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ])
    print(f"[P1] {len(images)} images à traiter dans : {presences_dir}")

    for img_name in images:
        img_path = os.path.join(presences_dir, img_name)
        autoValidID(img_path, desc_db, xlsx_path, results_dir, wb=wb)

    print(f"[P1] Résultats sauvegardés : {xlsx_path}")
    return xlsx_path


# ---------------------------------------------------------------------------
# Point d'entrée direct (test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 4:
        autoValidPresences(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Usage: python autoValidPresences.py <presences_dir> <sig_dir> <results_dir>")
