"""
Script de calibration du ROI Student ID pour les photos.

Lance ce script UNE FOIS pour produire des images de diagnostic.
Il normalise chaque photo et dessine dessus :
  - Rectangle ROUGE  = ROI_STUDENT_ID_PHOTO (position actuelle, peut être décalée)
  - Rectangle VERT   = ROI_STUDENT_ID (même ROI que pour les PDFs, pour comparaison)
  - Grille BLEUE     = découpage en 5 colonnes × 10 lignes du ROI photo

Ouvre les images résultantes dans Paint (Windows) : la barre de statut affiche
les coordonnées (x, y) du curseur. Repère le coin supérieur-gauche de la grille
Student ID (les petites cases à cocher), note x et y, mesure la largeur w et
la hauteur h totale des 5 colonnes × 10 lignes, puis mets à jour dans
utils/grid_decoder.py :
    ROI_STUDENT_ID_PHOTO = (x, y, w, h)

Usage :
    python calibrate_roi.py
    python calibrate_roi.py EXAM_FORM1_62445.jpg   (image spécifique)
"""

import os
import sys
import cv2
import numpy as np

# Ajout du répertoire courant au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.grid_decoder import (
    normalize_page,
    ROI_STUDENT_ID,
    ROI_STUDENT_ID_PHOTO,
    STUDENT_ID_ROWS,
    STUDENT_ID_COLS,
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "PROJECT 2026 -DATABASE-20260518", "FORM1")
OUTPUT_DIR  = os.path.join(BASE_DIR, "calibration_output")
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"}
MAX_IMAGES  = 5   # nombre max d'images à traiter (ajuste si besoin)


def draw_roi(img, roi, color, label, thickness=2):
    x, y, w, h = roi
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    cv2.putText(img, label, (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_grid(img, roi, rows, cols, color):
    x, y, w, h = roi
    cell_w = w // cols
    cell_h = h // rows
    for r in range(rows + 1):
        y0 = y + r * cell_h
        cv2.line(img, (x, y0), (x + w, y0), color, 1)
    for c in range(cols + 1):
        x0 = x + c * cell_w
        cv2.line(img, (x0, y), (x0, y + h), color, 1)


def process_image(img_path, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        print(f"  [WARN] Impossible de lire : {img_path}")
        return

    norm = normalize_page(img, is_photo=True)
    vis  = norm.copy()

    # ROI actuel pour photos (rouge)
    draw_roi(vis, ROI_STUDENT_ID_PHOTO, (0, 0, 255), "Photo ROI (actuel)")
    # Grille découpée sur ce ROI (bleu)
    draw_grid(vis, ROI_STUDENT_ID_PHOTO, STUDENT_ID_ROWS, STUDENT_ID_COLS, (255, 100, 0))
    # ROI PDF pour comparaison (vert)
    draw_roi(vis, ROI_STUDENT_ID, (0, 200, 0), "PDF ROI (ref)")

    # Afficher les coordonnées du ROI photo en haut de l'image
    info = (f"ROI_PHOTO=({ROI_STUDENT_ID_PHOTO[0]},{ROI_STUDENT_ID_PHOTO[1]},"
            f"{ROI_STUDENT_ID_PHOTO[2]},{ROI_STUDENT_ID_PHOTO[3]})")
    cv2.putText(vis, info, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    name    = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(out_dir, f"norm_{name}.png")
    cv2.imwrite(out_path, vis)
    print(f"  -> {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(sys.argv) > 1:
        # Image spécifiée en argument
        paths = [os.path.join(DATA_DIR, sys.argv[1])
                 if not os.path.isabs(sys.argv[1]) else sys.argv[1]]
    else:
        # Prendre les premières MAX_IMAGES photos du répertoire
        all_files = sorted([
            f for f in os.listdir(DATA_DIR)
            if os.path.splitext(f)[1] in IMG_EXTS
        ])
        paths = [os.path.join(DATA_DIR, f) for f in all_files[:MAX_IMAGES]]

    print(f"Traitement de {len(paths)} image(s)...")
    for p in paths:
        print(f"  {os.path.basename(p)}")
        process_image(p, OUTPUT_DIR)

    print(f"\nImages sauvegardées dans : {OUTPUT_DIR}")
    print("\nComment mesurer le bon ROI :")
    print("  1. Ouvre une image dans Paint (clic-droit → Ouvrir avec → Paint)")
    print("  2. Survole le coin supérieur-gauche de la grille de cases Student ID")
    print("     (les petites cases à cocher alignées en 5 colonnes × 10 lignes)")
    print("  3. Lis les coordonnées x,y dans la barre de statut en bas à gauche")
    print("  4. Mesure la largeur w (de la 1re à la 5e colonne) et")
    print("     la hauteur h (de la 1re à la 10e ligne)")
    print("  5. Mets à jour dans utils/grid_decoder.py :")
    print("     ROI_STUDENT_ID_PHOTO = (x, y, w, h)")


if __name__ == "__main__":
    main()
