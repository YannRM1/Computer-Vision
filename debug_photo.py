"""
Script de debug : sauvegarde UNE photo normalisée avec le ROI dessiné dessus.
Lance : python debug_photo.py
Les images sont sauvegardées à côté de ce fichier (même dossier).
"""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.grid_decoder import normalize_page, ROI_STUDENT_ID_PHOTO, ROI_STUDENT_ID

BASE = os.path.dirname(os.path.abspath(__file__))
FORM1 = os.path.join(BASE, "PROJECT 2026 -DATABASE-20260518", "FORM1")

# Choisir les photos à débugger (ID attendu dans le nom de fichier)
TARGETS = [
    "EXAM_FORM1_62445.jpg",   # attendu : 62445
    "EXAM_FORM1_63694.jpg",   # attendu : 63694
    "EXAM_FORM1_62440.jpeg",  # attendu : 62440
    "EXAM_FORM1_54331.jpg",   # attendu : 54331
    "EXAM_FORM1_62401.jpg",   # attendu : 62401
]

for fname in TARGETS:
    path = os.path.join(FORM1, fname)
    if not os.path.exists(path):
        print(f"[SKIP] {fname} introuvable")
        continue

    # Charger l'image
    img = cv2.imread(path)
    if img is None:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        print(f"[WARN] Impossible de lire {fname}")
        continue

    # Normaliser
    print(f"Normalisation de {fname}...", end=" ", flush=True)
    norm = normalize_page(img, is_photo=True)
    vis  = norm.copy()

    # Dessiner le ROI PHOTO actuel en ROUGE épais
    x, y, w, h = ROI_STUDENT_ID_PHOTO
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
    cv2.putText(vis, f"PHOTO ROI ({x},{y},{w},{h})", (x, max(0, y-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Dessiner le ROI PDF en VERT (pour comparaison)
    px, py, pw, ph = ROI_STUDENT_ID
    cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 200, 0), 2)
    cv2.putText(vis, f"PDF ROI ({px},{py},{pw},{ph})", (px, max(0, py-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # Dessiner la grille (bleu) sur le ROI photo pour voir les cellules
    cell_w = w // 5
    cell_h = h // 10
    for col in range(6):
        cx = x + col * cell_w
        cv2.line(vis, (cx, y), (cx, y + h), (255, 100, 0), 1)
    for row in range(11):
        ry = y + row * cell_h
        cv2.line(vis, (x, ry), (x + w, ry), (255, 100, 0), 1)

    # Sauvegarder dans le même dossier que ce script
    out_name = f"DEBUG_{os.path.splitext(fname)[0]}.png"
    out_path = os.path.join(BASE, out_name)
    cv2.imwrite(out_path, vis)
    print(f"sauvegardé : {out_path}")

print("\nTerminé !")
print("Ouvre les images DEBUG_*.png dans Paint.")
print("La barre de statut en bas de Paint affiche les coordonnées (x, y) au survol.")
print("Cherche la grille de cases à cocher (5 colonnes × 10 lignes) en haut à droite.")
