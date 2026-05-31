"""
Script d'évaluation quantitative du pipeline.

Pour chaque FORM (FORM1, FORM2, FORM3) :
  - Charge la base de signatures.
  - Pour chaque photo .jpg/.jpeg/.png ET pour chaque PDF :
      * extrait le Student ID grille
      * extrait la signature et l'identifie
      * compare au "true ID" du nom de fichier (EXAM_FORMX_NNNNN.*)
  - Calcule les accuracies par axe et écrit un CSV récapitulatif.

Usage :
  python evaluate.py [chemin_vers_PROJECT_2026-DATABASE]
"""

import os
import re
import sys
import time
import csv

import cv2
import numpy as np

# Imports projet
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.grid_decoder    import (normalize_page, read_student_id,
                                    extract_signature_roi)
from utils.signature_utils import (load_signatures, build_descriptor_db,
                                    identify_signature)

try:
    import fitz   # pymupdf
except ImportError:
    fitz = None


def render_pdf_p1(pdf_path, dpi=150):
    if fitz is None:
        raise RuntimeError("pymupdf manquant")
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR)
    doc.close()
    return arr


def process(img, desc_db, is_photo):
    """Renvoie (grid_id, sig_id, sig_score)."""
    norm = normalize_page(img, is_photo=is_photo)
    grid_id = read_student_id(norm, is_photo=is_photo)
    sig_roi = extract_signature_roi(norm)
    sig_id, score = identify_signature(sig_roi, desc_db)
    return grid_id, sig_id, score


def evaluate_form(form_dir, desc_db, form_name, out_rows):
    files = sorted(os.listdir(form_dir))
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")

    n_photos = grid_ok_p = sig_ok_p = sig_none_p = 0
    n_pdfs   = grid_ok_d = sig_ok_d = sig_none_d = 0

    for f in files:
        m = re.match(rf"EXAM_{form_name}_(\d+)\.([a-zA-Z]+)", f)
        if not m:
            continue
        exp_id, ext = m.group(1), m.group(2).lower()
        if exp_id == "00000":
            continue  # intrus / non-listé

        path = os.path.join(form_dir, f)
        try:
            if ext in ("jpg", "jpeg", "png", "bmp"):
                img = cv2.imread(path)
                if img is None:
                    try:
                        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = None
                if img is None:
                    continue
                kind = "photo"
                grid_id, sig_id, sc = process(img, desc_db, is_photo=True)
                n_photos += 1
                if str(grid_id) == exp_id: grid_ok_p += 1
                if sig_id is None: sig_none_p += 1
                if sig_id == exp_id: sig_ok_p += 1
            elif ext == "pdf":
                img = render_pdf_p1(path)
                kind = "pdf"
                grid_id, sig_id, sc = process(img, desc_db, is_photo=False)
                n_pdfs += 1
                if str(grid_id) == exp_id: grid_ok_d += 1
                if sig_id is None: sig_none_d += 1
                if sig_id == exp_id: sig_ok_d += 1
            else:
                continue
        except Exception as e:
            out_rows.append([form_name, f, "", "ERR", "", "", str(e)])
            continue

        out_rows.append([form_name, f, kind, exp_id,
                         grid_id if grid_id is not None else "",
                         sig_id  if sig_id  is not None else "",
                         f"{sc:.3f}"])

    return {
        "form": form_name,
        "n_photos": n_photos,
        "grid_acc_photo":  100 * grid_ok_p / max(1, n_photos),
        "sig_acc_photo":   100 * sig_ok_p  / max(1, n_photos),
        "sig_none_photo":  100 * sig_none_p / max(1, n_photos),
        "n_pdfs": n_pdfs,
        "grid_acc_pdf":   100 * grid_ok_d / max(1, n_pdfs),
        "sig_acc_pdf":    100 * sig_ok_d  / max(1, n_pdfs),
        "sig_none_pdf":   100 * sig_none_d / max(1, n_pdfs),
    }


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 \
        else "PROJECT 2026 -DATABASE-20260518"

    print(f"[EVAL] Données : {data_root}")
    t0 = time.time()
    raw = load_signatures(os.path.join(data_root, "SIGNATURES"))
    print(f"[EVAL] Base : {len(raw)} étudiants, "
          f"{sum(len(v) for v in raw.values())} signatures")
    desc_db = build_descriptor_db(raw)
    print(f"[EVAL] Descripteurs construits en {time.time()-t0:.1f}s")

    out_rows = [["form", "file", "kind", "expected_id",
                 "grid_id", "sig_id", "sig_score"]]
    summaries = []
    for form in ("FORM1", "FORM2", "FORM3"):
        form_dir = os.path.join(data_root, form)
        if not os.path.isdir(form_dir):
            continue
        print(f"\n[EVAL] {form}...")
        t_f = time.time()
        s = evaluate_form(form_dir, desc_db, form, out_rows)
        s["elapsed"] = time.time() - t_f
        summaries.append(s)
        print(f"  photos n={s['n_photos']:3d} | grid {s['grid_acc_photo']:5.1f}% | "
              f"sig {s['sig_acc_photo']:5.1f}% (none {s['sig_none_photo']:5.1f}%)")
        print(f"  pdfs   n={s['n_pdfs']:3d} | grid {s['grid_acc_pdf']:5.1f}% | "
              f"sig {s['sig_acc_pdf']:5.1f}% (none {s['sig_none_pdf']:5.1f}%) "
              f"| {s['elapsed']:.1f}s")

    out_csv = "evaluation_results.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(out_rows)
    print(f"\n[EVAL] Détails écrits dans : {out_csv}")

    # Récapitulatif final
    print("\n" + "=" * 70)
    print(f"{'FORM':<8}{'kind':<8}{'n':>5} | "
          f"{'gridAcc':>9} | {'sigAcc':>8} | {'sigNone':>9}")
    print("-" * 70)
    for s in summaries:
        for kind in ("photo", "pdf"):
            print(f"{s['form']:<8}{kind:<8}{s['n_'+('photos' if kind=='photo' else 'pdfs')]:>5} | "
                  f"{s['grid_acc_'+kind]:>8.1f}% | {s['sig_acc_'+kind]:>7.1f}% | "
                  f"{s['sig_none_'+kind]:>8.1f}%")
    print("=" * 70)
    print(f"Temps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
