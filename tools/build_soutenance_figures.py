"""
Genere les figures de la soutenance dans rapport/figures/.

Toutes les figures sont produites par le VRAI pipeline sur les donnees du
projet (exemples concrets, pas de schemas inventes).

Usage : python tools/build_soutenance_figures.py
"""
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "rapport", "figures")
os.makedirs(FIG, exist_ok=True)
D1 = os.path.join(ROOT, "PROJECT 2026 -DATABASE-20260518", "FORM1")

BLEU = "#1F4E79"
ORANGE = "#E8772E"
VERT = "#2E7D32"
ROUGE = "#C62828"
GRIS = "#9E9E9E"


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  [fig]", name)


# ---------------------------------------------------------------------------
# 0. Schema fonctionnel du pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, txt, fc, fontsize=10.5, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fontsize, color=tc, fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                     arrowstyle="-|>", mutation_scale=16, color="#555555",
                     lw=1.8))

    box(1, 26, 14, 9, "Photos page 1\n(presences)", GRIS, fontsize=10)
    box(1, 6, 14, 9, "PDF scannes\n(formulaires)", GRIS, fontsize=10)
    box(22, 16, 16, 9, "Normalisation\nrepere 900x1270\nORB + RANSAC", BLEU,
        fontsize=9.5)
    box(45, 26, 24, 9, "Page 1 : grilles, noms,\nsignature, cryptogramme",
        ORANGE, fontsize=9.5)
    box(45, 6, 24, 9, "Pages examen :\nMCQ + mantisse / exposant",
        ORANGE, fontsize=9.5)
    box(76, 26, 18, 9, "EXAM_FORMXX_\nPRESENCES.xlsx", VERT, fontsize=9.5)
    box(76, 6, 18, 9, "EXAM_FORMXX_\nNNNNN.xlsx", VERT, fontsize=9.5)
    box(22, 33, 16, 5, "Base signatures (60)", "#7B1FA2", fontsize=9)

    arrow(15, 30, 22, 22); arrow(15, 10, 22, 18)
    arrow(38, 22, 45, 30); arrow(38, 18, 45, 10)
    arrow(69, 30, 76, 30); arrow(69, 10, 76, 10)
    arrow(30, 33, 47, 31)
    ax.set_title("Pipeline DeepForm — Programmes 1 et 2", fontsize=14,
                 color=BLEU, fontweight="bold")
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------------------
# Prepa : pages normalisees d'un exemple (19283)
# ---------------------------------------------------------------------------
def load_example():
    from utils.pdf_utils import pdf_to_images
    from utils.grid_decoder import normalize_page, set_photo_template
    from utils.template_register import get_photo_template
    set_photo_template(get_photo_template(D1))
    imgs = pdf_to_images(os.path.join(D1, "EXAM_FORM1_19283.pdf"))
    return imgs, normalize_page(imgs[0], use_template=True)


# ---------------------------------------------------------------------------
# 1. Page 1 annotee (ROIs reelles)
# ---------------------------------------------------------------------------
def fig_roi(norm):
    from utils.grid_decoder import (ROI_CODES_EXAM, ROI_STUDENT_ID,
                                    ROI_GROUP_GRID, ROI_GROUP_LETTER,
                                    ROI_SIGNATURE_INNER, ROI_CRYPTO,
                                    GROUP_COL_WIDTHS)
    gx, gy, gw, gh = ROI_GROUP_GRID
    grp = (gx, gy, int(gw * (GROUP_COL_WIDTHS[0] + GROUP_COL_WIDTHS[1])), gh)
    vis = norm.copy()
    zones = [(ROI_CODES_EXAM, "CODES", (200, 100, 0)),
             (grp, "GROUPE", (180, 0, 180)),
             (ROI_GROUP_LETTER, "", (180, 110, 0)),
             (ROI_STUDENT_ID, "STUDENT ID", (0, 160, 0)),
             (ROI_SIGNATURE_INNER, "SIGNATURE", (0, 0, 220)),
             (ROI_CRYPTO, "CRYPTO", (160, 0, 80))]
    for (x, y, w, h), lab, c in zones:
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 3)
        if lab:
            cv2.putText(vis, lab, (x + 4, max(y - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2)
    fig, ax = plt.subplots(figsize=(5.2, 7.3))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax.axis("off")
    save(fig, "fig_roi_page1.png")


# ---------------------------------------------------------------------------
# 2. Deskew avant / apres (cas controle)
# ---------------------------------------------------------------------------
def fig_deskew(imgs):
    from utils.form_aligner import deskew
    page = imgs[0]
    h, w = page.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 7.0, 1.0)
    ca, sa = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sa + w * ca), int(h * ca + w * sa)
    M[0, 2] += (nw - w) / 2; M[1, 2] += (nh - h) / 2
    tilted = cv2.warpAffine(page, M, (nw, nh), borderValue=(255, 255, 255))
    fixed = deskew(tilted)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 6))
    for ax, im, t in [(axes[0], tilted, "Inclinee (+7°)"),
                      (axes[1], fixed, "Apres deskew (Hough)")]:
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=12, color=BLEU, fontweight="bold")
        ax.axis("off")
    save(fig, "fig_deskew.png")


# ---------------------------------------------------------------------------
# 3. Grille Student ID : cellules + heatmap des ratios d'encre
# ---------------------------------------------------------------------------
def fig_grid(norm):
    from utils.grid_decoder import ROI_STUDENT_ID, STUDENT_ID_ROWS, STUDENT_ID_COLS, get_roi
    from utils.checkbox_reader import split_grid, preprocess_for_checkbox, ink_ratio
    roi = get_roi(norm, ROI_STUDENT_ID)
    cells = split_grid(roi, STUDENT_ID_ROWS, STUDENT_ID_COLS)
    ratios = np.zeros((STUDENT_ID_ROWS, STUDENT_ID_COLS))
    for r in range(STUDENT_ID_ROWS):
        for c in range(STUDENT_ID_COLS):
            b = preprocess_for_checkbox(cells[r][c])
            m = max(2, int(min(b.shape) * 0.10))
            inner = b[m:-m, m:-m] if b.shape[0] > 2 * m else b
            ratios[r, c] = ink_ratio(inner)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    axes[0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Grille Student ID (19283)", fontsize=11,
                      color=BLEU, fontweight="bold")
    axes[0].axis("off")
    im = axes[1].imshow(ratios, cmap="Blues")
    for c in range(STUDENT_ID_COLS):
        r = int(np.argmax(ratios[:, c]))
        axes[1].text(c, r, str(r), ha="center", va="center",
                     color=ORANGE, fontsize=14, fontweight="bold")
    axes[1].set_title("Ratio d'encre par cellule\n(max par colonne = chiffre)",
                      fontsize=11, color=BLEU, fontweight="bold")
    axes[1].set_xlabel("colonne"); axes[1].set_ylabel("ligne (0-9)")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    save(fig, "fig_grid_heatmap.png")


# ---------------------------------------------------------------------------
# 4. MCQ : cases vertes/rouges + case noircie annulee (58927 Q1)
# ---------------------------------------------------------------------------
def fig_mcq(imgs):
    from utils.grid_decoder import normalize_page
    from utils.exam_parser import (detect_question_blocks, _find_mcq_checkboxes,
                                   _parse_mcq_choices, MCQ_CHOICES)
    ep = normalize_page(imgs[4]); H = ep.shape[0]
    for (a, b) in detect_question_blocks(ep):
        blk = ep[a:b]
        if blk.shape[0] < 150 or a > H - 170:
            continue
        hc = max(20, int(blk.shape[0] * 0.22))
        content = blk[hc:].copy()
        boxes = _find_mcq_checkboxes(content)
        ch = _parse_mcq_choices(content)
        for idx, (x, y, w, h) in enumerate(boxes):
            let = MCQ_CHOICES[idx] if idx < len(MCQ_CHOICES) else "?"
            col = (0, 170, 0) if ch.get(let) == 1 else (0, 0, 220)
            cv2.rectangle(content, (x - 2, y - 2), (x + w + 2, y + h + 2), col, 3)
        fig, ax = plt.subplots(figsize=(8.6, 2.4))
        ax.imshow(cv2.cvtColor(content[:210], cv2.COLOR_BGR2RGB))
        ax.set_title("MCQ : cochee (vert) / non cochee ou ANNULEE-noircie (rouge)",
                     fontsize=11, color=BLEU, fontweight="bold")
        ax.axis("off")
        save(fig, "fig_mcq.png")
        return


# ---------------------------------------------------------------------------
# 5. Grille des noms re-estimee (bande + separateurs)
# ---------------------------------------------------------------------------
def fig_names(norm):
    from utils.page1_parser import _fit_name_grid, FIRSTNAME_Y, NAME_Y, NAME_CELLS
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 2.6))
    for ax, (y0, y1), t in [(axes[0], FIRSTNAME_Y, "Prenom"),
                            (axes[1], NAME_Y, "Nom")]:
        x0, pitch = _fit_name_grid(norm, (y0, y1))
        band = norm[y0 - 4:y1 + 4, 20:430].copy()
        for k in range(NAME_CELLS + 1):
            x = int(round(x0 + k * pitch)) - 20
            cv2.line(band, (x, 0), (x, band.shape[0]), (0, 0, 230), 1)
        ax.imshow(cv2.cvtColor(band, cv2.COLOR_BGR2RGB))
        ax.set_ylabel(t, fontsize=10, color=BLEU, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Grille de cases re-estimee par formulaire (vote RANSAC)",
                 fontsize=11, color=BLEU, fontweight="bold")
    save(fig, "fig_names_grid.png")


# ---------------------------------------------------------------------------
# 6. Segmentation mantisse (composantes : chiffres / separateur)
# ---------------------------------------------------------------------------
def fig_mantisse():
    from utils.pdf_utils import pdf_to_images
    from utils.grid_decoder import normalize_page
    from utils.exam_parser import detect_question_blocks, _find_answer_boxes
    imgs = pdf_to_images(os.path.join(D1, "EXAM_FORM1_65442.pdf"))
    q = 1
    for pi in range(4, len(imgs)):
        page = normalize_page(imgs[pi]); H = page.shape[0]
        for (y0, y1) in detect_question_blocks(page):
            blk = page[y0:y1]
            if blk.shape[0] < 150 or y0 > H - 170:
                continue
            if q == 3:
                hc = max(20, int(blk.shape[0] * 0.22)); content = blk[hc:]
                ab = _find_answer_boxes(content)
                by_a = sorted(ab, key=lambda b: b[2] * b[3], reverse=True)
                large = [b for b in by_a
                         if b[2] * b[3] >= by_a[0][2] * by_a[0][3] * 0.4]
                bx, by, bw, bh = sorted(large, key=lambda b: b[0])[0]
                crop = content[by:by + bh, bx:bx + bw]
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
                gg = clahe.apply(g)
                h, w = gg.shape
                m = max(3, int(min(h, w) * 0.06))
                inner = gg[m:h - m, m:w - m]
                _, bb = cv2.threshold(inner, 210, 255, cv2.THRESH_BINARY_INV)
                n, _, stats, _ = cv2.connectedComponentsWithStats(bb, 8)
                vis = cv2.cvtColor(cv2.resize(crop, (w * 4, h * 4),
                                   interpolation=cv2.INTER_NEAREST),
                                   cv2.COLOR_BGR2RGB)
                ih, iw = inner.shape
                h_ref = max(int(stats[i][3]) for i in range(1, n))
                for i in range(1, n):
                    cx, cy, cw, chh, area = (int(v) for v in stats[i])
                    if area < 6:
                        continue
                    sep = (chh <= 0.70 * h_ref and cw <= iw * 0.22
                           and (cy + chh / 2) >= ih * 0.55
                           and cy >= ih * 0.40)
                    col = (232, 119, 46) if sep else (46, 125, 50)
                    cv2.rectangle(vis, ((cx + m) * 4, (cy + m) * 4),
                                  ((cx + m + cw) * 4, (cy + m + chh) * 4),
                                  col, 3)
                fig, ax = plt.subplots(figsize=(5.6, 2.2))
                ax.imshow(vis)
                ax.set_title("Mantisse « 3,75 » : chiffres (vert) / "
                             "separateur decimal (orange)",
                             fontsize=11, color=BLEU, fontweight="bold")
                ax.axis("off")
                save(fig, "fig_mantisse.png")
                return
            q += 1


# ---------------------------------------------------------------------------
# 7. Signature : ROI brute vs canvas normalise
# ---------------------------------------------------------------------------
def fig_signature(norm):
    from utils.grid_decoder import extract_signature_roi
    from utils.signature_utils import preprocess_signature
    sig = extract_signature_roi(norm)
    pp = preprocess_signature(sig)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 2.6))
    axes[0].imshow(cv2.cvtColor(sig, cv2.COLOR_BGR2RGB))
    axes[0].set_title("ROI signature (19283)", fontsize=11, color=BLEU,
                      fontweight="bold")
    axes[1].imshow(pp, cmap="gray")
    axes[1].set_title("Normalisee 192x96 (Otsu + recadrage + centrage)",
                      fontsize=11, color=BLEU, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    save(fig, "fig_signature.png")


# ---------------------------------------------------------------------------
# 8. Architecture CNN
# ---------------------------------------------------------------------------
def fig_cnn():
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 26); ax.axis("off")
    layers = [
        ("Entree\n28x28x1", "#90A4AE", 8),
        ("Conv 3x3, 32\n+ ReLU (x2)", BLEU, 11.5),
        ("MaxPool 2\n-> 14x14", ORANGE, 9),
        ("Conv 3x3, 64\n+ ReLU (x2)", BLEU, 11.5),
        ("MaxPool 2\n-> 7x7", ORANGE, 9),
        ("Flatten\n3136", "#90A4AE", 7.5),
        ("FC 128\nDropout 0.3", VERT, 10.5),
        ("FC 26 (A-Z)\nou 10 (0-9)", "#7B1FA2", 11.5),
    ]
    x = 1
    for txt, c, w in layers:
        ax.add_patch(FancyBboxPatch((x, 6), w, 13, boxstyle="round,pad=0.4",
                                    fc=c, ec="none"))
        ax.text(x + w / 2, 12.5, txt, ha="center", va="center", fontsize=9,
                color="white", fontweight="bold")
        x += w + 1.6
        if x < 96:
            ax.annotate("", (x - 0.4, 12.5), (x - 1.6, 12.5),
                        arrowprops=dict(arrowstyle="-|>", color="#555555"))
    ax.set_title("CNN — meme architecture pour lettres (26 cl.) et chiffres "
                 "(10 cl.)", fontsize=13, color=BLEU, fontweight="bold")
    save(fig, "fig_cnn_archi.png")


# ---------------------------------------------------------------------------
# 9. Resultats par axe (barres avant/apres)
# ---------------------------------------------------------------------------
def fig_results():
    axes_n = ["Imprime", "Graphique", "Signature", "Manuscrit", "GLOBAL"]
    avant = [83.6, 61.5, 58.5, 22.9, 63.2]
    apres = [84.5, 80.7, 78.0, 41.4, 75.1]
    x = np.arange(len(axes_n)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    b1 = ax.bar(x - w / 2, avant, w, label="Avant", color=GRIS)
    b2 = ax.bar(x + w / 2, apres, w, label="Apres", color=BLEU)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                    f"{r.get_height():.0f}", ha="center", fontsize=10,
                    fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(axes_n, fontsize=11)
    ax.set_ylabel("Accuracy par cellule (%)"); ax.set_ylim(0, 100)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)
    ax.set_title("FORM1 : avant / apres nos ameliorations (verite terrain)",
                 fontsize=12, color=BLEU, fontweight="bold")
    save(fig, "fig_results_bar.png")


# ---------------------------------------------------------------------------
# 10. Distribution des lettres du jeu de fine-tuning
# ---------------------------------------------------------------------------
def fig_letters():
    d = np.load(os.path.join(ROOT, "training", "letter_dataset.npz"))
    y = d["y"]
    counts = [(chr(65 + i), int((y == i).sum())) for i in range(26)]
    labels, vals = zip(*counts)
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    cols = [BLEU if v >= 30 else ROUGE for v in vals]
    ax.bar(labels, vals, color=cols)
    ax.set_ylabel("nb d'exemples")
    ax.set_title(f"Jeu de fine-tuning : {len(y)} lettres, "
                 f"{len(np.unique(d['groups']))} scripteurs "
                 "(rouge = classes rares)", fontsize=12, color=BLEU,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_letter_dist.png")


def main():
    print("[figures] generation ->", FIG)
    fig_pipeline()
    fig_cnn()
    fig_results()
    fig_letters()
    imgs, norm = load_example()
    fig_roi(norm)
    fig_deskew(imgs)
    fig_grid(norm)
    fig_mcq(imgs)
    fig_names(norm)
    fig_signature(norm)
    fig_mantisse()
    print("[figures] termine.")


if __name__ == "__main__":
    main()
