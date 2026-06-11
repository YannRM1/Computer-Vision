"""
Genere le PowerPoint de soutenance (rapport/Soutenance_IG2405_DeepForm.pptx).

Calibre sur le timing impose par le sujet (sec. 7.2) : 25 min de presentation
+ 15 min de questions. Le minutage de chaque slide est dans les notes orateur.
Les figures viennent de rapport/figures/ (tools/build_soutenance_figures.py),
toutes produites par le vrai pipeline sur les donnees du projet.

Usage :
    python tools/build_soutenance_figures.py   # 1. genere les figures
    python tools/build_soutenance_pptx.py      # 2. genere le pptx
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "rapport", "figures")
OUT = os.path.join(ROOT, "rapport", "Soutenance_IG2405_DeepForm.pptx")

BLEU = RGBColor(0x1F, 0x4E, 0x79)
BLEU_CLAIR = RGBColor(0xDC, 0xE6, 0xF1)
ORANGE = RGBColor(0xE8, 0x77, 0x2E)
GRIS_F = RGBColor(0x37, 0x37, 0x37)
GRIS = RGBColor(0x6E, 0x6E, 0x6E)
BLANC = RGBColor(0xFF, 0xFF, 0xFF)
VERT = RGBColor(0x2E, 0x7D, 0x32)

SW, SH = Inches(13.333), Inches(7.5)
_counter = {"n": 0}


def _rect(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _chrome(slide, title, minutes):
    """Bandeau titre bleu + liseret orange + pied de page."""
    _rect(slide, 0, 0, SW, Inches(1.0), BLEU)
    _rect(slide, 0, Inches(1.0), SW, Inches(0.07), ORANGE)
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.13),
                                  Inches(10.6), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(28); p.font.bold = True; p.font.color.rgb = BLANC

    if minutes:
        chip = _rect(slide, Inches(11.55), Inches(0.28),
                     Inches(1.45), Inches(0.46), ORANGE)
        tf = chip.text_frame
        tf.paragraphs[0].text = minutes
        tf.paragraphs[0].font.size = Pt(13)
        tf.paragraphs[0].font.bold = True
        tf.paragraphs[0].font.color.rgb = BLANC
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER

    _counter["n"] += 1
    ft = slide.shapes.add_textbox(Inches(0.4), Inches(7.12),
                                  Inches(12.5), Inches(0.32))
    p = ft.text_frame.paragraphs[0]
    p.text = f"IG.2405 — DeepForm — Soutenance        {_counter['n']}"
    p.font.size = Pt(10); p.font.color.rgb = GRIS


def _bullets(slide, items, x, y, w, h, size=18):
    body = slide.shapes.add_textbox(x, y, w, h)
    tf = body.text_frame
    tf.word_wrap = True
    first = True
    for b in items:
        lvl, txt = b if isinstance(b, tuple) else (0, b)
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.text = ("▪  " if lvl == 0 else "–  ") + txt
        para.level = lvl
        para.font.size = Pt(size if lvl == 0 else size - 3)
        para.font.color.rgb = GRIS_F if lvl == 0 else GRIS
        para.space_after = Pt(7)
    return body


def _fig(slide, name, x, y, w):
    path = os.path.join(FIG, name)
    if os.path.isfile(path):
        return slide.shapes.add_picture(path, x, y, width=w)
    tb = slide.shapes.add_textbox(x, y, w, Inches(1))
    tb.text_frame.paragraphs[0].text = f"[figure manquante : {name}]"
    return tb


def slide_fig_right(prs, title, minutes, bullets, figname, note,
                    fig_w=5.6, bsize=17):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, title, minutes)
    _bullets(s, bullets, Inches(0.45), Inches(1.35),
             Inches(12.4 - fig_w - 0.6), Inches(5.5), size=bsize)
    _fig(s, figname, Inches(12.95 - fig_w), Inches(1.45), Inches(fig_w))
    s.notes_slide.notes_text_frame.text = note
    return s


def slide_fig_bottom(prs, title, minutes, bullets, figname, note,
                     fig_w=8.8, bsize=17):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, title, minutes)
    _bullets(s, bullets, Inches(0.45), Inches(1.3),
             Inches(12.4), Inches(2.3), size=bsize)
    pic = _fig(s, figname, Inches(0.6), Inches(3.85), Inches(fig_w))
    # centre horizontalement
    try:
        pic.left = int((SW - pic.width) / 2)
    except Exception:
        pass
    s.notes_slide.notes_text_frame.text = note
    return s


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH

    # ===== 1. TITRE (0:00 -> 0:30) ==========================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, SW, SH, BLEU)
    _rect(s, 0, Inches(4.45), SW, Inches(0.09), ORANGE)
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(2))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "DeepForm"
    p.font.size = Pt(60); p.font.bold = True; p.font.color.rgb = BLANC
    p2 = tf.add_paragraph()
    p2.text = "Lecture automatique de formulaires d'examens semi-structures"
    p2.font.size = Pt(26); p2.font.color.rgb = BLEU_CLAIR
    tb2 = s.shapes.add_textbox(Inches(0.9), Inches(4.8), Inches(11.5), Inches(1.4))
    tf2 = tb2.text_frame
    p = tf2.paragraphs[0]
    p.text = "Projet Computer Vision IG.2405 — 2026"
    p.font.size = Pt(20); p.font.color.rgb = BLANC
    p2 = tf2.add_paragraph()
    p2.text = "Equipe : [Prenom NOM x4]"
    p2.font.size = Pt(18); p2.font.color.rgb = BLEU_CLAIR
    s.notes_slide.notes_text_frame.text = (
        "0:00 -> 0:30 (30 s) — Orateur 1\n"
        "Saluer, presenter l'equipe, annoncer le plan en une phrase.")

    # ===== 2. PROBLEME (0:30 -> 2:30) =======================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Position du probleme", "2 min")
    _bullets(s, [
        "Formulaires SEMI-structures : cases a cocher, grilles, champs manuscrits",
        "Programme 1 — presences : photo page 1 -> Student ID (grille) + signature authentifiee",
        "Programme 2 — lecture : PDF scanne -> un Excel par eleve (PAGE-01 + EXAM)",
    ], Inches(0.45), Inches(1.3), Inches(12.4), Inches(2.2), size=18)
    # encadre contrainte
    box = _rect(s, Inches(0.6), Inches(3.6), Inches(12.1), Inches(2.6), BLEU_CLAIR)
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Contrainte centrale (sec. 4.1)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = BLEU
    for txt in [
        "Elements GRAPHIQUES  ->  bas niveau uniquement : Otsu, morphologie, Hough, composantes connexes, correlation",
        "TEXTES imprimes / manuscrits  ->  haut niveau autorise : OCR, CNN fine-tune",
        "Challenge evalue sur 4 axes : signatures, imprime, manuscrit, graphique",
    ]:
        pp = tf.add_paragraph()
        pp.text = "▪  " + txt
        pp.font.size = Pt(15); pp.font.color.rgb = GRIS_F
    s.notes_slide.notes_text_frame.text = (
        "0:30 -> 2:30 (2 min) — Orateur 1\n"
        "Insister sur la contrainte bas/haut niveau : elle structure toutes "
        "nos decisions techniques.")

    # ===== 3. PIPELINE (2:30 -> 4:30) =======================================
    slide_fig_bottom(prs, "Vue systemique du pipeline", "2 min", [
        "Un seul repere canonique 900 x 1270 px pour photos ET PDF -> ROIs calibrees une fois",
        "Photos : recalage sur gabarit par points ORB + homographie RANSAC",
        "Sorties strictement conformes au cahier des charges (xlsx imposes)",
    ], "fig_pipeline.png",
        "2:30 -> 4:30 (2 min) — Orateur 1\n"
        "Derouler le schema bloc par bloc, de gauche a droite.",
        fig_w=10.6)

    # ===== 4. NORMALISATION (4:30 -> 7:00) ==================================
    slide_fig_right(prs, "Normalisation et robustesse", "2 min 30", [
        "Deskew : Hough -> angle dominant -> rotation inverse (cas controle +7° ci-contre)",
        "Recalage photo : ORB (6000 pts) + RANSAC",
        (1, "garde-fou : >= 30 inliers sinon repli (stoppe les recalages aberrants)"),
        "Entrees reelles encaissees :",
        (1, "HEIC deguises en .jpg -> pillow-heif"),
        (1, "photo de 8 pages empilees -> crop auto page 1"),
        (1, "photos a l'envers -> ORB invariant en rotation"),
        (1, "dates OCR aberrantes (52-01-62) rejetees"),
    ], "fig_deskew.png",
        "4:30 -> 7:00 (2 min 30) — Orateur 2\n"
        "Figure = cas controle : page droite inclinee artificiellement puis "
        "corrigee. Les 4 puces robustesse = cas reels de la base.",
        fig_w=5.4, bsize=16)

    # ===== 5. GRILLES (7:00 -> 9:00) ========================================
    slide_fig_right(prs, "Grilles Student ID / Groupe (bas niveau)", "2 min", [
        "Decoupage en cellules 10 x 5",
        "Binarisation Otsu -> ratio d'encre interieur",
        "Decision : maximum par colonne + seuil plancher anti-bruit",
        "Exemple ci-contre : 1-9-2-8-3 lu sans OCR",
        "Student ID : 90 % (PDF)",
    ], "fig_grid_heatmap.png",
        "7:00 -> 9:00 (2 min) — Orateur 2\n"
        "La heatmap montre que la decision est triviale une fois le ratio "
        "d'encre calcule : 100 % bas niveau (sec. 4.1).",
        fig_w=6.8, bsize=17)

    # ===== 6. MCQ (9:00 -> 11:00) ===========================================
    slide_fig_bottom(prs, "Cases MCQ : detection et decision", "2 min", [
        "Composantes connexes + filtre COLONNE ALIGNEE (vire les fragments d'enonce)",
        "Decision RELATIVE par question : cochee si encre >> mediane des cases (rattrape les coches legeres)",
        "Case entierement noircie (ratio ~0,9) = choix ANNULE par l'eleve, vs croix ~0,3-0,55  ->  MCQ : 61 % -> 79 %",
    ], "fig_mcq.png",
        "9:00 -> 11:00 (2 min) — Orateur 2\n"
        "Sur la figure : B cochee (vert), C entierement noircie = annulee "
        "correctement rejetee (rouge). Critere relatif = robuste a "
        "l'epaisseur du trait.",
        fig_w=9.6)

    # ===== 7. SIGNATURES (11:00 -> 13:00) ===================================
    slide_fig_bottom(prs, "Authentification des signatures", "2 min", [
        "Pretraitement : Otsu -> retrait des barres de cadre -> recadrage -> canvas 192x96 centre (centre de masse)",
        "Score = 0,6 x NCC + 0,35 x cos(HOG) + 0,05 x cos(Hu) ; max sur les references de chaque eleve (base 60)",
        "Bug corrige : le nettoyage effacait des signatures entieres (1 grande composante touchant le bord)  ->  58 % -> 78 %",
    ], "fig_signature.png",
        "11:00 -> 13:00 (2 min) — Orateur 3\n"
        "L'anecdote du bug 'le nettoyage de cadre effacait la signature "
        "cursive entiere' passe tres bien a l'oral.",
        fig_w=8.6)

    # ===== 8. IMPRIME (13:00 -> 14:30) ======================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Textes imprimes (OCR)", "1 min 30")
    _bullets(s, [
        "easyOCR sur image egalisee (CLAHE) et agrandie x4",
        "Extraction par motifs : Module, Professor, Date, Code",
        "Date : double source, l'en-tete haut de page prioritaire (valeurs reelles de l'examen)",
        "Validation jour <= 31 / mois <= 12 : rejette les lectures aberrantes",
        "Axe imprime : 84,5 % (FORM1) — 86,7 % (3 formulaires)",
    ], Inches(0.45), Inches(1.4), Inches(7.6), Inches(5.2), size=18)
    _fig(s, "fig_roi_page1.png", Inches(8.3), Inches(1.25), Inches(4.4))
    s.notes_slide.notes_text_frame.text = (
        "13:00 -> 14:30 (1 min 30) — Orateur 3\n"
        "Axe le plus simple, aller vite. La figure rappelle ou sont les "
        "zones lues (ROIs du pipeline).")

    # ===== 9. MANUSCRIT SEGMENTATION (14:30 -> 16:30) =======================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Manuscrit : segmentation bas niveau", "2 min")
    _bullets(s, [
        "Noms : grille re-estimee SUR CHAQUE formulaire (projection verticale + vote RANSAC sur origine/pas)",
        (1, "une grille fixe derivait d'une demi-case en fin de rangee"),
        "Mantisse / exposant : composantes connexes ; separateur decimal = composante basse et etroite ; signe moins = composante plate",
    ], Inches(0.45), Inches(1.3), Inches(12.4), Inches(2.0), size=17)
    _fig(s, "fig_names_grid.png", Inches(0.7), Inches(3.5), Inches(6.6))
    _fig(s, "fig_mantisse.png", Inches(7.6), Inches(3.7), Inches(5.2))
    s.notes_slide.notes_text_frame.text = (
        "14:30 -> 16:30 (2 min) — Orateur 3\n"
        "Gauche : grille ajustee (lignes rouges pile sur les separateurs). "
        "Droite : composantes vertes = chiffres envoyes au CNN, orange = "
        "virgule decimale detectee par position/taille.")

    # ===== 10. CNN (16:30 -> 18:30) =========================================
    slide_fig_bottom(prs, "Reconnaissance des caracteres : CNN", "2 min", [
        "Meme architecture pour chiffres et lettres (seule la tete change) — entree 28x28 type EMNIST",
        "Chiffres : EMNIST-digits -> 99,5 % (test EMNIST)",
        "Lettres : EMNIST seul insuffisant (fosse de domaine) : « Julien » lu SULIEN  ->  fine-tuning sur lettres reelles -> JULIEN",
    ], "fig_cnn_archi.png",
        "16:30 -> 18:30 (2 min) — Orateur 4\n"
        "Le schema d'architecture est exige aussi dans le rapport. Bien "
        "expliquer le fosse de domaine EMNIST vs stylo fin penche.",
        fig_w=11.6)

    # ===== 11. DATASET (18:30 -> 20:00) =====================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Jeu de fine-tuning : 2145 lettres reelles", "1 min 30")
    _bullets(s, [
        "Extraction AUTOMATIQUE des 221 formulaires (FORM1+2+3) — aucune annotation manuelle",
        (1, "labels = verite terrain Excel ; appariement case k <-> lettre k du nom"),
        "2145 lettres, 53 scripteurs — augmentation : rotations ±10°, translations ±2 px",
        (1, "37 noms composes exclus (l'espace laisse une case vide -> labels decales)"),
    ], Inches(0.45), Inches(1.3), Inches(12.4), Inches(1.9), size=17)
    _fig(s, "fig_letter_dist.png", Inches(0.55), Inches(3.45), Inches(7.3))
    _fig(s, "dataset_preview.png", Inches(8.2), Inches(3.45), Inches(4.6))
    s.notes_slide.notes_text_frame.text = (
        "18:30 -> 20:00 (1 min 30) — Orateur 4\n"
        "Gauche : distribution (rouge = classes rares Q/X/Z, limite "
        "assumee). Droite : vignettes 28x28 reellement extraites avec leur "
        "label (JULIEN, MOREAU, MIGUEL...).")
    # la preview vit dans training/, copie dans figures si absente
    import shutil
    src = os.path.join(ROOT, "training", "dataset_preview.png")
    dst = os.path.join(FIG, "dataset_preview.png")
    if os.path.isfile(src):
        shutil.copyfile(src, dst)

    # ===== 12. K-FOLD (20:00 -> 22:00) ======================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Methodologie : validation croisee par scripteur", "2 min")
    _bullets(s, [
        "Sec. 4.2 : bases distinctes + « privilegier la validation croisee »",
        "K-fold (K = 5) GROUPE PAR ETUDIANT :",
        (1, "toutes les lettres d'un meme scripteur dans le meme fold"),
        (1, "sinon score optimiste : le reseau reconnait l'ECRITURE, pas la lettre"),
        (1, "chaque fold repart des poids EMNIST purs (zero fuite)"),
        "Modele final : re-entraine sur 100 % des donnees avec la recette validee",
    ], Inches(0.45), Inches(1.3), Inches(7.5), Inches(4.4), size=17)
    box = _rect(s, Inches(8.3), Inches(1.4), Inches(4.6), Inches(4.4), BLEU_CLAIR)
    tf = box.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Resultats (par lettre)"
    p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = BLEU
    for txt, c, sz in [
        ("CV 5-fold, NOUVEAUX scripteurs :", GRIS_F, 15),
        ("62,7 % ± 4,2", ORANGE, 22),
        ("(folds : 69,9 / 60,4 / 64,7 / 60,4 / 58,0)", GRIS, 12),
        ("", GRIS_F, 8),
        ("Test FORM3, scripteurs CONNUS :", GRIS_F, 15),
        ("73,4 %  (70,5 % avant fine-tuning)", VERT, 18),
        ("", GRIS_F, 8),
        ("Labels nettoyes : 59,5 -> 62,7 % (+3,2).", VERT, 13),
        ("L'ecart vs scripteurs connus quantifie", GRIS_F, 13),
        ("la part d'ecriture personnelle apprise.", GRIS_F, 13),
    ]:
        pp = tf.add_paragraph()
        pp.text = txt
        pp.font.size = Pt(sz); pp.font.bold = True; pp.font.color.rgb = c
    s.notes_slide.notes_text_frame.text = (
        "20:00 -> 22:00 (2 min) — Orateur 4\n"
        "LA slide methodologie attendue par le jury. Message cle : le K-fold "
        "naif (lettres melangees) donnerait un score gonfle ; en groupant "
        "par scripteur on mesure la generalisation a une ECRITURE inconnue "
        "(59,5 %), alors que sur des scripteurs deja vus (FORM3, memes "
        "eleves que FORM1/2) on atteint 73,4 %. L'ecart est exactement la "
        "fuite que le GroupKFold elimine — c'est un resultat, pas un echec.")

    # ===== 13. RESULTATS (22:00 -> 24:00) ===================================
    slide_fig_bottom(prs, "Resultats par axe du challenge", "2 min", [
        "Mesure cellule par cellule contre la verite terrain fournie (tools/compare_to_truth.py)",
        "Generalisation 3 formulaires : 66,2 % global — temps d'execution ~12 min (43 PDF + 32 photos, CPU)",
    ], "fig_results_bar.png",
        "22:00 -> 24:00 (2 min) — Orateur 4\n"
        "Chaque barre 'apres' correspond a un levier presente dans les "
        "slides precedentes : MCQ -> graphique, cadre signature -> "
        "signature, segmentation+CNN -> manuscrit.",
        fig_w=8.8)

    # ===== 14. LIMITES (24:00 -> 25:00) =====================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _chrome(s, "Limites et perspectives", "1 min")
    _bullets(s, [
        "Noms en correspondance exacte : ~73 % par lettre -> un nom de 8 lettres reste difficile",
        (1, "perspective : plus de donnees, classes rares (Q, X, Z) sous-representees"),
        "Signatures : 8 cas ou le bon eleve est 2e a tres peu pres (ambiguite biometrique reelle)",
        "Verite terrain parfois incoherente (unite pre-imprimee « KHz » notee tantot vide, tantot KHz)",
        "Recalage photo : derive residuelle d'un chiffre sur ~7 photos difficiles",
    ], Inches(0.45), Inches(1.5), Inches(12.4), Inches(4.8), size=18)
    s.notes_slide.notes_text_frame.text = (
        "24:00 -> 25:00 (1 min) — Orateur 4\n"
        "Esprit critique = critere d'evaluation du sujet : assumer les "
        "limites avec des chiffres.")

    # ===== 15. QUESTIONS ====================================================
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, SW, SH, BLEU)
    _rect(s, 0, Inches(4.0), SW, Inches(0.09), ORANGE)
    tb = s.shapes.add_textbox(Inches(2.0), Inches(2.9), Inches(9.3), Inches(1.4))
    p = tb.text_frame.paragraphs[0]
    p.text = "Merci — Questions ?"
    p.font.size = Pt(48); p.font.bold = True; p.font.color.rgb = BLANC
    p.alignment = PP_ALIGN.CENTER
    s.notes_slide.notes_text_frame.text = (
        "25:00 — fin, 15 min de questions.\nGarder les notebooks 02/03/04 "
        "ouverts pour repondre en montrant.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    prs.save(OUT)
    print(f"[pptx] {len(prs.slides._sldIdLst)} slides -> {OUT}")


if __name__ == "__main__":
    main()
