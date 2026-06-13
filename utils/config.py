"""
Paramètres centralisés du projet (§8 : « paramètres bien définis et non codés en
dur »).

Ce module regroupe les hyper-paramètres et constantes de réglage des étages de
segmentation / lecture, afin d'avoir une source unique à ajuster pour le
challenge plutôt que des valeurs dispersées dans le code.

NB : les ROIs géométriques du formulaire (repère canonique 900x1270) restent
définis dans utils/grid_decoder.py (ROI_*), au plus près de leur usage ; ils
pourront être migrés ici dans un second temps.
"""

# ===========================================================================
# Programme 2 — conversion PDF & pages d'examen (autoReadForm)
# ===========================================================================

# Index (0-indexé) de la première page d'examen : page 5 du PDF.
EXAM_START_PAGE = 4
# Résolution de rendu PDF -> image.
PDF_DPI = 150


# ===========================================================================
# Lecture des pages d'examen (utils/exam_parser)
# ===========================================================================

# Bande d'en-tête (Module/Code/Date) en haut de chaque page d'examen.
HEADER_BAND_H = 82

# Détection des lignes horizontales séparant les blocs de question.
MIN_LINE_WIDTH = 380     # largeur min d'une ligne séparatrice (px)
LINE_MERGE_TOL = 60      # fusion des lignes proches (px)

# Cases à cocher MCQ (colonne de gauche du bloc).
MCQ_X_START = 12
MCQ_X_END   = 90
MCQ_MIN_SZ  = 12         # taille min d'un côté de case
MCQ_MAX_SZ  = 35         # taille max d'un côté de case
MCQ_MIN_AREA = 80
CHECKED_INK_THRESHOLD = 0.10   # ratio d'encre min pour considérer une case cochée.
                               # NB (mesuré via tools/sweep_mcq_threshold.py) : la
                               # détection MCQ est INSENSIBLE à ce seuil sur [0.08, 0.30].
                               # Les erreurs CHOIX viennent de la LOCALISATION des cases
                               # (_find_mcq_checkboxes / assignation des lettres), pas du seuil.

# Lettres de choix MCQ, en ordre alphabétique.
MCQ_CHOICES = "ABCDEFGH"

# Case MCQ entierement noircie = choix ANNULE par l'eleve (convention du
# formulaire : on noircit la case erronee puis on coche une autre avec un X).
# Une croix X atteint ~0.30-0.55 de ratio d'encre, une case noircie ~0.90.
MCQ_FILLED_CANCEL = 0.75

# Zones des réponses numériques (fractions du bloc) — mantisse / exposant / unité.
MANTISSE_X_FRAC = (0.05, 0.28)
MANTISSE_Y_FRAC = (0.50, 0.88)
EXPOSANT_X_FRAC = (0.28, 0.42)
EXPOSANT_Y_FRAC = (0.38, 0.66)
UNITE_X_FRAC    = (0.48, 0.72)
UNITE_Y_FRAC    = (0.55, 0.90)
