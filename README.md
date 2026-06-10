# Projet Computer Vision IG.2405 — 2026
**Lecture automatique de formulaires d'examens semi-structurés — DeepForm**

Système de vision par ordinateur qui (1) valide les présences à partir des photos
de première page et (2) lit automatiquement les formulaires d'examen numérisés
(PDF), en produisant les fichiers Excel imposés par le cahier des charges.

Conformément à la consigne (§4.1), les **éléments graphiques** (grilles, cases à
cocher, cryptogrammes) sont traités par des **méthodes de bas niveau** (seuillage
d'Otsu, morphologie, transformée de Hough, composantes connexes, corrélation
normalisée) ; les **textes** imprimés et manuscrits sont confiés à des méthodes
de **plus haut niveau** (OCR easyOCR ; réseaux de neurones convolutifs).

---

## Arborescence

```
Computer-Vision/
├── main.py                    # Point d'entrée — exécute les Programmes 1 et 2 (§3.6)
├── autoValidPresences.py      # PROGRAMME 1 — validation des présences (§3.3)
├── autoReadForm.py            # PROGRAMME 2 — lecture automatique des formulaires (§3.4)
├── requirements.txt
│
├── utils/                     # Modules de traitement
│   ├── config.py              # Hyper-paramètres centralisés (§8)
│   ├── form_aligner.py        # Deskew (Hough), L-brackets, perspective
│   ├── template_register.py   # Recalage photo par ORB + homographie RANSAC
│   ├── grid_decoder.py        # Grilles (Student ID, Groupe, conditions), cryptogramme
│   ├── checkbox_reader.py     # Détection bas niveau des cases (densité + motif X)
│   ├── signature_utils.py     # Signature : HOG + moments de Hu + NCC de gabarit
│   ├── page1_parser.py        # Assemblage de la page 1 (onglet PAGE-01)
│   ├── exam_parser.py         # Pages d'examen (MCQ + réponses numériques) → onglet EXAM
│   ├── ocr_utils.py           # OCR imprimé/manuscrit + segmentation des chiffres
│   ├── digit_cnn.py           # CNN chiffres (mantisse/exposant) — inférence
│   ├── letter_cnn.py          # CNN lettres (prénom/nom) — inférence
│   ├── pdf_utils.py           # Rendu PDF → images (PyMuPDF)
│   ├── image_io.py            # Lecture d'image robuste (JPEG/PNG/HEIF)
│   └── assets/form_template_ref.png   # Gabarit de référence (repère 900×1270)
│
├── models/                    # Poids entraînés (digit_cnn.pt, letter_cnn.pt)
├── training/                  # Entraînement / affinage des CNN
│   ├── train_digit_cnn.py     # CNN chiffres sur EMNIST-digits
│   ├── train_letter_cnn.py    # CNN lettres sur EMNIST-letters
│   └── build_letter_dataset.py# Jeu de lettres annotées (affinage par transfert)
│
├── eval/                      # Évaluation quantitative
│   ├── evaluate.py            # Accuracies Student ID + signature (FORM1/2/3)
│   ├── evaluation_results.csv # Résultats de référence (par image)
│   └── compare_results.csv    # Comparaison cellule par cellule (par axe)
├── tools/                     # Scripts de calibration / debug
├── notebooks/                 # Notebooks d'exploration et de développement
├── rapport/                   # Rapport (.docx) et figures (rapport/figures/)
├── docs/                      # Sujet du projet + notes de travail
└── PROJECT 2026 -DATABASE-20260518/   # Base fournie (FORM1/2/3 + SIGNATURES)
```

> Le dossier `.trash/` (s'il existe) contient des fichiers obsolètes mis de côté
> lors du rangement ; il est ignoré par Git et peut être supprimé.

---

## Utilisation

### Exécution complète
```bash
python main.py
```

### Adaptation pour le challenge (§5)
Les seules lignes à adapter sont dans `main.py` (nom de l'examen et répertoires) :
```python
EXAM_NAME      = "EXAM_FORM1"          # examen à traiter
SIGNATURES_DIR = ".../SIGNATURES"      # base STUDENT_CLASS_SIGNATURES
```
Les répertoires d'entrée/sortie sont déduits automatiquement ; le répertoire de
résultats `EXAM_FORMXX_RESULTS/` est créé et rempli des fichiers `.xlsx`.
On peut aussi passer les chemins en ligne de commande :
```bash
python main.py EXAM_FORM2 "PROJECT 2026 -DATABASE-20260518/SIGNATURES"
```

### Sorties générées (dans `EXAM_FORMXX_RESULTS/`)
| Fichier | Contenu |
|---|---|
| `EXAM_FORMXX_PRESENCES.xlsx` | `imageName`, `studentID_grid`, `studentID_signature` |
| `EXAM_FORMXX_NNNNN.xlsx` | Onglets `PAGE-01` (identité, conditions, notes, signature, cryptogramme) et `EXAM` (choix, mantisse, exposant, unité) |

---

## Reconnaissance manuscrite (CNN)

Deux petits CNN de même architecture (têtes différentes), entraînés sur EMNIST
puis utilisés en inférence ; repli automatique sur easyOCR si un modèle est absent.

| Champ | Module | Modèle |
|---|---|---|
| Prénom / Nom | `utils/letter_cnn.py` | `models/letter_cnn.pt` (26 classes A–Z) |
| Mantisse / Exposant | `utils/digit_cnn.py` | `models/digit_cnn.pt` (10 classes 0–9) |

```bash
python training/train_digit_cnn.py                                  # EMNIST-digits
python training/train_letter_cnn.py                                 # EMNIST-letters
python training/build_letter_dataset.py                             # jeu annoté (formulaires)
python training/train_letter_cnn.py --finetune letter_dataset.npz   # affinage par transfert
```

---

## Évaluation (reproduction)

```bash
python eval/evaluate.py "PROJECT 2026 -DATABASE-20260518"   # Student ID + signature
python tools/compare_to_truth.py                            # comparaison cellule/cellule par axe
```

Résultats de référence sur FORM1 (par cellule) : imprimé **83,6 %**, graphique
**61,5 %**, signature **58,5 %**, manuscrit **22,9 %**, global **63,2 %**.
CNN chiffres : **99,56 %** (test EMNIST). Détails et discussion dans `rapport/`.

---

## Dépendances
```
opencv-python  numpy  scikit-image  scikit-learn  openpyxl
pymupdf  easyocr  torch  torchvision  pandas  matplotlib  pillow-heif
```
Installation : `pip install -r requirements.txt`
