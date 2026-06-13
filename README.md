# Projet Computer Vision IG.2405 — 2026
**Lecture automatique de formulaires d'examens semi-structurés — DeepForm**

Système de vision par ordinateur qui (1) valide les présences à partir des photos
de première page et (2) lit automatiquement les formulaires d'examen numérisés
(PDF), en produisant les fichiers Excel imposés par le cahier des charges.

Conformément à la consigne (4.1), les **éléments graphiques** (grilles, cases à
cocher, signatures, cryptogrammes) sont traités par des **méthodes de bas niveau**
(seuillage d'Otsu, morphologie, transformée de Hough, composantes connexes,
corrélation normalisée) ; les **textes** imprimés et manuscrits sont confiés à des
méthodes de **plus haut niveau** (OCR easyOCR ; réseaux de neurones convolutifs).

**Équipe :** Yann ROQUIGNY--MARTIN · Oscar MILLET · Félix BLANCHIER · Victor POUSSIER

---

## Installation

```bash
pip install -r requirements.txt
```
Dépendances : `opencv-python numpy scikit-image scikit-learn openpyxl pymupdf
easyocr torch torchvision pandas matplotlib pillow-heif`.

Les poids entraînés (`models/letter_cnn.pt`, `models/digit_cnn.pt`) sont fournis :
aucun entraînement n'est nécessaire pour exécuter le projet.

---

## Exécution — une seule commande

`main.py` enchaîne le Programme 1 (présences) puis le Programme 2 (lecture des
formulaires) pour un examen, et crée le dossier `EXAM_FORMXX_RESULTS/` avec tous
les `.xlsx`. **Rien à modifier dans le code.**

```bash
# Examen FORM1 de la base fournie (P1 + P2 d'un coup)
python main.py EXAM_FORM1

# Les trois examens à la suite
for F in EXAM_FORM1 EXAM_FORM2 EXAM_FORM3; do python main.py "$F"; done
```

### Tester sur une AUTRE base de données (vérification par l'enseignant)

Les chemins se passent **en arguments**, sans toucher au code. Format :

```bash
python main.py <EXAM_NAME> <SIGNATURES_DIR> <PRESENCES_DIR> <PDF_DIR>
```

Exemple sur une base située ailleurs :

```bash
python main.py EXAM_FORM1 \
    "/chemin/vers/NOUVELLE_BASE/SIGNATURES" \
    "/chemin/vers/NOUVELLE_BASE/FORM1" \
    "/chemin/vers/NOUVELLE_BASE/FORM1"
```

- `SIGNATURES_DIR` : base `STUDENT_CLASS_SIGNATURES`. Le chargement est robuste au
  rangement — images à plat (`61992.png`, `61992_000.png`), sous-dossiers par
  étudiant (`61992/…`), ou archive(s) `.zip` ; l'identifiant est déduit du nom de
  fichier ou du dossier.
- `PRESENCES_DIR` : photos de la 1re page (`.jpg/.png/.heic`).
- `PDF_DIR` : formulaires scannés (`EXAM_FORMXX_NNNNN.pdf`).

Si `PRESENCES_DIR` et `PDF_DIR` sont identiques (cas de la base fournie), on peut
les passer deux fois, ou n'indiquer que les deux premiers arguments — les
répertoires sont alors déduits du nom de l'examen.

### Sorties générées (dans `EXAM_FORMXX_RESULTS/`)
| Fichier | Contenu |
|---|---|
| `EXAM_FORMXX_PRESENCES.xlsx` | `imageName`, `studentID_grid`, `studentID_signature` |
| `EXAM_FORMXX_NNNNN.xlsx` | Onglet `PAGE-01` (identité, conditions, notes, signature, cryptogramme) et onglet `EXAM` (choix MCQ, mantisse, exposant, unité) |

---

## Vérifier l'exactitude (comparaison aux vérités terrain)

`tools/compare_to_truth.py` compare **cellule par cellule** les `.xlsx` produits aux
`.xlsx` de référence (fournis à côté de chaque PDF), et affiche l'exactitude par
axe d'évaluation (imprimé / manuscrit / graphique / signature) et par formulaire.

```bash
# Sur la base fournie (RESULTS dans le dossier courant)
python tools/compare_to_truth.py

# Sur une autre base : <racine_base> <racine_resultats>
python tools/compare_to_truth.py "/chemin/vers/NOUVELLE_BASE" .
```
La comparaison est insensible à la casse et au format des nombres/dates
(« JULIEN » = « Julien », `2.3` = `"2.3"`).

---

## Arborescence

```
Computer-Vision/
├── main.py                    # Point d'entrée — Programmes 1 et 2 (§3.6)
├── autoValidPresences.py      # PROGRAMME 1 — validation des présences (§3.3)
├── autoReadForm.py            # PROGRAMME 2 — lecture des formulaires (§3.4)
├── requirements.txt
│
├── utils/                     # Modules de traitement
│   ├── config.py              # Hyper-paramètres centralisés (§8)
│   ├── form_aligner.py        # Deskew (Hough), L-brackets, perspective
│   ├── template_register.py   # Recalage par ORB + homographie RANSAC ; détection 180°
│   ├── grid_decoder.py        # Grilles (Student ID, Groupe, conditions), cryptogramme
│   ├── checkbox_reader.py     # Cases à cocher (densité d'encre + motif X)
│   ├── signature_utils.py     # Signature : NCC de gabarit + HOG + moments de Hu
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
├── training/                  # Entraînement / affinage des CNN (voir plus bas)
├── tools/compare_to_truth.py  # Évaluation cellule par cellule (par axe)
├── notebooks/                 # Notebooks d'exploration et de démonstration
├── rapport/                   # Rapport, soutenance, figures, progression d'accuracy
└── PROJECT 2026 -DATABASE-20260518/   # Base fournie (FORM1/2/3 + SIGNATURES)
```

---

## Reconnaissance manuscrite (CNN)

Deux petits CNN de même architecture (têtes différentes), pré-entraînés sur EMNIST
puis **affinés par transfert** sur des caractères réels découpés dans les
formulaires (étiquetés par les noms/valeurs des vérités terrain). Repli automatique
sur easyOCR si un modèle est absent.

| Champ | Module | Modèle |
|---|---|---|
| Prénom / Nom | `utils/letter_cnn.py` | `models/letter_cnn.pt` (26 classes A–Z) |
| Mantisse / Exposant | `utils/digit_cnn.py` | `models/digit_cnn.pt` (10 classes 0–9) |

Ré-entraînement (optionnel — les modèles sont déjà fournis) :
```bash
python training/train_letter_cnn.py                                       # base EMNIST
python training/build_letter_dataset.py                                   # jeu annoté (formulaires)
python training/train_letter_cnn.py --finetune training/letter_dataset.npz # affinage + K-fold
# idem pour les chiffres : train_digit_cnn.py / build_digit_dataset.py
```
L'affinage est validé par **GroupKFold par scripteur** (toutes les lettres d'un
même étudiant dans le même pli) : le score reflète la lecture d'écritures jamais
vues, sans fuite.

---

## Résultats

Exactitude cellule par cellule sur la base fournie (130 formulaires), via
`tools/compare_to_truth.py` :

| Axe | FORM1 | FORM2 | FORM3 | Global |
|---|---|---|---|---|
| Imprimé | 88,6 % | 95,5 % | 99,9 % | **95,1 %** |
| Manuscrit | 69,3 % | 61,5 % | 45,6 % | **57,2 %** |
| Graphique | 88,8 % | 87,0 % | 76,8 % | **83,8 %** |
| Signature | 92,7 % | 88,4 % | 93,5 % | **91,5 %** |
| **Global** | **85,3 %** | **84,4 %** | **79,0 %** | **82,7 %** |

CNN lettres : validation croisée 5-fold par scripteur **86,4 % ± 3,6**.
CNN chiffres : 5-fold par scripteur **87,3 % ± 3,7**.
La progression détaillée de l'exactitude à chaque amélioration est documentée dans
[`rapport/PROGRESSION.md`](rapport/PROGRESSION.md).
