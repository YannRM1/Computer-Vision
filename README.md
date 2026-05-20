# Projet Computer Vision IG.2405 – 2026
**Lecture automatique de formulaires d'examens semi-structurés – DeepForm**

---

## Structure du projet

```
Computer-Vision/
│
├── main.py                    # Point d'entrée – lance les Programmes 1 et 2
├── autoValidPresences.py      # Programme 1 : validation des présences
├── autoReadForm.py            # Programme 2 : lecture automatique des formulaires
│
├── utils/                     # Modules utilitaires
│   ├── form_aligner.py        # Deskew, détection zone active, normalisation
│   ├── grid_decoder.py        # Lecture grilles graphiques (Student ID, Groupe, conditions)
│   ├── checkbox_reader.py     # Détection cases cochées (bas niveau)
│   ├── signature_utils.py     # Descripteurs HOG + Hu, similarité cosinus
│   ├── ocr_utils.py           # OCR imprimé et manuscrit (EasyOCR)
│   ├── page1_parser.py        # Parser complet page 1
│   ├── exam_parser.py         # Parser pages d'examen (MCQ + réponses numériques)
│   └── pdf_utils.py           # Conversion PDF → images (PyMuPDF)
│
├── notebooks/                 # Notebooks d'analyse et de développement
│   ├── 01_exploration_donnees.ipynb
│   ├── 02_preprocessing_alignement.ipynb
│   ├── 03_decodage_grille.ipynb
│   ├── 04_signature_authentication.ipynb
│   ├── 05_ocr_textes.ipynb
│   ├── 06_programme1_presences.ipynb
│   ├── 07_programme2_formulaires.ipynb
│   └── 08_evaluation.ipynb
│
└── PROJECT 2026 -DATABASE-20260518/   # Base de données fournie
    ├── FORM1/                 # Formulaire 1 (images + PDFs + vérités terrain)
    ├── FORM2/                 # Formulaire 2
    ├── FORM3/                 # Formulaire 3
    └── SIGNATURES/            # Base de signatures (fichiers ZIP)
```

---

## Utilisation

### Lancement complet
```bash
python main.py
```

### Avec paramètres personnalisés (challenge)
```bash
python main.py EXAM_FORM2 "PROJECT 2026 -DATABASE-20260518/SIGNATURES"
```

### Configuration dans `main.py`
```python
EXAM_NAME      = "EXAM_FORM1"   # Nom de l'examen à traiter
SIGNATURES_DIR = "..."          # Chemin vers la base de signatures
```

Les répertoires d'entrée/sortie sont déduits automatiquement :
- `DATA_ROOT/FORMX/`  → images de présence + PDFs
- `EXAM_FORMX_RESULTS/` → fichiers xlsx générés

---

## Sorties générées

| Fichier | Description |
|---|---|
| `EXAM_FORMX_PRESENCES.xlsx` | Validation présences (imageName, studentID_grid, studentID_signature) |
| `EXAM_FORMX_XXXXX.xlsx` | Lecture formulaire (onglets PAGE-01 et EXAM) |

---

## Dépendances
```
opencv-python
numpy
openpyxl
pymupdf
easyocr
scikit-image
scikit-learn
pandas
matplotlib
```

---

## Notebooks

Les notebooks sont dans `notebooks/` et se lancent depuis Jupyter.
Ils chargent automatiquement la racine du projet dans `sys.path`.

| Notebook | Contenu |
|---|---|
| 01 | Exploration des données (images, PDFs, vérités terrain) |
| 02 | Prétraitement : binarisation, deskew (Hough), normalisation |
| 03 | Décodage grilles graphiques : Student ID, Groupe, MCQ, cryptogramme |
| 04 | Authentification signatures : HOG + Moments de Hu + similarité cosinus |
| 05 | OCR textes imprimés et manuscrits (EasyOCR) |
| 06 | Programme 1 complet : autoValidPresences |
| 07 | Programme 2 complet : autoReadForm |
| 08 | Évaluation quantitative : split train/val/test, métriques par axe |
