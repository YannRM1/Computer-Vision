# Récapitulatif — Ce qu'il reste à faire (code)
**Projet Computer Vision IG.2405 – 2026 (DeepForm)** · *MAJ : 02/06/2026*
Objectifs : (1) renforcer la partie manuscrite, (2) projet propre, (3) architecture & contraintes respectées.

---

## ✅ Fait dans cette session (partie manuscrite + correctif bloquant)

- **Correctif critique** : `utils/grid_decoder.py` ne définissait pas `read_note_maximale` ni `read_note_pour_valider`, alors que `page1_parser.parse_page1` les appelle → **le Programme 2 plantait dès l'import**. Les deux fonctions ont été ajoutées (lecture des cases « note maximale » / « note pour valider » via le helper `_read_two_digit_box` existant). Le pipeline s'importe de nouveau.
- **CNN de chiffres créé** : `utils/digit_cnn.py` (architecture identique au CNN de lettres, 10 classes 0-9). Repli automatique propre si `torch`/poids absents → aucune régression.
- **Script d'entraînement** : `train_digit_cnn.py` — lit directement les fichiers EMNIST-digits locaux (`emnist_data/EMNIST/raw/`, aucun téléchargement), split **train/val/test**, sauvegarde `models/digit_cnn.pt`, option `--finetune`. *Lecteur de données validé : 240 000 train / 40 000 test, classes équilibrées, normalisé [0,1].*
- **Intégration** : `ocr_handwritten_mantisse` et `ocr_handwritten_exposant` (dans `utils/ocr_utils.py`) utilisent désormais **le CNN de chiffres en priorité** (segmentation bas niveau conservée), avec repli heuristique/EasyOCR. Comportement de repli vérifié (sans modèle → ancien comportement, pas de régression).

> ⚠️ Le modèle `models/digit_cnn.pt` **n'a pas encore été entraîné** : l'environnement sandbox n'a pas pu installer `torch` (wheel CPU bloqué, wheel CUDA 532 Mo trop lent). À lancer sur ta machine (torch déjà installé).

---

## ⏳ Priorité 1 — Partie manuscrite : reste à faire

- [ ] **Entraîner le CNN de chiffres** sur ta machine :
  ```bash
  python train_digit_cnn.py            # EMNIST-digits complet (~99% test attendu)
  # ou rapide : python train_digit_cnn.py --subset 60000 --epochs 5
  ```
  → génère `models/digit_cnn.pt`, automatiquement utilisé par mantisse/exposant.
- [ ] **Fine-tuning des lettres** (jamais fait — `letter_dataset.npz` absent) :
  ```bash
  python build_letter_dataset.py                          # crée letter_dataset.npz
  python train_letter_cnn.py --finetune letter_dataset.npz
  ```
  Conserver les courbes loss/accuracy (val & test) pour le rapport.
- [ ] **(Optionnel) Fine-tuning des chiffres** sur les vrais formulaires : construire un `digit_dataset.npz` depuis les vérités terrain MANTISSE/EXPOSANT (onglet EXAM), puis `python train_digit_cnn.py --finetune digit_dataset.npz`. *(le `build_digit_dataset.py` reste à écrire, sur le modèle de `build_letter_dataset.py`.)*
- [ ] **Gestion du signe de l'exposant** (négatif) dans le chemin CNN : actuellement le signe `-` n'est repris que par le repli EasyOCR. Ajouter une détection de tiret bas niveau avant classification.
- [ ] **Évaluation quantitative** des champs manuscrits (Prénom L13, Nom L14, MANTISSE, EXPOSANT) contre la vérité terrain → tableau d'accuracy (axe 6.3).
- [ ] **Pour le rapport** : schéma de l'architecture CNN (commune lettres/chiffres) + hyperparamètres (lr, batch, epochs, augmentation) + bases d'apprentissage — exigé §4.1 / §7.1.

---

## ⏳ Priorité 2 — Propreté du projet & dépôt (inchangé)

- [x] **Doublons supprimés** : `utils/ocr_utils_backup.py` et `utils/ocr_utils_new.py` retirés (seul `utils/ocr_utils.py` subsiste, non importé ailleurs → sans risque).
- [x] **19 PNG de debug supprimés** de la racine (`block_*`, `p6_block_*`, `codes_*`, `crypto_*`, `header_hd`, `full_page_p1`, `top_*`, `exam_page*`, `brouillon_max_roi`, `emnist_orientation`). Template `utils/assets/` et `calibration_output/` **préservés**.
- [x] **`__pycache__` / `.pyc` supprimés** du disque + **`.gitignore` complété** (`__pycache__/`, `*.pyc`, `*.log`, datasets `.npz`, PNG de debug).
- [ ] **Finir le nettoyage git sur ta machine** (index cassé ici : `null sha1`) : `git rm -r --cached --ignore-unmatch "*.pyc"` puis `git add -A && git commit -m "chore: nettoyage (.pyc, doublons, PNG debug) + .gitignore"`. Si l'index reste corrompu : `rm .git/index && git reset`. Soigner les messages (éviter « commit », « commmit »).
- [ ] **Ranger les scripts de dev** (`debug_emnist.py`, `debug_photo.py`, `calibrate_roi.py`, `check_names.py`, `compare_to_truth.py`) dans `tools/` ; ranger `evaluate.py` dans `eval/`.
- [ ] **Nettoyer artefacts** régénérables : `executionduscriptresultat.txt`, `compare_results.csv` (les `*.log` sont désormais gitignorés).
- [ ] **Corriger `requirements.txt`** : ligne parasite `p` en fin de fichier.
- [ ] **Centraliser les paramètres codés en dur** (ROIs, pas de cases `24.45`, seuils `0.10`/`0.30`/`0.15`, `PDF_DPI=150`, `EXAM_START_PAGE`, fractions mantisse/exposant) dans un `utils/config.py` — §8 pénalise les paramètres « codés en dur ».
- [ ] **Mettre à jour le README** : aligner la description (Hough/HOG annoncés) sur le code réel (morphologie + composantes connexes + NCC). Ajouter `digit_cnn` / `train_digit_cnn.py`.

---

## ⏳ Priorité 3 — Conformité à l'architecture demandée (§3, inchangé)

- [ ] **Signatures des sous-fonctions** : la consigne impose `autoValidID(filename.jpg, STUDENT_CLASS_SIGNATURES, …)` et `autoReadFormID(EXAM_FORMXX_abcd.pdf, STUDENT_CLASS_SIGNATURES, …)`. Le code passe `desc_db`/`wb` → conformer (ou justifier l'optimisation).
- [ ] **`autoValidPresences`** : 4ᵉ paramètre `pdf_dir` hors spec (template de recalage) → documenter / rendre optionnel.
- [ ] **`main.py`** : `PRESENCES_DIR` = `PDF_DIR` = dossier `FORMx` (base mélangée). Tester le cas de répertoires distincts pour le challenge.
- [ ] **Vérifier l'onglet PAGE-01** : 18 lignes, ligne 12 vide, libellés/ordre = Figure 2.

---

## ⏳ Contraintes transverses (§4.1, méthodo, §8)

- [x] **Éléments graphiques en bas niveau** (grille, cases, cryptogramme, lignes) : conforme.
- [x] **Texte manuscrit → CNN** : lettres ✅, chiffres ✅ (code) — *reste à entraîner les modèles*.
- [ ] **train/val/test + validation croisée** à généraliser et documenter pour l'éval.
- [ ] **Paramètres non codés en dur** (→ `config.py`).
- [ ] **Exécution sans plantage / sans correction manuelle** sur données neuves (challenge §6) : retester `main.py` de bout en bout **après** le correctif d'import.

---

### Prochaine action immédiate
1. `python train_digit_cnn.py` → `models/digit_cnn.pt`.
2. `python build_letter_dataset.py` puis `python train_letter_cnn.py --finetune letter_dataset.npz`.
3. Relancer `main.py` sur FORM1 et vérifier les xlsx (mantisse/exposant + noms).
