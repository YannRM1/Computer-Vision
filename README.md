# Projet Computer Vision IG.2405 — DeepForm
**Lecture automatique de formulaires d'examens semi-structurés**

Équipe : Yann ROQUIGNY--MARTIN · Oscar MILLET · Félix BLANCHIER · Victor POUSSIER

## 1. Installation
pip install -r requirements.txt
Les modèles CNN (`models/*.pt`) sont fournis : aucun entraînement nécessaire.

## 2. Exécution
Une seule commande enchaîne le Programme 1 (présences) puis le Programme 2
(lecture des formulaires) et crée le dossier `EXAM_FORMXX_RESULTS/`.

# un examen de la base fournie
python main.py EXAM_FORM1

## 3. Tester sur une autre base

En éditant la config en haut de `main.py` (`BDD`, `EXAM_NAME`, `SIGNATURES`),
puis faire

python main.py

## Sorties dans EXAM_FORMXX_RESULTS

## (Optionnel) Vérifier l'exactitude

python tools/compare_to_truth.py                   
