# Progression de l'exactitude — projet DeepForm (IG.2405)

Ce document trace, étape par étape, comment l'exactitude a progressé au fil des
améliorations. Toutes les mesures « de bout en bout » proviennent de
`tools/compare_to_truth.py` (comparaison cellule par cellule aux vérités terrain,
130 formulaires des trois examens). Les scores des CNN sont mesurés par
**validation croisée 5-fold par scripteur** (GroupKFold) : toutes les écritures
d'un même étudiant restent dans le même pli, donc le score reflète la lecture
d'écritures jamais vues, sans fuite optimiste.

---

## 1. Vue d'ensemble (exactitude globale, 3 formulaires)

| Étape | Global |
|---|---|
| Lecture initiale | 67,2 % |
| + grilles par localisation de cases, recalage ORB | 75,3 % |
| + CNN chiffres affiné, anti-décalage du pied de page | 77,8 % |
| + CNN lettres recadré (voir §2) | 80,2 % |
| + tri des pages désordonnées (§4) | 81,1 % |
| + localisation exposant (§3) + vérification signature (§5) | **82,7 %** |

Le tableau final par axe et par formulaire est en dernière section.

---

## 2. Lettres manuscrites (prénom / nom)

Le plus gros gain de tout le projet — **sans jamais changer l'architecture du
réseau**, uniquement en améliorant la qualité des données et le cadrage des
lettres. Illustration du principe « garbage in, garbage out ».

| Ajustement | CV 5-fold / scripteur |
|---|---|
| Base EMNIST seule (aucun affinage) | 59,5 % |
| + exclusion des noms composés (étiquettes décalées) | 62,7 % |
| + préservation des lettres en trait plein (I, T) au prétraitement | 66,0 % |
| + appariement strict case↔lettre (suite contiguë de longueur exacte) | 66,5 % |
| + recalage **vertical** de la bande de cases | 77,5 % |
| + recalage des photos (Otsu, paire de lignes de base, fit horizontal) | **86,4 % ± 3,6** |

Chaque correctif a été déclenché par l'inspection visuelle de l'aperçu du jeu de
données (`training/dataset_preview.png`) : un décalage d'une case ou une lettre
coupée corrompait silencieusement les étiquettes. Les deux derniers correctifs —
recalage vertical puis recalage des photos — ont à eux seuls apporté +20 points,
car le recalage global laissait jusqu'à ~30 px de jeu vertical (la bande lisait
parfois le label imprimé au lieu des cases) et les séparateurs des photos étaient
sous le seuil de détection.

**De bout en bout (champs Prénom + Nom)** : 46 % → **68,5 %**.

---

## 3. Réponses numériques (mantisse / exposant)

Les chiffres sont lus par un CNN affiné par transfert (CV 5-fold par scripteur
**87,3 % ± 3,7**). Le gain de bout en bout est venu d'un correctif de
**localisation de la case exposant**, qui touchait surtout l'exposant (la case
de la mantisse, plus grande, était déjà localisée de façon fiable).

| Composant | Avant | Après |
|---|---|---|
| EXPOSANT | 39,3 % | **53,5 %** |
| MANTISSE | 58,6 % | 58,6 % (stable) |

**Cause** : la case de l'exposant était sélectionnée par un seuil d'aire fragile
(« plus petite que la moitié de la mantisse »). Une case la frôlant n'était pas
reconnue, et le repli lisait le « .10 » imprimé à la place — d'où des exposants
négatifs systématiquement faux (« -1 » lu « 101 » ou « 1 ») : seulement 13/87 des
exposants négatifs étaient corrects.
**Correctif** : sélection par **position verticale** — l'exposant est écrit en
superscript, donc sa case est nettement plus haute que celles de la mantisse et de
l'unité (propriété structurelle du formulaire, pas un seuil arbitraire). Le signe
« − » se lit alors correctement dès que la bonne case est extraite.

---

## 4. Structure des pages (ordre + comptage des questions)

**Ordre des pages.** Trois PDF avaient leurs pages d'examen inversées par paires
(artefact de scan recto-verso) : les réponses sortaient dans le désordre. Le
recalage ORB de la page 1 révèle gratuitement une rotation à 180° ; pour l'ordre,
on lit le **numéro de page imprimé** (OCR, texte autorisé §4.1) et on réordonne.
Procédure conservatrice : on ne réordonne que si tous les numéros sont lus et
forment une permutation réellement désordonnée.

| PDF | Avant | Après |
|---|---|---|
| FORM1_62380 | 31,1 % | 57,9 % |
| FORM2_62766 | 61,4 % | 85,7 % |
| FORM3_62578 | 45,8 % | 84,8 % |

**Comptage des questions.** Un pied de page (numéro + cryptogramme + équerres)
était parfois compté comme une question, décalant toute la numérotation.
Discriminant robuste : une vraie question porte un en-tête « QUESTION N » juste
sous sa ligne supérieure (densité d'encre ≥ 0,05) ; un pied de page y est vide
(≤ 0,022). Plus fiable que l'encre totale, contaminée par la ligne séparatrice.

---

## 5. Signatures

L'exactitude était limitée par une **erreur de cadrage de la tâche** : on faisait
de l'identification 1-parmi-~60 (« quel étudiant a la signature la plus proche ? »)
et on ne validait que si l'étudiant attendu sortait premier. Or l'identité est
**déjà revendiquée** sur le formulaire (grille STUDENT ID). On rejetait donc à tort
des signatures authentiques classées 2ᵉ-4ᵉ malgré un bon score propre (0,39-0,47).

**Correctif** : vérification 1-contre-1 — on compare l'image de la signature à la
**seule référence de l'étudiant revendiqué** (NCC + HOG + moments de Hu sur les
pixels) et on valide si le score dépasse un seuil. La décision reste 100 % fondée
sur la comparaison d'**images** ; le numéro étudiant ne sert qu'à choisir la
référence. Les scores propres mesurés (0,27-0,72) valident toutes les signatures
authentiques tout en rejetant une case vide (seuil d'encre minimal).

Exactitude signature : 67,7 % → **91,5 %**.

---

## 6. Tableau final par axe et par formulaire

| Axe | FORM1 | FORM2 | FORM3 | Global |
|---|---|---|---|---|
| Imprimé | 88,6 % | 95,5 % | 99,9 % | **95,1 %** |
| Manuscrit | 69,3 % | 61,5 % | 45,6 % | **57,2 %** |
| Graphique | 88,8 % | 87,0 % | 76,8 % | **83,8 %** |
| Signature | 92,7 % | 88,4 % | 93,5 % | **91,5 %** |
| **Global** | **85,3 %** | **84,4 %** | **79,0 %** | **82,7 %** |

Détail manuscrit : Noms 68,5 % · MANTISSE 58,6 % · EXPOSANT 53,5 % · UNITÉ 45,5 %.

---

## 7. Limites connues

- **Manuscrit (exposant, unité)** : reste l'axe le plus difficile ; certaines
  écritures sont ambiguës même à l'œil.
- **7 PDF sous-comptés par fusion** : un séparateur horizontal pâle manquant colle
  deux questions, décalant les réponses suivantes. Distinct des correctifs ci-dessus.
- **Signatures** : le descripteur bas niveau valide les vraies signatures mais ne
  rejetterait pas un faussaire habile (scores authentique/imposteur partiellement
  superposés) — plafond raisonnable d'une méthode bas niveau.
- **Quelques vérités terrain fournies sont corrompues** (fichiers `.xlsx`
  illisibles) et exclues de la comparaison.
