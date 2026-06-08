// Génère le rapport scientifique (.docx) — Projet Computer Vision IG.2405 / DeepForm
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, Header, Footer, TableOfContents,
  PageBreak, Bookmark
} = require("docx");

const NAVY = "13334f", BLUE = "2c6fbb", GREY = "555555";
const CW = 9026; // largeur de contenu A4 (marges 1")

// ---------- helpers ----------
const P = (text, opts = {}) => new Paragraph({
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 276 },
  children: Array.isArray(text) ? text : [new TextRun({ text, ...opts.run })],
});
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });
const r = (text, o = {}) => new TextRun({ text, ...o });
const it = (text) => new TextRun({ text, italics: true });
const b = (text) => new TextRun({ text, bold: true });
// équation centrée (notation mathématique unicode, en italique)
const EQ = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 140 },
  children: [new TextRun({ text, italics: true, font: "Cambria Math", size: 24 })],
});
const bullet = (children) => new Paragraph({
  numbering: { reference: "bul", level: 0 }, spacing: { after: 80, line: 272 },
  alignment: AlignmentType.JUSTIFIED,
  children: Array.isArray(children) ? children : [new TextRun(children)],
});
const num = (children) => new Paragraph({
  numbering: { reference: "n1", level: 0 }, spacing: { after: 80, line: 272 },
  alignment: AlignmentType.JUSTIFIED,
  children: Array.isArray(children) ? children : [new TextRun(children)],
});
function fig(path, w, h, caption) {
  const data = fs.readFileSync(path);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ type: "png", data, transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: caption } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, size: 19, color: GREY })] }),
  ];
}
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, { head = false, alignc = false } = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: { fill: head ? "D5E8F0" : "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({ alignment: alignc ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: Array.isArray(text) ? text : [new TextRun({ text, bold: head, size: 19 })] })],
  });
}
function table(cols, rows) {
  const widths = cols.map(c => c.w);
  return new Table({
    width: { size: widths.reduce((a, x) => a + x, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: cols.map(c => cell(c.t, c.w, { head: true, alignc: c.c })) }),
      ...rows.map(row => new TableRow({ children: row.map((t, i) =>
        cell(typeof t === "string" ? t : t, cols[i].w, { alignc: cols[i].c })) })),
    ],
  });
}

// ---------- contenu ----------
const children = [];

// Page de titre
children.push(
  new Paragraph({ spacing: { before: 1600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("Projet Computer Vision — IG.2405 (2026)", { color: BLUE, bold: true, size: 24 })] }),
  new Paragraph({ spacing: { before: 240, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("Lecture automatique de formulaires", { bold: true, size: 44, color: NAVY })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("d’examens semi-structurés", { bold: true, size: 44, color: NAVY })] }),
  new Paragraph({ spacing: { before: 200, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("Société DeepForm — validation des présences et lecture automatisée des formulaires", { italics: true, size: 22, color: GREY })] }),
  new Paragraph({ spacing: { before: 1200, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("Équipe : ", { bold: true }), r("[Noms et prénoms à compléter]")] }),
  new Paragraph({ spacing: { before: 80, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("Module IG.2405 — Vision par ordinateur — ISEP")] }),
  new Paragraph({ spacing: { before: 80, after: 0 }, alignment: AlignmentType.CENTER,
    children: [r("[Date]")] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// Table des matières
children.push(
  new Paragraph({ spacing: { after: 160 }, children: [r("Table des matières", { bold: true, size: 30, color: NAVY })] }),
  new TableOfContents("Sommaire", { hyperlink: true, headingStyleRange: "1-3" }),
  new Paragraph({ children: [new PageBreak()] }),
);

// Résumé
children.push(H1("Résumé"));
children.push(P("Ce rapport présente un système de vision par ordinateur conçu pour la correction automatique de formulaires d’examens semi-structurés. À partir des photos de première page (validation des présences) et des formulaires numérisés au format PDF (lecture complète), le système authentifie les signatures, identifie l’étudiant, lit les informations imprimées, graphiques et manuscrites, et produit des fichiers tableurs structurés conformes au cahier des charges. Les éléments graphiques (grilles d’identifiant, cases à cocher, cryptogrammes) sont traités par des méthodes de bas niveau (seuillage, morphologie mathématique, transformée de Hough, analyse de composantes connexes, corrélation normalisée). Les textes imprimés et manuscrits sont traités par des méthodes de plus haut niveau : reconnaissance optique de caractères pour l’imprimé, et réseaux de neurones convolutifs (CNN) entraînés sur EMNIST puis affinés sur les données du projet pour le manuscrit. L’authentification des signatures repose sur une combinaison de descripteurs (HOG, moments de Hu, corrélation de gabarit). Le pipeline complet s’exécute sans intervention manuelle et atteint, sur le formulaire FORM1, une exactitude globale de 63,2 % par cellule, avec des performances contrastées selon la nature de l’information (imprimé 83,6 %, graphique 61,5 %, signature 58,5 %, manuscrit 22,9 %). Une analyse critique met en évidence les facteurs limitants — difficulté intrinsèque de la reconnaissance d’écriture, faible volume de données annotées, et une erreur dans la vérité terrain fournie — et propose des perspectives d’amélioration."));

// 1. Position du problème
children.push(H1("1. Position du problème"));
children.push(P("La société DeepForm développe un outil de lecture automatique de formulaires d’examens semi-structurés. « Semi-structuré » signifie que le document combine des éléments bien définis (en-têtes, grilles, cases à cocher) avec une mise en page variable et des champs renseignés à la main (chiffres, lettres, signatures). Énoncés et zones de réponse cohabitent dans un même document ; il n’existe pas de grille de réponse indépendante."));
children.push(P("Le scénario comporte deux flux de données. D’une part, chaque étudiant photographie la première page d’identification de son sujet et la dépose en ligne : l’ensemble des photos constitue le répertoire des présences. D’autre part, le professeur numérise les copies papier en un fichier PDF par formulaire. Le système doit alors remplir trois objectifs :"));
children.push(num([b("Valider les présences"), r(" et authentifier les signatures à partir des photos de première page.")]));
children.push(num([b("Authentifier"), r(" chaque formulaire scanné (examen et étudiant) et le "), b("lire automatiquement"), r(".")]));
children.push(num([b("Attribuer une note"), r(" (programme de fusion, hors périmètre de ce projet).")]));
children.push(P([r("Le présent travail implémente les "), b("Programmes 1 et 2"), r(" du cahier des charges. Le Programme 1 ("), it("autoValidPresences"), r(") produit un fichier de présences à trois colonnes — nom de l’image, identifiant lu sur la grille, identifiant déduit de la signature. Le Programme 2 ("), it("autoReadForm"), r(") génère, pour chaque PDF, un classeur à deux onglets : "), it("PAGE-01"), r(" (identité, conditions d’examen, notes, validation de signature et de cryptogramme) et "), it("EXAM"), r(" (réponses : choix multiples, mantisse, exposant, unité).")]));
children.push(P([b("Contraintes méthodologiques. "), r("Conformément au cahier des charges, les éléments graphiques doivent être traités exclusivement par des fonctions de bas niveau (filtrage, morphologie, Hough, rotation, composantes connexes) ; l’usage de détecteurs de haut niveau prêts à l’emploi (détection automatique de rectangles ou de cases cochées) est proscrit. À l’inverse, les textes imprimés ou manuscrits peuvent être traités par des méthodes de plus haut niveau, notamment des réseaux de neurones, sous réserve de justifier les choix, les paramètres et les architectures.")]));
children.push(P([b("Difficultés. "), r("Le problème cumule plusieurs difficultés classiques de la vision par ordinateur : géométrie variable des photos prises à main levée (rotation, perspective, ombres), faible contraste de certaines encres, diversité des écritures manuscrites, ressemblance possible entre signatures, et présence — annoncée pour le challenge — d’usurpations d’identité, d’échanges de pages et de fausses signatures.")]));

// 2. État de l'art
children.push(H1("2. État de l’art"));
children.push(P("L’analyse automatique de documents structurés constitue un domaine établi de la vision par ordinateur et de la reconnaissance de formes [1]. Le traitement d’un formulaire se décompose classiquement en trois étapes : prétraitement et normalisation géométrique, segmentation des zones d’intérêt, puis reconnaissance des contenus."));
children.push(P("La normalisation d’images de documents acquises à main levée s’appuie couramment sur la mise en correspondance de points caractéristiques entre l’image et un gabarit de référence. Les détecteurs SIFT [2] et ORB [3], associés à l’estimation d’une homographie robuste par RANSAC [4], permettent de recaler une photographie déformée dans un repère canonique. Le redressement d’inclinaison (deskew) repose fréquemment sur la transformée de Hough [5] appliquée aux contours."));
children.push(P("La lecture de cases à cocher relève de la reconnaissance optique de marques (Optical Mark Recognition). Les approches de bas niveau exploitent le taux de remplissage (densité d’encre) et la morphologie de la marque [6]. La reconnaissance de caractères manuscrits a été popularisée par les bases MNIST et EMNIST [7] et par les réseaux convolutifs, dont l’architecture fondatrice LeNet-5 [8] ; l’apprentissage par transfert permet d’adapter un réseau pré-entraîné à un domaine cible disposant de peu d’exemples [9]. La reconnaissance de texte imprimé est aujourd’hui assurée par des architectures convolutives-récurrentes (CRNN) intégrées dans des bibliothèques open-source telles qu’easyOCR. Enfin, la vérification de signatures hors-ligne combine des descripteurs de forme et de texture — histogrammes de gradients orientés (HOG) [10], moments invariants de Hu [11] — et des mesures de similarité par corrélation de gabarit."));

// 3. Méthodes
children.push(H1("3. Description des méthodes"));

children.push(H2("3.1. Architecture générale et schéma fonctionnel"));
children.push(P([r("Le système est organisé de façon modulaire autour d’un programme principal ("), it("main"), r(") qui définit le nom de l’examen et le répertoire des signatures, en déduit les répertoires d’entrée et de sortie, crée le répertoire de résultats, puis exécute successivement les deux programmes. Chaque programme appelle une sous-fonction par document — "), it("autoValidID"), r(" pour une photo, "), it("autoReadFormID"), r(" pour un PDF. La Figure 1 résume le flux de traitement, des entrées jusqu’aux fichiers produits, en distinguant les trois étages (prétraitement, segmentation, reconnaissance) et la nature des méthodes employées (bas niveau pour le graphique, haut niveau pour le texte).")]));
fig("rapport/figures/fig1_pipeline.png", 580, 390, "Figure 1 — Schéma fonctionnel du système : prétraitement, segmentation et reconnaissance, puis génération des fichiers de résultats par les Programmes 1 et 2.").forEach(p => children.push(p));

children.push(H2("3.2. Prétraitement et normalisation géométrique"));
children.push(P([r("Toutes les lectures s’effectuent dans un "), b("repère canonique"), r(" de dimensions fixes "), it("W"), r("×"), it("H"), r(" = 900×1270 pixels, dans lequel les régions d’intérêt (ROI) de chaque champ sont calibrées une fois pour toutes. Cette normalisation rend les positions stables et évite de redétecter chaque champ.")]));
children.push(P([b("Formulaires PDF. "), r("Chaque page est rendue en image à 150 points par pouce puis ramenée au repère canonique. Les pages étant des numérisations propres, un redimensionnement suffit, complété si nécessaire d’un redressement d’inclinaison.")]));
children.push(P([b("Photographies. "), r("Les photos, prises à main levée, subissent rotation, perspective et ombrage. On les recale sur un gabarit de référence (image normalisée d’une première page) par mise en correspondance de points : des points caractéristiques sont extraits par ORB sur le gabarit et sur la photo, appariés au plus proche voisin avec le test de ratio de Lowe (seuil 0,75), puis une homographie "), it("H"), r(" est estimée de façon robuste par RANSAC. La photo est enfin projetée dans le repère canonique. Pour un point "), it("x"), r(" de la photo et son correspondant "), it("x′"), r(" dans le repère, la relation s’écrit, en coordonnées homogènes :")]));
children.push(EQ("x′ ≃ H · x ,   H ∈ ℝ³ˣ³ ,   estimée par RANSAC (seuil de reprojection 5 px)."));
children.push(P([r("Un chemin de repli (utilisé si le recalage par points échoue) détecte les quatre marques de coin en « L » par corrélation de gabarits synthétiques multi-échelles, puis applique une transformation perspective vers leurs positions attendues. Le redressement d’inclinaison repose sur la transformée de Hough : après détection des contours (Canny) et des droites, l’angle dominant "), it("θ"), r(" est estimé par la médiane des orientations proches de l’horizontale :")]));
children.push(EQ("θ = médiane { αᵢ − 90° : |αᵢ − 90°| < 45° } ,   puis rotation d’angle −θ."));

children.push(H2("3.3. Lecture des éléments graphiques (méthodes de bas niveau)"));
children.push(P([r("Le socle commun est la "), b("binarisation"), r(" d’une ROI par seuillage d’Otsu (inverse), produisant une image "), it("I_b"), r(" où l’encre vaut 1. Le "), b("taux d’encre"), r(" d’une région "), it("R"), r(" est défini par :")]));
children.push(EQ("ρ(R) = (1 / |R|) · Σ_{p ∈ R} 1[ I_b(p) = 1 ]  ∈ [0, 1]."));
children.push(P([b("Identifiant étudiant et groupe (grilles). "), r("L’identifiant et le groupe sont inscrits dans des grilles graphiques : pour chaque position, l’étudiant marque, dans une colonne, la ligne correspondant au chiffre (0–9) ou à la lettre (A–J). Chaque colonne "), it("c"), r(" est binarisée globalement puis découpée en "), it("R"), r(" cellules ; le symbole retenu est la ligne de taux d’encre maximal :")]));
children.push(EQ("d_c = arg max_{r ∈ {0,…,R−1}}  ρ( C_{c,r} ) ,   accepté si  ρ( C_{c,d_c} ) > τ_grille (= 0,04)."));
children.push(P([r("L’identifiant est la concaténation des chiffres lus colonne par colonne ; le code groupe combine deux chiffres et une lettre, soit « G », deux chiffres et une lettre.")]));
children.push(P([b("Cases à cocher (conditions d’examen, choix multiples). "), r("Après binarisation et suppression d’une marge interne (pour exclure la bordure de la case), une case est jugée cochée si son taux d’encre dépasse un seuil "), it("τ_encre"), r(" et qu’elle présente soit une croix, soit un remplissage dense :")]));
children.push(EQ("cochée ⇔ ρ(int) ≥ τ_encre  ∧  ( motifX(int)  ∨  ρ(int) > 0,30 )."));
children.push(P([r("La détection du motif en croix érode légèrement la région (suppression des bords), puis mesure la densité d’encre le long des deux diagonales "), it("ρ_d⁺"), r(" et "), it("ρ_d⁻"), r(" ; une croix est reconnue si les deux diagonales sont suffisamment marquées : motifX ⇔ ρ_d⁺ > τ_diag ∧ ρ_d⁻ > τ_diag, avec τ_diag = 0,12. Les carrés pleins (réponse « OUI » des conditions) sont détectés par un seuil de remplissage plus élevé (0,35).")]));
children.push(P([b("Cryptogramme. "), r("Un petit motif graphique en pied de page doit être identique sur toutes les pages. On compare le cryptogramme de la page 1 à celui de chaque page par corrélation croisée normalisée (NCC) de leurs versions binaires recadrées à taille fixe :")]));
children.push(EQ("NCC(a, b) = Σ (a − ā)(b − b̄) / √( Σ(a − ā)²  ·  Σ(b − b̄)² )  ∈ [−1, 1]."));
children.push(P([r("Le cryptogramme est validé si une majorité des pages exploitables vérifie NCC ≥ 0,50, ce qui tolère les pages où le motif est absent à cette position.")]));

children.push(H2("3.4. Authentification des signatures"));
children.push(P([r("La signature est extraite de l’intérieur de sa boîte (hors label et cadre), puis "), b("normalisée"), r(" : binarisation d’Otsu, suppression des composantes assimilables au cadre, recadrage sur la boîte englobante de l’encre avec marge, redimensionnement à ratio préservé vers un canevas fixe de 192×96 pixels, et recentrage sur le centre de masse afin de réduire la sensibilité à un décalage du tracé.")]));
children.push(P([r("Trois descripteurs complémentaires sont calculés sur l’image normalisée : un "), b("histogramme de gradients orientés"), r(" (HOG, 9 orientations, cellules 16×16, blocs 2×2, normalisation L2-Hys), le vecteur des "), b("sept moments de Hu"), r(" (invariants à la translation, à l’échelle et à la rotation), et l’image binaire elle-même servant de "), b("gabarit"), r(". La similarité entre une signature requête "), it("q"), r(" et une signature de référence "), it("ρ"), r(" combine la corrélation de gabarit et les similarités cosinus des descripteurs :")]));
children.push(EQ("s(q, ρ) = w₁ · NCC(q, ρ) + w₂ · cos( h_q , h_ρ ) + w₃ · cos( u_q , u_ρ ) ,"));
children.push(P([r("avec les poids (w₁, w₂, w₃) = (0,60 ; 0,35 ; 0,05) accordant le rôle principal à la corrélation de forme, et cos(a, b) = ⟨a, b⟩ / (‖a‖·‖b‖). Chaque étudiant possédant plusieurs signatures de référence, on agrège par le maximum, puis on identifie l’étudiant le plus probable :")]));
children.push(EQ("S(id) = max_{ρ ∈ Réf(id)} s(q, ρ) ,   îd = arg max_id S(id) ,   accepté si S(îd) ≥ τ_sig."));
children.push(P([r("L’identifiant ainsi déduit de la signature ("), it("studentID_signature"), r(") est confronté à celui lu sur la grille ("), it("studentID_grid"), r("). L’égalité confirme la présence ; une signature non reconnue laisse la colonne vide ; deux identifiants différents signalent une possible usurpation, que le professeur devra vérifier.")]));

children.push(H2("3.5. Reconnaissance du texte manuscrit (réseaux de neurones)"));
children.push(P([r("Le formulaire est rempli en majuscules, un caractère par case. La reconnaissance des "), b("lettres"), r(" (prénom, nom) et des "), b("chiffres"), r(" (mantisse, exposant) est donc un problème de classification de caractères isolés, traité par deux réseaux de neurones convolutifs partageant la "), b("même architecture"), r(" (Figure 2), seule la tête de classification différant : 26 classes (A–Z) pour les lettres, 10 classes (0–9) pour les chiffres.")]));
fig("rapport/figures/fig2_cnn.png", 600, 180, "Figure 2 — Architecture commune des CNN de reconnaissance de caractères (entrée 28×28, deux blocs convolutifs, classifieur dense). K = 26 pour les lettres, K = 10 pour les chiffres.").forEach(p => children.push(p));
children.push(P([r("Le réseau enchaîne deux blocs convolutifs (deux convolutions 3×3 suivies d’un sous-échantillonnage 2×2) faisant passer la résolution de 28×28 à 7×7 avec 64 canaux, puis un classifieur dense (128 neurones, ReLU, abandon de 0,3) et une couche de sortie à "), it("K"), r(" classes. La probabilité de la classe "), it("k"), r(" est donnée par la fonction softmax sur les logits "), it("z"), r(", et la prédiction est la classe la plus probable :")]));
children.push(EQ("p_k = exp(z_k) / Σ_j exp(z_j) ,   ŷ = arg max_k p_k ."));
children.push(P([b("Apprentissage. "), r("Chaque réseau est entraîné sur la base EMNIST (variante « letters » pour 26 classes, « digits » pour 10 classes), selon la méthodologie imposée de séparation en trois bases distinctes : une base d’apprentissage, une base de validation (10 % du train, pour le suivi de la généralisation et la sauvegarde du meilleur modèle) et une base de test indépendante pour l’évaluation finale. L’optimisation minimise l’entropie croisée")]));
children.push(EQ("L = − Σ_k y_k · log p_k"));
children.push(P([r("par l’algorithme Adam (taux d’apprentissage 10⁻³). Les images EMNIST étant stockées transposées, on les redresse pour qu’elles correspondent au sens de lecture des cases. Le CNN de chiffres atteint "), b("99,56 % d’exactitude sur la base de test"), r(" EMNIST.")]));
children.push(P([b("Affinage sur les données du projet (lettres). "), r("Pour réduire l’écart de domaine entre EMNIST et l’écriture réelle des formulaires, le réseau de lettres est affiné par apprentissage par transfert sur un jeu construit à partir des vérités terrain : les prénoms et noms étant connus, on apparie chaque case non vide à sa lettre lorsque le nombre de cases coïncide avec celui des lettres du nom. On obtient 1 105 lettres issues de 221 formulaires. L’apprentissage utilise une augmentation de données par transformations affines aléatoires (rotation α ∈ [−10°, +10°], translation t ∈ [−2, +2] px), un taux d’apprentissage réduit (3·10⁻⁴) et 25 époques. La validation atteint 65,76 % — chiffre qui reflète la difficulté réelle de l’écriture manuscrite des formulaires et le faible volume, avec des classes rares fortement sous-représentées (F, Q, V, X, Z).")]));
children.push(P([b("Prétraitement des cases. "), r("Avant classification, chaque case est mise au format EMNIST : binarisation d’Otsu inverse, suppression des fragments de cadre (composantes très allongées), recadrage sur la boîte englobante, redimensionnement à ratio préservé vers ~20 px, centrage dans une image 28×28, épaississement du trait et léger flou pour imiter l’anti-aliasing d’EMNIST. La "), b("segmentation"), r(" amont, de bas niveau, isole les caractères par analyse de composantes connexes : une composante est retenue comme chiffre si sa hauteur est proche de la plus grande composante de la case ; le séparateur décimal est identifié comme une composante petite, étroite et située dans la moitié basse de la case. Une garde de « case vide » (taux d’encre, hors cadre, inférieur à 1 %) empêche le réseau de produire des chiffres parasites dans une case non renseignée.")]));

children.push(H2("3.6. Reconnaissance du texte imprimé"));
children.push(P([r("Les champs imprimés de la première page (module, professeur, date, code, notes maximale et de validation) sont lus par reconnaissance optique de caractères (easyOCR), méthode de haut niveau autorisée pour le texte par le cahier des charges. Les ROIs concernées sont agrandies et contrastées (CLAHE) avant reconnaissance ; la date et le code, plus fiables dans l’en-tête, y sont lus en priorité.")]));

children.push(H2("3.7. Paramètres du système"));
children.push(P("Les paramètres de réglage sont centralisés (et non codés en dur de façon dispersée), afin d’être ajustés et justifiés aisément. Les principaux sont rassemblés ci-dessous."));
children.push(table(
  [{ t: "Étape", w: 3000 }, { t: "Paramètre", w: 3526 }, { t: "Valeur", w: 2500, c: true }],
  [
    ["Recalage photo (ORB)", "ratio de Lowe / seuil RANSAC", "0,75 / 5 px"],
    ["Repère canonique", "dimensions (W × H)", "900 × 1270"],
    ["Lecture de grille", "seuil d’acceptation du pic ρ", "0,04"],
    ["Cases à cocher", "seuil d’encre τ_encre / pleine", "0,10 / 0,30"],
    ["Motif en croix", "seuil diagonal τ_diag", "0,12"],
    ["Cryptogramme", "seuil NCC / majorité", "0,50 / 50 %"],
    ["Signature", "poids (NCC, HOG, Hu)", "0,60 / 0,35 / 0,05"],
    ["CNN", "optimiseur / taux d’apprentissage", "Adam / 10⁻³"],
    ["CNN (affinage)", "taux / époques / augmentation", "3·10⁻⁴ / 25 / affine"],
    ["Rendu PDF", "résolution", "150 ppp"],
  ]
));

// 4. Protocole expérimental
children.push(H1("4. Protocole expérimental"));
children.push(P([b("Bases d’apprentissage. "), r("Les CNN sont mis au point sur EMNIST en respectant la séparation apprentissage / validation / test. Les hyper-paramètres (nombre d’époques, sauvegarde du meilleur modèle) sont choisis sur la base de validation ; la base de test, jamais vue à l’apprentissage, mesure la capacité de généralisation. L’affinage des lettres utilise un jeu annoté à partir des vérités terrain du projet, lui-même scindé apprentissage / validation.")]));
children.push(P([b("Évaluation système. "), r("La base fournie (FORM1, FORM2, FORM3) contient les vérités terrain au format tableur. On compare, cellule par cellule, les fichiers produits par le pipeline aux fichiers de référence. La comparaison est rendue robuste aux formats : les dates sont uniformisées, les nombres écrits en texte sont comparés numériquement (par exemple « 2.3 » et 2,3 sont jugés égaux), et le préfixe « G » du code de groupe est neutralisé. Les résultats sont agrégés selon les quatre axes du challenge : authentification des signatures, lecture de l’imprimé, lecture du manuscrit, lecture du graphique. Le temps d’exécution est également mesuré (≈ 900 s pour 43 formulaires).")]));
children.push(P([b("Métrique. "), r("L’exactitude d’un axe est la proportion de cellules correctement lues parmi les cellules renseignées en vérité ou en production :")]));
children.push(EQ("Exactitude(axe) = (nombre de cellules correctes) / (nombre de cellules évaluées)."));

// 5. Résultats et discussion
children.push(H1("5. Résultats expérimentaux et discussion"));
children.push(P("Le tableau ci-dessous présente l’exactitude par axe sur le formulaire FORM1 (43 formulaires PDF), ainsi que les performances des réseaux de reconnaissance de caractères."));
children.push(table(
  [{ t: "Axe / modèle", w: 4026 }, { t: "Mesure", w: 3000 }, { t: "Résultat", w: 2000, c: true }],
  [
    ["Lecture de l’imprimé", "exactitude par cellule (FORM1)", "83,6 %"],
    ["Lecture du graphique", "exactitude par cellule (FORM1)", "61,5 %"],
    ["Authentification signatures", "exactitude par cellule (FORM1)", "58,5 %"],
    ["Lecture du manuscrit", "exactitude par cellule (FORM1)", "22,9 %"],
    [[b("Global (FORM1)")], [b("exactitude par cellule")], [b("63,2 %")]],
    ["CNN chiffres", "exactitude (test EMNIST)", "99,56 %"],
    ["CNN lettres (affiné)", "exactitude (validation, réel)", "65,76 %"],
    ["Lecture de l’identifiant", "grille, PDF (3 formulaires)", "92–100 %"],
  ]
));
children.push(P("Plusieurs enseignements se dégagent de ces résultats."));
children.push(P([b("Le réseau n’est pas le maillon faible du manuscrit. "), r("Le CNN de chiffres atteint 99,56 % sur EMNIST, alors que la lecture manuscrite des formulaires plafonne à 22,9 %. L’écart s’explique par la chaîne autour du réseau (segmentation des cases, détection du séparateur décimal, gestion des cases vides) et par l’écart de domaine entre EMNIST et l’écriture réelle. Les corrections apportées — détection du point décimal par sa position basse, garde anti-parasite sur les cases vides, filtrage des fragments de cadre — ont fait progresser cet axe (de 18,8 % à 22,9 %). Il reste néanmoins l’axe le plus difficile, ce qui est conforme à la littérature : la reconnaissance d’écriture manuscrite libre, sur peu de données annotées et déséquilibrées, est intrinsèquement ardue.")]));
children.push(P([b("Le seuil de détection des cases n’influence pas le résultat. "), r("Un balayage systématique du seuil d’encre des cases à cocher sur l’intervalle [0,08 ; 0,30] laisse l’exactitude des choix multiples strictement inchangée (≈ 59 %). Ce résultat négatif, important, indique que les erreurs ne proviennent pas de la sensibilité d’encre mais de la localisation des cases ; il oriente les travaux futurs vers l’amélioration de la segmentation des cases plutôt que vers le réglage du seuil.")]));
children.push(P([b("Une erreur dans la vérité terrain fournie. "), r("L’examen détaillé des écarts a révélé que, pour le champ « Module », le formulaire imprime « IG.1103 », valeur que le système lit correctement, alors que la vérité terrain indique « IG.2405 » (le numéro du module du projet, et non celui imprimé sur la copie). Ce champ, identique pour tous les formulaires, pénalise donc à tort une quarantaine de cellules. En corrigeant cette incohérence et en comparant les valeurs de façon rigoureuse, l’exactitude globale réelle se situe plutôt autour de 65 %. Cette observation illustre l’importance d’un regard critique sur les données d’évaluation elles-mêmes.")]));
children.push(P([b("Performances par nature d’information. "), r("La lecture de l’imprimé (83,6 %) est la plus fiable et constitue le meilleur candidat à une bonne place sur l’axe correspondant. Le graphique (61,5 %) souffre principalement de la localisation des cases MCQ et de quelques champs de comptage. L’authentification des signatures (58,5 %) est limitée par la qualité variable des photos et la ressemblance de certaines signatures, mais le système distingue correctement présence, non-reconnaissance et divergence d’identité.")]));
children.push(P([b("Coût et robustesse. "), r("Le pipeline complet s’exécute sans aucune correction manuelle et sans plantage sur l’ensemble des formulaires, conformément à la condition d’éligibilité au challenge. Le temps de traitement (~900 s pour 43 formulaires) est dominé par le rendu des PDF et l’inférence des réseaux.")]));

// 6. Conclusion
children.push(H1("6. Conclusion et perspectives"));
children.push(P("Nous avons conçu et mis en œuvre un système complet de vision par ordinateur répondant au cahier des charges de DeepForm : validation des présences et lecture automatisée de formulaires d’examens semi-structurés. L’architecture est modulaire et conforme aux spécifications (programme principal, deux programmes et leurs sous-fonctions, formats de fichiers imposés). Les contraintes méthodologiques sont respectées : les éléments graphiques sont traités exclusivement par des méthodes de bas niveau (seuillage, morphologie, Hough, composantes connexes, corrélation normalisée), tandis que les textes sont confiés à des méthodes de plus haut niveau (OCR pour l’imprimé, CNN pour le manuscrit) dont les architectures et les paramètres sont justifiés et documentés."));
children.push(P("Sur le plan expérimental, le système atteint une exactitude globale de l’ordre de 63 à 65 %, avec des performances solides sur l’imprimé et plus modestes sur le manuscrit — l’axe intrinsèquement le plus difficile. L’étude a également mis en évidence, par une démarche d’analyse critique, deux faits saillants : l’invariance des résultats au seuil de détection des cases, et une erreur dans la vérité terrain fournie."));
children.push(P("Plusieurs perspectives se dégagent pour améliorer les performances :"));
children.push(bullet([b("Affiner le CNN de chiffres"), r(" sur les chiffres réels des formulaires, comme cela a été fait pour les lettres, afin de réduire l’écart de domaine.")]));
children.push(bullet([b("Améliorer la localisation des cases MCQ"), r(" (segmentation et appariement des cases aux lettres), levier identifié comme déterminant pour l’axe graphique.")]));
children.push(bullet([b("Rééquilibrer et augmenter le jeu de lettres"), r(" (classes rares), et étendre l’augmentation de données, pour relever l’exactitude du manuscrit.")]));
children.push(bullet([b("Recaler finement certaines ROIs"), r(" (champs de comptage des conditions) et gérer le signe négatif de l’exposant.")]));
children.push(bullet([b("Adopter une validation croisée"), r(" pour fiabiliser le choix des hyper-paramètres et estimer la variance des performances.")]));

// Références
children.push(H1("Références"));
const refs = [
  "R. Kasturi, L. O’Gorman, V. Govindaraju, « Document image analysis: A primer », Sadhana, 2002.",
  "D. G. Lowe, « Distinctive Image Features from Scale-Invariant Keypoints », International Journal of Computer Vision, 2004.",
  "E. Rublee, V. Rabaud, K. Konolige, G. Bradski, « ORB: An efficient alternative to SIFT or SURF », ICCV, 2011.",
  "M. A. Fischler, R. C. Bolles, « Random Sample Consensus (RANSAC) », Communications of the ACM, 1981.",
  "R. O. Duda, P. E. Hart, « Use of the Hough Transformation to Detect Lines and Curves in Pictures », Comm. of the ACM, 1972.",
  "R. Smith, « An Overview of the Tesseract OCR Engine », ICDAR, 2007.",
  "G. Cohen, S. Afshar, J. Tapson, A. van Schaik, « EMNIST: an extension of MNIST to handwritten letters », IJCNN, 2017.",
  "Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, « Gradient-Based Learning Applied to Document Recognition », Proceedings of the IEEE, 1998.",
  "J. Yosinski, J. Clune, Y. Bengio, H. Lipson, « How transferable are features in deep neural networks? », NeurIPS, 2014.",
  "N. Dalal, B. Triggs, « Histograms of Oriented Gradients for Human Detection », CVPR, 2005.",
  "M.-K. Hu, « Visual Pattern Recognition by Moment Invariants », IRE Transactions on Information Theory, 1962.",
];
refs.forEach((t, i) => children.push(new Paragraph({
  numbering: { reference: "refs", level: 0 }, spacing: { after: 70, line: 264 },
  alignment: AlignmentType.JUSTIFIED, children: [new TextRun({ text: t, size: 20 })],
})));

// ---------- document ----------
const doc = new Document({
  creator: "Équipe IG.2405",
  title: "Lecture automatique de formulaires d’examens semi-structurés",
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: BLUE },
        paragraph: { spacing: { before: 220, after: 110 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: NAVY },
        paragraph: { spacing: { before: 160, after: 90 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 280 } } } }] },
      { reference: "n1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] },
      { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "[%1]", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 420 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Projet IG.2405 — DeepForm    •    ", size: 16, color: GREY }),
                 new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: GREY })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("rapport/Rapport_IG2405_DeepForm.docx", buf);
  console.log("OK -> rapport/Rapport_IG2405_DeepForm.docx", buf.length, "octets");
});
