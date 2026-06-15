"""
Compare les xlsx produits par ton pipeline aux xlsx vérités terrain.

Vérité terrain  : PROJECT 2026 -DATABASE-20260518/FORMX/EXAM_FORMX_NNNNN.xlsx
                  (fourni par les profs, à côté de chaque pdf/photo)
Production      : EXAM_FORMX_RESULTS/EXAM_FORMX_NNNNN.xlsx
                  (généré par autoReadForm)

Pour chaque xlsx :
  - Onglet PAGE-01 : 18 lignes (Module, Professor, Date, ..., STUDENT ID)
  - Onglet EXAM    : grille des réponses (CHOIX A-H, MANTISSE, EXPOSANT, UNITE)

Sortie :
  - Récap par axe (imprimé / manuscrit / graphique / signature)
  - CSV détaillé : compare_results.csv  (champ par champ, OK/KO/MISSING)

Usage :
  python compare_to_truth.py [chemin_DATABASE] [chemin_RESULTS]
"""

import os
import sys
import csv
import re
from datetime import datetime

import openpyxl


# Mapping ligne PAGE-01 -> axe d'évaluation (cf. §6 du sujet)
PAGE01_AXES = {
    1:  ("Module",                  "imprime"),
    2:  ("Professor",               "imprime"),
    3:  ("Date",                    "imprime"),
    4:  ("Code",                    "imprime"),
    5:  ("Notes de cours",          "graphique"),
    6:  ("Notes manuscrites",       "graphique"),
    7:  ("Ordinateur portable",     "graphique"),
    8:  ("Calculatrice",            "graphique"),
    9:  ("Feuilles brouillon",      "graphique"),
    10: ("Note maximale",           "imprime"),
    11: ("Note pour valider",       "imprime"),
    13: ("Prenom",                  "manuscrit"),
    14: ("Nom",                     "manuscrit"),
    15: ("Validation signature",    "signature"),
    16: ("Group",                   "graphique"),
    17: ("STUDENT ID",              "graphique"),
    18: ("Validation cryptogramme", "graphique"),
}


_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
                 "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y")


def normalize_value(v):
    """Normalise pour comparaison (date->str, str->lower trimmed, etc.)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        # Normaliser toutes les dates au format DD/MM/YYYY
        return v.strftime("%d/%m/%Y")
    if isinstance(v, str):
        s = v.strip()
        # Tenter de parser comme date pour uniformiser le format
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt).strftime("%d/%m/%Y")
            except ValueError:
                continue
        # Valeur numérique stockée en texte ('2.3', '17.0') -> nombre, pour
        # comparer 2.3 (vérité texte) == 2.3 (float produit) numériquement.
        try:
            return float(s.replace(",", "."))
        except ValueError:
            pass
        return s.lower()
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def cells_equal(a, b):
    na = normalize_value(a)
    nb = normalize_value(b)
    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    if isinstance(na, (int, float)) and isinstance(nb, (int, float)):
        return abs(na - nb) < 1e-3
    return na == nb


def _norm_group(v):
    """Group : ignore le préfixe 'G' (la vérité terrain est incohérente :
    'G04E' ici, '05C' là). Comparer le code de groupe sans ce préfixe."""
    if v is None:
        return None
    s = str(v).strip().upper()
    return s[1:] if s.startswith("G") else s


def compare_page1(prod_ws, truth_ws):
    """Retourne liste de tuples (ligne, libellé, axe, truth, prod, ok)."""
    rows = []
    for r, (label, axis) in PAGE01_AXES.items():
        t = truth_ws.cell(r, 2).value
        p = prod_ws.cell(r, 2).value
        if label == "Group":                       # comparer sans le préfixe 'G'
            ok = cells_equal(_norm_group(t), _norm_group(p))
        else:
            ok = cells_equal(t, p)
        rows.append((r, label, axis, t, p, ok))
    return rows


def compare_exam(prod_ws, truth_ws):
    """Compare l'onglet EXAM par cellule. Retourne (cellule, axe, truth, prod, ok)."""
    rows = []
    # Headers: col 1=QUESTION, col 2-9=CHOIX A-H, col 10=MANTISSE, col 11=EXPOSANT, col 12=UNITE
    axis_map = {1: "imprime", 10: "manuscrit", 11: "manuscrit", 12: "manuscrit"}
    for c in range(2, 10): axis_map[c] = "graphique"  # CHOIX A-H

    def _is_checked(v):
        """Une case MCQ est 'cochée' qu'elle soit notée 1, 'X' ou 'x' dans la
        vérité terrain (convention incohérente d'un fichier à l'autre)."""
        if v is None:
            return False
        s = str(v).strip().lower()
        return s not in ("", "0", "none")

    max_r = max(truth_ws.max_row, prod_ws.max_row)
    for r in range(2, max_r + 1):
        # Skip empty truth rows
        if truth_ws.cell(r, 1).value is None:
            continue
        for c in range(1, 13):
            t = truth_ws.cell(r, c).value
            p = prod_ws.cell(r, c).value if r <= prod_ws.max_row else None
            if t is None and p is None:
                continue
            axis = axis_map.get(c, "autre")
            if 2 <= c <= 9:                       # CHOIX A-H : comparer "coché/non"
                ok = (_is_checked(t) == _is_checked(p))
            else:
                ok = cells_equal(t, p)
            cell = f"R{r}C{c}"
            rows.append((cell, axis, t, p, ok))
    return rows


def _collect_xlsx(root, production):
    """Indexe les EXAM_<FORM>_<ID>.xlsx sous `root` par (form, id).
    production=True : ceux dans un dossier *_RESULTS ; False : les autres."""
    found = {}
    for dirpath, _dirs, files in os.walk(root):
        in_results = os.path.basename(dirpath.rstrip("/\\")).upper().endswith("_RESULTS")
        if production != in_results:
            continue
        for f in files:
            m = re.match(r"(?:EXAM_)?(\w+?)_(\d+)\.xlsx$", f)
            if m:
                found[(m.group(1), m.group(2))] = os.path.join(dirpath, f)
    return found


def find_pairs(data_root, results_root, only_form=None):
    """Paires (form, id, vérité, production), appariées par nom de fichier.
    only_form ('FORM1') restreint à un formulaire ; None/'ALL' = tous."""
    truth = _collect_xlsx(data_root, production=False)
    prod  = _collect_xlsx(results_root, production=True)
    keys = truth.keys() & prod.keys()
    if only_form and only_form.upper() != "ALL":
        keys = {(f, i) for (f, i) in keys if f.upper() == only_form.upper()}
    return [(f, i, truth[(f, i)], prod[(f, i)]) for (f, i) in sorted(keys)]


def _pairs_in_dirs(truth_dir, results_dir):
    """Paires entre deux dossiers précis, appariées par nom de fichier
    (le jeton FORM des noms peut différer du dossier ; on ne s'y fie pas)."""
    pairs = []
    if not (os.path.isdir(truth_dir) and os.path.isdir(results_dir)):
        return pairs
    for f in sorted(os.listdir(truth_dir)):
        m = re.match(r"(?:EXAM_)?(\w+?)_(\d+)\.xlsx$", f)
        if m and os.path.isfile(os.path.join(results_dir, f)):
            pairs.append((m.group(1), m.group(2),
                          os.path.join(truth_dir, f),
                          os.path.join(results_dir, f)))
    return pairs


def _main_config():
    """(base, formulaire) repris de main.py : sans argument on compare le
    seul examen configuré là-bas (EXAM_NAME)."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from main import DATA_ROOT, EXAM_NAME, _form_folder
        return DATA_ROOT, _form_folder(EXAM_NAME)
    except Exception:
        return "PROJECT 2026 -DATABASE-20260518", None


def main():
    # Sans argument : config de main.py -> on cible les dossiers de l'examen
    # choisi (<base>/<form> et EXAM_<form>_RESULTS), appariés par nom de
    # fichier. Avec arguments : base donnée, 3e argument = formulaire (sinon tous).
    if len(sys.argv) > 1:
        data_root    = sys.argv[1]
        results_root = sys.argv[2] if len(sys.argv) > 2 else "."
        only_form    = sys.argv[3] if len(sys.argv) > 3 else None
        pairs = find_pairs(data_root, results_root, only_form)
        base_label, form_label = data_root, (only_form or "tous")
    else:
        data_root, form = _main_config()
        results_root = "."
        # dossier vérité : 'FORM1' ou 'EXAM_FORM1' selon le nommage de la prof
        truth_dir = next((os.path.join(data_root, n) for n in (form, "EXAM_" + form)
                          if os.path.isdir(os.path.join(data_root, n))),
                         os.path.join(data_root, form))
        pairs = _pairs_in_dirs(truth_dir,
                               os.path.join(results_root, f"EXAM_{form}_RESULTS"))
        base_label = os.path.basename(data_root.rstrip("/\\")) or data_root
        form_label = form

    if not pairs:
        print(f"[!] Aucun fichier de production trouvé pour {form_label}.")
        print(f"    Vérité : {base_label}  |  Production : {results_root}")
        print(f"    -> Lance d'abord 'python main.py' pour générer les xlsx de prod.")
        return

    # Stats par axe
    axis_stats = {}  # axis -> [ok, total]
    csv_rows = [["form", "id", "sheet", "field_or_cell", "axis",
                 "truth", "prod", "ok"]]

    skipped = 0
    for form, sid, tp, pp in pairs:
        try:
            twb = openpyxl.load_workbook(tp, data_only=True)
        except Exception as e:
            print(f"  [WARN] Vérité terrain illisible, ignoré : {os.path.basename(tp)} ({e})")
            skipped += 1
            continue
        try:
            pwb = openpyxl.load_workbook(pp, data_only=True)
        except Exception as e:
            print(f"  [WARN] Production illisible, ignoré : {os.path.basename(pp)} ({e})")
            skipped += 1
            continue

        # PAGE-01
        if "PAGE-01" in twb.sheetnames and "PAGE-01" in pwb.sheetnames:
            for r, label, axis, t, p, ok in compare_page1(pwb["PAGE-01"],
                                                          twb["PAGE-01"]):
                axis_stats.setdefault(axis, [0, 0])
                axis_stats[axis][1] += 1
                if ok: axis_stats[axis][0] += 1
                csv_rows.append([form, sid, "PAGE-01", label, axis,
                                 str(t), str(p), int(ok)])
        # EXAM
        if "EXAM" in twb.sheetnames and "EXAM" in pwb.sheetnames:
            for cell, axis, t, p, ok in compare_exam(pwb["EXAM"], twb["EXAM"]):
                axis_stats.setdefault(axis, [0, 0])
                axis_stats[axis][1] += 1
                if ok: axis_stats[axis][0] += 1
                csv_rows.append([form, sid, "EXAM", cell, axis,
                                 str(t), str(p), int(ok)])

    if skipped:
        print(f"\n  [INFO] {skipped} fichier(s) ignoré(s) (xlsx corrompu ou manquant)")

    out_csv = "compare_results.csv"
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    print(f"\n=== Comparaison vérité vs production ({len(pairs)} xlsx) ===")
    print(f"Base : {base_label}  |  Formulaire : {form_label}\n")
    print(f"{'AXE':<12}{'OK':>6}{'TOTAL':>8}{'ACC':>10}")
    print("-" * 36)
    for axis in ("imprime", "manuscrit", "graphique", "signature", "autre"):
        if axis not in axis_stats: continue
        ok, tot = axis_stats[axis]
        print(f"{axis:<12}{ok:>6}{tot:>8}{100*ok/max(1,tot):>9.1f}%")
    print("-" * 36)
    total_ok = sum(v[0] for v in axis_stats.values())
    total    = sum(v[1] for v in axis_stats.values())
    print(f"{'GLOBAL':<12}{total_ok:>6}{total:>8}{100*total_ok/max(1,total):>9.1f}%")

    # ---- Vue par formulaire (axe en colonnes) -----------------------------
    data = csv_rows[1:]
    forms = sorted({r[0] for r in data})
    axes  = ("imprime", "manuscrit", "graphique", "signature")
    print(f"\n{'FORM':<8}" + "".join(f"{a:>11}" for a in axes) + f"{'GLOBAL':>9}")
    for fo in forms:
        cells, tot = [], [0, 0]
        for a in axes:
            sub = [r for r in data if r[0] == fo and r[4] == a]
            o = sum(r[7] for r in sub)
            tot[0] += o; tot[1] += len(sub)
            cells.append(f"{100*o/max(1,len(sub)):>10.1f}%")
        print(f"{fo:<8}" + "".join(cells)
              + f"{100*tot[0]/max(1,tot[1]):>8.1f}%")

    # ---- Détail de l'axe manuscrit (composants) ---------------------------
    def comp(r):
        if r[3] in ("Prenom", "Nom"): return "Noms"
        if r[3].endswith("C10"): return "MANTISSE"
        if r[3].endswith("C11"): return "EXPOSANT"
        return "UNITE"
    print(f"\n{'MANUSCRIT':<12}{'OK':>6}{'TOTAL':>8}{'ACC':>10}")
    man = [r for r in data if r[4] == "manuscrit"]
    for k in ("MANTISSE", "EXPOSANT", "Noms", "UNITE"):
        sub = [r for r in man if comp(r) == k]
        o = sum(r[7] for r in sub)
        print(f"{k:<12}{o:>6}{len(sub):>8}{100*o/max(1,len(sub)):>9.1f}%")

    print(f"\nDétails -> {out_csv}")


if __name__ == "__main__":
    main()
