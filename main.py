"""
Programme principal – Correction automatique d'examens
PROJET COMPUTER VISION IG.2405 – 2026

Usage :
    python main.py [exam_name] [sig_dir] [presences_dir] [pdf_dir]

  exam_name     (optionnel) : nom de l'examen, ex. 'EXAM_FORM2'
  sig_dir       (optionnel) : chemin vers la base de signatures
  presences_dir (optionnel) : répertoire des photos de 1re page
  pdf_dir       (optionnel) : répertoire des formulaires PDF scannés
"""

import os
import sys
import time

from autoValidPresences import autoValidPresences
from autoReadForm       import autoReadForm

# =============================================================================
# CONFIGURATION – adapter ces chemins pour le challenge (§3.6 de la consigne)
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nom de l'examen à traiter
EXAM_NAME = "EXAM_FORM1"

# Répertoire racine des données
DATA_ROOT = os.path.join(BASE_DIR, "PROJECT 2026 -DATABASE-20260518")

# Répertoire des signatures
SIGNATURES_DIR = os.path.join(DATA_ROOT, "SIGNATURES")

# Répertoires des données d'examen.
# Dans la base actuelle, photos et PDFs sont dans le même dossier.
# Pour le challenge avec répertoires distincts, modifier directement ces deux lignes :
#   PRESENCES_DIR = r"C:\...\EXAM_FORMXX_PRESENCES"
#   PDF_DIR       = r"C:\...\EXAM_FORMXX_PDF"
_form_num     = EXAM_NAME.split("_")[-1]              # ex: "FORM1"
PRESENCES_DIR = os.path.join(DATA_ROOT, _form_num)    # photos de 1re page
PDF_DIR       = os.path.join(DATA_ROOT, _form_num)    # formulaires scannés

# =============================================================================
# Ne pas modifier ci-dessous
# =============================================================================

def _dirs_for_exam(exam_name: str) -> tuple[str, str]:
    """
    Déduit (presences_dir, pdf_dir) du nom d'examen, comme le fait la section
    CONFIGURATION pour EXAM_FORM1. Indispensable pour que
    `python main.py EXAM_FORM2 ...` traite bien FORM2 et non FORM1 : sans cela,
    les répertoires restaient figés sur ceux dérivés au chargement du module.
    """
    form_num = exam_name.split("_")[-1]            # ex: "FORM2"
    d = os.path.join(DATA_ROOT, form_num)
    return d, d


def main(exam_name: str = EXAM_NAME,
         sig_dir: str = SIGNATURES_DIR,
         presences_dir: str | None = None,
         pdf_dir: str | None = None) -> None:
    t0 = time.time()

    # Répertoires déduits du nom d'examen si non fournis explicitement.
    if presences_dir is None or pdf_dir is None:
        _pres, _pdf = _dirs_for_exam(exam_name)
        presences_dir = presences_dir or _pres
        pdf_dir = pdf_dir or _pdf

    results_dir = os.path.join(BASE_DIR, exam_name + "_RESULTS")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f" Traitement : {exam_name}")
    print(f" Présences  : {presences_dir}")
    print(f" PDFs       : {pdf_dir}")
    print(f" Signatures : {sig_dir}")
    print(f" Résultats  : {results_dir}")
    print("=" * 60)

    # ---- PROGRAMME 1 : Validation des présences --------------------------
    print("\n--- PROGRAMME 1 : Validation des présences ---")
    presences_xlsx = autoValidPresences(
        presences_dir=presences_dir,
        signatures_dir=sig_dir,
        results_dir=results_dir,
        pdf_dir=pdf_dir,
    )

    # ---- PROGRAMME 2 : Lecture automatique des formulaires ---------------
    print("\n--- PROGRAMME 2 : Lecture automatique des formulaires ---")
    generated_xlsx = autoReadForm(
        pdf_dir=pdf_dir,
        signatures_dir=sig_dir,
        results_dir=results_dir,
    )

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f" Terminé en {elapsed:.1f}s")
    print(f" Fichier présences   : {os.path.basename(presences_xlsx)}")
    print(f" Fichiers formulaires: {len(generated_xlsx)} xlsx générés")
    print("=" * 60)


if __name__ == "__main__":
    exam = sys.argv[1] if len(sys.argv) > 1 else EXAM_NAME
    sigs = sys.argv[2] if len(sys.argv) > 2 else SIGNATURES_DIR
    # Si les répertoires ne sont pas passés, ils sont déduits de exam_name
    # dans main() (None -> dérivation), pour éviter de rester figé sur FORM1.
    pres = sys.argv[3] if len(sys.argv) > 3 else None
    pdfs = sys.argv[4] if len(sys.argv) > 4 else None
    main(exam_name=exam, sig_dir=sigs, presences_dir=pres, pdf_dir=pdfs)
