"""
Programme principal – Correction automatique d'examens
PROJET COMPUTER VISION IG.2405 – 2026

Usage :
    python main.py [exam_name] [sig_dir]

  exam_name (optionnel) : nom de l'examen, ex. 'EXAM_FORM2'
                          (défaut défini dans la section CONFIG)
  sig_dir   (optionnel) : chemin vers la base de signatures
                          (défaut défini dans la section CONFIG)
"""

import os
import sys
import time

from autoValidPresences import autoValidPresences
from autoReadForm       import autoReadForm

# =============================================================================
# CONFIGURATION – adapter ces chemins pour le challenge
# =============================================================================

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))

# Nom de l'examen à traiter
EXAM_NAME = "EXAM_FORM2"

# Répertoire racine des données (changer si nécessaire)
DATA_ROOT = os.path.join(BASE_DIR, "PROJECT 2026 -DATABASE-20260518")

# Répertoire des signatures
SIGNATURES_DIR = os.path.join(DATA_ROOT, "SIGNATURES")

# =============================================================================
# Ne pas modifier ci-dessous (déduit automatiquement)
# =============================================================================

def main(exam_name: str = EXAM_NAME, sig_dir: str = SIGNATURES_DIR) -> None:
    t0 = time.time()

    # Déduire les répertoires d'entrée et de sortie
    form_num = exam_name.split("_")[-1]   # ex: 'FORM2'
    data_dir = os.path.join(DATA_ROOT, form_num)

    presences_dir = data_dir   # les photos (.jpg/.jpeg) et PDFs sont dans le même dossier
    pdf_dir       = data_dir   # idem
    results_dir   = os.path.join(BASE_DIR, exam_name + "_RESULTS")

    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f" Traitement : {exam_name}")
    print(f" Signatures : {sig_dir}")
    print(f" Résultats  : {results_dir}")
    print("=" * 60)

    # ---- PROGRAMME 1 : Validation des présences --------------------------
    print("\n--- PROGRAMME 1 : Validation des présences ---")
    presences_xlsx = autoValidPresences(
        presences_dir=presences_dir,
        signatures_dir=sig_dir,
        results_dir=results_dir,
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
    # Permettre de surcharger exam_name et sig_dir depuis la ligne de commande
    exam = sys.argv[1] if len(sys.argv) > 1 else EXAM_NAME
    sigs = sys.argv[2] if len(sys.argv) > 2 else SIGNATURES_DIR
    main(exam_name=exam, sig_dir=sigs)
