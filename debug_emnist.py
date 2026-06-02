"""
Diagnostic d'orientation EMNIST.

Sauvegarde 'emnist_orientation.png' : pour quelques echantillons, 4 variantes
d'orientation cote a cote avec leur vraie lettre. On choisit la variante ou la
lettre est DROITE et LISIBLE -> c'est la transformation a appliquer a
l'entrainement.

Usage : python debug_emnist.py
"""
import os, sys
import numpy as np
import cv2


def main():
    import torch
    from torchvision import datasets, transforms

    ds = datasets.EMNIST("./emnist_data", split="letters", train=False,
                         download=True, transform=transforms.ToTensor())

    # quelques echantillons varies
    idxs = [0, 1, 2, 3, 4, 5, 6, 7]
    variants = {
        "raw":            lambda a: a,
        "transpose":      lambda a: a.T,
        "rot90CW+flipH":  lambda a: np.fliplr(np.rot90(a, k=3)),
        "rot90CCW":       lambda a: np.rot90(a, k=1),
    }
    names = list(variants)
    cell = 56
    rows = []
    # entete
    for i in idxs:
        img, lab = ds[i]
        letter = chr(ord('A') + (lab - 1))
        a = (img[0].numpy() * 255).astype(np.uint8)
        col = []
        for nm in names:
            v = variants[nm](a).astype(np.uint8)
            v = cv2.resize(v, (cell, cell), interpolation=cv2.INTER_NEAREST)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
            cv2.putText(v, letter, (2, 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 0, 255), 1)
            col.append(v)
        rows.append(np.vstack(col))
    montage = np.hstack(rows)
    # bandeau de noms de variantes a gauche
    label_col = np.full((montage.shape[0], 120, 3), 255, np.uint8)
    for r, nm in enumerate(names):
        cv2.putText(label_col, nm, (2, r * cell + cell // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    montage = np.hstack([label_col, montage])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "emnist_orientation.png")
    cv2.imwrite(out, montage)
    print("Sauvegarde:", out)
    print("Lignes = variantes d'orientation ; la lettre rouge = verite.")
    print("Repere la ligne ou les lettres sont DROITES et lisibles.")


if __name__ == "__main__":
    main()
