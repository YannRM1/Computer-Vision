"""
Entrainement d'un CNN de reconnaissance de lettres manuscrites (A-Z) sur EMNIST.

Conforme a la consigne (sec. 4.1) : les textes manuscrits peuvent etre traites
par reseau de neurones. Le formulaire est rempli en MAJUSCULES, une lettre par
case -> classification 26 classes.

Usage (sur une machine disposant de torch + torchvision) :
    pip install torch torchvision
    python train_letter_cnn.py            # entrainement complet
    python train_letter_cnn.py --epochs 8 --batch 256

Sortie : models/letter_cnn.pt (poids charges par utils/letter_cnn.py).

Methodologie (sec. 4.2.1) : EMNIST fournit deja un split train/test. On reserve
en plus une partie du train comme validation pour suivre la generalisation.

NB orientation : les images EMNIST sont stockees transposees. On applique une
transposition pour obtenir des lettres "droites", cohérentes avec les cases du
formulaire (lues a l'endroit a l'inference).
"""

import argparse
import os

import numpy as np


def _fix_orientation(t):
    """Redresse les images EMNIST (stockees transposees). t : (1,28,28)."""
    import torch
    return torch.transpose(t, 1, 2)


def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.letter_cnn import build_model, MODEL_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", default="./emnist_data")
    ap.add_argument("--finetune", default=None,
                    help="chemin du letter_dataset.npz pour fine-tuner sur tes formulaires")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    if args.finetune:
        finetune(args); return

    # Correction d'orientation EMNIST : transposer (H<->W) pour redresser.
    # (fonction au niveau module -> picklable, requis par DataLoader Windows)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_fix_orientation),
    ])

    # EMNIST 'letters' : labels 1..26 -> on mappe vers 0..25.
    full_train = datasets.EMNIST(args.data, split="letters", train=True,
                                 download=True, transform=tfm)
    test = datasets.EMNIST(args.data, split="letters", train=False,
                           download=True, transform=tfm)

    n_val = len(full_train) // 10
    n_tr = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42))
    print(f"[train] train={n_tr} val={n_val} test={len(test)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test, batch_size=512)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def evaluate(dl):
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), (y - 1).to(device)   # 1..26 -> 0..25
                ok += (model(x).argmax(1) == y).sum().item()
                tot += y.numel()
        return 100.0 * ok / tot

    best = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), (y - 1).to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        val_acc = evaluate(val_dl)
        print(f"[train] epoch {ep}/{args.epochs} "
              f"loss={running/n_tr:.4f} val_acc={val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"          -> sauvegarde {MODEL_PATH} (best val {best:.2f}%)")

    # Recharge le meilleur modele et evalue sur le test.
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[train] TEST accuracy = {evaluate(test_dl):.2f}%")
    print(f"[train] termine. Poids : {MODEL_PATH}")


def finetune(args):
    """Fine-tune le CNN pre-entraine EMNIST sur les lettres reelles des
    formulaires (letter_dataset.npz). Petite base -> augmentation + LR faible.

    Methodologie (sec. 4.2.2 : « privilegier la validation croisee ») :
      1. Validation croisee K-fold PAR ETUDIANT (GroupKFold) : toutes les
         lettres d'un meme scripteur restent dans le meme fold, sinon le score
         est optimiste (le reseau reconnait l'ecriture, pas la lettre).
         Chaque fold repart des poids EMNIST PURS (models/letter_cnn_emnist.pt)
         pour eviter toute fuite via des poids deja affines.
         -> rapporte accuracy moyenne +/- ecart-type.
      2. Modele FINAL : re-entraine sur 100 %% des donnees avec la meme
         recette, sauvegarde dans models/letter_cnn.pt.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.letter_cnn import build_model, MODEL_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.finetune)
    X = torch.from_numpy(data["X"]).float().unsqueeze(1)   # (N,1,28,28)
    y = torch.from_numpy(data["y"]).long()
    if "groups" not in data:
        print("[finetune] ERREUR : letter_dataset.npz sans 'groups' "
              "(re-generer avec build_letter_dataset.py).")
        return
    groups = data["groups"]
    print(f"[finetune] {len(X)} lettres, {len(set(y.tolist()))} classes, "
          f"{len(np.unique(groups))} etudiants")

    # Poids de depart du transfert : EMNIST purs (jamais vus les formulaires).
    base_path = os.path.join(os.path.dirname(os.path.abspath(MODEL_PATH)),
                             "letter_cnn_emnist.pt")
    if not os.path.isfile(base_path):
        print(f"[finetune] ERREUR : base EMNIST absente ({base_path}).")
        return
    base_state = torch.load(base_path, map_location=device)

    def augment(batch):
        # petites perturbations affines (rot/translation) pour densifier
        import torchvision.transforms.functional as TF
        out = []
        for img in batch:
            ang = float(np.random.uniform(-10, 10))
            tx = int(np.random.uniform(-2, 2)); ty = int(np.random.uniform(-2, 2))
            out.append(TF.affine(img, angle=ang, translate=[tx, ty],
                                 scale=1.0, shear=[0.0, 0.0]))
        return torch.stack(out)

    crit = nn.CrossEntropyLoss()
    epochs = max(args.epochs if args.epochs else 30, 25)

    def train_once(tr_idx, va_idx=None, tag=""):
        """Entraine un modele frais (base EMNIST) sur tr_idx ; renvoie la
        meilleure accuracy sur va_idx (ou None si pas de validation)."""
        model = build_model().to(device)
        model.load_state_dict({k: v.clone() for k, v in base_state.items()})
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        tr_dl = DataLoader(TensorDataset(X[tr_idx], y[tr_idx]),
                           batch_size=64, shuffle=True, num_workers=0)
        va_dl = (DataLoader(TensorDataset(X[va_idx], y[va_idx]),
                            batch_size=256, num_workers=0)
                 if va_idx is not None else None)

        def acc(dl):
            model.eval(); ok = tot = 0
            with torch.no_grad():
                for xb, yb in dl:
                    xb, yb = xb.to(device), yb.to(device)
                    ok += (model(xb).argmax(1) == yb).sum().item()
                    tot += yb.numel()
            return 100.0 * ok / max(1, tot)

        best = 0.0
        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in tr_dl:
                xb = augment(xb).to(device); yb = yb.to(device)
                opt.zero_grad(); loss = crit(model(xb), yb)
                loss.backward(); opt.step()
            if va_dl is not None:
                best = max(best, acc(va_dl))
        if va_dl is not None:
            print(f"[finetune]{tag} val_acc(best)={best:.2f}%")
        return model, best

    # ---- 1) Validation croisee par etudiant (GroupKFold, K=5) -------------
    from sklearn.model_selection import GroupKFold
    K = 5
    scores = []
    for k, (tr_idx, va_idx) in enumerate(
            GroupKFold(n_splits=K).split(X, y, groups), start=1):
        _, sc = train_once(torch.as_tensor(tr_idx),
                           torch.as_tensor(va_idx), tag=f" fold {k}/{K}")
        scores.append(sc)
    mean, std = float(np.mean(scores)), float(np.std(scores))
    print(f"[finetune] CV {K}-fold (par etudiant) : "
          f"{mean:.2f}% +/- {std:.2f}  (folds: "
          + " ".join(f"{s:.1f}" for s in scores) + ")")

    # ---- 2) Modele final : re-entrainement sur TOUTES les donnees ---------
    model, _ = train_once(torch.arange(len(X)), None, tag=" final")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[finetune] modele final (100% des donnees) -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
