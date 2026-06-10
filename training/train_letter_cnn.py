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
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    formulaires (letter_dataset.npz). Petite base -> augmentation + LR faible."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.letter_cnn import build_model, MODEL_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.finetune)
    X = torch.from_numpy(data["X"]).float().unsqueeze(1)   # (N,1,28,28)
    y = torch.from_numpy(data["y"]).long()
    print(f"[finetune] {len(X)} lettres, {len(set(y.tolist()))} classes presentes")

    ds = TensorDataset(X, y)
    n_val = max(1, len(ds) // 6)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))

    def augment(batch):
        # petites perturbations affine (rot/translation) pour densifier
        import torchvision.transforms.functional as TF
        out = []
        for img in batch:
            ang = float(np.random.uniform(-10, 10))
            tx = int(np.random.uniform(-2, 2)); ty = int(np.random.uniform(-2, 2))
            out.append(TF.affine(img, angle=ang, translate=[tx, ty],
                                 scale=1.0, shear=[0.0, 0.0]))
        return torch.stack(out)

    tr_dl = DataLoader(tr, batch_size=64, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=256, num_workers=0)

    model = build_model().to(device)
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[finetune] poids EMNIST charges (transfert).")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = nn.CrossEntropyLoss()

    def acc(dl):
        model.eval(); ok = tot = 0
        with torch.no_grad():
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                ok += (model(xb).argmax(1) == yb).sum().item(); tot += yb.numel()
        return 100.0 * ok / max(1, tot)

    best = 0.0
    epochs = args.epochs if args.epochs else 30
    epochs = max(epochs, 25)
    for ep in range(1, epochs + 1):
        model.train(); run = 0.0
        for xb, yb in tr_dl:
            xb = augment(xb).to(device); yb = yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            run += loss.item() * xb.size(0)
        va = acc(va_dl)
        print(f"[finetune] epoch {ep}/{epochs} loss={run/len(tr):.4f} val_acc={va:.2f}%")
        if va >= best:
            best = va
            torch.save(model.state_dict(), MODEL_PATH)
    print(f"[finetune] termine. best val_acc={best:.2f}% -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
