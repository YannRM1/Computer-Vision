"""
Entrainement d'un CNN de reconnaissance de chiffres manuscrits (0-9) sur
EMNIST-digits.

Conforme a la consigne (sec. 4.1) : les textes manuscrits peuvent etre traites
par reseau de neurones. Les reponses numeriques (mantisse, exposant de l'onglet
EXAM) sont remplies a la main, un chiffre par case -> classification 10 classes.

Architecture : identique au CNN de lettres (utils/digit_cnn.build_model), seule
la tete de sortie change (10 classes). Un seul schema a decrire dans le rapport.

Methodologie (sec. 4.2.1) : EMNIST fournit deja un split train/test. On reserve
en plus une partie du train comme VALIDATION pour suivre la generalisation et
sauvegarder le meilleur modele. Le TEST n'est utilise qu'a la fin.

Les fichiers EMNIST sont lus directement depuis ./emnist_data/EMNIST/raw/
(format idx, deja telecharge) : aucune dependance reseau.

NB orientation : les images EMNIST sont stockees transposees. On applique une
transposition (H<->W) pour obtenir des chiffres "droits", coherents avec les
cases du formulaire lues a l'endroit a l'inference.

Usage :
    pip install torch torchvision
    python train_digit_cnn.py                 # entrainement complet
    python train_digit_cnn.py --epochs 5 --subset 60000
    python train_digit_cnn.py --finetune digit_dataset.npz
"""

import argparse
import os
import struct

import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "emnist_data", "EMNIST", "raw")


# ---------------------------------------------------------------------------
# Lecture directe des fichiers EMNIST au format idx (pas de telechargement)
# ---------------------------------------------------------------------------

def _read_idx_images(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"magic image inattendu: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"magic label inattendu: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_emnist_digits(raw_dir: str = RAW):
    """
    Charge EMNIST-digits depuis les fichiers idx locaux.
    Renvoie (Xtr, ytr, Xte, yte) avec X float32 (N,28,28) dans [0,1], chiffres
    redresses (transpose H<->W). Labels 0..9.
    """
    def _load(split):
        Xi = os.path.join(raw_dir, f"emnist-digits-{split}-images-idx3-ubyte")
        Yi = os.path.join(raw_dir, f"emnist-digits-{split}-labels-idx1-ubyte")
        X = _read_idx_images(Xi).astype(np.float32) / 255.0
        X = np.transpose(X, (0, 2, 1))          # redressement EMNIST
        y = _read_idx_labels(Yi).astype(np.int64)
        return X, y

    Xtr, ytr = _load("train")
    Xte, yte = _load("test")
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------

def main():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.digit_cnn import build_model, MODEL_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0,
                    help="limite le nb d'images d'entrainement (0 = tout)")
    ap.add_argument("--finetune", default=None,
                    help="chemin d'un digit_dataset.npz pour fine-tuner sur tes formulaires")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device = {device}")

    if args.finetune:
        finetune(args)
        return

    Xtr, ytr, Xte, yte = load_emnist_digits()
    print(f"[train] EMNIST-digits charge : train={len(Xtr)} test={len(Xte)}")

    # Sous-echantillonnage optionnel (entrainement rapide sur CPU).
    if args.subset and args.subset < len(Xtr):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xtr), size=args.subset, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"[train] sous-echantillon train = {len(Xtr)}")

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(1)   # (N,1,28,28)
    ytr_t = torch.from_numpy(ytr)
    Xte_t = torch.from_numpy(Xte).unsqueeze(1)
    yte_t = torch.from_numpy(yte)

    full = TensorDataset(Xtr_t, ytr_t)
    n_val = len(full) // 10
    n_tr = len(full) - n_val
    train_ds, val_ds = random_split(
        full, [n_tr, n_val],
        generator=torch.Generator().manual_seed(42))
    test_ds = TensorDataset(Xte_t, yte_t)
    print(f"[train] train={n_tr} val={n_val} test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    def evaluate(dl):
        model.eval()
        ok = tot = 0
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                ok += (model(x).argmax(1) == y).sum().item()
                tot += y.numel()
        return 100.0 * ok / tot

    best = 0.0
    os.makedirs(os.path.dirname(os.path.abspath(MODEL_PATH)), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        val_acc = evaluate(val_dl)
        print(f"[train] epoch {ep}/{args.epochs} "
              f"loss={running/n_tr:.4f} val_acc={val_acc:.2f}%", flush=True)
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"          -> sauvegarde {MODEL_PATH} (best val {best:.2f}%)")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[train] TEST accuracy = {evaluate(test_dl):.2f}%")
    print(f"[train] termine. Poids : {MODEL_PATH}")


def finetune(args):
    """Fine-tune le CNN pre-entraine EMNIST sur les chiffres reels des
    formulaires (digit_dataset.npz). Petite base -> augmentation + LR faible."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.digit_cnn import build_model, MODEL_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.load(args.finetune)
    X = torch.from_numpy(data["X"]).float().unsqueeze(1)   # (N,1,28,28)
    y = torch.from_numpy(data["y"]).long()
    print(f"[finetune] {len(X)} chiffres, classes presentes : "
          f"{sorted(set(y.tolist()))}")

    ds = TensorDataset(X, y)
    n_val = max(1, len(ds) // 6)
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))

    def augment(batch):
        import torchvision.transforms.functional as TF
        out = []
        for img in batch:
            ang = float(np.random.uniform(-10, 10))
            tx = int(np.random.uniform(-2, 2)); ty = int(np.random.uniform(-2, 2))
            out.append(TF.affine(img, angle=ang, translate=[tx, ty],
                                 scale=1.0, shear=[0.0, 0.0]))
        return torch.stack(out)

    tr_dl = DataLoader(tr, batch_size=64, shuffle=True)
    va_dl = DataLoader(va, batch_size=256)

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
    epochs = max(args.epochs if args.epochs else 30, 25)
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
