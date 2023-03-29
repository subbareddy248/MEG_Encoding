#!/usr/bin/env python
# coding: utf-8
import argparse
import random
import logging
from pathlib import Path

import numpy as np

from tqdm import tqdm
from sklearn.metrics import r2_score


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-r", "--reportdir", type=str, help="Index (sub=id%8+1, feat=id//8) of job"
)

BLOCKSIZE = 10
ITERS = 5000
N_SPLITS = 4


def compute_sig(report):
    random.seed(654)
    np.random.seed(654)

    base_dir = Path(report)

    sub = str(base_dir.parent.stem).split("-")[1]
    feat = str(base_dir.stem)

    print("Subject {} Feature {}".format(sub, feat))

    if not base_dir.exists():
        print(f"Not found: {base_dir}")
        return None

    save_dir = base_dir / "sig.npy"
    if save_dir.exists():
        return 1 / ITERS

    reals = []
    for split_num in range(N_SPLITS):
        reals.append(np.load(base_dir / f"y_test_{split_num}.npy"))
    all_reals = np.vstack(reals)

    preds = []
    for split_num in range(N_SPLITS):
        preds.append(np.load(base_dir / f"y_pred_{split_num}.npy"))

    real_r2 = np.load(base_dir / f"r2s.npy")
    greater = np.zeros(preds[0].shape[1])

    blocks = []
    for i in range(N_SPLITS):
        p = [preds[i][j : j + BLOCKSIZE] for j in range(0, len(preds[i]), BLOCKSIZE)]
        blocks.append(p)

    for s in tqdm(range(ITERS)):
        new_preds = []

        for i in range(N_SPLITS):
            perm = np.random.permutation(len(blocks[i]))
            bl = [blocks[i][p] for p in perm]
            shuffled_preds = np.vstack(bl)
            new_preds.append(shuffled_preds)

        new_preds = np.vstack(new_preds)

        shuffled_r2 = r2_score(all_reals, new_preds, multioutput="raw_values")
        shuffled_r2[real_r2 == 0] = 0

        greater[shuffled_r2 >= real_r2] += 1

    sig = greater / ITERS
    sig[sig == 0] = 1 / ITERS

    np.save(base_dir / "sig.npy", sig)

    return np.min(sig)


if __name__ == "__main__":
    args = parser.parse_args()

    report = args.reportdir

    compute_sig(report)
