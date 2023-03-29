#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import random
from pathlib import Path

import numpy as np

from tqdm import tqdm
from sklearn.metrics import r2_score

parser = argparse.ArgumentParser()
parser.add_argument("--subject", "-s", type=str, help="Subject id")
parser.add_argument("--feat", "-f", type=str, help="Feature id")


def compute_sig(sub, feat):
    print("Subject {} Feature {}".format(sub, feat))

    base_dir = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions"

    if not base_dir.exists():
        print(f"Not found: {base_dir}")
        return None

    save_dir = base_dir / "0_sig.npy"
    if save_dir.exists():
        return 0.0002

    reals = []
    for split_num in range(4):
        reals.append(np.load(base_dir / f"0_y_test_{split_num}.npy"))
    all_reals = np.vstack(reals)

    preds = []
    for split_num in range(4):
        preds.append(np.load(base_dir / f"0_y_pred_{split_num}.npy"))
    all_preds = np.vstack(preds)

    real_r2 = np.load(base_dir / f"0_r2s.npy")
    greater = np.zeros(preds[0].shape[1])

    blocksize = 10
    iters = 5000

    blocks = []
    for i in range(4):
        p = [preds[i][j : j + blocksize] for j in range(0, len(preds[i]), blocksize)]
        blocks.append(p)

    for s in tqdm(range(iters)):
        new_preds = []
        for i in range(4):
            perm = np.random.permutation(len(blocks[i]))
            bl = [blocks[i][p] for p in perm]
            shuffled_preds = np.vstack(bl)
            new_preds.append(shuffled_preds)
        new_preds = np.vstack(new_preds)
        shuffled_r2 = r2_score(all_reals, new_preds, multioutput="raw_values")
        shuffled_r2[real_r2 == 0] = 0
        greater[shuffled_r2 >= real_r2] += 1
    sig = greater / iters
    sig[sig == 0] = 1 / iters
    np.save(base_dir / "0_sig.npy", sig)
    return np.min(sig)


if __name__ == "__main__":
    args = parser.parse_args()

    random.seed(654)
    np.random.seed(654)

    REPORTS = Path("/beegfs/ntrouvai/MEG-analysis/reports/")

    all_features = [
        # "bert",
        # "glove",
        # "postag",
        # "cm",
        # "deptags",
        # "bertseq20rh",
        # "bertseq5",
        # "bertseq5rh",
        # "bertseq4",
        # "bertseq_residuals",
        # "bertseq1",
        # "bertseqlag1_5",
        # "bertseqlag5_2_5",
        # "bertseqlag5_3_5",
        # "bertseq5_lag1rh",
        # "bertseq5_lag2rh",
        # "bertseq5_lag3rh",
        "bertseq2",
        "bertseq3",
        "bertseq4",
    ]

    features = all_features[int(args.feat) // 8]
    subject = str(int(args.subject) % 8 + 1)

    compute_sig(subject, features)
