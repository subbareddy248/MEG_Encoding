#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import random
from pathlib import Path

import numpy as np

from sklearn.metrics import r2_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--subject", "-s", type=str, help="Subject id")
parser.add_argument("--pair", "-p", type=str, help="Feature pair id")


def compute_diff_sig(sub, feat1, feat2):
    print("Subject {} Feature {} - Features {}".format(sub, feat1, feat2))
    
    base_dir = REPORTS / f"sub-{sub}"
    
    feat1_dir = base_dir / f"{feat1}_s0_predictions"
    feat2_dir = base_dir / f"{feat2}_s0_predictions"
    
    if not feat1_dir.exists():
        print(f"Not found: {feat1_dir}")
        return None

    if not feat2_dir.exists():
        print(f"Not found: {feat2_dir}")
        return None

    save_dir = base_dir / f"{feat1}_diff_{feat2}"
    if not save_dir.exists():
        save_dir.mkdir()
    
    if (save_dir / "0_sig_boot.npy").exists():
        return 0.0002
    
    reals = []
    for split_num in range(4):
        reals.append(np.load(feat2_dir / "0_y_test_{}.npy".format(split_num)))
    # all_reals = np.vstack(reals)

    preds_feat1 = []
    for split_num in range(4):
        preds_feat1.append(np.load(feat1_dir / "0_y_pred_{}.npy".format(split_num)))
    # all_preds_feat1 = np.vstack(preds_feat1)
    
    preds_feat2 = []
    for split_num in range(4):
        preds_feat2.append(np.load(feat2_dir / "0_y_pred_{}.npy".format(split_num)))
    # all_preds_feat2 = np.vstack(preds_feat2)

    real_r2_feat1 = np.load(feat1_dir / "0_r2s.npy")
    real_r2_feat2 = np.load(feat2_dir / "0_r2s.npy")
    
    real_diff = real_r2_feat1 - real_r2_feat2
    
    greater = np.zeros(real_diff.shape[0])
    
    blocksize = 10
    
    blocks_feat1 = []
    for i in range(4):
        p = [preds_feat1[i][j:j+blocksize] for j in range(0,len(preds_feat1[i]), blocksize)]
        blocks_feat1.append(p)
        
    blocks_feat2 = []
    for i in range(4):
        p = [preds_feat2[i][j:j+blocksize] for j in range(0,len(preds_feat2[i]), blocksize)]
        blocks_feat2.append(p)
        
    blocks_real = []
    for i in range(4):
        p = [reals[i][j:j+blocksize] for j in range(0,len(reals[i]), blocksize)]
        blocks_real.append(p)
        
    iters = 5000
    
    for s in tqdm(range(iters)):
        new_preds_feat1 = []
        new_preds_feat2 = []
        new_reals = []
        for i in range(4):
            sample = np.random.choice(len(blocks_feat1[i]), len(blocks_feat1[i]))
            
            bl1 = [blocks_feat1[i][s] for s in sample]
            shuffled_preds_feat1 = np.vstack(bl1)
            new_preds_feat1.append(shuffled_preds_feat1)
            
            bl2 = [blocks_feat2[i][s] for s in sample]
            shuffled_preds_feat2 = np.vstack(bl2)
            new_preds_feat2.append(shuffled_preds_feat2)
            
            blr = [blocks_real[i][s] for s in sample]
            shuffled_reals = np.vstack(blr)
            new_reals.append(shuffled_reals)
            
        new_preds_feat1 = np.vstack(new_preds_feat1)
        new_preds_feat2 = np.vstack(new_preds_feat2)
        new_reals = np.vstack(new_reals)
        
        shuffled_r2_feat1 = r2_score(new_reals, new_preds_feat1, multioutput="raw_values")
        shuffled_r2_feat1[real_r2_feat1 == 0] = 0
        
        shuffled_r2_feat2 = r2_score(new_reals, new_preds_feat2, multioutput="raw_values")
        shuffled_r2_feat2[real_r2_feat2 == 0] = 0
        
        shuffled_diff = shuffled_r2_feat1 - shuffled_r2_feat2        
        greater[shuffled_diff <= 0] += 1            
    sig = greater/iters
    sig[sig == 0] = 1/iters
    np.save(save_dir / "0_sig_boot.npy", sig)
    return np.min(sig)

if __name__ == "__main__":

    args = parser.parse_args()

    random.seed(654)
    np.random.seed(654)

    REPORTS = Path("/beegfs/ntrouvai/MEG-analysis/reports/")

    all_pairs = [
        ("bert", "glove"),
        ("postag", "deptags"),
        ("postag", "cm"),
        ("deptags", "cm"),
    ]
    
    features = all_pairs[int(args.pair) // 8]
    subject = str(int(args.subject) % 8 + 1)

    compute_diff_sig(subject, *features)
