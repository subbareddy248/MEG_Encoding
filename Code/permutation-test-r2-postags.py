#!/usr/bin/env python
# coding: utf-8
import os
import argparse
from pathlib import Path

import numpy as np

from sklearn.metrics import r2_score
import random

parser = argparse.ArgumentParser()
parser.add_argument("--subject", "-s", type=str, help="Subject id")
parser.add_argument("--feat", "-f", type=str, help="Feature id")

POSTAGS = dict(
    noun=['NN', 'NNS', 'NNP', 'NNPS'], 
    verb=['MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    adj=['JJ', 'JJR', 'JJS'],
)


def compute_sig(sub, feat):
    print("Subject {} Feature postag-{}".format(sub,feat))
    
    base_dir = REPORTS / f"sub-{sub}" / f"postag_s0_predictions" 
    
    if not base_dir.exists():
        print(f"Not found: {base_dir}")
        return None
     
    save_dir = base_dir / f"0_{feat}_sig.npy"
    if save_dir.exists():
        return 0.0002
    
    # POS tags mappings and epochs indices
    pos_all = np.load(REPORTS / "pos_tags_all.npy")
    pos_all_mappings = np.load(REPORTS / "pos_tags_mappings_all.npy", allow_pickle=True).item()
    
    tag_ids = sorted([pos_all_mappings[k] for k in POSTAGS[feat] if k in pos_all_mappings])
    tag_idxs = np.any(pos_all[:, tag_ids], axis=1)
    
    reals = []
    offset = 0
    for split_num in range(4):
        y = np.load(base_dir / f"0_y_test_{split_num}.npy")
        story_len = y.shape[0]
        reals.append(y[tag_idxs[offset:offset+story_len]])
        offset = story_len
    all_reals = np.vstack(reals)

    preds = []
    offset = 0
    for split_num in range(4):
        y = np.load(base_dir / f"0_y_pred_{split_num}.npy")
        story_len = y.shape[0]
        preds.append(y[tag_idxs[offset:offset+story_len]])
        offset = story_len
    all_preds = np.vstack(preds)

    real_r2 = r2_score(all_reals, all_preds, multioutput="raw_values")
    greater = np.zeros(preds[0].shape[1])
        
    blocksize = 10
    iters = 5000

    blocks = []
    for i in range(4):
        p = [preds[i][j:j+blocksize] for j in range(0, len(preds[i]), blocksize)]
        blocks.append(p)

    for s in range(iters):
        new_preds = []
        for i in range(4):
            perm = np.random.permutation(len(blocks[i]))
            bl = [blocks[i][p] for p in perm]
            shuffled_preds = np.vstack(bl)
            new_preds.append(shuffled_preds)
        new_preds = np.vstack(new_preds)
        shuffled_r2 = r2_score(all_reals,new_preds,multioutput="raw_values")
        shuffled_r2[real_r2 == 0] = 0
        greater[shuffled_r2 >= real_r2] += 1            
    sig = greater/iters
    sig[sig == 0] = 1/iters
    np.save(save_dir, sig)
    return np.min(sig)


if __name__ == "__main__":

    args = parser.parse_args()

    random.seed(654)
    np.random.seed(654)

    REPORTS = Path("/beegfs/ntrouvai/MEG-analysis/reports/")
    # REPORTS = Path("reports/")

    all_features = ["noun", "verb", "adj"]
    
    features = all_features[int(args.feat) // 8]
    subject = str(int(args.subject) % 8 + 1)

    compute_sig(subject, features)
