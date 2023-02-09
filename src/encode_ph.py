import argparse
from pathlib import Path

import numpy as np

from tools.dataset import as_task_data
from tools.parameters import P_ph_meg as P
from tools.model import MEG2phoneme_seq, MEG2phoneme_vec

parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject", "-s", type=int, help="Subject id.", default="12"
)
parser.add_argument(
    "--root", type=str, help="Data root directory.", default="./data/MEG-MASC"
)
parser.add_argument(
    "--xdata", "-x", type=str, help="Formatted data directory."
)
parser.add_argument(
    "--ydata", "-y", type=str, help="Formatted targets directory."
)
parser.add_argument(
    "--method", "-m", type=str, help="RNN decoding method (seq2seq or seq2vec)"
)


def train_phoneme_decoder(X_train, X_test, y_train, y_test, method, params):
    if method == "seq2seq":
        model = MEG2phoneme_seq(params)
        warmup = round(
            np.abs(np.subtract(*params.epochs.baseline)) * params.sfreq
        )
    if method == "seq2vec":
        model = MEG2phoneme_vec(params)
        warmup = 0  # useless, we take only the last vector

    y_pred = model.fit(X_train, y_train, warmup=warmup).run(X_test)

    y_pred = np.r_[y_pred]

    return y_pred
