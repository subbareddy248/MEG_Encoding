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

    y_pred = np.r_[raw_pred]

    return y_pred


if __name__ == "__main__":

    args = parser.parse_args()

    if args.xdata is None or args.ydata is None:
        X, y = as_task_data(args.subject, "MEG->phoneme", args.root, P)

    if args.xdata is not None:
        X = np.load(Path(args.xdata))
    if args.ydata is not None:
        y = np.load(Path(args.ydata))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=P.seed, stratify=phonemes
    )

    n_train, time_length, n_channels = X_train.shape
    n_test = X_test.shape[0]

    robust_scaler = RobustScaler(quantile_range=P.quantile_range)

    X_train = robust_scaler.fit_transform(X_train.reshape(-1, n_channels))
    X_test = robust_scaler.transform(X_test.reshape(-1, n_channels))

    th = P.threshold
    X_train = np.clip(X_train, -th, th).reshape(
        n_train, time_length, n_channels
    )
    X_test = np.clip(X_test, -th, th).reshape(n_test, time_length, n_channels)
