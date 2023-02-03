import argparse
import logging
from pathlib import Path

import numpy as np

from tools.parameters import P_ph_meg as P
from tools.dataset import as_task_data, split

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument(
    "--subject", "-s", type=str, help="Subject id.", default="12"
)
parser.add_argument(
    "--root",
    "-r",
    type=str,
    help="Data root directory.",
    default="./data/MEG-MASC",
)
parser.add_argument("--output", "-o", type=str, help="Output directory.")
parser.add_argument(
    "--task",
    "-t",
    type=str,
    help="Either 'MEG->phoneme', 'MEG->word', 'word->'MEG', 'phoneme->MEG'",
)
parser.add_argument(
    "--postprocess",
    help="Split the dataset in train and test sets and apply rescaling.",
    action="store_true",
)


if __name__ == "__main__":

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    logger.info(args)

    task_directory = Path(args.output, args.task)
    task_directory.mkdir(parents=True, exist_ok=True)

    X, y = as_task_data(args.subject, task=args.task, root=args.root, params=P)

    dtype = np.dtype([("x", f"{X.shape[1:]}f4"), ("y", f"{y.shape[1:]}i2")])

    if args.postprocess:
        X_train, X_test, y_train, y_test = split(X, y, P)

        train_records = np.core.records.fromarrays(
            [X_train, y_train], dtype=dtype
        )
        test_records = np.core.records.fromarrays(
            [X_test, y_test], dtype=dtype
        )

        np.save(task_directory / "train.npy", train_records)
        np.save(task_directory / "test.npy", test_records)

    else:
        records = np.core.records.fromarrays([X, y], dtype=dtype)
        np.save(task_directory / "data.npy", records)

    logger.info("Done.")
