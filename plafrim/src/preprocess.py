import argparse
import logging

from pathlib import Path

import numpy as np
import pandas as pd
import mne
import mne_bids


logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("root", type=str, help="Data root directory.")
parser.add_argument("output", type=str, help="Output directory.")
parser.add_argument("--subject", "-s", type=str, help="Subject id.")

SESSION = 0
BANDPASS_FILTER_LOW = 0.5
BANDPASS_FILTER_HIGH = 30.0
EPOCH_TMIN = -0.2
EPOCH_TMAX = 0.6
EPOCH_BASELINE = (-0.2, 0.0)
EPOCH_DECIM = 10
RESCALE_QUANTILE = 0.95
RESCALE_FACTOR = 1e-13


def _preprocess_annotations(raw):
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    meta.loc[words.index + 1, "is_word"] = True

    meta = meta.query('kind=="phoneme"')

    return meta


def _load_raw(subject, session, task, root):
    logger.info(f"Loading: sub{subject}-sess{session}-task{task}")

    bids_path = mne_bids.BIDSPath(
        subject=str(subject).zfill(2),
        session=str(session),
        task=str(task),
        datatype="meg",
        root=root,
    )

    try:
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    except FileNotFoundError:
        logger.critical(
            f"File not found: sub{subject}-sess{session}-task{task}. Skipping."
        )
        return None

    raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)

    raw.load_data().filter(
        BANDPASS_FILTER_LOW, BANDPASS_FILTER_HIGH, n_jobs=1, verbose=False
    )

    meta = _preprocess_annotations(raw)

    events = np.c_[meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        decim=EPOCH_DECIM,
        baseline=EPOCH_BASELINE,
        metadata=meta,
        preload=True,
        event_repeated="drop",
        verbose=False,
    )

    epochs.metadata["half"] = np.round(np.linspace(0, 1.0, len(epochs))).astype(int)
    epochs.metadata["task"] = task
    epochs.metadata["session"] = session

    epochs = rescale(epochs)

    m = epochs.metadata
    label = (
        "t"
        + m.task.astype(str)
        + "_s"
        + m.session.astype(str)
        + "_h"
        + m.half.astype(str)
    )
    epochs.metadata["label"] = label

    X = epochs["is_word"].get_data() * RESCALE_FACTOR

    return X


def rescale(epochs):
    # as in JR King code
    th = np.percentile(np.abs(epochs._data), RESCALE_QUANTILE)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    logger.info(f"Clipped MEG data outside {-th, th}.")
    epochs.apply_baseline()

    return epochs


def save_epochs(data, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    data_path = str(Path(output) / f"{name}.npy")
    np.save(data_path, data, allow_pickle=True)
    logger.info(f"Saved Epochs data to {data_path}.")


def preprocess_epochs(root, output, name, subject):
    data = []
    for task in range(4):
        X = _load_raw(
            subject,
            SESSION,
            task,
            root,
        )

        if X is not None:
            data.append(X)

    logger.info("Epochs created.")

    save_epochs(
        data=data,
        output=output,
        name=name,
    )


if __name__ == "__main__":
    args = parser.parse_args()

    logger.info("Preprocessing MEG data...")

    name = f"sub{args.subject.zfill(2)}-meg-data-ses{SESSION}"

    preprocess_epochs(
        root=args.root,
        output=args.output,
        subject=args.subject,
        name=name,
    )

    logger.info("Done !")
