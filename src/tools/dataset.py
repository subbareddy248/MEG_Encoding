import logging
import itertools

from pathlib import Path

import pandas as pd
import numpy as np
import librosa as lbr
import mne
import mne_bids

from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .utils import tqdm_joblib

logger = logging.getLogger(__name__)


def _concatenate(all_epochs):
    epochs = mne.concatenate_epochs(all_epochs)
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

    return epochs


def _preprocess_annotations(raw, phoneme_list):

    df_raw = raw.annotations.to_data_frame()
    df_raw["onset"] = raw.annotations.onset
    df_desc = pd.DataFrame(df_raw.description.apply(eval).to_list())
    df = pd.concat([df_raw.drop("description", axis=1), df_desc], axis=1)

    df[["index", "subject"]] = df[["index", "subject"]].fillna(method="ffill")

    df_ = df.copy().query("kind != 'sound'")

    df_["word"] = df_["word"].shift(-1)
    df_filled = df_["word"].fillna(method="ffill")

    df.loc[df["kind"] != "sound", "word"] = df_filled

    def as_short_phoneme(x):
        if isinstance(x, str):
            return x.split("_")[0]
        return x

    def as_phoneme_code(x):
        if isinstance(x, str):
            return phoneme_list.index(x.split("_")[0])
        return x

    df["s_phoneme"] = df["phoneme"].apply(as_short_phoneme)
    df["phoneme_code"] = df["phoneme"].apply(as_phoneme_code)

    return df


def _segment(raw, meta, params):
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=params.epochs.tmin,
        tmax=params.epochs.tmax,
        decim=params.epochs.decim,
        baseline=params.epochs.baseline,
        metadata=meta,
        preload=True,
        event_repeated="drop",
        verbose=False,
    )

    return epochs


@delayed
def _load_raw(subject, session, task, root, params, phoneme_list, level):
    logger.info(f"Loading: sub{subject}-sess{session}-task{task}")

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=str(session),
        task=str(task),
        datatype="meg",
        root=root,
    )

    try:
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    except FileNotFoundError:
        logger.critical(
            f"File not found: sub{subject}-sess{session}-task{task}"
        )
        return

    raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)

    raw.load_data().filter(
        params.meg.bandpass.low, params.meg.bandpass.high, n_jobs=1, verbose=False
    )

    annotations = _preprocess_annotations(raw, phoneme_list)

    w_epochs = ph_epochs = None
    if level in ["word", "all"]:
        words = annotations.query("kind=='word'")
        w_epochs = _segment(raw, words, params)
        w_epochs.metadata["half"] = np.round(
            np.linspace(0, 1.0, len(w_epochs))
        ).astype(int)
        w_epochs.metadata["task"] = task
        w_epochs.metadata["session"] = session

    if level in ["phoneme", "all"]:
        phonemes = annotations.query("kind=='phoneme'")
        ph_epochs = _segment(raw, phonemes, params)
        ph_epochs.metadata["half"] = np.round(
            np.linspace(0, 1.0, len(ph_epochs))
        ).astype(int)
        ph_epochs.metadata["task"] = task
        ph_epochs.metadata["session"] = session

    return w_epochs, ph_epochs


def get_epochs(subject, root, params, level="all"):

    if Path(root).exists() and Path(root, "phoneme_info.csv").exists():
        phoneme_info = pd.read_csv(Path(root) / "phoneme_info.csv")
        phoneme_list = phoneme_info.phoneme.tolist()
    else:
        raise FileNotFoundError("phoneme_info.csv")

    sessions_tasks = list(itertools.product(range(1), range(4)))

    with tqdm_joblib(
        tqdm(desc="Loading MEG", total=len(sessions_tasks))
    ):
        epochs = Parallel(n_jobs=-1, backend="loky")(
            _load_raw(
                subject,
                session,
                task,
                root,
                params,
                phoneme_list,
                level,
            )
            for session, task in sessions_tasks
        )

    all_w_epochs = [e[0] for e in epochs if e[0] is not None]
    all_ph_epochs = [e[1] for e in epochs if e[1] is not None]

    if len(all_ph_epochs) == 0 and len(all_w_epochs) == 0:
        raise FileNotFoundError(f"No file found at root {root}")

    w_epochs = None
    if len(all_w_epochs) > 0:
        w_epochs = _concatenate(all_w_epochs)

    ph_epochs = None
    if len(all_ph_epochs) > 0:
        ph_epochs = _concatenate(all_ph_epochs)

    logger.info("Epochs created.")

    return w_epochs, ph_epochs


def get_phoneme_sounds(root, meta, params):
    # start column in metadata is the phoneme/word onset
    # in the sound (one of the 4 audio bouts)
    from functools import partial

    import librosa as lbr

    from tqdm import tqdm

    params.sound_sfreq = 22050

    # Params from Gwilliams, King, Poeppel
    # "Neural dynamics of phoneme sequences reveal
    # position-invariant code for content and order"

    params.hop_length = 128
    params.n_fft = 2048
    params.n_mels = 208
    params.window = "hamming"

    meta["l_story"] = meta.story.str.lower()
    meta.sound_id = meta.sound_id.astype(int)

    def segment_audio(meta_row, x, sfreq, audio_list):
        onset = meta_row.start
        start = round((onset - np.abs(params.epochs.tmin)) * sfreq)
        end = start + params.epoch_length

        assert end - start == params.epoch_length, f"{end - start} != {params.epoch_length}"

        if start < 0:
            # 0-pad
            x_ = x[0:end]
            y = np.pad(x_, ((-start, 0), (0, 0)), constant_values=(0, 0))
        elif end > x.shape[0]:
            x_ = x[start:]
            y = np.pad(x_, ((0, end - x.shape[0]), (0, 0)), constant_values=(0, 0))
        else:
            y = x[start:end]

        # meta_row.name is the epoch index
        # should not mix anything
        audio_list[meta_row.name] = y

    audio_files = Path(root) / "stimuli" / "audio"
    audios = [None] * len(meta)
    for file in tqdm(list(audio_files.glob("*.wav"))):
        f = file.stem
        story = "_".join(f.split("_")[:-1])
        if len(story) < 1:
            logger.warning("Wrong filename format:", file)
            continue

        sound_id = int(f.split("_")[-1])
        meta_story = meta[meta.l_story.str.contains(story.lower())]
        meta_sound = meta_story.query("sound_id==@sound_id")

        sfreq = None
        y, sfreq = lbr.load(file, sr=sfreq)
        mel = lbr.feature.melspectrogram(
            y=y,
            sr=sfreq,
            n_fft=params.n_fft,
            hop_length=params.hop_length,
            window=params.window,
            n_mels=params.n_mels,
            )

        mel = lbr.resample(
            mel,
            orig_sr=params.sound_sfreq // params.hop_length,
            target_sr=params.raw_sfreq // params.epochs.decim,
            axis=1,
            res_type="soxr_vhq"
            )

        mel = np.clip(mel, 0, None)  # remove very few negative amplitude values
        mel = np.log10(1000 * mel + 1.0)  # rescale (approx between 0 ~ 6)

        meta_sound.apply(
            partial(
                segment_audio,
                x=mel.T,
                sfreq=params.raw_sfreq // params.epochs.decim,
                audio_list=audios
                ),
            axis=1
            )

    return np.r_[audios]


def as_meg_phoneme(subject, root, params):

    _, epochs = get_epochs(subject, root=root, level="phoneme", params=params)

    meta = epochs.metadata

    ph_count = meta.groupby("s_phoneme").count()
    lonely_phons = ph_count[ph_count["onset"] < 2]

    logger.info(f"Phoneme occurrences:\n{ph_count['onset'].sort_values()}")

    lonely_index = list()
    if len(lonely_phons) > 0:
        lonely_phons = lonely_phons.index.values.tolist()

        logger.critical(
            "Some phonemes appear only once !"
            f"\n{lonely_phons}"
            "\nThey will be removed."
        )

        lonely_index = meta[meta["s_phoneme"].isin(lonely_phons)].index.values

    X = np.swapaxes(epochs.get_data()[:, :208, :], 1, 2)
    X = np.delete(X, lonely_index, axis=0)

    n_epochs, n_timepoints, n_channels = X.shape

    phonemes = meta.phoneme_code.values
    phonemes = np.delete(phonemes, lonely_index, axis=0)

    y = phonemes.reshape(n_epochs, 1, 1).astype(int)

    logger.info(f"Data dimensions:\n\tX : {X.shape}\n\ty : {y.shape}")

    assert X.shape[0] == y.shape[0]

    return X, y


def as_sound_meg(subject, root, params):
    ...


def split(X, y, params):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params.test_size,
        random_state=params.seed,
        stratify=y.flatten(),
    )

    n_train, time_length, n_channels = X_train.shape
    n_test = X_test.shape[0]

    robust_scaler = RobustScaler(quantile_range=params.quantile_range)

    X_train = robust_scaler.fit_transform(X_train.reshape(-1, n_channels))
    X_test = robust_scaler.transform(X_test.reshape(-1, n_channels))

    th = params.threshold
    X_train = np.clip(X_train, -th, th).reshape(
        n_train, time_length, n_channels
    )
    X_test = np.clip(X_test, -th, th).reshape(n_test, time_length, n_channels)

    return X_train, X_test, y_train, y_test


def as_task_data(subject, task, root, params, **kwargs):

    task_table = {
        "MEG->phoneme": as_meg_phoneme,
        "sound->MEG": as_sound_meg,
    }

    if task in task_table:
        logger.info(f"Loading data for task: {task}")
    else:
        raise KeyError(
            f"{task} is not a task."
            f"\nAvailable tasks are '{' '.join(list(task_table.keys()))}'."
        )

    return task_table[task](subject, root, params, **kwargs)
