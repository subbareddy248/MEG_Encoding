import argparse
import logging
import itertools
import pickle
import gc

from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import librosa as lbr
import mne
import mne_bids

from tqdm import tqdm
from sklearn.preprocessing import RobustScaler

from tools.parameters import load_parameters

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument("root", type=str, help="Data root directory.")
parser.add_argument("output", type=str, help="Output directory.")
parser.add_argument("--data_id", "-i", type=str, help="Unique id.")
parser.add_argument("--subject", "-s", type=str, help="Subject id.")
parser.add_argument("--level", "-l", type=str, help="'phoneme' or 'word'.")
parser.add_argument("--params", "-p", type=str, help="YAML parameters file.")
parser.add_argument(
    "--split",
    action="store_true",
    help="Split the dataset between for training."
)
parser.add_argument("--meg", action="store_true")
parser.add_argument("--audios", action="store_true")


def _preprocess_annotations(raw, phoneme_list):

    df_raw = raw.annotations.to_data_frame()
    df_raw["onset"] = raw.annotations.onset
    df_desc = pd.DataFrame(df_raw.description.apply(eval).to_list())
    df = pd.concat([df_raw.drop("description", axis=1), df_desc], axis=1)

    df[["index", "subject"]] = df[["index", "subject"]].fillna(method="ffill")

    df_ = df.query("kind != 'sound'").copy()

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
            f"File not found: sub{subject}-sess{session}-task{task}. Skipping."
        )
        return

    raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)

    raw.load_data().filter(
        params.meg.bandpass.low, params.meg.bandpass.high, n_jobs=1, verbose=False
    )

    annotations = _preprocess_annotations(raw, phoneme_list)

    if level in ["phoneme", "word"]:
        meta = annotations.query("kind==@level")
        events = np.c_[
            meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
        ].astype(int)
        
        baseline = params.epochs.baseline
        if hasattr(baseline, "__iter__"):
            baseline = tuple(baseline)

        epochs = mne.Epochs(
            raw,
            events,
            tmin=params.epochs.tmin,
            tmax=params.epochs.tmax,
            decim=params.epochs.decim,
            baseline=baseline,
            metadata=meta,
            preload=True,
            event_repeated="drop",
            verbose=False,
            )

        epochs.metadata["half"] = np.round(
            np.linspace(0, 1.0, len(epochs))
        ).astype(int)
        epochs.metadata["task"] = task
        epochs.metadata["session"] = session

        if params.meg.scaler == "const":
            epochs = rescale_before(epochs, params)

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
    else:
        raise ValueError(f"Level must be either 'phoneme' or 'word', not {level}")

    return epochs


def train_test_split(X, meta, params):

    test_story = params.test_story

    logger.info(f"Testing on story: {test_story}")

    test_idxs = meta.query("story==@test_story").index.values
    train_idxs = meta.query("story!='@test_story'").index.values

    logger.info(f"Test size: {len(test_idxs)}")
    logger.info(f"Train size: {len(train_idxs)}")

    meta["split"] = "train"
    meta.loc[test_idxs, "split"] = "test"

    X_test = X[test_idxs]
    X_train = X[train_idxs]

    return X_train, X_test, meta


def rescale(X_train, X_test, params):

    logger.info("Rescaling...")

    N_train, T, C = X_train.shape  # samples, timesteps, channels
    N_test = X_test.shape[0]

    robust_scaler = RobustScaler(
        quantile_range=tuple(params.meg.scaler_params.quantile_range)
    )

    X_train = robust_scaler.fit_transform(X_train.reshape(-1, C))
    X_test = robust_scaler.transform(X_test.reshape(-1, C))

    logger.info(f"Rescaled MEG data using RobustScaler, "
                f"quantiles: {params.meg.scaler_params.quantile_range}.")

    low, high = params.meg.clip
    np.clip(X_train, low, high, out=X_train)
    np.clip(X_test, low, high, out=X_test)

    logger.info(f"Clipped MEG data outside {low, high}.")

    return X_train.reshape(N_train, T, C), X_test.reshape(N_test, T, C)


def rescale_before(epochs, params):
    # as in JR King code
    last_quant = int(params.meg.scaler_params.quantile_range[1])
    th = np.percentile(np.abs(epochs._data), last_quant)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    logger.info(f"Clipped MEG data outside {-th, th}.")
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), last_quant)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    logger.info(f"Clipped MEG data outside {-th, th}.")
    epochs.apply_baseline()
    return epochs


def save_epochs(data, output, name, suffix="", info=None, meta=None):

    Path(output).mkdir(parents=True, exist_ok=True)

    if info is not None:
        info_path = Path(output) / f"info-{name}.pkl"
        with open(info_path, "wb+") as fp:
            pickle.dump(info, fp)
        logger.info(f"Saved Epochs.info to {info_path}.")

    if meta is not None:
        meta_path = Path(output) / f"meta-{name}.csv"
        meta.to_csv(meta_path, index=False)
        logger.info(f"Saved Epochs.metadata to {meta_path}.")

    data_path = str(Path(output) / f"epochs-{name+suffix}.npy")
    np.save(data_path, data)
    logger.info(f"Saved Epochs.data to {data_path}.")


def preprocess_epochs(root, output, data_id, subject, level, params_path, split):

    params = load_parameters(params_path)

    if Path(root).exists() and Path(root, "phoneme_info.csv").exists():
        phoneme_info = pd.read_csv(Path(root) / "phoneme_info.csv")
        phoneme_list = phoneme_info.phoneme.tolist()
    else:
        raise FileNotFoundError("phoneme_info.csv")

    sessions_tasks = list(itertools.product(range(2), range(4)))

    epochs = []
    for session, task in tqdm(sessions_tasks, "Segmenting MEG"):
        epoch = _load_raw(
            subject,
            session,
            task,
            root,
            params,
            phoneme_list,
            level,
        )

        if epoch is not None:
            epochs.append(epoch)

    epochs = mne.concatenate_epochs(epochs)

    logger.info("Epochs created.")

    X = epochs.get_data()

    X = np.swapaxes(X, 1, 2)  # N, time, channels
    metadata = epochs.metadata
    info = epochs.info
    
    del epochs
    gc.collect()

    if split:
        X_train, X_test, meta = train_test_split(X, metadata, params)
        
        del X  # too big for my computer (decim=4)
        gc.collect()
	
        if params.meg.scaler == "const":
            coef = params.meg.scaler_params.coef
            X_train, X_test = X_train * coef, X_test * coef
        else:
            X_train, X_test = rescale(X_train, X_test, params)

        save_epochs(X_train, output, data_id, suffix=".train", info=info, meta=meta)
        save_epochs(X_test, output, data_id, suffix=".test")
    else:
        save_epochs(
            X,
            output,
            data_id,
            info=info,
            meta=metadata
        )


def save_audios(data, output, name, suffix=""):
    Path(output).mkdir(parents=True, exist_ok=True)

    audio_path = str(Path(output) / f"audio-{name+suffix}.npy")
    np.save(audio_path, data)
    logger.info(f"Saved audio segments to {audio_path}.")


def preprocess_audio(root, output, data_id, meta, params_path, split):
    # start column in metadata is the phoneme/word onset
    # in the sound (one of the 4 audio bouts)

    params = load_parameters(params_path)

    p_sound = params.sound
    p_mel = p_sound.mel_spec

    sound_sfreq = p_sound.sfreq

    hop_length = p_mel.hop_length
    n_fft = p_mel.n_fft
    n_mels = p_mel.n_mels
    window = p_mel.window

    res_type = p_mel.resampling_type
    low, high = p_mel.clip
    scaling = p_mel.scaling

    meta["l_story"] = meta.story.str.lower()
    meta.sound_id = meta.sound_id.astype(int)

    def segment_audio(meta_row, x, sfreq, audio_list):
        onset = meta_row.start
        start = round((onset - np.abs(params.epochs.tmin)) * sfreq)
        end = start + params.epochs.epoch_length

        assert end - start == params.epochs.epoch_length,\
            f"Inconsistent audio segment size (different from epoch): " \
            f"{end - start} != {params.epochs.epoch_length}."

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
    for file in tqdm(list(audio_files.glob("*.wav")), "Segmenting audio"):
        f = file.stem
        story = "_".join(f.split("_")[:-1])
        if len(story) < 1:
            logger.warning(f"Wrong filename format: {str(file)}")
            continue

        sound_id = int(f.split("_")[-1])
        meta_story = meta[meta.l_story.str.contains(story.lower())]
        meta_sound = meta_story.query("sound_id==@sound_id")

        sfreq = None
        y, sfreq = lbr.load(file, sr=sfreq)

        assert sfreq == sound_sfreq

        mel = lbr.feature.melspectrogram(
            y=y,
            sr=sfreq,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            n_mels=n_mels,
            )

        mel = lbr.resample(
            mel,
            orig_sr=sound_sfreq // hop_length,
            target_sr=params.epochs.sfreq,
            axis=1,
            res_type=res_type
            )

        mel = np.clip(mel, low, high)  # remove very few negative amplitude values
        mel = np.log10(scaling * mel + 1.0)  # rescale (approx between 0 ~ 6)

        meta_sound.apply(
            partial(
                segment_audio,
                x=mel.T,
                sfreq=params.epochs.sfreq,
                audio_list=audios
            ),
            axis=1
        )

    audios = np.r_[audios]

    logger.info(f"Audios segments shape: {audios.shape}")

    if split:
        X_train, X_test, _ = train_test_split(audios, meta, params)
        logger.info(f"Train shape: {X_train.shape}")
        logger.info(f"Test shape: {X_test.shape}")
        save_audios(X_train, output, data_id, suffix=".train")
        save_audios(X_test, output, data_id, suffix=".test")
    else:
        save_audios(audios, output, data_id)


if __name__ == '__main__':

    args = parser.parse_args()

    if args.meg:
        logger.info("Preprocessing MEG data...")
        preprocess_epochs(
            args.root,
            args.output,
            "-".join((args.subject, args.level, args.data_id)),
            args.subject,
            args.level,
            args.params,
            args.split
        )

        logger.info("Done !")

    if args.audios:
        logger.info("Preprocessing audio data...")
        metadata = pd.read_csv(Path(args.output)
                / f"meta-{'-'.join((args.subject, args.level, args.data_id))}.csv")
        preprocess_audio(
            args.root,
            args.output,
            "-".join((args.subject, args.level, args.data_id)),
            metadata,
            args.params,
            args.split
        )

        logger.info("Done !")
