import itertools

import mne_bids

def get_epochs(subject, level="all"):
    all_w_epochs, all_ph_epochs = list(), list()
    for session, task in itertools.product(range(1), range(1)):
        print(f"Loading: sub{subject}-sess{session}-task{task}")

        bids_path = mne_bids.BIDSPath(
            subject=subject, 
            session=str(session), 
            task=str(task),
            datatype="meg",
            root=ROOT
        )

        try:
            raw = mne_bids.read_raw_bids(bids_path)
        except FileNotFoundError:
            print(f"Not found: sub{subject}-sess{session}-task{task}")
            continue

        raw = raw.pick_types(
            meg=True, misc=False, eeg=False, eog=False, ecg=False
        )

        raw.load_data().filter(
            P.bandpass.low, 
            P.bandpass.high, 
            n_jobs=1
        )

        annotations = preprocess_annotations(raw)

        w_epochs = ph_epochs = None
        if level in ["word", "all"]:
            words = annotations.query("kind=='word'")
            w_epochs = segment(raw, words)
            w_epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(w_epochs))
            ).astype(int)
            w_epochs.metadata["task"] = task
            w_epochs.metadata["session"] = session

            all_w_epochs.append(w_epochs)

        if level in ["phoneme", "all"]:
            phonemes = annotations.query("kind=='phoneme'")
            ph_epochs = segment(raw, phonemes)
            ph_epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(ph_epochs))
            ).astype(int)
            ph_epochs.metadata["task"] = task
            ph_epochs.metadata["session"] = session

            all_ph_epochs.append(ph_epochs)
    
    w_epochs = None
    if len(all_w_epochs) > 0:
        w_epochs = concatenate(all_w_epochs)
        
    ph_epochs = None
    if len(all_ph_epochs) > 0:
        ph_epochs = concatenate(all_ph_epochs)

    return w_epochs, ph_epochs, annotations

def concatenate(all_epochs):
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
    
def preprocess_annotations(raw):
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
    
def segment(raw, meta):
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=P.epochs.tmin,
        tmax=P.epochs.tmax,
        decim=P.epochs.decim,
        baseline=P.epochs.baseline,
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )
    
    return epochs