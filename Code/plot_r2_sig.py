import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mne
import pickle
from tqdm import tqdm


for sub in tqdm([str(i) for i in range(1, 9)]):
    feat = "postag"
    
    nchan = 208
    ntime = 81

    try:
        sig = np.load(f"reports/sub-{sub}/{feat}_s0_predictions/0_sig_group_corrected.npy")
    except FileNotFoundError:
        continue
    
    sig = sig.reshape(nchan, ntime)
    print()

    plt.figure(figsize=(5,5))

    plt.imshow(sig, cmap="gray", interpolation="None", origin="lower", aspect="auto")
    plt.colorbar()

    #t = np.arange(-0.2, 0.6, 0.1)
    #tpos = np.round((t+0.2)*100)
    #plt.xticks(tpos, t)

    plt.xlabel("Time (s)")
    plt.ylabel("MEG channel")
    plt.savefig(f"figures/corrected-r2-p-value-sub-{sub}-{feat}.pdf")

    with open("data/formatted/word/info-12-word-100Hz-800ms-no-baseline.pkl", "rb") as fp:
        info = pickle.load(fp)

    plt.clf()
    times = np.arange(0.0, 0.8, 0.05)
    mne.EvokedArray(sig, info).plot_topomap(times, nrows=4, ncols=4, cmap="gray", cnorm=matplotlib.colors.LogNorm())
    plt.savefig(f"figures/corrected-sig-topomap-{sub}-{feat}.pdf")
