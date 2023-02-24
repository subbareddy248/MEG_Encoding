from pathlib import Path
import numpy as np

from tqdm import tqdm

REPORTS = Path("reports")

def FDR(vector, q, do_correction = False):
    original_shape = vector.shape
    vector = vector.flatten()
    N = vector.shape[0]
    sorted_vector = sorted(vector)
    if do_correction:
        C = np.sum([1.0/i for i in range(N)])
    else:
        C = 1.0
    thresh = 0
    #a=b
    for i in range(N-1, 0, -1):
        if sorted_vector[i]<= (i*1.0)/N*q/C:
            thresh = sorted_vector[i]
            break
    thresh_vector = vector<=thresh
    thresh_vector = thresh_vector.reshape(original_shape)
    thresh_vector = thresh_vector*1.0
    print("FDR threshold is : {}, {} voxels rejected".format(thresh, thresh_vector.sum()))
    return thresh_vector, thresh


sub_data = "sub_space_data/"

all_subjects = [str(i) for i in range(1, 9)]
all_features = ["bert", "glove", "postag", "deptag", "cm"]

missing_sig = []

all_uncorrected_sig = []
for sub in all_subjects:
    for feat in tqdm(all_features, f"Subject: {sub}"):
        sig_file = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions" / "0_sig.npy" 
        if not sig_file.exists():
            missing_sig.append((sub, feat))
            continue
        uncorrected_sig = np.load(sig_file)
        all_uncorrected_sig.append(uncorrected_sig)
        
q = 0.05
all_corrected_sig, _ = FDR(np.hstack(all_uncorrected_sig), q)

last_end = 0
ind = 0
for sub in all_subjects:
    for feat in all_features:
        if (sub, feat) in missing_sig:
            continue
        corrected_file = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions" / "0_sig_group_corrected.npy" 
        corrected_sig = all_corrected_sig[last_end:last_end + all_uncorrected_sig[ind].shape[0]]
        np.save(corrected_file, corrected_sig)
        last_end += all_uncorrected_sig[ind].shape[0]
        ind += 1
    
        print(f"{feat, sub}: num rejected = {corrected_sig.sum()}")
