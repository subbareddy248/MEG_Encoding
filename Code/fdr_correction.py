from pathlib import Path
import numpy as np
from statsmodels.stats.multitest import fdrcorrection

REPORTS = Path("reports")

all_subjects = [str(i) for i in range(1, 9)]
all_single = ["bert", "glove", "postag", "deptags", "cm"]
#all_pairs = [
#        ("bert", "glove"),
#        ("postag", "deptags"),
#        ("postag", "cm"),
#        ("deptags", "cm"),
#    ]
all_pairs = []

all_features = all_single + all_pairs

missing_sig = []

all_uncorrected_sig = []
for sub in all_subjects:
    for feat in all_features:
        if isinstance(feat, tuple):
            sig_file = REPORTS / f"sub-{sub}" / f"{feat[0]}_diff_{feat[1]}" / "0_sig_boot.npy" 
        else:
            sig_file = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions" / "0_sig.npy" 
        
        if not sig_file.exists():
            missing_sig.append((sub, feat))
            print(f"Not found: {sig_file}")
            continue
        uncorrected_sig = np.load(sig_file)
        all_uncorrected_sig.append(uncorrected_sig)

q = 0.05
rejected, corrected = fdrcorrection(np.hstack(all_uncorrected_sig), q)

last_end = 0
ind = 0
for sub in all_subjects:
    for feat in all_features:
        if (sub, feat) in missing_sig:
            continue
        if isinstance(feat, tuple):
            corrected_file = REPORTS / f"sub-{sub}" / f"{feat[0]}_diff_{feat[1]}" / "0_sig_group_corrected.npy" 
            rejected_file = REPORTS / f"sub-{sub}" / f"{feat[0]}_diff_{feat[1]}" / "0_h0_group_corrected.npy" 
        else:
            corrected_file = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions" / "0_sig_group_corrected.npy" 
            rejected_file = REPORTS / f"sub-{sub}" / f"{feat}_s0_predictions" / "0_h0_group_corrected.npy" 
        
        corrected_sig = corrected[last_end:last_end + all_uncorrected_sig[ind].shape[0]]
        rejected_h0 = rejected[last_end:last_end + all_uncorrected_sig[ind].shape[0]]
        np.save(corrected_file, corrected_sig)
        np.save(rejected_file, rejected_h0)
        last_end += all_uncorrected_sig[ind].shape[0]
        ind += 1
    
        print(f"{feat, sub}: num rejected = {rejected_h0.sum()}")
