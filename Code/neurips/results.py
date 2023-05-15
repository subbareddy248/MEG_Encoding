import itertools

import numpy as np
import pandas as pd


def get_noise_ceiling(data, subject):
    
    if subject == "all":
        subject = [str(f.name).split("_")[1] for f in data.glob("noise_ceiling/subject_*")]
    elif isinstance(subject, str):
        subject = [subject]
    
    subject = sorted([str(s).zfill(2) for s in subject])
    
    ceilings = []
    for sub in subject:
        noise_ceiling = np.load(data / "noise_ceiling" / f"subject_{int(sub)}_kernel_ridge.npy")
        ceilings.append({"nc": noise_ceiling})
    
    return pd.DataFrame(ceilings)
    

def get_results(reports, feat, layer, subject):
    
    if subject == "all":
        subject = [str(f.name).split("-")[1] for f in reports.glob("sub-*")]
    elif isinstance(subject, str):
        subject = [subject]
    
    subject = sorted([str(s).zfill(2) for s in subject])
    
    if feat == "all":
        feat = []
        for s in subject:
            feat += [
                        "_".join(str(f.name).split("_")[:-3]) 
                        for f in reports.glob(f"sub-{s}-predictions/*")
                    ]
        feat = np.unique(feat)
 
    elif isinstance(feat, str):
        feat = [feat]
    
    if layer == "all":
        layer = np.arange(0, 12)
    elif isinstance(layer, int):
        layer = [layer]
        
    all_comb = itertools.product(subject, feat, layer)
    
    results = []
    for sub, f, l in all_comb:
        
        if "bert" in f or "concat" in f:
            results_dir = reports / f"sub-{sub}-predictions" / f"{f}_layer{l}_s{sub}_predictions"
        else:
            results_dir = reports / f"sub-{sub}-predictions" / f"{f}_s{sub}_predictions"
            
        
        if not (results_dir / "corr.npy").exists():
            #print(f"Not found: {results_dir / 'norm_corr.npy'}")
            continue

        comb_res = {"feat": f,
                       "sub": int(sub),
                       "layer": l,
                       "corr": np.nan,
                       "r2": np.nan
                    }
        
        try:
            corr = np.load(results_dir / "corr.npy")
            comb_res["corr"] = corr
        except Exception as e:
            print(e)
            
        try:
            r2 = np.load(results_dir / "r2s.npy")
            comb_res["r2"] = r2
        except Exception as e:
            print(e)
            
        results.append(comb_res)

    return pd.DataFrame(results)  