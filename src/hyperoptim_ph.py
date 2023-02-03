from pathlib import Path

import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from decode_ph import train_phoneme_decoder
from tools.parameters import Parameters


SEED = 864351
RND = np.random.default_rng(SEED)
DATA_ROOT = Path("data/formatted/")
PHONEME_INFO = Path("data/MASC-MEG/phoneme_info.csv")


def kfold(num_folds, random_state=SEED):
    if num_folds > 1:
        return StratifiedKFold(num_folds, random_state=SEED)
    else:
        def gen(X, y):
            yield np.arange(len(X)), np.arange(len(y))
        return gen


def load_data(task, decim=1.0):

    data_dir = DATA_ROOT / task
    data = np.load(data_dir / "train.npy")
    categories = pd.read_csv(PHONEME_INFO)
    category_codes = np.arange(len(categories)).tolist()
    
    encoder = OneHotEncoder(
        categories=[category_codes],
        sparse_output=False,
    )

    X, y = data.x, data.y

    if decim < 1.0:
        n_samples = X.shape[0]
        size = round(n_samples * decim)
        idx = RND.sample(np.arange(n_samples), size, replace=False)
        X, y = X[idx], y[idx]
  
    shape = y.shape
    y = encoder.fit_transform(y.reshape(-1, 1)).reshape(shape[:-1], -1)

    return X, y


def optim_MEG2phoneme_vec(args):
    
    from tools.parameters import P_ph_meg as P

    X, y = load_data("MEG->phoneme", args.decim)
    
    num_trials = args.trials
    num_folds = args.folds
    num_inits = args.inits

    study_name = args.name
    mysql_url = args.db
    
    study = optuna.create_study(
        study_name=study_name,
        sampler=optuna.samplers.RandomSampler(seed=SEED),
        storage="mysql://" + mysql_url,
    )
    
    seeds=RND.choice(99999, size=num_inits, replace=False)

    def objective(trial):

        folder = kfold(num_folds, random_state=SEED)
        
        mean_top1 = []
        mean_top10 = []
        for train_idxs, val_idxs in folder(X, y):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_val, y_val = X[val_idxs], y[val_idxs]

            top1 = [0.0] * num_folds
            top10 = [0.0] * num_folds
            for i in range(num_inits):
                  
                params = Parameters(
                    sr=trial.suggest_float("spectral_radius", 1e-4, 1e2, log=True),
                    lr=trial.suggest_float("leaking_rate", 1e-6, 1.0, log=True),
                    input_scaling=trial.suggest_float("input_scaling", 1e-4, 1e2, log=True),
                    ridge=trial.suggest_float("ridge", 1e-8, 1e-2, log=True),
                    input_bias=False,
                    bias_scaling=0.0,
                    units=300,
                    seed=seeds[i],
                )
                
                P.update(params)

                y_pred = train_phoneme_decoder(X_train, X_val, y_train, y_val, method="seq2vec", params=P)
                
                n_categories = y_pred.shape[-1]

                accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=1)
                top10accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=10)
                
                top1[i] = accuracy
                top10[i] = top10accuracy
                
            mean_top1.append(np.mean(top1))
            mean_top10.append(np.mena(top10))
        
        

                

if __name__ == "__main__":
    
    ...