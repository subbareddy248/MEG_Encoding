import argparse
import logging

from pathlib import Path

import optuna
import numpy as np
import pandas as pd
import joblib
import reservoirpy as rpy

from sklearn.metrics import top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder

from decode_ph import train_phoneme_decoder
from tools.parameters import Parameters

logger = logging.getLoger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("--trials", "-t", type=int, help="Number of trials")
parser.add_argument("--folds", "-f", type=int, help="Number of folds", default=1)
parser.add_argument("--inits", "-i", type=int, help="Number of random initializations")
parser.add_argument("--name", "-n", type=str, help="Study name")
parser.add_argument("--db", type=str, help="MySQL url", default="optuna:password@localhost/optuna")
parser.add_argument("--dryrun", action="store_true", help="Dry run study (to test objective function)")
parser.add_argument("--decim", type=float, help="Data decimation ratio", default=1.0)
parser.add_argument("task", type=str, help="Task to run")


SEED = 864351
RND = np.random.default_rng(SEED)
DATA_ROOT = Path("data/formatted/")
REPORT_ROOT = Path("reports/")
PHONEME_INFO = Path("data/MASC-MEG/phoneme_info.csv")
VALIDATION_SIZE = 0.2


def kfold(num_folds, random_state=SEED):
    if num_folds > 1:
        return StratifiedKFold(num_folds, shuffle=True, random_state=SEED)
    else:
        class ValidationSplit:
            def split(X, y):
                n_samples = len(X)
                n_val = round(n_samples * VALIDATION_SIZE)
                idxs = np.random.default_rng(SEED).shuffle(np.arange(len(X)))
                yield idxs[:n_val], idxs[n_val:]
        return ValidationSplit()


def load_data(task, decim=1.0, tile=False):

    data_dir = DATA_ROOT / task
    data = np.load(data_dir / "train.npy")
    categories = pd.read_csv(PHONEME_INFO)
    category_codes = np.arange(len(categories)).tolist()
    
    encoder = OneHotEncoder(
        categories=[category_codes],
        sparse_output=False,
    )

    X, y = data["x"], data["y"]

    if decim < 1.0:  # 100% data
        n_samples = X.shape[0]
        size = round(n_samples * decim)
        idx = RND.choice(np.arange(n_samples), size, replace=False)
        X, y = X[idx], y[idx]
        
    if tile:  # repeatlabel along time axis
        n_samples, n_timepoints, _ = X.shape
        y = np.tile(y, (n_timepoints, 1)).astype(int)

    # shape = y.shape
    # y_encoded = encoder.fit_transform(y.reshape(-1, 1)).reshape(*shape[:-1], -1)
    encoder = encoder.fit(y.reshape(-1, 1))
    return X, y, encoder, category_codes


def optim_MEG2phoneme_vec(args):
    """Seq2Vec phoneme decoding; Random search script."""
    
    from tools.parameters import P_ph_meg as P

    X, y, encoder, classes = load_data("MEG->phoneme", args.decim)
    
    num_trials = args.trials
    num_folds = args.folds
    num_inits = args.inits

    study_name = args.name
    mysql_url = args.db
    
    seeds=RND.choice(99999, size=num_inits, replace=False)

    def objective(trial):

        folder = kfold(num_folds, random_state=SEED)
        
        mean_top1 = []
        mean_top10 = []
        
        params = Parameters(
            sr=trial.suggest_float("spectral_radius", 1e-4, 1e2, log=True),
            lr=trial.suggest_float("leaking_rate", 1e-6, 1.0, log=True),
            input_scaling=trial.suggest_float("input_scaling", 1e-4, 1e2, log=True),
            ridge=trial.suggest_float("ridge", 1e-8, 1e-2, log=True),
            input_bias=False,
            bias_scaling=0.0,
            units=300,
        )
        
        for train_idxs, val_idxs in folder.split(X, y.flatten()):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_val, y_val = X[val_idxs], y[val_idxs]
            
            shape = y_train.shape
            y_train = encoder.transform(y_train.reshape(-1, 1)).reshape(*shape[:-1], -1)

            top1 = [0.0] * num_folds
            top10 = [0.0] * num_folds
            for i in range(num_inits):
                
                params.seed = seeds[i]
                
                P.update(params)

                y_pred = train_phoneme_decoder(X_train, X_val, y_train, y_val, method="seq2vec", params=P)
                
                n_categories = y_pred.shape[-1]
                
                accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=1, labels=classes)
                top10accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=10, labels=classes)
                
                top1[i] = accuracy
                top10[i] = top10accuracy
                
            mean_top1.append(np.mean(top1))
            mean_top10.append(np.mean(top10))
            
        trial.set_user_attr("top10_accuracy", np.mean(mean_top10))
        trial.set_user_attr("n_folds", num_folds)
        trial.set_user_attr("n_inits", num_inits)
        trial.set_user_attr("seeds", seeds.tolist())
        
        return np.mean(mean_top1)
    
    if args.dryrun:
        logger.info(objective(optuna.trial.FixedTrial({"spectral_radius": 0.9, "leaking_rate": 0.5, "input_scaling": 1.0, "ridge": 1e-8})))F
    else:   
        logger.info(f"Starting study: {study_name}")
        
        sampler_path = REPORT_ROOT / ("sampler" + study_name + ".pkl")
        if sampler_path.is_file():
            with open(REPORT_ROOT / ("sampler" + study_name + ".pkl"), "w+") as fp:
                sampler = joblib.load(fp)
            logger.info("Found a saved Sampler instance for this study.")
        else:
            logger.info("No existing sampler available for this study. Creating a new one.")
            sampler = optuna.samplers.RandomSampler(seed=SEED)
        
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            direction="maximize",
            storage="mysql://" + mysql_url,
            load_if_exists=True,
        )
        try:
            logger.info("Running...")
            return study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
        except KeyboardInterrupt as e:
            study.trials_dataframe().to_csv(REPORT_ROOT / ("optim" + study_name + ".csv"))
            with open(REPORT_ROOT / ("sampler" + study_name + ".pkl"), "w+") as fp:
                joblib.dump(sampler, fp)
        finally:
            raise KeyboardInterrupt
    
    
def optim_MEG2phoneme_seq(args):
    """Seq2Seq phoneme decoding; Random search script."""
    
    from tools.parameters import P_ph_meg as P

    # tile labels along time axis for seq2seq prediction
    X, y, encoder, classes = load_data("MEG->phoneme", args.decim, tile=True)
    
    num_trials = args.trials
    num_folds = args.folds
    num_inits = args.inits

    study_name = args.name
    mysql_url = args.db
    
    seeds=RND.choice(99999, size=num_inits, replace=False)

    def objective(trial):

        folder = kfold(num_folds, random_state=SEED)
        
        mean_top1 = []
        mean_top10 = []
        
        params = Parameters(
            sr=trial.suggest_float("spectral_radius", 1e-4, 1e2, log=True),
            lr=trial.suggest_float("leaking_rate", 1e-6, 1.0, log=True),
            input_scaling=trial.suggest_float("input_scaling", 1e-4, 1e2, log=True),
            ridge=trial.suggest_float("ridge", 1e-8, 1e-2, log=True),
            input_bias=False,
            bias_scaling=0.0,
            units=300,
        )
        
        for train_idxs, val_idxs in folder.split(X, y[:, 0, :].flatten()):
            X_train, y_train = X[train_idxs], y[train_idxs]
            X_val, y_val = X[val_idxs], y[val_idxs]
            
            shape = y_train.shape
            y_train = encoder.transform(y_train.reshape(-1, 1)).reshape(*shape[:-1], -1)

            top1 = [0.0] * num_folds
            top10 = [0.0] * num_folds
            for i in range(num_inits):
                
                params.seed = seeds[i]
                
                P.update(params)

                y_pred = train_phoneme_decoder(X_train, X_val, y_train, y_val, method="seq2seq", params=P)
                
                n_categories = y_pred.shape[-1]
                
                accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=1, labels=classes)
                top10accuracy = top_k_accuracy_score(y_val.reshape(-1, 1), y_pred.reshape(-1, n_categories), k=10, labels=classes)
                
                top1[i] = accuracy
                top10[i] = top10accuracy
                
            mean_top1.append(np.mean(top1))
            mean_top10.append(np.mean(top10))
            
        trial.set_user_attr("top10_accuracy", np.mean(mean_top10))
        trial.set_user_attr("n_folds", num_folds)
        trial.set_user_attr("n_inits", num_inits)
        trial.set_user_attr("seeds", seeds.tolist())
        
        return np.mean(mean_top1)
    
    if args.dryrun:
        logger.info(objective(optuna.trial.FixedTrial({"spectral_radius": 0.9, "leaking_rate": 0.5, "input_scaling": 1.0, "ridge": 1e-8})))
    else:
        logger.info(f"Starting study: {study_name}")
        
        sampler_path = REPORT_ROOT / ("sampler" + study_name + ".pkl")
        if sampler_path.is_file():
            with open(REPORT_ROOT / ("sampler" + study_name + ".pkl"), "w+") as fp:
                sampler = joblib.load(fp)
            logger.info("Found a saved Sampler instance for this study.")
        else:
            logger.info("No existing sampler available for this study. Creating a new one.")
            sampler = optuna.samplers.RandomSampler(seed=SEED)
        
        study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            direction="maximize",
            storage="mysql://" + mysql_url,
            load_if_exists=True,
        )
        try:
            logger.info("Running...")
            return study.optimize(objective, n_trials=num_trials, show_progress_bar=True)
        except KeyboardInterrupt as e:
            logger.warning(f"Interrutping study. Saving sampler and trials to {REPORT_ROOT}")
            study.trials_dataframe().to_csv(REPORT_ROOT / ("optim" + study_name + ".csv"))
            with open(REPORT_ROOT / ("sampler" + study_name + ".pkl"), "w+") as fp:
                joblib.dump(sampler, fp)
        finally:
            raise KeyboardInterrupt


if __name__ == "__main__":
    
    rpy.verbosity(0)

    args = parser.parse_args()
    
    tasks_table = {
        "optim-MEG2phoneme-vec": optim_MEG2phoneme_vec,
        "optim-MEG2phoneme-seq": optim_MEG2phoneme_seq,
    }
    
    tasks_table[args.task](args)

    logger.info("Done")
