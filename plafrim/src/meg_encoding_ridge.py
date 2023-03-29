#!/usr/bin/env python
# coding: utf-8
from __future__ import division
import argparse
import logging
import warnings

from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy import stats
from sklearn.model_selection import KFold


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("meg_data", type=str)
parser.add_argument("feat_data", type=str)
parser.add_argument("output", type=str)
parser.add_argument(
    "-s", "--subject", type=str, help="Index (sub=id%8+1, feat=id//8) of job"
)
parser.add_argument(
    "-f", "--feature", type=str, help="Index (sub=id%8+1, feat=id//8) of job"
)
parser.add_argument(
    "-l", "--layer", type=int, help="Index (sub=id%8+1, feat=id//8) of job"
)

T_DIM = 81
MEG_DIM = 208
SESSION = "ses0"
LAMBDA_CV_SPLITS = 10


def ridge_by_lambda(X, Y, Xval, Yval, lambdas):
    scores = np.zeros((lambdas.shape[0], Y.shape[1]))
    for idx, lmbda in enumerate(lambdas):
        # supress LinAlgWarning for ill-conditioned matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Ridge(alpha=lmbda, solver="cholesky").fit(X, Y)
        scores[idx] = 1.0 - r2_score(model.predict(Xval), Yval)
    return scores


def cross_val_ridge(
    train_features,
    train_data,
    n_splits=10,
    lambdas=np.array([10**i for i in range(-6, 10)]),
):
    nL = lambdas.shape[0]
    scores = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)

    for trn, val in kf.split(train_data):
        sc = ridge_by_lambda(
            train_features[trn],
            train_data[trn],
            train_features[val],
            train_data[val],
            lambdas=lambdas,
        )

        scores += sc

    # one lambda per timepoint and sensor (best based on R^2)
    argmax_lambda = np.argmax(scores, axis=0)
    final_lambda = np.array([lambdas[i] for i in argmax_lambda])

    model = Ridge(alpha=final_lambda, solver="cholesky")
    model.fit(train_features, train_data)

    return model, np.array([lambdas[i] for i in argmax_lambda])


def get_info(feature):
    feature_type = feature.split("-")[0].split(".")[0]

    if feature_type == "bert":
        feature_name = feature.split("-")[-1].split(".")[0]
        feature_name = feature_type + "_" + feature_name
    elif feature_type == "concat":
        feature_name = "-".join(feature.split("-")[1:]).split(".")[0]
        feature_name = feature_type + "_" + feature_name
    else:
        feature_name = feature.split(".")[0]

    return feature_type, feature_name


def select_data(data, features, train, test, has_layers, layer):
    if has_layers:
        x_train = np.concatenate(
            [
                np.array(features[train[0]][layer]),
                np.array(features[train[1]][layer]),
                np.array(features[train[2]][layer]),
            ],
            axis=0,
        )
        x_test = np.array(features[test[0]][layer])
    else:
        x_train = np.concatenate(
            [
                np.array(features[train[0]]),
                np.array(features[train[1]]),
                np.array(features[train[2]]),
            ],
            axis=0,
        )
        x_test = np.array(features[test[0]])

    y_train = np.concatenate(
        [
            np.array(data[train[0]]).reshape(len(data[train[0]]), MEG_DIM * T_DIM),
            np.array(data[train[1]]).reshape(len(data[train[1]]), MEG_DIM * T_DIM),
            np.array(data[train[2]]).reshape(len(data[train[2]]), MEG_DIM * T_DIM),
        ],
        axis=0,
    )
    y_test = np.array(data[test[0]]).reshape(len(data[test[0]]), MEG_DIM * T_DIM)

    return x_train, y_train, x_test, y_test


def encode_meg(sub, feature, layer, meg_source, feat_source, output):
    meg_source = Path(meg_source)
    feat_source = Path(feat_source)
    output = Path(output)

    sub = str(sub).zfill(2)

    features = np.load(feat_source / feature, allow_pickle=True)
    data = np.load(meg_source / f"sub{sub}-meg-data-{SESSION}.npy", allow_pickle=True)

    feature_type, feature_name = get_info(feature)
    logger.info(f"Prediction with: {feature_name}")

    results_save_dir = output / f"sub-{sub}-predictions"
    results_save_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(9)
    kf = KFold(n_splits=4)

    has_layers = False if layer == -1 else True

    if has_layers:
        logger.info("Layer: " + str(layer))
        save_dir = results_save_dir / f"{feature_name}_layer{layer}_s{sub}_predictions"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = results_save_dir / f"{feature_name}_s{sub}_predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "weights_0.npy").exists():
        logger.info("Already computed. Skipping.")
        return

    logger.info(f"MEG shape: {data.shape}\nFeat shape: {features.shape}")

    r2s = np.zeros(MEG_DIM * T_DIM)
    corr = np.zeros((MEG_DIM * T_DIM, 2))

    all_preds = []
    all_reals = []

    for split_num, (train, test) in enumerate(kf.split(np.arange(4))):
        x_train, y_train, x_test, y_test = select_data(
            data=data,
            features=features,
            train=train,
            test=test,
            has_layers=has_layers,
            layer=layer,
        )

        model, lbda = cross_val_ridge(
            train_features=x_train, train_data=y_train, n_splits=LAMBDA_CV_SPLITS
        )
        y_pred = model.predict(x_test)

        np.save(save_dir / f"y_pred_{split_num}.npy", np.nan_to_num(y_pred))
        np.save(save_dir / f"y_test_{split_num}.npy", y_test)
        np.save(save_dir / f"weights_{split_num}.npy", model.coef_)
        np.save(save_dir / f"intercept_{split_num}.npy", model.intercept_)
        np.save(save_dir / f"lbda_{split_num}.npy", lbda)

        all_reals.append(y_test)
        all_preds.append(y_pred)

    all_reals = np.vstack(all_reals)
    all_preds = np.vstack(all_preds)

    r2s = r2_score(all_reals, all_preds, multioutput="raw_values")

    for i in range(all_reals.shape[1]):
        if np.nonzero(all_reals[:, i])[0].size > 0:
            corr[i] = stats.pearsonr(all_reals[:, i], all_preds[:, i])
        else:
            r2s[i] = 0.0
            corr[i][1] = 1.0

    logger.info(f"Max R2: {np.max(r2s)}")

    np.save(save_dir / "r2s.npy", np.nan_to_num(r2s))
    np.save(save_dir / "corr.npy", np.nan_to_num(corr))
    
    return save_dir

    
if __name__ == "__main__":
    args = parser.parse_args()

    subject = args.subject.zfill(2)
    feature = args.feature
    layer = args.layer

    meg_data = Path(args.meg_data)
    feat_data = Path(args.feat_data)
    output = Path(args.output)

    try:
        save_dir = encode_meg(
            sub=subject,
            feature=feature,
            layer=layer,
            meg_source=meg_data,
            feat_source=feat_data,
            output=output,
        )
        
        print(save_dir)

    except Exception as e:
        logger.critical(f"Sub {subject}, Feat {feature}, Layer {layer}")
        logger.critical("Something bad happened...")
        raise e
