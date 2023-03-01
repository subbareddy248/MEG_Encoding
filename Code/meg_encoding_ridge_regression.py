#!/usr/bin/env python
# coding: utf-8
from __future__ import division  
import os   
import time 
import argparse

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy import stats                             
from numpy.linalg import inv, svd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
from scipy.stats import zscore

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--id", type=str, help="Index (sub=id%8+1, feat=id//8) of job")

#REPORTS = Path("/beegfs/ntrouvai/MEG-analysis/reports")
#DATA = Path("/beegfs/ntrouvai/MEG-analysis/data")
REPORTS = Path("reports")
DATA = Path("data")

def corr(X,Y):
    return np.mean(zscore(X)*zscore(Y),0)

def R2(Pred,Real):
    SSres = np.mean((Real-Pred)**2,0)
    SStot = np.var(Real,0)
    return np.nan_to_num(1-SSres/SStot)

def R2r(Pred,Real):
    R2rs = R2(Pred,Real)
    ind_neg = R2rs<0
    R2rs = np.abs(R2rs)
    R2rs = np.sqrt(R2rs)
    R2rs[ind_neg] *= - 1
    return R2rs

def ridge(X,Y,lmbda):
    return np.dot(inv(X.T.dot(X)+lmbda*np.eye(X.shape[1])),X.T.dot(Y))

def ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error

def ridge_sk(X,Y,lmbda):
    rd = Ridge(alpha = lmbda)
    rd.fit(X,Y)
    return rd.coef_.T

def ridgeCV_sk(X,Y,lmbdas):
    rd = RidgeCV(alphas = lmbdas,solver = 'svd')
    rd.fit(X,Y)
    return rd.coef_.T

def ridge_by_lambda_sk(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = ridge_sk(X,Y,lmbda)
        error[idx] = 1 -  R2(np.dot(Xval,weights),Yval)
    return error

def ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))

def ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(Vt,np.diag(d).dot(U.T.dot(Y)))
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge(X,Y,lmbda):
    return np.dot(X.T.dot(inv(X.dot(X.T)+lmbda*np.eye(X.shape[0]))),Y)

def kernel_ridge_by_lambda(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    for idx,lmbda in enumerate(lambdas):
        weights = kernel_ridge(X,Y,lmbda)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def kernel_ridge_svd(X,Y,lmbda):
    U, s, Vt = svd(X.T, full_matrices=False)
    d = s / (s** 2 + lmbda)
    return np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)

def kernel_ridge_by_lambda_svd(X, Y, Xval, Yval, lambdas=np.array([0.1,1,10,100,1000])):
    error = np.zeros((lambdas.shape[0],Y.shape[1]))
    U, s, Vt = svd(X.T, full_matrices=False)
    for idx,lmbda in enumerate(lambdas):
        d = s / (s** 2 + lmbda)
        weights = np.dot(np.dot(U,np.diag(d).dot(Vt)),Y)
        error[idx] = 1 - R2(np.dot(Xval,weights),Yval)
    return error


def cross_val_ridge(train_features,train_data, n_splits = 10, 
                    lambdas = np.array([10**i for i in range(-6,10)]),
                    method = 'plain',
                    do_plot = False):
    
    ridge_1 = dict(plain = ridge_by_lambda,
                   svd = ridge_by_lambda_svd,
                   kernel_ridge = kernel_ridge_by_lambda,
                   kernel_ridge_svd = kernel_ridge_by_lambda_svd,
                   ridge_sk = ridge_by_lambda_sk)[method]
    ridge_2 = dict(plain = ridge,
                   svd = ridge_svd,
                   kernel_ridge = kernel_ridge,
                   kernel_ridge_svd = kernel_ridge_svd,
                   ridge_sk = ridge_sk)[method]
    
    n_voxels = train_data.shape[1]
    nL = lambdas.shape[0]
    r_cv = np.zeros((nL, train_data.shape[1]))

    kf = KFold(n_splits=n_splits)
    start_t = time.time()
    for icv, (trn, val) in enumerate(kf.split(train_data)):
#         print('ntrain = {}'.format(train_features[trn].shape[0]))
        cost = ridge_1(train_features[trn],train_data[trn],
                               train_features[val],train_data[val], 
                               lambdas=lambdas)
        if do_plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(cost,aspect = 'auto')
        r_cv += cost
#         if icv%3 ==0:
#             print(icv)
#         print('average iteration length {}'.format((time.time()-start_t)/(icv+1)))
    if do_plot:
        plt.figure()
        plt.imshow(r_cv,aspect='auto',cmap = 'RdBu_r');

    argmin_lambda = np.argmin(r_cv,axis = 0)
    weights = np.zeros((train_features.shape[1],train_data.shape[1]))
    for idx_lambda in range(lambdas.shape[0]): # this is much faster than iterating over voxels!
        idx_vox = argmin_lambda == idx_lambda
        weights[:,idx_vox] = ridge_2(train_features, train_data[:,idx_vox],lambdas[idx_lambda])
    if do_plot:
        plt.figure()
        plt.imshow(weights,aspect='auto',cmap = 'RdBu_r',vmin = -0.5,vmax = 0.5);

    return weights, np.array([lambdas[i] for i in argmin_lambda])




def GCV_ridge(train_features,train_data,lambdas = np.array([10**i for i in range(-6,10)])):
    
    n_lambdas = lambdas.shape[0]
    n_voxels = train_data.shape[1]
    n_time = train_data.shape[0]
    n_p = train_features.shape[1]

    CVerr = np.zeros((n_lambdas, n_voxels))

    # % If we do an eigendecomp first we can quickly compute the inverse for many different values
    # % of lambda. SVD uses X = UDV' form.
    # % First compute K0 = (X'X + lambda*I) where lambda = 0.
    #K0 = np.dot(train_features,train_features.T)
    print('Running svd',)
    start_time = time.time()
    [U,D,Vt] = svd(train_features,full_matrices=False)
    V = Vt.T
    print(U.shape,D.shape,Vt.shape)
    print('svd time: {}'.format(time.time() - start_time))

    for i,regularizationParam in enumerate(lambdas):
        regularizationParam = lambdas[i]
        print('CVLoop: Testing regularization param: {}'.format(regularizationParam))

        #Now we can obtain Kinv for any lambda doing Kinv = V * (D + lambda*I)^-1 U'
        dlambda = D**2 + np.eye(n_p)*regularizationParam
        dlambdaInv = np.diag(D / np.diag(dlambda))
        KlambdaInv = V.dot(dlambdaInv).dot(U.T)
        
        # Compute S matrix of Hastie Trick  H = X(XT X + lambdaI)-1XT
        S = np.dot(U, np.diag(D * np.diag(dlambdaInv))).dot(U.T)
        denum = 1-np.trace(S)/n_time
        
        # Solve for weight matrix so we can compute residual
        weightMatrix = KlambdaInv.dot(train_data);


#         Snorm = np.tile(1 - np.diag(S) , (n_voxels, 1)).T
        YdiffMat = (train_data - (train_features.dot(weightMatrix)));
        YdiffMat = YdiffMat / denum;
        CVerr[i,:] = (1/n_time)*np.sum(YdiffMat * YdiffMat,0);


    # try using min of avg err
    minerrIndex = np.argmin(CVerr,axis = 0);
    r=np.zeros((n_voxels));

    for nPar,regularizationParam in enumerate(lambdas):
        ind = np.where(minerrIndex==nPar)[0];
        if len(ind)>0:
            r[ind] = regularizationParam;
            print('{}% of outputs with regularization param: {}'.format(int(len(ind)/n_voxels*100),
                                                                        regularizationParam))
            # got good param, now obtain weights
            dlambda = D**2 + np.eye(n_p)*regularizationParam
            dlambdaInv = np.diag(D / np.diag(dlambda))
            KlambdaInv = V.dot(dlambdaInv).dot(U.T)

            weightMatrix[:,ind] = KlambdaInv.dot(train_data[:,ind]);


    return weightMatrix, r

def sub_to_subjectnum(sub):
    for i in range(len(all_subjects)):
        if all_subjects[i] == sub:
            return i+1
    return 0


def eval(feature_name, feature_file, sub, save_regressed_y = False):  
    results_save_dir = REPORTS / f"sub-{sub}"

    print(feature_name)
    np.random.seed(9)
    kf = KFold(n_splits=4)
    features = np.load(DATA / feature_file, allow_pickle=True)
    
    save_dir = os.path.join(results_save_dir, feature_name)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    data = np.load(DATA / f"meg/sub0{sub}-meg-data-ses0.npy", allow_pickle=True)
    
    print("Subject " + str(sub))
    
    for sub in np.arange(0,1):
        
        if os.path.exists(os.path.join(save_dir, str(sub) + "_r2s.npy")):
            continue
        
        print(data.shape)
        print(features.shape)

        r2s = np.zeros(208*81)   
        corr = np.zeros((208*81,2))

        split_num = 0

        all_preds = []
        all_reals = []

        for train, test in kf.split(np.arange(4)):
            #for BERT
            x_train = np.concatenate([
                np.array(features[train[0]][6]), 
                np.array(features[train[1]][6]), 
                np.array(features[train[2]][6])
                ], axis=0)
            
            x_test = np.array(features[test[0]][6])
            #for POS, DEP, CM
#             x_train = np.concatenate([np.array(features[train[0]]), np.array(features[train[1]]), np.array(features[train[2]])], axis=0)
#             x_test = np.array(features[test[0]])
            #for Nodecount, Wordlen
#             x_train = np.concatenate([np.array(features[train[0]]).reshape(-1,1), np.array(features[train[1]]).reshape(-1,1), np.array(features[train[2]]).reshape(-1,1)], axis=0)
#             x_test = np.array(features[test[0]]).reshape(-1,1)

            y_train = np.concatenate([
                np.array(data[train[0]]).reshape(len(data[train[0]]),208*81), 
                np.array(data[train[1]]).reshape(len(data[train[1]]),208*81),
                np.array(data[train[2]]).reshape(len(data[train[2]]),208*81)
                ], axis=0)
            y_test = np.array(data[test[0]]).reshape(len(data[test[0]]),208*81)
            
#             x_train = np.random.rand(y_train.shape[0],768)
#             x_test = np.random.rand(y_test.shape[0],768)
            #data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2]))
            
#             x_train, x_test = features[train_index], features[test_index]
#             y_train, y_test = data[train_index], data[test_index]
            
#             x_train = stats.zscore(x_train,axis=0)
#             x_train = np.nan_to_num(x_train)

#             x_test = stats.zscore(x_test,axis=0)
#             x_test = np.nan_to_num(x_test)

#             y_train = stats.zscore(y_train,axis=0)
#             y_train = np.nan_to_num(y_train)

#             y_test = stats.zscore(y_test,axis=0) 
#             y_test = np.nan_to_num(y_test)
            
            print(x_train.shape)
            print(y_train.shape)
            print(x_test.shape)
            print(y_test.shape)

            weights, lbda = cross_val_ridge(x_train,y_train)        
            y_pred = np.dot(x_test,weights)

            np.save(os.path.join(save_dir, "{}_y_pred_{}".format(str(sub), split_num)),np.nan_to_num(y_pred))
            np.save(os.path.join(save_dir, "{}_y_test_{}".format(str(sub), split_num)),y_test)
            #np.save(os.path.join(save_dir, "{}_weights_{}".format(str(sub), split_num)),weights)
            np.save(os.path.join(save_dir, "{}_lbda_{}".format(str(sub), split_num)),lbda)
            
            if save_regressed_y:
                y_pred_train = np.dot(x_train,weights)
                np.save(os.path.join(save_dir, "{}_y_regress_test_{}".format(str(sub), split_num)),y_test - y_pred)
                np.save(os.path.join(save_dir, "{}_y_regress_train_{}".format(str(sub), split_num)),y_train - y_pred_train)

            split_num += 1

            all_reals.append(y_test)
            all_preds.append(y_pred)
            #break

        all_reals = np.vstack(all_reals)
        all_preds = np.vstack(all_preds)

        r2s = r2_score(all_reals,all_preds,multioutput="raw_values")

        for i in range(all_reals.shape[1]):
            if np.nonzero(all_reals[:,i])[0].size > 0:
                corr[i] = stats.pearsonr(all_reals[:,i],all_preds[:,i])
            else:
                r2s[i] = 0
                corr[i][1] = 1

        print(np.max(r2s))

        np.save(os.path.join(save_dir, str(sub) + "_r2s"),np.nan_to_num(r2s))
        np.save(os.path.join(save_dir, str(sub) + "_corr"),np.nan_to_num(corr))


if __name__ == "__main__":

    allfeatures = [
        'bert-base-lw-5.npy',
        'bert-base-lw-rh-5.npy',
        'bert-base-lw-rh-20.npy',
        'bert-base-lw-4.npy',
        ]
    
    args = parser.parse_args()
    
    subject = int(args.id) % 8 + 1
    feature = allfeatures[int(args.id) // 8]

    if 'rh' in feature:
        eval(
            'bertseq'+feature.split('-')[-1].split('.')[0]+'rh_s0_predictions',
            "bert-vectors/" + feature,
            subject
        )
    else:
        eval(
            'bertseq'+feature.split('-')[-1].split('.')[0]+'_s0_predictions',
            "bert-vectors/" + feature,
            subject
        )
