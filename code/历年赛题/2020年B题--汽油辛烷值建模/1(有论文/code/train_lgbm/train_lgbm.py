#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
import pylab as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.preprocessing import QuantileTransformer
from joblib import dump
import warnings
import seaborn as sns

warnings.filterwarnings("ignore")
X = pd.read_csv("../processed.csv")
y_s = X.pop("S")
RON = X.pop("RON")
y = X.pop("RON_loss")


def func(X: pd.DataFrame, y: pd.Series, selection_times=3, title="RON_loss", del_abnormal=False,
         abnormal_threshold=0.08):
    y = np.array(y)
    selector = SelectFromModel(
        estimator=GradientBoostingRegressor(random_state=0))
    X_ = QuantileTransformer(n_quantiles=1000).fit_transform(X)
    X_ = pd.DataFrame(X_, columns=X.columns)
    for i in range(selection_times):
        X_d = selector.fit_transform(X_, y)
        X_ = pd.DataFrame(X_d, columns=X_.columns[selector.get_support()])
    X_d = QuantileTransformer(n_quantiles=1000).fit_transform(X[X_.columns])
    # X_d=X[X_.columns].values
    X_ = pd.DataFrame(X_d, columns=X_.columns)
    print(f"{title} | {selection_times}次特征筛选后的X_.shape = {X_.shape}")
    print(f"{title} | 特征筛选后保留的列： {X_.columns.tolist()}")
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    pipeline = LGBMRegressor(random_state=0, n_estimators=100, learning_rate=0.1)
    pipeline.fit(X_, y)
    y_pred = pipeline.predict(X_)
    train_score = r2_score(y, y_pred)
    pearson_correlation = pearsonr(y, y_pred)[0]
    print(f"{title} | 在训练集上，r2 = {train_score}, pearson 相关系数 = {pearson_correlation}")
    y_ = y.copy()
    y_pred_ = y_pred.copy()
    if del_abnormal:
        y_pred = pipeline.predict(X_)
        err = np.abs(y - y_pred)
        mask = err > abnormal_threshold
        print(f"{title} | 异常样本数 = {np.count_nonzero(mask)}")
        plt.rcParams['figure.figsize'] = (7, 4.5)
        plt.grid(alpha=0.2)
        plt.scatter(y[mask], y_pred[mask], label="abnormal samples", c="r")
        plt.scatter(y[~mask], y_pred[~mask], label="normal samples", c="b")
        plt.legend(loc="best")
        print(f"{title} | 删除异常样本前的表现 = {cross_val_score(pipeline, X_, y, cv=cv).mean()}")
        X_ = X_.loc[~mask, :]
        y = y[~mask]
        print(f"{title} | 删除异常样本后的表现 = {cross_val_score(pipeline, X_, y, cv=cv).mean()}")
        plt.title(f"{title} abnormal samples")
        plt.xlabel("y true")
        plt.ylabel("y pred")
        plt.savefig(f"{title}_abnormal.pdf")
        plt.close()
    valid_scores = []
    plt.rcParams['figure.figsize'] = (18, 12)
    for i, (train_ix, valid_ix) in enumerate(cv.split(X_, y)):
        X_train = X_.iloc[train_ix, :].copy()
        X_valid = X_.iloc[valid_ix, :].copy()
        y_train = y[train_ix].copy()
        y_valid = y[valid_ix].copy()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_valid)
        plt.subplot(2, 3, i + 1)
        sns.regplot(x="y true", y="y pred",
                    data=pd.DataFrame({"y true": y_valid, "y pred": y_pred}))
        plt.title(f"fold-{i + 1}")
        valid_scores.append(r2_score(y_valid, y_pred))
    plt.subplot(2, 3, 6)
    sns.regplot(x="y true", y="y pred",
                data=pd.DataFrame({"y true": y_, "y pred": y_pred_}))
    plt.title(f"train-set")
    plt.suptitle(f"{title} cross-validation")
    print(f"{title} | 5折交叉验证后，在验证集上的平均r2 = {np.mean(valid_scores)}\n"
          f"{title} | 每折的r2 = {valid_scores}")
    plt.savefig(f"{title}_cross-validation.pdf")
    plt.close()
    X_["label"] = y
    X_.to_csv(f"{title}_data.csv", index=False)
    dump(pipeline, f"{title}_model.bz2")


func(X, y, selection_times=3, title="RON_loss", del_abnormal=True, abnormal_threshold=0.08)
func(X, y_s, selection_times=2, title="S", del_abnormal=False)
