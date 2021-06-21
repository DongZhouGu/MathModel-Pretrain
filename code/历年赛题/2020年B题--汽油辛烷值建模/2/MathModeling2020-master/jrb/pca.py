import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from pandas.core.frame import DataFrame
import pandas as pd


def do_pca(mat):
    pca = PCA(n_components=60)
    pca.fit(mat)
    new_mat = pca.transform(mat)
    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))
    data = DataFrame(new_mat)
    data.to_csv('new_data.csv', index=False, header=False, sep='\t')


def normalization(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_min_max = min_max_scaler.fit_transform(x)
    return x_min_max


def main():
    df = pd.read_excel('data.xlsx')
    # df.drop(df.columns[9], axis=1, inplace=True)
    df_new = DataFrame()
    df_new[8] = df[8]
    df_new[9] = df[9]
    mat = df_new.values
    mat = normalization(mat)
    do_pca(mat)


main()
