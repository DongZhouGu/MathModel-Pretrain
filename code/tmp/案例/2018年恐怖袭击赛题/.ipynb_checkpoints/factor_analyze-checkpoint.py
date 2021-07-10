from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import numpy as np
from sklearn import preprocessing
from matplotlib import cm
import matplotlib.pyplot as plt
def Factor_analyzer(data):
    print("\n原始数据:\n",data)
    # 皮尔森相关系数
    data_corr=data.corr()
    print("\n相关系数:\n",data_corr)
    #热力图
    cmap = cm.Blues
    fig=plt.figure()
    ax=fig.add_subplot(111)
    map = ax.imshow(df2_corr, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title('correlation coefficient--headmap')
    ax.set_yticks(range(len(df2_corr.columns)))
    ax.set_yticklabels(df2_corr.columns)
    ax.set_xticks(range(len(df2_corr)))
    ax.set_xticklabels(df2_corr.columns)
    plt.colorbar(map)
    plt.show()
    # KMO测度
    def kmo(dataset_corr):
        corr_inv = np.linalg.inv(dataset_corr)
        nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        dataset_corr = np.asarray(dataset_corr)
        kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        kmo_value = kmo_num / kmo_denom
        return kmo_value

    print("\nKMO测度:", kmo(df2_corr))

        
        
        