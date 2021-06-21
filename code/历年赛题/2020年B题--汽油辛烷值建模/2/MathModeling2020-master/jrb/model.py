import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, preprocessing


def normalization(x):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_min_max = min_max_scaler.fit_transform(x)
    return x_min_max


###########1.数据生成部分##########
df_x = pd.read_excel("x.xlsx")
df_y = pd.read_excel("y.xlsx")
# df_x = (df_x - df_x.min()) / (df_x.max() - df_x.min())
# df_y = (df_y - df_y.min()) / (df_y.max() - df_y.min())
x_train, x_test, y_train, y_test = model_selection.train_test_split(df_x.values, df_y[1].values, test_size=0.3,
                                                                    random_state=0)


###########2.回归部分##########
def try_different_method(model):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()


###########3.具体方法选择##########
####3.1决策树回归####
from sklearn import tree

model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2线性回归####
from sklearn import linear_model

model_LinearRegression = linear_model.LinearRegression()
####3.3SVM回归####
from sklearn import svm

model_SVR = svm.SVR()
####3.4KNN回归####
from sklearn import neighbors

model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5随机森林回归####
from sklearn import ensemble

model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)  # 这里使用20个决策树
####3.6Adaboost回归####
from sklearn import ensemble

model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
####3.7GBRT回归####
from sklearn import ensemble

model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
####3.8Bagging回归####
from sklearn.ensemble import BaggingRegressor

model_BaggingRegressor = BaggingRegressor()
####3.9ExtraTree极端随机树回归####
from sklearn.tree import ExtraTreeRegressor

model_ExtraTreeRegressor = ExtraTreeRegressor()
###########4.具体方法调用部分##########
try_different_method(model_LinearRegression)

print(model_LinearRegression.coef_)
