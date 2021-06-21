import joblib
import pandas
import matplotlib.pyplot as plt
from sklearn import tree, linear_model, svm, neighbors, ensemble, metrics
from sklearn.ensemble import BaggingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import ExtraTreeRegressor
import util


# 读取数据并划分训练集和测试集
def read_data(type: str):
    # excel_data = pandas.read_excel(path, header=0)
    # # 将特征划分到 X 中，标签划分到 Y 中
    # x = excel_data.iloc[:, 0:15]
    # y = excel_data['ret']
    # # 使用train_test_split函数划分数据集(训练集占75%，测试集占25%)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    x_train, x_test, y_train, y_test = None, None, None, None
    if type == "roh":
        # 特征
        df_x = pandas.read_excel("./data/x1.xlsx")
        # todo: 使用随机森林进行特征选择
        # index_list = util.feature_selector()
        # df_x = pandas.read_excel("/Users/faye/Desktop/output.xlsx", header=[0, 1])
        # df_x = df_x.drop(["样本编号", "时间"], axis=1)
        # df_x = df_x.iloc[:, index_list]
        # 标签
        df_y = pandas.read_excel("./data/y1.xlsx")
        # 去掉变化在 1 以下的
        # drop_index = df_y[df_y[1] < 1.0].index.values
        # df_x = df_x.drop(drop_index)
        # df_y = df_y.drop(drop_index)
        # todo: 增加特征多项式
        poly = PolynomialFeatures(degree=1, include_bias=False)
        df_x = poly.fit_transform(df_x)
        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y.values, test_size=0.2, random_state=0)
    elif type == "s":
        # 特征
        df_x = pandas.read_excel("./data/x2.xlsx")
        # 标签
        df_y = pandas.read_excel("./data/y2.xlsx")
        # 处理 df_y，替换为 0 和 1
        df_y.loc[df_y[1] <= 5, 1] = 1
        df_y.loc[df_y[1] > 5, 1] = 0
        x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.2, random_state=0)
    elif type == "s_3.2":
        # 特征
        df_x = pandas.read_excel("./data/x2.xlsx")
        # 标签
        df_y = pandas.read_excel("./data/y2.xlsx")
        # 处理 df_y，替换为 0 和 1
        df_y.loc[df_y[1] != 3.2, 1] = 0
        df_y.loc[df_y[1] == 3.2, 1] = 1
        x_train, x_test, y_train, y_test = train_test_split(df_x.values, df_y.values, test_size=0.2, random_state=0)

    return x_train, x_test, y_train, y_test


def fit_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    result = model.predict(x_test)
    plt.figure()
    plt.plot(np.arange(len(result)), y_test, 'go-', label='true value')
    plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
    plt.title('score: %f' % score)
    plt.legend()
    plt.show()

    return model, result


# 二分类（离散数据），针对于硫含量
def classification(model, x_train, x_test, y_train, y_test, model_name):
    # 训练模型，并返回训练后的模型和预测结果集
    model, result = fit_model(model, x_train, x_test, y_train, y_test)
    # 模型持久化
    joblib.dump(model, model_name + '.pkl')

    print("精确率：%f" % metrics.precision_score(y_test, result))
    print("召回率：%f" % metrics.recall_score(y_test, result))
    print(metrics.classification_report(y_test, result))


# 回归（连续数据），针对于 ROH
def regression(model, x_train, x_test, y_train, y_test):
    model, result = fit_model(model, x_train, x_test, y_train, y_test)
    joblib.dump(model, 'regression_model.pkl')

    plt.scatter(y_test, result)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel('true')
    plt.ylabel('predict')
    plt.show()

    print(metrics.mean_absolute_error(y_test, result))
    print(metrics.mean_squared_error(y_test, result))


if __name__ == '__main__':
    # 回归预测 roh
    x_train, x_test, y_train, y_test = read_data("roh")
    model = util.get_model("model_LinearRegression")
    regression(model, x_train, x_test, y_train, y_test)
    # 二分类预测硫含量(是否大于 5)
    x_train, x_test, y_train, y_test = read_data("s")
    model = util.get_model("model_LogisticRegression_weight")
    classification(model, x_train, x_test, y_train, y_test, "classification_model_s_5")
    # 二分类预测硫含量(是否大于 3.2)
    x_train, x_test, y_train, y_test = read_data("s_3.2")
    model = util.get_model("model_LogisticRegression")
    classification(model, x_train, x_test, y_train, y_test, "classification_model_s_3.2")
