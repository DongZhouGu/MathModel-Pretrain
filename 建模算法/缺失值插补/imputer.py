import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


### ����ȱʧ��������ͼ
def miss_heatamp(data, title):
    cols = data.columns
    colours = ['#006699', '#ffff99']  ## ��һ��ûȱʧ����ɫ���ڶ��ȱʧ����ɫ

    plt.figure(figsize=(12, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ����
    plt.rcParams['axes.unicode_minus'] = False  # ����޷���ʾ���ŵ�����
    sns.set(font='SimHei', font_scale=1)
    f = sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))
    f.set_title(title)

### ȱʧֵ��ֱ��ͼ
def miss_bar(data, title):
    '''
    data : dataframe��ʽ�����ݣ���Ϊ���ݣ���Ϊ����
    title : ͼ������
    '''
    missValue2miss_num = {}
    for col in data.columns:
        missing = data[col].isnull()
        num_missing = np.sum(missing)
        if num_missing > 0:
            missValue2miss_num[col] = num_missing
    df = pd.DataFrame([missValue2miss_num])
    df.index = ['miss_num']
    df = df.T

    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ����
    plt.rcParams['axes.unicode_minus'] = False  # ����޷���ʾ���ŵ�����
    sns.set(font='SimHei', font_scale=1)

    f = sns.barplot(x=df.index, y=df['miss_num'], color='#336699')
    f.set_title(title)


### ���ɭ�ֲ�ֵ
def rf_impute(data):
    '''
    data:dataframe��ʽ
    '''

    copy_data = data.copy()
    miss_columns = copy_data.isnull().sum()[data.isnull().sum() != 0].sort_values().index.tolist()
    unmiss_columns = copy_data.isnull().sum()[data.isnull().sum() == 0].sort_values().index.tolist()
    for col in miss_columns:
        X_train = copy_data[copy_data[col].notnull()][unmiss_columns].values
        Y_train = copy_data[copy_data[col].notnull()][col].values
        X_test = copy_data[copy_data[col].isnull()][unmiss_columns][unmiss_columns].values
        rfr=RandomForestRegressor()
        rfr.fit(X_train,Y_train)
        predict_value = rfr.predict(X_test)
        copy_data.loc[(copy_data[col].isnull()),col] = predict_value
        unmiss_columns.append(col)
    return copy_data