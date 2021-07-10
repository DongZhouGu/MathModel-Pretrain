# 时间序列简介

时间序列 是指将同一统计指标的数值按其先后发生的时间顺序排列而成的数列。时间序列分析的主要目的是根据已有的历史数据对未来进行预测。

## 常用的时间序列模型

常用的时间序列模型有四种：**自回归模型 AR(p)**、**移动平均模型 MA(q)**、**自回归移动平均模型 ARMA(p,q)**、**自回归差分移动平均模型 ARIMA(p,d,q)**, 可以说前三种都是 ARIMA(p,d,q)模型的特殊形式。模型的具体方程可以查找相关的专业书籍及网上的资料。

## 时间序列建模基本步骤

- 获取被观测系统时间序列数据；
- 对数据绘图，观测是否为平稳时间序列；对于非平稳时间序列要先进行d阶差分运算，化为平稳时间序列；
- 经过第二步处理，已经得到平稳时间序列。要对平稳时间序列分别求得其自相关系数ACF 和偏自相关系数PACF ，通过对自相关图和偏自相关图的分析，得到最佳的阶层 p 和阶数 q
- 由以上得到的d、q、p ，得到ARIMA模型。然后开始对得到的模型进行模型检验。

自回归模型的限制 
1、自回归模型是用自身的数据进行预测 
2、必须具有平稳性 
3、必须具有相关性，如果自相关系数（φi）小于0.5，则不宜采用 
4、自回归只适用于预测与自身前期相关的现象

## ARIMA

ARIMA模型的参数定义如下：

- **p**：模型中包含的滞后观察数，也称为滞后顺序。
- **d**：原始观测值的差异次数，也称为差分程度。
- **q**：移动平均窗口的大小，也称为移动平均值的顺序。

### 1️⃣平稳性检验

 平稳：就是围绕着一个常数上下波动且波动范围有限，即有常数均值和常数方差。如果有明显的趋势或周期性，那它通常不是平稳序列。一般有三种方法：

> （1）直接画出时间序列的趋势图，看趋势判断。
>
> （2）画自相关和偏自相关图：平稳的序列的自相关图（Autocorrelation）和偏相关图（Partial Correlation）要么拖尾，要么是截尾。
>
> （3）单位根检验ADF检验：检验序列中是否存在单位根，如果存在单位根就是非平稳时间序列。

ADF检验全称是 Augmented Dickey-Fuller test，顾名思义，ADF是 Dickey-Fuller检验的增广形式。DF检验只能应用于一阶情况，当序列存在高阶的滞后相关时，可以使用ADF检验，所以说ADF是对DF检验的扩展。

```python
t=sm.tsa.stattools.adfuller(time_series, )
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)
```

输出

```
                                value
Test Statistic Value         0.807369
p-value                      0.991754
Lags Used                           1
Number of Observations Used        31
Critical Value(1%)           -3.66143
Critical Value(5%)           -2.96053
Critical Value(10%)          -2.61932
```

ADF数值都比这些标准值大。。说明接受原假设（ADF原假设是存在单位根）
所以存在单位根 原数列不平稳。。做差分

```python
time_series = time_series.diff(1)
time_series = time_series.dropna(how=any)
time_series.plot(figsize=(8,6))
plt.show()
t=sm.tsa.stattools.adfuller(time_series)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)
```

```
                                 value
Test Statistic Value           -3.52276
p-value                      0.00742139
Lags Used                             0
Number of Observations Used          31
Critical Value(1%)             -3.66143
Critical Value(5%)             -2.96053
Critical Value(10%)            -2.61932
```

-3.52276小于置信度区间1%的 -3.66143 ，拒绝原假设（有单位根），所以一阶差分后平稳了

### 2️⃣确定ARMA的阶数

根据时间序列的识别规则，采用 ACF 图、PAC 图，AIC 准则（赤道信息量准则）和 BIC 准则（贝叶斯准则）相结合的方式来确定 ARMA 模型的阶数, 应当选取 AIC 和 BIC 值达到最小的那一组为理想阶数。

#### （1）利用自相关图和偏自相关图



#### （2）利用AIC、BIC自动定阶



### 3️⃣建立ARMA模型并预测





### 4️⃣对残差进行ADF检验



### 程序实现

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
import statsmodels.tsa.stattools as st
import scipy.stats as scs
from statsmodels.tsa.arima_model import ARIMA


class Arima:
    def __init__(self, data, n):
        """
            :param data: Series/np/list
            :param n: 预测数量
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        self.check()
        self.pre_model()
        self.build_model(n)
        print("返回值为dataframe，可通过.res_df拿到, 可通过.plot_res画预测图\n", self.res_df)

    def check(self):
        series = pd.Series(self.data.reshape(-1))
        # 平稳性ADF检验
        print('+++++++++++++++++++++++++++++++++开始进行平稳性ADF检验+++++++++++++++++++++++++++++++')
        d = 0
        while (True):
            if (d > 0):
                series = series.diff(1)
                series = series.dropna(how=any)
            t = sm.tsa.stattools.adfuller(series, )
            output = pd.DataFrame(
                index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                       "Critical Value(1%)",
                       "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
            output['value']['Test Statistic Value'] = t[0]
            output['value']['p-value'] = t[1]
            output['value']['Lags Used'] = t[2]
            output['value']['Number of Observations Used'] = t[3]
            output['value']['Critical Value(1%)'] = t[4]['1%']
            output['value']['Critical Value(5%)'] = t[4]['5%']
            output['value']['Critical Value(10%)'] = t[4]['10%']
            print(output)
            if t[1] > 0.05:
                print(f'单位根检验中p值为{t[1]}，大于0.05，为非平稳序列,进行{d + 1}阶差分')
                d += 1
            else:
                print('单位根检验中p值为%.2f，小于0.05，为平稳序列' % (t[1]))
                self.d = d
                break
        print(f'++++++++++++++++++++++++++ADF检验完成，{d}阶差分后已为平稳序列+++++++++++++++++++++++++++++++')
        print(f'++++++++++++++++++++++++++开始白噪声检验+++++++++++++++++++++++++++++++')
        noiseP = acorr_ljungbox(series, lags=1)[-1]
        if noiseP <= 0.05:
            print('白噪声检验中p值为%.2f，小于0.05，为非白噪声' % noiseP)
        else:
            print('白噪声检验中%.2f，大于0.05，为白噪声' % noiseP)
        print(f'++++++++++++++++++++++++++白噪声检验完成+++++++++++++++++++++++++++++++')
        self.data_diff = series

    def pre_model(self):
        series = self.data_diff
        self.time_plot(series)
        import warnings
        warnings.filterwarnings("ignore")
        pMax = int(series.shape[0] / 10)  # 一般阶数不超过length/10
        qMax = pMax  # 一般阶数不超过length/10
        order = st.arma_order_select_ic(series, max_ar=pMax, max_ma=qMax, ic=['aic', 'bic', 'hqic'])
        p, q = order.aic_min_order
        print('AIC准则下确定p,q为%s,%s' % (p, q))
        p, q = order.bic_min_order
        print('BIC准则下确定p,q为%s,%s' % (p, q))
        self.q = q
        self.p = p

    # 借助AIC、BIC统计量自动确定p,q
    def build_model(self, n):
        print(f'++++++++++++++++++++++++++开始建立ARIMA模型+++++++++++++++++++++++++++++++')
        series = pd.Series(self.data.reshape(-1))
        print('ARIMA建模使用参数：p=%s,d=%s,q=%s' % (self.p, self.d, self.q))
        model = ARIMA(series, order=(self.p, self.d, self.q)).fit()
        predict_n = model.forecast(n)[0]

        fit_v = model.fittedvalues
        for _ in range(self.d):
            fit_v = fit_v.cumsum()
        fit_v += series[0]
        fit_res = [series[0]]
        fit_res.extend(x for x in fit_v)
        fit_res.extend(x for x in predict_n)

        delta = [np.nan]
        delta.extend(x for x in model.resid)
        self.res_df = pd.concat([pd.DataFrame({'原始值': self.data}), pd.DataFrame({'预测值': fit_res}),
                                 pd.DataFrame({'残差': delta}),
                                 pd.DataFrame({'相对误差': list(map(lambda x: '{:.2%}'.format(x), np.abs(delta / self.data)))})
                                 ], axis=1)
        self.verify(model.resid)

    # 模型验证，针对残差
    def verify(self, resid):
        print(f'++++++++++++++++++++++++++开始模型验证+++++++++++++++++++++++++++++++')
        t = sm.tsa.stattools.adfuller(resid, )
        output = pd.DataFrame(
            index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                   "Critical Value(1%)",
                   "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
        output['value']['Test Statistic Value'] = t[0]
        output['value']['p-value'] = t[1]
        output['value']['Lags Used'] = t[2]
        output['value']['Number of Observations Used'] = t[3]
        output['value']['Critical Value(1%)'] = t[4]['1%']
        output['value']['Critical Value(5%)'] = t[4]['5%']
        output['value']['Critical Value(10%)'] = t[4]['10%']
        print(output)
        resid = pd.Series(resid)
        self.time_plot(resid, title='ARIMA残差')

    def time_plot(self, series, title=''):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig = plt.figure(figsize=(10, 8))
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        series.plot(ax=ts_ax)
        ts_ax.set_title(f'{title}时序图')
        plot_acf(series, ax=acf_ax, alpha=0.5)
        acf_ax.set_title('自相关系数')
        plot_pacf(series.values, ax=pacf_ax, nlags=series.shape[0]-2,alpha=0.5)
        pacf_ax.set_title('偏自相关系数')
        sm.qqplot(series, line='s', ax=qq_ax)
        qq_ax.set_title('QQ 图')
        scs.probplot(series, sparams=(series.mean(),
                                      series.std()), plot=pp_ax)
        pp_ax.set_title('PP 图')
        plt.tight_layout()
        plt.show()

    def plot_res(self, xlabel='', ylabel=''):
        res_df = self.res_df
        f, ax = plt.subplots(figsize=(8, 5))
        sns.lineplot(x=res_df.index.tolist(), y=res_df['预测值'], linewidth=2, ax=ax)
        sns.scatterplot(x=res_df.index.tolist(), y=res_df['原始值'], s=60, color='r', marker='v', ax=ax)
        plt.fill_between(np.where(np.isnan(res_df["原始值"]))[0], y1=min(plt.yticks()[0]), y2=max(plt.yticks()[0]),
                         color='orange', alpha=0.2)
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        plt.show()
```

