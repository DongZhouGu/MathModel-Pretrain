# 灰色预测

## 1️⃣灰色预测概述

灰色预测是用灰色模型GM(1,1)来进行定量分析的，通常分为以下几类：
**(1) 灰色时间序列预测**。用等时距观测到的反映预测对象特征的一系列数量（如产量、销量、人口数量、存款数量、利率等）构造灰色预测模型，预测**未来某一时刻的特征量**，或者**达到某特征量的时间**。
**(2) 畸变预测（灾变预测**）。通过模型预测异常值出现的时刻，预测异常值什么时候出现在特定时区内。
 (3) 波形预测，或称为拓扑预测，它是通过灰色模型预测**事物未来变动的轨迹**。
(4) 系统预测，对系统行为特征指标建立一族相互关联的灰色预测理论模型，在预测系统整体变化的同时，预测系统各个环节的变化。
上述灰色预测方法的共同特点是：
1）允许**少数据预测**；
2）允许对**灰因果律事件进行预测**，例如：

- 灰因白果律事件：在粮食生产预测中，影响粮食生产的因子很多，多到无法枚举，故为灰因，然而粮食产量却是具体的，故为白果。粮食预测即为灰因白果律事件预测。

- 白因灰果律事件：在开发项目前-景预测时，开发项目的投入是具体的，为白因，而项目的效益暂时不很清楚，为灰果。项目前景预测即为灰因白果律事件预测。

 3）**具有可检验**性，包括：建模可行性的级比检验（事前检验），建模精度检验（模型检验），预测的滚动检验（预测检验）。

## 2️⃣GM(1,1)模型理论

`G：Grey（灰色）；M：模型；(1,1)：只含有一个变量的一阶微分方程模型`

详见https://www.jianshu.com/p/a35ba96d852b

https://blog.csdn.net/qq_41196612/article/details/105637329

![img](https://img-blog.csdnimg.cn/20200420160606521.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMTk2NjEy,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20200420160726361.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMTk2NjEy,size_16,color_FFFFFF,t_70)

![img](https://img-blog.csdnimg.cn/20200420160823554.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxMTk2NjEy,size_16,color_FFFFFF,t_70)

## 3️⃣算法步骤

### (1) 数据的级比检验

 为了保证灰色预测的可行性，需要对原始序列数据进行级比检验。
 对原始数据列$X^{(0)}=(x^{(0)}(1),x^{(0)}(2),...,x^{(0)}(3))$，计算序列的级比：
$$
\lambda(k)=\frac{x^{0}(k-1)}{x^{(0)}(k)}, \quad k=2,...,n
$$
若所有的级比$\lambda(k)$都落在可容覆盖$\Theta=(e^{-2/(n+1)},e^{2/(n+2)})$内，则可进行灰色预测；否则需要对$X^{(0)}$做平移变换，$Y^{(0)}=X^{(0)}+c$，使得满足级比$Y^{(0)}$要求。并不是说有的数据没落入区间之内就不能建模，只是落在区间之内建模效果比较好

### (2) 建立GM(1,1)模型，计算出预测值列。

### (3) 检验预测值：



![img](https://img-blog.csdn.net/20180826140757559?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NjY2NzU2/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![image-20210708135733502](F:\notebook\tpora img\image-20210708135733502.png)

![image-20210708135752264](F:\notebook\tpora img\image-20210708135752264.png)

# 4️⃣程序实现

```python

class GrayForecast():
    # 初始化
    def __init__(self, data, n):
        """
        :param data: Series/np/list
        :param n: 预测数量
        """
        if isinstance(data, pd.Series):
            self.data = data.values
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, list):
            self.data = np.array(data)
        self.level_check()
        self.GM_11_build_model(n)
        print("返回值为dataframe\n", self.res_df)

    def level_check(self):
        # 数据级比校验
        b = self.data[0]
        n = len(self.data)
        lambda_k = np.zeros(n - 1)
        while (True):
            for i in range(n - 1):
                lambda_k[i] = self.data[i] / self.data[i + 1]
            if max(lambda_k) < np.exp(2 / (n + 2)) and min(lambda_k) > np.exp(-2 / (n + 1)):
                self.c = self.data[0] - b
                print(f"完成数据 级比校验, 平移变换c={self.c}")
                break
            else:
                self.data = self.data + 0.1

    # GM(1,1)建模
    def GM_11_build_model(self, n):
        '''
            灰色预测
            x：序列，numpy对象
            n:需要往后预测的个数
        '''
        x = self.data
        # 累加生成（1-AGO）序列
        x1 = x.cumsum()
        # 紧邻均值生成序列
        z1 = z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0
        z1 = z1.reshape((len(z1), 1))
        B = np.append(-z1, np.ones_like(z1), axis=1)
        Y = x[1:].reshape((len(x) - 1, 1))
        # a为发展系数 b为灰色作用量
        [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)  # 计算参数
        # 预测数据
        fit_res = [x[0]]
        for index in range(1, len(x) + n):
            fit_res.append((x[0] - b / a) * (1 - np.exp(a)) * np.exp(-a * (index)))
        # 数据还原
        self.data -= self.c
        fit_res -= self.c
        self.res_df = pd.concat([pd.DataFrame({'原始值': self.data}), pd.DataFrame({'预测值': fit_res})], axis=1)
        print(f"发展系数a={a}, 灰色作用量b={b}\n")
        self.verfify(self.data, fit_res, a)
        return self.res_df

    def verfify(self, x, predict, a):
        S1_2 = x.var()  # 原序列方差
        e = list()  # 残差序列
        for index in range(x.shape[0]):
            e.append(x[index] - predict[index])
        S2_2 = np.array(e).var()  # 残差方差
        C = S2_2 / S1_2  # 后验差比
        if C <= 0.35:
            assess = '后验差比<=0.35，模型精度等级为好'
        elif C <= 0.5:
            assess = '后验差比<=0.5，模型精度等级为合格'
        elif C <= 0.65:
            assess = '后验差比<=0.65，模型精度等级为勉强'
        else:
            assess = '后验差比>0.65，模型精度等级为不合格'
        print(f"后验差比={C}, {assess} \n")

        # 级比偏差
        a_ = (1 - 0.5 * a) / (1 + 0.5 * a)
        delta = [np.nan]
        for i in range(x.shape[0] - 1):
            delta.append(1 - a_ * (x[i] / x[i + 1]))

        self.res_df = pd.concat([self.res_df, pd.DataFrame({'残差': e}),
                                 pd.DataFrame({'相对误差': list(map(lambda x: '{:.2%}'.format(x), np.abs(e / x)))}),
                                 pd.DataFrame({'级比偏差': delta})
                                 ],
                                axis=1)
```

