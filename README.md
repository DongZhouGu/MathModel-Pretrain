# 华为杯研究生数学建模竞赛

![GitHub repo size](https://img.shields.io/github/repo-size/DongZhouGu/MathModel-Pretrain?style=for-the-badge)
![GitHub stars](https://img.shields.io/github/stars/DongZhouGu/MathModel-Pretrain?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/DongZhouGu/MathModel-Pretrain?style=for-the-badge)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/DongZhouGu/MathModel-Pretrain?style=for-the-badge)
![Bitbucket  issues](https://img.shields.io/github/issues-closed/DongZhouGu/MathModel-Pretrain?style=for-the-badge)

<p align="left">
<a href="https://github.com/DongZhouGu/MathModel-Pretrain" target="_blank">
	<img src="https://cdn.jsdelivr.net/gh/dongzhougu/imageuse1/img20211231164708.png#pic_center" width=""/>
</a>
</p>


## 简介

> :smiley: 此项目用于我们队伍数学建模比赛准备阶段，在此期间，我们整理了许多算法demo和资料。
>
> :clap: 很幸运获得了2021年D题（抗乳腺癌候选药物的优化建模）一等奖。
>
> :star: 现在将此项目开源，以帮助更多数模er，祝愿大家都能取得好成绩！如果觉得有用，请点个star 吧！感谢！！
>
> :triangular_flag_on_post: 同时，也欢迎大家PR，共同将这个项目壮大！
>

<details>
  <summary>文档目录</summary>
  <ol>
    <li><a href="#简介">简介</a></li>
    <li><a href="#-待办事项">待办事项</a></li>
    <li><a href="#-项目目录说明">项目目录说明</a></li>
    <li>
      <a href="#-算法索引">算法索引</a>
      <ul>
        <li><a href="#特征相关">特征相关</a></li>
        <li><a href="#时间序列">时间序列</a></li>
        <li><a href="#分类/预测">分类/预测</a></li>
        <li><a href="#优化问题">优化问题</a></li>
        <li><a href="#可视化">可视化</a></li>
      </ul>
    </li>
    <li><a href="#-如何参与开源项目">参与此项目</a></li>
    <li><a href="#-贡献者">贡献者</a></li>
    <li><a href="#-鸣谢">鸣谢</a></li>
  </ol>
</details>



## 🔨 待办事项
- [ ] 整理仓库
- [x] 上传2021年D题一等奖（数模之星）论文，论文最后有代码
- [x] 上传论文word模板




## 💻 项目目录说明

```
filetree 
├── README.md       你现在看到的内容
├── README_OLD.md   比赛准备阶段的README
└── 建模算法         各种机器学习的demo和代码
└── 论文模板         2021论文word模板
└── 实用工具         一些值得分享的工具或网站
└── 杂七杂八         经验、PPT资料等
└── code  
    └── 历年赛题         整理搜罗的历年赛题代码
    └── tmp		团队初期用于学习的代码
```



## 🚀 算法索引

[数学建模模型总结](https://github.com/DongZhouGu/MathModel-Pretrain/blob/master/%E5%BB%BA%E6%A8%A1%E7%AE%97%E6%B3%95/%E6%95%B0%E5%AD%A6%E5%BB%BA%E6%A8%A1%20%E5%9B%9B%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%80%BB%E7%BB%93%20-%20%E7%99%BE%E5%BA%A6%E6%96%87%E5%BA%93.pdf)

### 特征相关

- [组合赋权法](./建模算法/组合赋权法/组合赋权.md)—[code](./建模算法/组合赋权法/组合赋权.md)
- [因子分析法](./建模算法/降维/因子分析法.md)—[code](./建模算法/降维/factor_analyze.py)—[demo](./建模算法/降维/terrorism.ipynb)
- PCA—[code](./建模算法/降维/terrorism.ipynb)—[demo](./建模算法/组合赋权法/组合赋权.md)
- [聚类wkmeans](./建模算法/聚类/划分聚类/wkmeans.md)—[code](./建模算法/聚类/划分聚类/wkmeans.py)
- [灰色关联](./建模算法/灰色关联度分析/灰色关联度分析.md)—[code](./建模算法/灰色关联度分析/GRA.py)
- [逻辑回归关联](./建模算法/logistic_similar/logstic_similar.md)—[code](./建模算法/logistic_similar/logstic_similar.py)
- [缺失值](./建模算法/缺失值插补/插补.md)—[code](./建模算法/缺失值插补/imputer.py)—[demo](./建模算法/缺失值插补/imputer.ipynb)

### 时间序列

- [灰色预测](./建模算法/预测/灰色预测/灰色预测.md)—[code](./建模算法/预测/灰色预测/gm.py)—[demo](./建模算法/预测/灰色预测/灰色预测&线性回归&函数拟合.ipynb)
- 线性回归&函数拟合—[code](./建模算法/预测/灰色预测/灰色预测&线性回归&函数拟合.ipynb)
- [ARIMA](./建模算法/预测/整合移动平均自回归(ARIMA)/ARIMA.md)—[code](./建模算法/预测/整合移动平均自回归(ARIMA)/arima.py)—[demo](./建模算法/预测/整合移动平均自回归(ARIMA)/ARIMA.ipynb)
- Lstm—[code](./建模算法/lstm/lstm.py)—[demo](./建模算法/预测/kaggle预测题/Rossmann_Store_Sales.ipynb)

### 分类/预测

- [线性回归、多项式回归、神经网络回归](./建模算法/回归/线性回归/线性回归.md)—[code](./建模算法/回归/线性回归/pytorch_linear_distributed.py)—[demo](./建模算法/回归/线性回归/boston.ipynb)
- [神经网络集成学习](./建模算法/集成学习/集成学习.md)—[demo](./建模算法/集成学习/classify_regression_visual.ipynb)
- 机器学习pipline—[demo](./建模算法/pipline/mlFlow.ipynb)

### 优化问题

- [约束优化gurobi求解器](./建模算法/约束优化/gurobi求解器.md)—[demo](./建模算法/约束优化/gurobi.ipynb)
- 贪心—[demo](./建模算法/约束优化/贪心.ipynb)
- Dijkstra—[code](./建模算法/Dijkstra/dijkstra.py)—[demo](./建模算法/Dijkstra/question1.1.ipynb)

### 可视化

- [matplotlib](./code/tmp/matplotlib/README.md)—[demo](matplotlib-beginner.ipynb)

- [seaborn](./code/tmp/seaborn/README.md)—[demo](./code/tmp/seaborn/Searborn.ipynb)

- 可视化以恐怖袭击为例—[demo](./建模算法/可视化/可视化以恐怖袭击为例.ipynb)
- 可视化以covid-19为例—[demo](./code/tmp/案例/covid19/coronavirus-covid-19-visualization-prediction.ipynb)
- 自动EDA包：pandas-profiling、sweetviz、dataprep、lux、AutoViz 



## 📫 如何参与开源项目

贡献使开源社区成为一个学习、激励和创造的绝佳场所。

1. Fork这个项目
2. 创建您的单独分支  (`git checkout -b your_branch`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 上传到您的分支中 (`git push origin your_branch`)
5. 创建拉取请求，请参阅 如何[创建拉取请求](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request)。



## 🤝 贡献者

我们感谢以下对这个项目做出贡献的人：

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/yysfff">
        <img src="https://avatars.githubusercontent.com/u/56332682?v=4" width="100px;"/><br>
        <sub>
          <b>YYS</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/xiao-aBoy">
        <img src="https://avatars.githubusercontent.com/u/45626011?v=4" width="100px;"/><br>
        <sub>
          <b>LX</b>
        </sub>
      </a>
    </td>
  </tr>
</table>



## ☕  鸣谢

感谢以下项目，我们从中得到了很大的帮助：

- [scikit-learn 机器学习 常用算法及编程实战](https://github.com/DongZhouGu/scikit-learn-ml)
- [黄海广老师的机器学习项目](https://github.com/fengdu78/machine_learning_beginner)
- [最全的特征工程&特征选择demo](https://github.com/Yimeng-Zhang/feature-engineering-and-feature-selection)
- [智能启发算法scikit-opt库](https://github.com/guofei9987/scikit-opt)以及[作者博客](https://www.guofei.site/)
- [sklearn 中文文档](https://www.scikitlearn.com.cn/)
- [AI learning](https://github.com/apachecn/AiLearning)
- [数据竞赛Top解决方案开源整理](https://github.com/Smilexuhc/Data-Competition-TopSolution)

