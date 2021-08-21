## 介绍

Gurobi 全球用户超过2600家，广泛应用在金融、物流、制造、航空、石油石化、商业服务等多个领域，为智能化决策提供了坚实的基础，成为上千个成熟应用系统的核心优化引擎。

**Gurobi 是全局优化器，支持的模型类型包括：**

（1）连续和混合整数线性问题

（2）凸目标或约束连续和混合整数二次问题

（3）非凸目标或约束连续和混合整数二次问题

（4）含有对数、指数、三角函数、高阶多项式目标或约束，以及任何形式的分段约束的非线性问题

（5）含有绝对值、最大值、最小值、逻辑与或非目标或约束的非线性问题

　

**Gurobi 技术优势：**

（1）可以求解大规模线性问题，二次型问题和混合整数线性和二次型问题

（2）支持非凸目标和非凸约束的二次优化

（3）支持多目标优化

（4）支持包括SUM, MAX, MIN, AND, OR等广义约束和逻辑约束

（5）支持包括高阶多项式、指数、三角函数等的广义函数约束

（6）问题尺度只受限制于计算机内存容量，不对变量数量和约束数量有限制

（7）采用最新优化技术，充分利用多核处理器优势。支持并行计算

（8）提供了方便轻巧的接口，支持 C++, Java, Python, .Net, Matlab 和R，内存消耗少

（9）支持多种平台，包括 Windows, Linux, Mac OS X



## 激活安装

### 激活

使用教育邮箱申请https://zhuanlan.zhihu.com/p/212191049免费使用

中文官网：http://www.gurobi.cn/

### 安装

打开Anaconda Prompt，并输入以下两条指令：

```powershell
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

**即可在anaconda中完成Gurobi包的下载安装。**



## 使用demo

官方文档：https://www.gurobi.com/documentation/9.1/quickstart_windows/cs_simple_python_example.html

- Model(name=“”)
  name是模型的名称, 返回值是一个model对象，初始情况下没有变量和约束条件
  调用方式：model1 = Model()
- Model.addVar(lb, ub, obj, vtype, name,column)
  lb是下限，ub是上限，obj是目标的优化系数，vtype是变量类型，name是变量的名称，column是变量参与的约束以及优化系数。
  所有的参数都是可选的，不指定就是默认值
  变量的类型有GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER, GRB.SEMICONT, or GRB.SEMIINT
- Model.addConstr( lhs, sense=None, rhs=None, name="" )
  lhs是约束的右侧，rhs是约束的左侧，sense是约束的类型，有(GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL
  name是约束的名称
- Model.addConstrs( generator, name="" )
  generator是Python表达式。这种表达方式比 2 直观一点，返回值是tupledict类型的
  例：model.addConstrs(x[i] + x[j] <= 1 for i in range(5) for j in range(5))
  注意：generator表达式只能由一个比较关系。
- Model.update()
  对修改了的模型进行更新
  Model.setObjective(expr, sense)
  设置模型的优化函数
  expr是优化目标表达式，sense是优化类型，优化类型有GRB.MINIMIZE和GRB.MAXIMIZE，如果省略sense可以使用ModelSense函数来指定优化的类型。
- Model.write(filename)
  将优化模型，解向量，基向量，起始向量或参数设置写入文件，文件的类型有.mps, .rew, .lp .rlp，也可以只保存的模型的一部分，详见refman
- Model.getVars()
  返回模型中的所有变量

**总结**

```python
import gurobipy

# 创建模型
MODEL = gurobipy.Model()

# 创建变量
X = MODEL.addVar(vtype=gurobipy.GRB.INTEGER,name="X")

# 更新变量环境
MODEL.update()

# 创建目标函数
MODEL.setObjective('目标函数表达式', gurobipy.GRB.MINIMIZE)

# 创建约束条件
MODEL.addConstr('约束表达式，逻辑运算')

# 执行线性规划模型
MODEL.Params.LogToConsole=True # 显示求解过程
MODEL.Params.TimeLimit=100 # 限制求解时间为 100s
MODEL.optimize()

# 输出模型结果
print("Obj:", MODEL.objVal)
for x in X:
    print(f"{x.varName}：{round(x.X,3)}")

```

### 数据结构

Gurobi封装了更高级的Python数据结构，即Multidict、Tuplelist、Tupledict。在对复杂或大规模问题建模时，可以大大提高模型求解效率。



#### multdict

multdict函数允许在一个语句中初始化一个或多个字典，举例如下：

```python3
import gurobipy as grb

# mutidict 用法
student, chinese, math, english = grb.multidict({
    'student1': [10, 2, 300],
    'student2': [20, 3, 400],
    'student3': [30, 4, 500],
    'student4': [40, 5, 600]
})

print(student)  # 字典的键
# 输出
# ['student1', 'student2', 'student3', 'student4']

print(chinese)  # 语文成绩的字典
# 输出
# {'student1': 10, 'student2': 20, 'student3': 30, 'student4': 40}

print(math)  # 数学成绩的字典
# 输出
# {'student1': 2, 'student2': 3, 'student3': 4, 'student4': 5}

print(english)  # 英语成绩的字典
# 输出
# {'student1': 300, 'student2': 400, 'student3': 500, 'student4': 600}
```

![img](https://gitee.com/ma_tung_zhou/imageuse1/raw/master/imgg/20210814140744.jpg)

#### tuplelist

tuplelist：元组列表，list元素的tuple类型，可以高效地在元组列表中构建子列表。

可以使用tuplelist对象的select方法进行元组检索，很想SQL语句中的select-where操作。

tulpelist继承自list，所以向tuplelist中添加新元素和list的用法一样，有append、pop等方法

```python3
import gurobipy as grb
# 创建tuplelist对象
tl = grb.tuplelist([(1, 2), (1, 3), (2, 3), (2, 5)])

# 取出第一个值是1的元素
print(tl.select(1, '*'))
# 输出
# <gurobi.tuplelist (2 tuples, 2 values each):
#  ( 1 , 2 )
#  ( 1 , 3 )

# 取出第二个值是3的元素
print(tl.select('*', 3))
# 输出
# <gurobi.tuplelist (2 tuples, 2 values each):
#  ( 1 , 3 )
#  ( 2 , 3 )
# -----------------------------------------------------------------------
# 添加一个元素
tl.append((3, 5))
print(tl.select(3, '*'))
# 输出 <gurobi.tuplelist (1 tuples, 2 values each):
#  ( 3 , 5 )

# 使用迭代的方式实现select功能
print(tl.select(1, '*'))
# 输出 <gurobi.tuplelist (2 tuples, 2 values each):
#  ( 1 , 2 )
#  ( 1 , 3 )

# 对应的迭代语法是这样的
print([(x, y) for x, y in tl if x == 1])
```



![img](https://gitee.com/ma_tung_zhou/imageuse1/raw/master/imgg/20210814140726.jpg)

从上面的代码可以看出，tuplelist在内部存储上和list是一样的，指数Gurobi在继承list类的基础上添加了select方法，因此可以把tuplelist看作list对象，可以使用迭代、添加或删除元素等方法。

#### tupledict

tupledict是Python的dict的子类。

Gurobi变量一般都是tupledict类型。

tupledict的key（键）在内部的存储格式是tuplelist，因此可以使用tuplelist的select方法筛选。在实际应用中，通过将元组与每个Gurobi变量关联起来，可以有效地创建包含匹配变量子集的表达式。

如创建一个3×3的矩阵，里面每个元素表示线性表达式的变量，取其中一部分变量的操作就显得很方便了，

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bbmatrix%7Dx_%7B11%7D%5C+x_%7B12%7D+%5C++x_%7B13%7D%5C%5Cx_%7B21%7D%5C+x_%7B22%7D+%5C++x_%7B23%7D%5C%5Cx_%7B31%7D%5C+x_%7B32%7D+%5C++x_%7B33%7D%5Cend%7Bbmatrix%7D) 

对第一行求和的代码如下：

```python3
import gurobipy as grb

model = grb.Model()
# 定义变量的下标
tl = [(1, 1), (1, 2), (1, 3),
      (2, 1), (2, 2), (2, 3),
      (3, 1), (3, 2), (3, 3)]
vars = model.addVars(tl, name = "x")
model.update()
print(vars) #vars是tupledict类型的数据
# 输出
# {(1, 1): <gurobi.Var x[1,1]>,
# (1, 2): <gurobi.Var x[1,2]>,
# (1, 3): <gurobi.Var x[1,3]>,
# (2, 1): <gurobi.Var x[2,1]>,
# (2, 2): <gurobi.Var x[2,2]>,
# (2, 3): <gurobi.Var x[2,3]>,
# (3, 1): <gurobi.Var x[3,1]>,
# (3, 2): <gurobi.Var x[3,2]>,
# (3, 3): <gurobi.Var x[3,3]>}
# # 基于元素下标的操作
print(grb.quicksum(vars.select(1, '*')))
print(sum(vars.select(1, '*')))
print(vars.sum(1, '*'))
# 输出
# <gurobi.LinExpr: x[1,1] + x[1,2] + x[1,3]>
```

![img](https://gitee.com/ma_tung_zhou/imageuse1/raw/master/imgg/20210814140732.jpg)

























