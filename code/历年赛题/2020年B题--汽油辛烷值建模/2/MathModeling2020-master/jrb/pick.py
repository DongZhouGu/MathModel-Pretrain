import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

df = pd.read_excel("data.xlsx")
df_x = df.copy()
col_y = df_x[9]
# col_y2 = df_x[8]
df_x = df_x.drop([8, 9, 10], axis=1)

multi_valve = 0.2
low_var_valve = 0.01
corr_valve = 0.4
high_corr_valve = 0.9

# 多重复值过滤
valve = len(df_x) * multi_valve
for col in df_x.columns:
    duplicate_v = df_x[col][0]
    duplicate_num = 1
    for e in df_x[col]:
        if e == duplicate_v:
            duplicate_num += 1
            if duplicate_num > valve:
                del df_x[col]
                break
        else:
            duplicate_v = e
            duplicate_num = 1

# 低方差过滤
for col in df_x.columns:
    if df_x[col].var() < low_var_valve:
        del df_x[col]

# 与结果的相关性
df_all = df_x.copy()
df_all[0] = col_y
corr_y = df_all.corr()[0]
corr_y = corr_y.drop(0)
corr_y = corr_y.sort_values(ascending=False)
pick_col = []
negative_flag = True
for e in corr_y.items():
    if abs(e[1]) > corr_valve:
        pick_col.append(e[0])
        if negative_flag and e[1] < 0:
            print(e[0])
            negative_flag = False
df_new_x = DataFrame()
for e in pick_col:
    df_new_x[e] = df_x[e]
df_x = df_new_x

# 高相关变量
corr = df_x.corr().values
high_corr_array = []
high_corr_set = set()
for i in range(len(corr)):
    high_corr = [df_x.columns[i]]
    for j in range(i + 1, len(corr[i])):
        if corr[i][j] > high_corr_valve:
            if df_x.columns[j] in high_corr_set:
                break
            high_corr.append(df_x.columns[j])
            high_corr_set.add(df_x.columns[j])
    if len(high_corr) > 1:
        high_corr_array.append(high_corr)
print(high_corr_array)

print(df_x)
# df_x.to_csv('data_new.csv', index=False, header=True, sep='\t')
df_x.to_excel('x.xlsx', index=False, header=True)

# y
df_y = DataFrame()
df_y[1] = col_y
# df_y[2] = col_y2
df_y.to_excel('y.xlsx', index=False, header=True)
