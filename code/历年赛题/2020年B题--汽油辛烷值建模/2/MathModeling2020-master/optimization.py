import copy
import json
from pprint import pprint

import joblib
import pandas
import numpy as np

from data_processing import get_variable_range


def variable_list_sort(var_list, range_data, mapping_data):
    model = joblib.load("./regression_model.pkl")
    dict = {}
    for i in range(len(var_list)):
        # 非操作变量
        if 1 <= var_list[i] <= 14:
            delta = 0
        else:
            delta = range_data[mapping_data[var_list[i]][1]]["delta"]
        dict[var_list[i]] = model.coef_[0][i] * delta
    tmp = sorted(dict.items(), key=lambda kv: (kv[1], kv[0]))
    return [tuple_item[0] for tuple_item in tmp]


def greedy(sample_dict: dict, ron: float, row_index: int):
    # 记录变化过程
    df = pandas.DataFrame()
    # 只记录变量的初始值和最终值，忽略过程量
    df_result = pandas.DataFrame()

    # 描述变化量
    delta_sum_dict = {}
    # 优化流程（dict 列表）
    tmp_dict = copy.deepcopy(sample_dict)
    tmp_dict["ron损失"] = ron
    tmp_dict["硫"] = check_sulfur(sample_dict, row_index, "3.2")
    df = df.append(tmp_dict, ignore_index=True)
    # 优化前属性
    tmp_dict = copy.deepcopy(sample_dict)
    tmp_dict["ron损失"] = ron
    tmp_dict["样本编号"] = row_index + 1
    tmp_dict["降幅"] = str(0)
    df_result = df_result.append(tmp_dict, ignore_index=True)

    initial_ron = ron
    range_data = get_variable_range("./data/354个操作变量信息.xlsx")
    mapping_data = pandas.read_excel("./data/title.xlsx")
    variable_list = []
    for key, val in sample_dict.items():
        variable_list.append(key)
        delta_sum_dict[key] = 0
    variable_list = variable_list_sort(variable_list, range_data, mapping_data)
    variable_len = len(variable_list)

    count = 0
    random = 0
    while True:
        # 随机挑选一个变量进行调整
        # random = np.random.randint(low=0, high=variable_len)
        variable_pick_no = variable_list[random]
        # index 1-14 为非操作变量，无法调整
        if 1 <= variable_pick_no <= 14:
            random += 1
            continue
        variable_pick = mapping_data[variable_pick_no][1]
        # print("pick_index: %d, pick item: %s" % (variable_pick_no, variable_pick))

        # 获取该变量的相关信息
        low = range_data[variable_pick]["lower_bound"]
        high = range_data[variable_pick]["upper_bound"]
        delta = range_data[variable_pick]["delta"]
        cur_val = sample_dict[variable_pick_no]

        # 获取变化后的数据
        # new_sample = sample_dict
        sample_dict[variable_pick_no] += delta
        new_ron = get_new_ron(sample_dict, row_index)

        # print(low <= sample_dict[variable_pick_no] <= high)
        # print(check_sulfur(sample_dict, row_index, "5"))
        # print(new_ron <= ron)
        # print()

        # 变更完满足如下条件：
        # 1. 变量仍在范围内
        # 2. 硫含量小于 5
        # 3. ron 损失有降低
        if low <= sample_dict[variable_pick_no] <= high and check_sulfur(sample_dict, row_index,
                                                                         "5") is True and new_ron <= ron:
            ron = new_ron
            # sample_dict = sample_dict
            delta_sum_dict[variable_pick_no] += delta
            tmp_dict = copy.deepcopy(sample_dict)
            tmp_dict["ron损失"] = ron
            tmp_dict["硫"] = check_sulfur(sample_dict, row_index, "3.2")
            df = df.append(tmp_dict, ignore_index=True)
        else:
            count += 1
            random += 1
            sample_dict[variable_pick_no] -= delta

        if count > 10 or random >= len(variable_list):
            break

    # with open('./result.txt', 'a') as f:
    #     f.write("row_index: %d, original: %.5f, new: %.5f, percent: %.2f%%" % (row_index, initial_ron, ron, (initial_ron - ron)/initial_ron*100))
    #     f.write("\n")
    #     f.write(json.dumps(delta_sum_dict))
    #     f.write("\n\n")

    # 优化后属性
    tmp_dict = copy.deepcopy(sample_dict)
    tmp_dict["ron损失"] = ron
    tmp_dict["样本编号"] = row_index + 1
    tmp_dict["降幅"] = str((initial_ron - ron) / initial_ron * 100) + "%"
    df_result = df_result.append(tmp_dict, ignore_index=True)

    df.to_excel("./data/result/" + str(row_index + 1) + "_process.xlsx", index=False, header=True)
    df_result.to_excel("./data/result/" + str(row_index + 1) + "_result.xlsx", index=False, header=True)

    print("row_index: %d, original: %.5f, new: %.5f, percent: %.2f%%" % (
        row_index, initial_ron, ron, (initial_ron - ron) / initial_ron * 100))
    print(delta_sum_dict)
    print()


# 判断硫含量
def check_sulfur(sample_dict: dict, index: int, type: str):
    df_x = pandas.read_excel("./data/x2.xlsx")
    for key, val in sample_dict.items():
        if key in df_x.columns:
            df_x.loc[index, key] = val

    model = None
    if type == "5":
        model = joblib.load("./classification_model_s_5.pkl")
    elif type == "3.2":
        # model = joblib.load("./classification_model_s_3.2.pkl")
        return get_rest_s_dict(df_x.to_dict('records')[index])

    if model.predict(df_x[index:index + 1])[0] == 1.0:
        return True
    else:
        return False


# 判断 ron 损失
def get_new_ron(sample_dict: dict, index: int):
    model = joblib.load("./regression_model.pkl")
    df_x = pandas.read_excel("./data/x1.xlsx")

    for key, val in sample_dict.items():
        df_x.loc[index, key] = val

    return model.predict(df_x[index:index + 1])[0][0]


def get_rest_s(x):
    b0 = -17.1393441637
    b = [-0.0575303098, 0.0043942943, -0.0000278449, -0.0328484906, 0.0176854694, -0.0022627343, 0.0239644130,
         -0.0035410290, -0.2210221642, 0.0052330259, 0.0079804443, 0.0035524819, 0.0114921149, 1.1437657356,
         -0.0331726202, 0.0148461304, 0.0110701114, 0.0144670189, -0.0099540070, 0.0330228034, 0.0005910554]

    y = b0
    for i in range(len(b)):
        y += b[i] * x[i]
    return y


def get_rest_s_dict(x_dict: dict):
    index = [30, 40, 46, 62, 82, 83, 90, 153, 159, 183, 185, 222, 251, 252, 265, 271, 282, 294, 325, 4, 6]
    x = []
    for e in index:
        x.append(x_dict[e])
    return get_rest_s(x)


if __name__ == "__main__":
    df_x = pandas.read_excel("./data/x1.xlsx")
    df_y = pandas.read_excel("./data/y1.xlsx")

    # 将 df 转为字典数组，列为 key
    record_list = df_x.to_dict('records')

    for i in range(325):
        # 此处使用计算值作为基准，而不是 excel 文件里的值
        base = get_new_ron(record_list[i], i)
        greedy(record_list[i], base, i)

    # greedy(record_list[132], get_new_ron(record_list[132], 132), 132)
