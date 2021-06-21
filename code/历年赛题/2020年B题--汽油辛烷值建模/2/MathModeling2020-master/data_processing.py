from pprint import pprint

import numpy
from pandas import DataFrame
import pandas


# 读附件四，获取变量的范围
def get_variable_range(path: str) -> dict:
    data = pandas.read_excel(path)
    range_dict = {}
    for index, row in data.iterrows():
        var_id, var_name, var_range, delta = row[1], row[2], row[3], row[5]
        # 范围去括号
        var_range = var_range.replace('(', '')
        var_range = var_range.replace(')', '')
        var_range = var_range.replace('（', '')
        var_range = var_range.replace('）', '')
        # 存在负数情况
        split_list = var_range.split("-")
        range_dict[var_id] = {}
        range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = 0, 0
        range_dict[var_id]["delta"] = float(delta)
        # left + right +
        if len(split_list) == 2:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = float(split_list[0]), float(
                split_list[1])
        # left - right +
        elif len(split_list) == 3:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = -float(split_list[1]), float(
                split_list[2])
        # left - right -
        elif len(split_list) == 4:
            range_dict[var_id]["lower_bound"], range_dict[var_id]["upper_bound"] = -float(split_list[1]), -float(
                split_list[3])
        else:
            print("can't handle, range is %s" % var_range)
    # pprint(range_dict)
    return range_dict


# (4) 数据范围筛选
def check_range(excel_data: DataFrame, range_dict: dict):
    # 遍历列，col_name 为列名，col_content 为该列的具体内容
    for col_name, col_content in excel_data.iteritems():
        # print("col_name: " + col_name[0])
        if col_name[0] in ['时间', '样本编号']:
            continue

        range_lower, range_upper = range_dict[col_name[0]]['lower_bound'], range_dict[col_name[0]]['upper_bound']
        for row_index in range(len(col_content)):
            val = float(col_content[row_index])
            if val < range_lower or val > range_upper:
                print("[invalid row] var_id: %s, lower: %f, upper: %f, index: %d, val: %f" %
                      (col_name[0], range_lower, range_upper, row_index, val))
                excel_data.loc[row_index, col_name] = numpy.nan
                # excel_data = excel_data.drop([row_index], axis=0)

    return excel_data


# (5) 拉伊达准则去除异常值
def check_exceptional_data(excel_data: DataFrame):
    for col_name, col_content in excel_data.iteritems():
        if col_name[0] in ['时间', '样本编号']:
            continue

        # 求平均值
        mean_value = col_content.mean()
        # 求标准差
        std_value = col_content.std()
        # 位于(μ-3σ,μ+3σ)区间内的数据是正常的，不在该区间的数据是异常的
        # ser1中的数值小于μ-3σ或大于μ+3σ均为异常值
        # 一旦发现异常值就标注为True，否则标注为False
        rule = (mean_value - 3 * std_value > col_content) | (mean_value + 3 * std_value < col_content)
        # 返回异常值的位置索引
        row_index_list = numpy.arange(col_content.shape[0])[rule]
        for index in row_index_list:
            print("[exceptional data] var_id: %s, index: %d, val: %f" %
                  (col_name[0], index, col_content[index]))
            # todo: 异常值视为缺失值，利用缺失值的处理方法修正异常值
            excel_data.loc[index, col_name] = numpy.nan

    return excel_data


# (1) 删掉全为 0 或多数为 0 的列
@DeprecationWarning
def remove_zero(excel_data: DataFrame):
    # 遍历列，col_name 为列名，col_content 为该列的具体内容
    for col_name, col_content in excel_data.iteritems():
        # print("col_name: " + col_name[0])
        if col_name[0] == '时间':
            continue

        count = 0
        # 遍历该列的每一行
        for row_index in range(len(col_content)):
            # print("row_index: " + str(row_index))
            # print("row_content: " + str(col_content[row_index]))
            if float(col_content[row_index]) == 0:
                count += 1
        # 超过一半为 0，即删掉该点位
        if count > 40 * 0.5:
            print("delete column " + col_name[0])
            excel_data = excel_data.drop(col_name[0], axis=1)
            continue

    return excel_data


# (3) 使用列平均值填充缺失值
def fill_nan(excel_data: DataFrame):
    for col_name, col_content in excel_data.iteritems():
        # print("col_name: " + col_name[0])
        if col_name[0] in ['时间', '样本编号']:
            continue

        mean_val = col_content.mean()
        col_content.fillna(mean_val, inplace=True)

    return excel_data


# 读附件三，对 285 和 313 数据进行处理
def process_original_data(path: str, sample_no: str, range_dict: dict):
    excel_data = None
    data_count = 40
    if sample_no == '285':
        # nrows 不包含表头
        excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, nrows=40)
    elif sample_no == '313':
        excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, skiprows=range(3, 44), nrows=40)
    elif sample_no == 'all':
        excel_data = pandas.read_excel(path, header=[1, 2])
        data_count = 325

    # 使用 nan 替换 0，方便 pandas 处理
    excel_data = excel_data.replace(0, numpy.nan, inplace=False)

    # (1,2) 去除缺失值(nan)数据较多的位点
    excel_data = excel_data.dropna(axis=1, how='any', thresh=data_count * 0.7, inplace=False)
    # (3) todo: 使用（这一列的）平均值填充缺失值
    excel_data = fill_nan(excel_data)
    # (4) 数据范围筛选，异常值会设置为缺失值(nan)
    excel_data = check_range(excel_data, range_dict)
    # (5) 拉伊达准则筛选，异常值会设置为缺失值(nan)
    excel_data = check_exceptional_data(excel_data)
    # (1,2) 去除缺失值(nan)数据较多的位点
    excel_data = excel_data.dropna(axis=1, how='any', thresh=data_count * 0.7, inplace=False)
    # (3) todo: 使用（这一列的）平均值填充缺失值
    excel_data = fill_nan(excel_data)

    if sample_no != 'all':
        # 添加平均值到文件末尾
        avg_data = []
        for col_name, col_content in excel_data.iteritems():
            if col_name[0] == '时间':
                avg_data.append("")
            else:
                avg_data.append(format(col_content.mean(), '.4f'))
        excel_data.loc[40] = avg_data

    return excel_data


if __name__ == '__main__':
    range_data = get_variable_range("/Users/faye/Downloads/数模题/附件四：354个操作变量信息.xlsx")
    # process_original_data("D:\\Downloads\\2020年中国研究生数学建模竞赛赛题\\2020年B题\\数模题\\附件三：285号和313号样本原始数据.xlsx")
    # processed_data = process_original_data("/Users/faye/Downloads/数模题/test.xlsx", "313", range_data)
    processed_data = process_original_data("/Users/faye/Downloads/数模题/test_325个样本数据的副本.xlsx", "all", range_data)

    # print(excel_data)
    pandas.DataFrame(processed_data).to_excel("/Users/faye/Desktop/output.xlsx", sheet_name='Sheet1', header=True)
