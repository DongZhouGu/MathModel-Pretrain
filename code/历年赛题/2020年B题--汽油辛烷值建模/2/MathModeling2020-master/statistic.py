import numpy
import pandas

from data_processing import get_variable_range, process_original_data, fill_nan, check_range, check_exceptional_data

if __name__ == "__main__":
    range_data = get_variable_range("./data/354个操作变量信息.xlsx")
    path = "D:\\Downloads\\2020年中国研究生数学建模竞赛赛题\\2020年B题\\数模题\\附件三：285号和313号样本原始数据.xlsx"
    # excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, nrows=40)
    excel_data = pandas.read_excel(path, header=[1, 2], sheet_name=4, skiprows=range(3, 44), nrows=40)
    # print(excel_data.info())

    excel_data = excel_data.replace(0, numpy.nan, inplace=False)
    # print("缺失比例统计：")
    # excel_data.isna().sum().to_excel("./data/statistic/test1.xlsx")

    excel_data = excel_data.dropna(axis=1, how='any', thresh=40 * 0.7, inplace=False)
    excel_data = fill_nan(excel_data)
    excel_data = check_range(excel_data, range_data)
    excel_data = check_exceptional_data(excel_data)
    # print("异常比例统计：")
    # excel_data.isna().sum().to_excel("./data/statistic/exceptional_data.xlsx")

    excel_data = excel_data.dropna(axis=1, how='any', thresh=40 * 0.7, inplace=False)
    excel_data = fill_nan(excel_data)
    # print(excel_data.info())

    df_list = []
    for i in range(1, 326):
        df = pandas.read_excel("./data/result/" + str(i) + "_result.xlsx")
        df_list.append(df)
    pandas.concat(df_list).to_excel("./data/result/all_result.xlsx")
