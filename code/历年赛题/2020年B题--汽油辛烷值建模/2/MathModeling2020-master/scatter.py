import pandas
import numpy as np
import matplotlib.pyplot as plt


def get_relation(path: str, index: int):
    excel_data = pandas.read_excel(path, header=0)
    col_content = excel_data["x"]
    result = excel_data["y"]

    plt.scatter(col_content, result)
    plt.show()


if __name__ == '__main__':
    get_relation("./325个样本数据.xlsx", 162)
