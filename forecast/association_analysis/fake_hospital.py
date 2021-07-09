# -*- coding: utf-8 -*-

"""
pip install efficient-apriori
pip install dataclasses
"""

import pandas as pd
# import numpy as np
from efficient_apriori import apriori
from datetime import datetime

data = pd.read_csv(r"./跨院排除透析.csv")  # 读取数据
data.columns = ('visit_date', 'clientid')  # 修改列名
data_group = data['clientid'].apply(lambda x: str(x).split(','))  # 将每一个clientid以逗号分隔
'''将数据转化成Aprior算法所需的列表形式'''
lst = []
for j in data_group:
    lst.append(j)
print(len(lst), "rows")


def data_generator(filename):
    """
    Data generator, needs to return a generator to be called several times.
    Use this approach if data is too large to fit in memory. If not use a list.
    """
    def data_gen():
        with open(filename) as file:
            for line in file:
                yield tuple(k.strip() for k in line.split(','))

    return data_gen


def main():
    """计算频繁项集并保存结果"""
    itemsets, rules = apriori(lst, min_support=20 / len(lst), min_confidence=1)  # 求频繁项集及关联规则，可以修改最小支持度和置信度
    print(itemsets)
    print(rules)
    df = pd.DataFrame(columns=["groups", "times"])  # 创建一个空数据框用于后面保存频繁项集
    for k in itemsets.keys():  # m表示每个频繁项集的长度
        if k > 1:  # 只输出长度大于1的频繁项集
            for key, val in itemsets[k].items():  # 保存频繁项集
                df.loc[k - 1] = [",".join(key), val]
                # df = df.append({"groups": key, "times": val}, ignore_index=True)  # 逐行插入数据
    print(df)
    df.to_csv(r"./跨院假住院的频繁项集.csv", index=False)


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print(datetime.now() - start_time)
