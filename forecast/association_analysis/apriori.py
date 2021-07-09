# coding=utf-8
# python3

"""
Ck = [['a','b'], ['a','c'], ['b','c']]
Lk = pd.DataFrame(data, columns=["combination", "num", "row_num", "support", "k"])

1、基于最底层的所有item生成2项集、2项集生成3项集等
2、优点：
3、缺点：
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from itertools import combinations


# from sqlalchemy import create_engine, pool

# from pandarallel import pandarallel
# pandarallel.initialize()
# pandarallel.initialize(nb_workers=8)

def bin_search(data_list, val):
    """二分查找，data_list必须排序，简直不要太快"""
    low = 0  # 最小数下标
    high = len(data_list) - 1  # 最大数下标
    while low <= high:
        mid = (low + high) // 2  # 中间数下标
        if data_list[mid] == val:  # 如果中间数下标等于val, 返回
            return mid
        elif data_list[mid] > val:  # 如果val在中间数左边, 移动high下标
            high = mid - 1
        else:  # 如果val在中间数右边, 移动low下标
            low = mid + 1
    return  # val不存在, 返回None


def calc_count(x: list, combinations: list) -> list:
    # method1
    # res = []
    # for i, combination in enumerate(combinations):
    #     # print(len(combinations), i, combination, x)
    #     if combination.issubset(x):
    #         res.append(combination)
    # return res

    # method2，和method1类似，只是写法精简
    # return [combination for combination in combinations if combination.issubset(x)]

    # method3
    # ck_unique = get_ck_unique(combinations)
    # x_filter = list(set(x) & ck_unique)
    # return [combination for combination in combinations if combination.issubset(x_filter)]

    # method4，每次遍历缩小dataset的结果集来提高issubset的效率。
    # df["items"].apply(lambda x: list(set(x) & ck_unique)
    # logger.info("combinations {}, x {}".format(len(combinations), len(x)))
    return [combination for combination in combinations if combination.issubset(x)]

    # method5，使用numpy.array数组，np.intersect1d(a,b)
    # return [combination for combination in combinations if len(np.intersect1d(list(combination), x)) == len(combination)]

    # method6，二分查找
    # result = []
    # for combination in combinations:
    #     flag = 1
    #     for val in combination:
    #         if bin_search(x, val) is None:
    #             flag = 0
    #             break
    #     if flag == 1:
    #         result.append(combination)
    # return result


def get_ck_unique(ck: list) -> set:
    """
    :param ck: [[a,b], [a,c], [b,c]]
    :return: (a, b, c)
    辅助函数，缩小每行的元素数，提高issubset性能

    arr = np.array([[1, 2, 1], [2, 3, 4]])
    print(arr)
    print(np.unique(arr))

    减枝步
    """
    ck_unique = set()
    for x in ck:
        for y in x:
            ck_unique.add(y)
    return ck_unique


def generate_ck(ck_unique_list: list, k: int) -> list:
    # 返回list，不如返回迭代器
    sorted_combinations = list(combinations(sorted(ck_unique_list), k))
    return sorted_combinations


def generate_ck_with_depends(ck_last: list, k: int) -> list:
    """
    :param ck_last:  [['a'],['b'],['c'],['d']]
    :param k: k>=2
    :return: [['a','b'],['a','c'],['a','d'],['b','c'],['b','d'],['c','d']]

    :param ck_last:  [frozenset({'a'}), frozenset({'b'}), frozenset({'c'}), frozenset({'d'})]
    :param k: k>=2
    :return: [frozenset({'a','b'}), frozenset({'a','c'}), frozenset({'a','d'}), frozenset({'b','c'}), frozenset({'b','d'}), frozenset({'c','d'})]

    连接步
    """
    ck_data = []
    ck_last_len = len(ck_last)
    for i in range(ck_last_len):
        for j in range(i + 1, ck_last_len):
            l1 = list(ck_last[i])[:k - 2]
            l2 = list(ck_last[j])[:k - 2]
            l1.sort()
            l2.sort()
            if l1 == l2:
                ck_data.append(ck_last[i] | ck_last[j])
                # t = copy.deepcopy(ck_last[i])
                # t.append(ck_last[j][-1])
                # ck_data.append(t)

    # # method1
    # else:
    #     df1 = self.dataset.explode(self.column)
    #     df1.drop_duplicates(subset=[self.column], keep="first", inplace=True)
    #     df1[self.column] = df1[self.column].apply(lambda x: [x])
    #     ck_data = df1[self.column].tolist()
    #     ck_data.sort()
    #
    # # method2，遍历效率是非常非常慢的，强烈推荐使用apply进行矢量计算
    # # for index, val in self.dataset.iterrows():
    # #     # items = set(val["items"].split(","))
    # #     print("c1", "row id", index, "length ", len(val[self.column]))
    # #     for item in val[self.column]:
    # #         if [item] not in ck_data:
    # #             ck_data.append([item])
    # #     ck_data.sort()
    # print(f"c{k}_data", ck_data)
    logger.info(f"C{k} size: {len(ck_data)}")
    return list(map(frozenset, ck_data))  # 将C1各元素转换为frozenset格式，注意frozenset作用对象为可迭代对象


class Apriori:

    def __init__(self, dataset: pd.DataFrame, column: str = "items", min_support: float = 0.5, min_confidence: float = 0.7, k: int = 10):
        """
        :param dataset: DataFrame(data, columns=[...,"items",...])
        :param min_support:
        :param min_confidence:
        """
        # 最小支持度
        self.min_support = min_support
        # 最小置信度
        self.min_confidence = min_confidence
        # 最大项集数
        self.k = k
        # 计时器
        self.timer = {}

        self.dataset = dataset
        self.column = column
        self.dataset[self.column] = dataset[column].apply(lambda x: list(set(str(x).split(','))))
        self.dataset["length"] = self.dataset[self.column].apply(lambda x: len(x))
        # self.column_src = column + "_src"
        # self.dataset[self.column_src] = dataset[column].apply(lambda x: str(x).split(','))
        # self.dataset["length_src"] = self.dataset[self.column_src].apply(lambda x: len(x))
        self.dataset_num = len(self.dataset)

        self.white_list = []
        self.frequent_set = pd.DataFrame()
        self.association_rules = pd.DataFrame()
        # print(self.dataset.describe())

    def set_white_list(self):
        self.white_list = []

    def create_l1(self):
        """
        必须提前生成l1，第一次扫描结果集生成c1的同时统计次数，直接过滤成l1速度会提升很多
        :return:
        """
        logger.info("L1 dataset {}".format(self.dataset.shape))
        l1_df = self.dataset.drop("length", axis=1).copy()
        l1_df = l1_df.explode(self.column, ignore_index=True)
        l1_df["num"] = 1
        l1_df = l1_df.groupby(by=[self.column]).sum()
        logger.info("{sep} Total number of {number} {sep}".format(number=len(l1_df), sep="*" * 6))

        l1_df["row_num"] = self.dataset_num
        l1_df["support"] = round(l1_df["num"] / l1_df["row_num"], 4)  # 计算支持度
        l1_df["k"] = 1
        l1_df = l1_df[l1_df["support"] >= self.min_support]
        l1_df = l1_df.sort_values(by=["support"], ascending=[False])
        return l1_df

    def generate_lk(self, ck: list, k: int) -> pd.DataFrame:
        # 筛选过滤dataset数据
        ck_unique = get_ck_unique(ck)
        logger.info("ck unique length: " + str(len(ck_unique)))
        self.dataset[self.column] = self.dataset[self.column].apply(lambda x: sorted(list(set(x) & ck_unique)))
        self.dataset["length"] = self.dataset[self.column].apply(lambda x: len(x))
        self.dataset = self.dataset[self.dataset["length"] >= k]
        logger.info(f"{k} filter dataset\n" + str(self.dataset[[self.column, "length"]].sort_values(by=["length"], ascending=[False])))

        if k == 2:
            self.dataset[self.column + "_groups_1"] = self.dataset[self.column].apply(lambda x: [frozenset([v]) for v in x])
            self.dataset[self.column + "_groups_1_length"] = self.dataset["length"]

        # method1
        # TODO  # 瓶颈，比较耗时，最后通过提前过滤dataset解决
        groups = self.column + "_groups_" + str(k)
        groups_length = self.column + "_groups_" + str(k) + "_length"
        # self.dataset[groups] = self.dataset[self.column].apply(calc_count, args=(ck,))
        # self.dataset[groups] = self.dataset[self.column].parallel_apply(calc_count, args=(ck,))  # df必须要有数据，否则会报错raise ValueError("Number of processes must be at least 1")
        # self.dataset[groups] = self.dataset[self.column + "_groups_" + str(k - 1)].apply(lambda x: generate_ck_with_depends(x, k))
        self.dataset[groups] = self.dataset[self.column].apply(lambda x: generate_ck(x, k))
        self.dataset[groups_length] = self.dataset[groups].apply(lambda x: len(x))

        lk_df = self.dataset[[groups, groups_length, "length"]].copy()
        logger.info("L{} dataset {}".format(k, lk_df.shape))
        if lk_df.empty:
            return lk_df
        else:
            logger.info(f"L{k} filter df\n" + str(lk_df[[groups_length, "length"]].sort_values(by=[groups_length], ascending=[False])))
        lk_df = lk_df.explode(groups, ignore_index=True)
        lk_df["num"] = 1
        lk_df = lk_df[[groups, "num"]].groupby(by=[groups]).sum()

        # method2，遍历效率是非常非常慢的，强烈推荐使用apply进行矢量计算
        # for index, val in lk_df.iterrows():
        #     # items = set(val["items"].split(","))
        #     print(f"generate_lk {k}: index {index}, item num " + str(len(val[self.column])))
        #     for combination in ck:
        #         if combination.issubset(val[self.column]):
        #             lk_data[combination] = lk_data.get(combination, 0) + 1
        # lk_df = pd.DataFrame.from_dict(lk_data, orient="index", columns=["num"])
        # 新增一列，索引列变成字段名为"index"的列
        # lk_df = lk_df.reset_index()

        lk_df["row_num"] = self.dataset_num
        lk_df["support"] = round(lk_df["num"] / lk_df["row_num"], 4)  # 计算支持度
        lk_df["k"] = k
        # 添加自增列
        # lk_df['id'] = range(len(lk_df))

        lk_df = lk_df[lk_df["support"] >= self.min_support]
        lk_df = lk_df.sort_values(by=["support"], ascending=[False])
        # print(f"l{i}_data", lk)
        logger.info(f"L{k} size: {len(lk_df)}")
        if not lk_df.empty:
            logger.info(f"L{k} RESULT:\n" + str(lk_df))
        return lk_df

    def run_frequent_set(self):
        s1_time = datetime.now()
        logger.info("starting loop 1 ...")
        logger.info("\n" + str(self.dataset.sort_values(by=["length"], ascending=[False])))
        l1 = self.create_l1()
        logger.info("L1 RESULT:\n" + str(l1))
        logger.info("L1 size: {}".format(len(l1)))
        c1 = [[x] for x in l1.index.tolist()]
        c1.sort()
        ck = list(map(frozenset, c1))
        self.timer["L1"] = str(datetime.now() - s1_time)
        for i in range(2, self.k + 1):
            si_time = datetime.now()
            logger.info("starting loop {} ...".format(i))
            # ck = generate_ck(ck, i)
            lk_df = self.generate_lk(ck, i)
            if len(lk_df) == 0:
                break
            self.frequent_set = pd.concat([self.frequent_set, lk_df], ignore_index=False, sort=False)
            self.timer[f"L{i}"] = str(datetime.now() - si_time)
            lk = list(map(frozenset, lk_df.index))
            ck = lk
            i += 1
        # logger.info("frequent set")
        logger.info("\n============ FREQUENT SET ============\n" + str(self.frequent_set))
        # self.frequent_set["k"] = self.frequent_set["index"].values.map(lambda x: len(x))
        # self.frequent_set = self.frequent_set.sort_values(by=["support","k"], ascending=[False,True])
        self.frequent_set.to_csv("./frequent_set.csv", encoding="utf-8", header=True, index=True)  # index为group组

    # def run_association_rules(self):
    #     print("-" * 100, "association details")
    #     rules_data = []
    #     for index, row in self.frequent_set.iterrows():
    #         if row["k"] == 1:
    #             continue
    #         # print(index)
    #         for sorted_combinations_iter in get_sorted_combinations_all(index):
    #             for sorted_combination in sorted_combinations_iter:
    #                 if len(index) == len(sorted_combination):
    #                     continue
    #
    #                 x = sorted_combination
    #                 y = tuple(set(index) - set(sorted_combination))
    #                 numerator = self.frequent_set.at[index, "num"]
    #                 denominator = self.frequent_set.at[sorted_combination, "num"]
    #                 confidence = round(numerator / denominator, 4)  # 计算置信度
    #                 print(x, "==>", y, confidence)
    #
    #                 if confidence <= self.min_confidence:
    #                     continue
    #                 rules_data_row = [x, y, numerator, denominator, confidence]
    #                 rules_data.append(rules_data_row)
    #
    #     self.association_rules = pd.DataFrame(rules_data, columns=["x", "y", "numerator", "denominator", "confidence"])
    #     print("-" * 100, "association rules")
    #     print(self.association_rules)


if __name__ == "__main__":
    logger.info("Starting Calc ... ...")
    start_time = datetime.now()

    # df = pd.DataFrame(["1,3,4", "2,3,5", "1,2,3,5", "2,5"], columns=["items"])
    # print(df)
    # df = pd.read_csv("./example_data.csv", names=["id", "items"], sep="\t")  # 0.5, 0.75
    # df = pd.read_csv("./groceries.csv", names=["items"], sep="\t")  # 0.05, 0.25
    df = pd.read_csv(r"./跨院排除透析-items.csv", names=["items"])
    # conn = create_engine("impala://10.63.82.200:21050/medical", echo=False, poolclass=pool.NullPool)
    # df = pd.read_sql("select * from tmp.transactions", conn)

    print(df.head())
    m_support = 20 / 700  # 200/9835, 20/700,  700/136361
    m_confidence = 1
    far = Apriori(df, min_support=m_support, min_confidence=m_confidence)
    # far.dataset_overview()
    far.run_frequent_set()
    # far.run_association_rules()
    logger.info(json.dumps(far.timer, indent=4))

    logger.info("Total time taken {}".format(datetime.now() - start_time))
