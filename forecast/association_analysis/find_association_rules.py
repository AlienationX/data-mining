# coding=utf-8
# python3

"""
1、基于最底层的所有item生成2项集、3项集等
2、优点：有现成的函数combinations生成k项集
3、缺点：生成的项集太多，每次循环组合的项集再判断明细项中是否包含。太耗时，且生成的组合项很多没用
"""

from itertools import combinations
import pandas as pd


def get_sorted_combinations(item_set, k):
    # 返回list，不如返回迭代器
    # sorted_combinations = list(combinations(sorted(item_set), k))
    # return sorted_combinations
    return combinations(sorted(item_set), k)


def get_sorted_combinations_all(item_set):
    sorted_combinations_all = []
    for i in range(len(item_set)):
        sorted_combinations = get_sorted_combinations(item_set, i + 1)
        sorted_combinations_all.append(sorted_combinations)
    return sorted_combinations_all


class FindAssociationRules:

    def __init__(self, transactions, min_support=0.5, min_confidence=0.7):
        """
        :param transactions: DataFrame(data, columns=[...,"items",...])
        :param min_support:
        :param min_confidence:
        """
        # 最小支持度
        self.min_support = min_support
        # 最小置信度
        self.min_confidence = min_confidence

        self.transactions = transactions
        self.transactions_num = len(self.transactions)

        self.frequent_set = pd.DataFrame()
        self.association_rules = pd.DataFrame()

    def create_lk(self, k):
        lk_data = {}
        for index, val in self.transactions.iterrows():
            items = set(val["items"].split(","))
            for sorted_combination in get_sorted_combinations(items, k):
                if sorted_combination in lk_data:
                    lk_data[sorted_combination] = [lk_data[sorted_combination][0] + 1]
                else:
                    lk_data[sorted_combination] = [1]

        lk_df = pd.DataFrame.from_dict(lk_data, orient="index", columns=["num"])
        # 新增一列，索引列变成字段名为"index"的列
        # lk_df = lk_df.reset_index()
        lk_df["row_num"] = self.transactions_num
        lk_df["support"] = round(lk_df["num"] / lk_df["row_num"], 2)  # 计算支持度
        lk_df["k"] = k
        # print(lk_df)

        lk_df = lk_df[lk_df["support"] >= self.min_support]
        # print(lk_df)

        return lk_df

    def run_frequent_set(self, k=20):
        for i in range(1, k + 1):
            lk_df = self.create_lk(i)
            if len(lk_df) == 0:
                break
            else:
                self.frequent_set = pd.concat([self.frequent_set, lk_df], ignore_index=False, sort=False)
            i += 1
        print("-" * 100, "frequent set")
        # self.frequent_set["k"] = self.frequent_set["index"].values.map(lambda x: len(x))
        # self.frequent_set = self.frequent_set.sort_values(by=["support","k"], ascending=[False,True])
        print(self.frequent_set)

    def run_association_rules(self):
        print("-" * 100, "association details")
        rules_data = []
        for index, row in self.frequent_set.iterrows():
            if row["k"] == 1:
                continue
            # print(index)
            for sorted_combinations_iter in get_sorted_combinations_all(index):
                for sorted_combination in sorted_combinations_iter:
                    if len(index) == len(sorted_combination):
                        continue

                    x = sorted_combination
                    y = tuple(set(index) - set(sorted_combination))
                    numerator = self.frequent_set.at[index, "num"]
                    denominator = self.frequent_set.at[sorted_combination, "num"]
                    confidence = round(numerator / denominator, 2)  # 计算置信度
                    print(x, "==>", y, confidence)

                    if confidence <= self.min_confidence:
                        continue
                    rules_data_row = [x, y, numerator, denominator, confidence]
                    rules_data.append(rules_data_row)

        self.association_rules = pd.DataFrame(rules_data, columns=["x", "y", "numerator", "denominator", "confidence"])
        print("-" * 100, "association rules")
        print(self.association_rules)


if __name__ == "__main__":
    print(list(combinations(['a', 'b', 'c', 'd'], 3)))
    print(get_sorted_combinations_all(['a', 'b', 'c', 'd']))

    df = pd.read_csv("./example_data.csv", names=["id", "items"], sep="\t")  # chunksize=500
    far = FindAssociationRules(df, 0.5, 0.7)
    far.run_frequent_set()
    far.run_association_rules()

    # 随机生成的数据效果太差，不推荐
    # df = pd.read_csv("./generate_data.csv", names=["id", "items"], sep="\t")  # chunksize=500
    # far = FindAssociationRules(df, 0.2, 0.7)
    # far.run_frequent_set()
    # far.run_association_rules()
