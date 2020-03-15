# coding=utf-8
# python3

"""
某个项集是频繁的，那么它的所有子集也是频繁的。
如果一个项集是 非频繁项集，那么它的所有超集也是非频繁项集。
1、k项集基于k-1项集生成，且删除非频繁组合
2、优点：
3、缺点：
"""

import pandas as pd
import json
from itertools import combinations


class Apriori:

    def __init__(self):
        self.dataset = None
        self.dataset_length = None
        self.items = []
        # 支持度
        self.support = 0.5
        # 置信度
        self.confidence = 0.7

        self.result = {}

    def set_support(self, support):
        self.support = support

    def set_confidence(self, confidence):
        self.confidence = confidence

    def set_dataset(self, df):
        self.dataset = df
        self.dataset_length = len(self.dataset)

    def set_items(self):
        for index, row in self.dataset.iterrows():
            for item in row["items"].split(","):
                if item not in self.items:
                    self.items.append(item)
        # self.items.sort()

    def create_ck(self, input_ck, k):
        ck = list(combinations(input_ck, k))
        print(ck)
        print(len(ck))
        return ck

    def create_lk(self, dataset, ck, min_support):
        lk_tmp = {}
        for index, val in dataset.iterrows():
            for group_items in ck:
                set_group_items = set(group_items)
                set_detail_items = set(val["items"].split(","))
                if set_group_items.issubset(set_detail_items):
                    # key = ",".join(group_items)
                    lk_tmp[group_items] = lk_tmp.get(group_items, 0) + 1

        print("lk_tmp", json.dumps({",".join(k): v for k, v in lk_tmp.items()}, indent=4, ensure_ascii=False))

        # lk = {}
        # for k, v in lk_tmp.items():
        #     if v / self.dataset_length >= self.support:
        #         lk[k] = (v, round(v / self.dataset_length, 2))
        #     else:
        #         # 去掉不符合支持度的item，以便减少组合数
        #         # 这么处理存在问题，生成组合需要基于上一层，而不是第一层
        #         for infrequent in k:
        #             print(infrequent)
        #             if infrequent in self.items:
        #                 self.items.remove(infrequent)
        lk = {k: round(v / self.dataset_length, 2) for k, v in lk_tmp.items() if v / self.dataset_length >= self.support}
        print("lk", json.dumps({",".join(k): str(v) for k, v in lk.items()}, indent=4, ensure_ascii=False))

        print(self.items)
        return lk

    def run(self):
        i = 1
        while True:
            ck = self.create_ck(self.items, i)
            lk = self.create_lk(self.dataset, ck, self.support)
            if len(lk) == 0:
                break
            else:
                self.result.update(lk)
            i += 1
        print("-" * 100)
        print("result", json.dumps({",".join(k): str(v) for k, v in self.result.items()}, indent=4, ensure_ascii=False))


def load_data():
    df = pd.read_csv("./example_data.csv", names=["id", "items"], sep="\t")  # chunksize=500
    print(df.head())
    return df


if __name__ == "__main__":
    # combinations的实现？
    print(list(combinations(range(4), 3)))
    print(list(combinations(["A", "B", "C", "D"], 3)))

    ap = Apriori()
    ap.set_dataset(load_data())
    ap.set_items()
    # ap.generate_group_k(1)
    # ap.generate_group_k(2)
    # ap.generate_group_k(3)
    # ap.generate_group_k(4)
    # ap.calc_ck(1)
    ap.run()
