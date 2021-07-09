# coding:utf-8
# python3

import numpy as np
import pandas as pd


def convert():
    df = pd.read_csv(r"./跨院排除透析.csv", encoding="utf-8")
    print(df)
    df[["clientid"]].to_csv(r"./跨院排除透析-items.csv", encoding="utf-8", header=False, index=False)
    df.to_csv(r"./跨院排除透析-hive.csv", encoding="utf-8", header=False, index=False, sep="\001")


def save_impala(df: pd.DataFrame):
    def get_dict_type(df: pd.DataFrame):
        from sqlalchemy.dialects.sqlite import INTEGER, VARCHAR, FLOAT
        from sqlalchemy.types import TEXT
        type_dict = {}
        for idx, row in df.iterrows():
            data_type = row["type"]
            if "string" in data_type or "char" in data_type:
                data_type = "str"
            elif "int" in data_type:
                data_type = "int"
            elif "float" in data_type or "double" in data_type or "decimal" in data_type:
                data_type = "float"
            else:
                data_type = "str"
            type_dict[row["name"]] = data_type
        return type_dict

    from sqlalchemy import create_engine, pool
    engine_impala = create_engine(
        "impala://{user}@{host}:{port}/{database}".format(user="work", host="10.63.82.200", port="21050", database="tmp"),
        poolclass=pool.NullPool)
    to_table_name = "app_fk_inhospital_together"
    to_df = pd.read_sql("describe " + to_table_name, engine_impala)
    print(to_df)

    df.columns = to_df["name"].tolist()
    dtypes_map = get_dict_type(to_df)
    for k, v in dtypes_map.items():
        df[k] = df[k].astype(v)
    print(df.dtypes)
    # 巨慢无比，强烈不推荐
    df.to_sql("app_fk_inhospital_together", engine_impala, if_exists="append", index=False)


def df_split():
    data = [[1, "apple,orange,banana"], [2, "apple,mile"]]
    df = pd.DataFrame(data, columns=["id", "items"])
    print(df)
    print("-" * 20)
    # 拆分一列数据
    print(df["items"].str.split(",", expand=True).stack().reset_index(level=1, drop=True).rename("item"))
    # 拆分一列数据，并和其他数据组合拼成新的dataframe
    print(df.drop("items", axis=1).join(df["items"].str.split(",", expand=True).stack().reset_index(level=1, drop=True).rename("item")))

    def calc_count(x, combinations):
        res = []
        for combination in combinations:
            if combination.issubset(x):
                res.append(combination)
        return res

    print("=" * 20)
    df = pd.DataFrame(data, columns=["id", "items"])
    df["items"] = df["items"].apply(lambda x: x.split(","))
    print(df)
    ck_data = [["apple"], ["orange"], ["apple", "orange"]]
    df["items"] = df["items"].apply(calc_count, args=(list(map(frozenset, ck_data)),))
    print(df)
    df = df.explode("items")
    df["cnt"] = 1
    print(df)
    print(df[["items", "cnt"]].groupby(["items"]).sum())


def analysis_data():
    from loguru import logger
    df = pd.read_csv(r"./跨院排除透析.csv", encoding="utf-8")
    df["length"] = df["clientids"].apply(lambda x: len(x.split(",")))
    df["length_unique"] = df["clientids"].apply(lambda x: len(set(x.split(","))))
    df["clientids_unique"] = df["clientids"].apply(lambda x: ",".join(set(x.split(","))))
    df.sort_values(by=["length"], ascending=False, inplace=True)
    print(df.head(10))
    print(df[["visitdate", "length", "length_unique"]].head(10))

    df_clientid_unique = df["clientids_unique"].str.split(",", expand=True).stack().reset_index(level=1, drop=True).rename("clientid")
    df_clientid = df["clientids"].str.split(",", expand=True).stack().reset_index(level=1, drop=True).rename("clientid")
    print(df_clientid_unique.shape)
    print(df_clientid.shape)

    df_detail = df[["visitdate"]].join(df_clientid_unique)  # 默认使用index关联
    print(df_detail)

    df1 = df_detail.groupby("clientid").agg({"clientid": np.size})
    df1.columns = ["cnt"]
    df1.sort_values(by="cnt", ascending=False, inplace=True)
    df1 = df1[df1["cnt"] >= 3]
    print(df1)

    def filter_items(x: str, combinations: list) -> list:
        res = []
        item_list = x.split(",")
        for item in item_list:
            if item in combinations:
                res.append(item)
        return res

    def filter_items1(x: str, combinations: list) -> list:
        item_list = x.split(",")
        return list(set(item_list) & set(combinations))

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

    def filter_items2(x: str, combinations: list) -> list:
        res = []
        item_list = x.split(",")
        for item in item_list:
            if bin_search(combinations, item):
                res.append(item)
        return res

    clientid_list = df1.index.tolist()
    print(type(np.array(clientid_list)))
    print(len(clientid_list))

    # df = df[:10]
    clientid_list.sort()  # 二分查找必须排序

    logger.info(df.columns.tolist())
    logger.info(df[["visitdate", "length", "length_unique"]])
    df["clientids_f2"] = df["clientids_unique"].apply(filter_items2, args=(clientid_list,))
    df["length_clientids_f2"] = df["clientids_f2"].apply(lambda x: len(x))
    logger.info(df[["visitdate", "length", "length_unique", "length_clientids_f2"]])


if __name__ == "__main__":
    # convert()
    # df_split()
    analysis_data()
