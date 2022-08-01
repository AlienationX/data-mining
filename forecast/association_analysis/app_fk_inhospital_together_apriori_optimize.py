#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   结伴Apriori.py
@Time    :   2021/06/29 17:39:07
@Author  :   lshu 
@Version :   1.0
@Contact :   lshu@abc.com
@Desc    :    

yum install -y python3-devel
pip3 install pandas efficient_apriori dataclasses impyla
'''

import pandas as pd
import numpy as np
from efficient_apriori import apriori
import time
import re
import hashlib
import datetime
import sys

"""
1. 读取数据
2. 频繁项集计算
3. 去除具有包含关系的重复结伴组
4. 查找明细数据
"""

# INPUT_NUM = sys.argv[1] # 参数 - 最小支持度(表示最少同时出现几次)
# visittype_id = sys.argv[2] # ZY住院 or GY购药 or MZ门诊
# group_data = sys.argv[3]  #获取初始路径
# detail_data = sys.argv[4] #获取data表路径
# outputpath = sys.argv[5]  #输出路径名称

INPUT_NUM=6
VISITTYPE="ZY"
group_data = r"C:\Users\Admin\Desktop\ods_jieban_date.csv"
detail_data = r"C:\Users\Admin\Desktop\ods_jieban_hos.csv"
output_file = r"C:\Users\Admin\Desktop\result.csv"

if VISITTYPE not in ("MZ", "ZY", "GY"):
    print("visittype_id must be in [MZ, ZY, GY]")
    sys.exit(1)
    
    
def load_group_data() -> pd.DataFrame:
    group_df = pd.read_csv(group_data, sep="\t", header=0)
    group_df.columns = ["year", "visittype_id", "orgname", "visitdate", "clientids"]
    group_df = group_df[group_df["visittype_id"].str.contains(VISITTYPE)]
    group_df = group_df.reset_index(drop=True)  # 重置索引
    group_df = group_df[["orgname", "visitdate", "clientids"]]
    group_df["clientid_list"] = group_df["clientids"].map( lambda x: x.split(",") )
    print(group_df)
    return group_df
    

def load_detail_data() -> pd.DataFrame:
    detail_df = pd.read_csv(detail_data, sep="\t", header=0)
    detail_df.columns = ["year", "visit", "visittype", "visittype_id", "visitid", "clientid", "clientname", "sex", "yearage", 
                         "visitdate", "leavedate", "zy_days", "diseasename", "chronicdis", "orgid", "orgname", "fundpay", "totalfee"]
    detail_df = detail_df[detail_df["visittype_id"].str.contains(VISITTYPE)]
    detail_df = detail_df.reset_index(drop=True)  # 重置索引
    detail_df["visitid"] = detail_df["visitid"].astype(str)  # 避免科学计数法之类的bug
    detail_df["visitdate_day"] = detail_df["visitdate"].map( lambda x: str(x)[:10] )  # 截取到天
    # print(detail_df)
    return detail_df
    

def calc_itemsets(group_df: pd.DataFrame) -> pd.DataFrame:
    # [[a], [a,b,c], [b,c], [a,d]]
    data_list = group_df["clientid_list"].to_list()
    # print(data_list)
    start_time = datetime.datetime.now()
    print("Start time: " + str(start_time))
    itemsets, rules = apriori(data_list, min_support=int(INPUT_NUM)/len(data_list), min_confidence=1)
    end_time = datetime.datetime.now()
    print("End time: " + str(end_time))
    print("Take time: " + str((end_time - start_time).seconds))
    print(type(itemsets))  # <class 'dict'>
    # print(itemsets.get(1))
    # print(itemsets.get(2))
    # {
    #     1:{('a',): 10, ('b',): 1, ('c',): 8, ('d',): 2}, 
    #     2:{('a', 'b'): 10, ('a', 'c'): 8, ('a', 'd'): 2} ,
    #     3:{('a', 'b', 'c'): 8},
    #     ... ...
    # }
    print(type(rules))     # <class 'list'>
    # print(rules)         # [{a} -> {b}, {a, c} -> {b}]
    
    # 去除具有包含关系的重复结伴组
    itemsets = drop_contain_group(itemsets)
    
    frequent_clientid_list = []
    cnt_list = []
    for k in itemsets.keys():
        if k > 1:
            for clientid_tuple in itemsets[k].keys():
                frequent_clientid_list.append(clientid_tuple)
                cnt_list.append(itemsets[k][clientid_tuple])
                
    frequent_df = pd.DataFrame({
        "frequent_clientid_tuple": frequent_clientid_list,
        "group_cnt": cnt_list
    })
    print(frequent_df.sort_values("group_cnt", ascending=False))
    return frequent_df


def split_frequent_df(group_df: pd.DataFrame, frequent_itemsets) -> pd.DataFrame:
    
    def add_frequent_groups(clientid_list, frequent_itemsets):
        frequent_groups = []
        for frequent_id_tuple in frequent_itemsets["frequent_clientid_tuple"].to_list():
            # 如果 clientid_list 包含 frequent_id_set 则添加到新的字段中
            if set(clientid_list) > set(frequent_id_tuple):
                frequent_groups.append(frequent_id_tuple)
        return frequent_groups
        
    group_df["frequent_groups"] = group_df["clientid_list"].apply(lambda x: add_frequent_groups(x, frequent_itemsets))
    group_df["frequent_groups_size"] = group_df["frequent_groups"].map( lambda x: len(x))
    simplify_df = group_df[ group_df["frequent_groups_size"] > 0 ]
    simplify_df = simplify_df.reset_index(drop=True)
    simplify_df = simplify_df[["orgname", "visitdate", "frequent_groups"]]
    print("simplify_df =========== >")
    print(simplify_df)
    
    def explode_df(df: pd.DataFrame, split_column: str):
        """拆分成行
        :param df:DataFrame  原始数据
        :param column:str    拆分的列名
        """
        data = {}
        for column in df.columns:
            # 保留拆分的原始列
            data[column] = df[column].repeat(df[split_column].str.len())
            # 新增拆分的列，并命名为 split_column + "_split"
            if column == split_column:
                # 降维操作：[[1,2], [3,4]] => [1,2,3,4]
                # np.concatenate( np.array([[[1,2],[2,3]],[[1,2]]]) )
                # np.vstack( np.array([[[1,2],[2,3]],[[1,2]]]) )
                
                # 如果元素个数不相同，则会报错 np.concatenate( np.array([[[1,2],[2,3,4]],[[1,2]]]) )
                # TODO optimize
                tmp = []
                for values in df[split_column].values:
                    for value in values:
                        tmp.append(value)
                data[split_column + "_split"] = tmp
        return pd.DataFrame(data)
    
    # 拆分 frequent_groups
    split_groups_df = explode_df(simplify_df, "frequent_groups")
    print("split_groups_df =========== >")
    print(split_groups_df)
    
    # 拆分 frequent_groups_split
    split_clientid_df = explode_df(split_groups_df, "frequent_groups_split")
    split_clientid_df = pd.merge(
        split_clientid_df, 
        frequent_itemsets, 
        left_on=["frequent_groups_split"], 
        right_on=["frequent_clientid_tuple"], 
        how="inner"
    )
    # split_clientid_df.columns = ["orgname", "visitdate", "clientid"]
    print("split_clientid_df =========== >")
    print(split_clientid_df)
    return split_clientid_df


def find_detail(split_clientid_df: pd.DataFrame):
    detail_df = load_detail_data()
    result = pd.merge(
        detail_df, 
        split_clientid_df, 
        left_on=["orgname", "visitdate_day", "clientid"], 
        right_on=["orgname", "visitdate", "frequent_groups_split_split"], 
        how="inner"
    )
    
    result["group_size"] = result["frequent_groups_split"].map( lambda x: len(x) )
    result["groupname"] = result["frequent_groups_split"].map( lambda x: ",".join(x) )
    result["groupid"] = result["groupname"].map( lambda x: VISITTYPE + hashlib.md5(str(x).encode('utf-8')).hexdigest() )
    result["id"] = result["groupid"] + result["visit"]
    # result["group_cnt"] = 
    print(result.dtypes)
    result.to_csv(output_file, encoding="utf-8", sep="\t", header=True, index=True)
    
    
def apriori_test():
    data = [
        ['a', 'c', 'd'],
        ['b', 'c', 'e'],
        ['a', 'b', 'c', 'e'],
        ['b', 'e']
    ]
    itemsets, rules = apriori(data, min_support=0.5, min_confidence=1)
    print(itemsets)
    print(rules)
    return itemsets


def drop_contain_group(itemsets: dict):
    """
    去除具有包含关系的重复项目组. 注意如果去除包换关系且不加频次, 输入的num不同, 每次跑的数据都会不同.
    example:
        a,c   2
        b,c   2  --> drop
        b,e   3  --> 如果不保留频次, 该条也需要drop
        c,e   2  --> drop
        b,c,e 2
    """
    
    # 删除1项集
    itemsets.pop(1)
    
    # {1: {('a',): 2, ('c',): 3, ('b',): 3, ('e',): 3}, 2: {('a', 'c'): 2, ('b', 'c'): 2, ('b', 'e'): 3, ('c', 'e'): 2}, 3: {('b', 'c', 'e'): 2}}
    # 将所有id组转换成key，频次为value。其实也可以不需要频次
    convert_dict = {}
    for v_dict in itemsets.values():
        for ids_set, cnt in v_dict.items():
            convert_dict[ids_set] = cnt
    
    result_dict = {}
    for k, v_dict in itemsets.items():
        level2_dict = {}
        for id_set, cnt in v_dict.items():
            flag = 0
            for key in convert_dict.keys():  # TODO 先从组合数大的开始遍历，减少循环次数提高效率
                # 判断id_set包含且频次相同则删除。必须使用set进行比较，不能使用tuple
                if set(key) > set(id_set):
                    flag = 1
                    break
            if flag == 0:
                level2_dict[id_set] = cnt
        result_dict[k] = level2_dict
    # print(result_dict)
    return result_dict


if __name__ == "__main__":
    # drop_contain_group(apriori_test())
    group_df = load_group_data()
    frequent_df = calc_itemsets(group_df)
    split_clientid_df = split_frequent_df(group_df, frequent_df)
    find_detail(split_clientid_df)
