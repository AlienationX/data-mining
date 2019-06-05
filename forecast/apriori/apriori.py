# coding=utf-8
# python3

import pandas as pd


def load_data():
    df = pd.read_csv("./example_data.csv", names=["id", "items"])  # chunksize=500
    print(df.head())


if __name__ == "__main__":
    load_data()
