# coding=utf-8
# python3

import random

meta_items = [
    "面包",
    "牛奶",
    "冰激凌",
    "苹果",
    "香蕉",
    "土豆",
    "梨",
    "西红柿",
    "西瓜"
]

count = len(meta_items)
rows = 1000


def rand_row():
    row_items = []
    item_count = random.randint(1, count - 1)
    for i in range(item_count):
        j = random.randint(1, count - 1)
        row_items.append(meta_items[j])
    return list(set(row_items))


def main():
    with open("./example_data.csv", "w", encoding="utf8") as f:
        for i in range(1, rows + 1):
            row_items_format = str(rand_row())[1:-1].replace("'", "").replace(",", ";")
            row = "{id},{items}".format(id=str(i), items=row_items_format)
            print(row)
            if i == rows:
                f.writelines(row)
            else:
                f.writelines(row + "\n")


if __name__ == "__main__":
    main()
