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
    "西瓜",
    "电视机",
    "冰箱",
    "空调",
    "微波炉",
    "洗衣机",
    "电饭煲",
    "热水器"
]

# meta_items = [str(x) for x in range(1, 101)]

ROW_NUM = 1000
COUNT = len(meta_items)


def rand_row():
    row_items = []
    item_count = random.randint(1, COUNT - 1)
    for i in range(item_count):
        j = random.randint(1, COUNT - 1)
        row_items.append(meta_items[j])
    return list(set(row_items))


def main():
    with open("./generate_data.csv", "w", encoding="utf8") as f:
        for i in range(1, ROW_NUM + 1):
            row_items_format = ",".join(rand_row())
            row = "{id}\t{items}".format(id=str(i), items=row_items_format)
            print(row)
            if i == ROW_NUM:
                f.writelines(row)
            else:
                f.writelines(row + "\n")


if __name__ == "__main__":
    main()
