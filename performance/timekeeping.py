# coding=utf-8
# python3

from datetime import datetime


def run():
    """
    反复测试大概需要30s，根据不同的机器配置和性能会有所不同
    :return:
    """
    start_time = datetime.now()
    print(start_time)

    # for i in range(10000):
    #     cnt = 0
    #     for j in range(10000*10):
    #         cnt += j
    #     print(i, cnt)

    cnt = 0
    for i in range(10000 * 10000):
        cnt += i
    print(cnt)

    end_time = datetime.now()
    print(end_time)
    print(end_time - start_time)

