# 编译python文件为c

```shell
pip3 install cython
gcc -v

# 使用python测试
python3 test.py

2020-04-26 14:02:17.975454
4999999950000000
2020-04-26 14:02:25.917488
0:00:07.942034

# 编译成c
python3 compile_to_c.py build_ext

# 拷贝生成的c文件到当前目录
# linux
cp build/lib.linux-x86_64-3.6/timekeeping.cpython-36m-x86_64-linux-gnu.so .
# windows
cp build/lib.win-amd64-3.6/timekeeping.cpython-36m-win_amd64.pyd .

# 重新测试（可以不用删除原始的python文件，默认会优先使用c文件，当然也可以删除原始的python文件不影响）
python3 test.py

2020-04-26 14:32:34.538042
4999999950000000
2020-04-26 14:32:39.894104
0:00:05.356062
```