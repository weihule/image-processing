import os
import argparse
import json
import os
import ast

# parse = argparse.ArgumentParser(description="命令行传入一个数字")
#
# parse.add_argument("integers", type=int, nargs="+", help="input an integer")
# args = parse.parse_args()
#
# # 获得传入的参数
# print(args)
#
# # 获得integers参数
# print(args.integers)


parse = argparse.ArgumentParser(description="姓名")
parse.add_argument("--param1", type=str, default="张", help="姓")
parse.add_argument("--param2", type=str, default="三", help="名")
args = parse.parse_args()

print(eval(args.param1))
# 打印姓名
print(args.param1+args.param2)

parse.set_defaults()

