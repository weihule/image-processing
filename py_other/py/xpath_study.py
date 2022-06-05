import os
from socket import herror
from lxml import etree
import requests
import numpy as np
import cv2
import time

# 爬取58二手房信息
def main():
    # 获取页面源码数据
    hearders = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    url = 'https://bj.58.com/ershoufang/'
    page_text = requests.get(url=url, headers=hearders).text
    tree = etree.HTML(page_text)
    # res = tree.xpath('//div[@class="property-content-detail"]//@title')
    titles = tree.xpath('//div[@class="property-content-detail"]/div[@class="property-content-title"]/h3/@title')
    house_prices = tree.xpath('//div[@class="property-content-info"]/p[last()]/text()')
    # for title in titles:
    #     print(title)
    for house_type in house_prices:
        print(house_type)


# 爬取4K美女
def beautiful():
    save_root = 'D:\\workspace\\data\\DL\\beautiful_girl'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    url = 'http://pic.netbian.com/4kmeinv/'
    page_text = requests.get(url=url, headers=headers).text  
    tree = etree.HTML(page_text)
    imgs = tree.xpath('//div[@class="slist"]//img/@src')
    for idx, i in enumerate(imgs):
        img_url = 'http://pic.netbian.com' + i
        img_bin = requests.get(url=img_url, headers=headers).content
        img_buf = np.frombuffer(img_bin, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        save_path = os.path.join(save_root, str(idx+1).rjust(3, '0')+'.png')
        cv2.imwrite(save_path, img)
        print(idx+1, save_path)
        time.sleep(3)
        # cv2.imshow('res', img)
        # cv2.waitKey(0)



def countries():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    url = 'https://www.aqistudy.cn/historydata/'
    page_text = requests.get(url=url, headers=headers).text 
    tree = etree.HTML(page_text)
    roi_cities = tree.xpath('//div[@class="hot"]//ul[@class="unstyled"]/li')
    print(len(roi_cities))
    for roi_city in roi_cities:
        city_name = roi_city.xpath('.//text()')[0]
        print(city_name)

    all_cities = tree.xpath('//div[@class="all"]//ul[@class="unstyled"]//li')
    for i in all_cities:
        print(i.xpath('./a/text()')[0])



def get_resumes():
    """
    爬取简历模板
    """
    save_root = 'D:\\workspace\\data\\scipy\\resumes'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    c = 0
    for i in range(1, 6):
        net = 'free' if i == 1 else 'free_' + str(i)
        url = 'https://sc.chinaz.com/jianli/' + net +'.html'
        page_text = requests.get(url=url, headers=headers).text
        tree = etree.HTML(page_text)
        resumes_urls = tree.xpath('//div[@id="main"]//p/a/@href')
        # 进入每一个模板的界面 
        for idx, resumes_url in enumerate(resumes_urls):
            resume_dl_page = requests.get(url='https:'+resumes_url, headers=headers).text
            resume_tree = etree.HTML(resume_dl_page)
            dl = resume_tree.xpath('//div[@class="clearfix mt20 downlist"]//li[1]/a/@href')[0]
            print(dl)
            data_bin = requests.get(dl, headers=headers).content
            # c += 1 
            # save_path = os.path.join(save_root, str(c).rjust(3, '0')+'.rar')
            # with open(save_path, 'wb') as f:
            #     f.write(data_bin)
            # print('page{} index:{}'.format(i, idx+1))
            
            # time.sleep(3)
            break
        break


if __name__ == "__main__":
    # main()
    # beautiful()

    # countries()

    get_resumes()