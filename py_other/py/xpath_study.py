import os
from socket import herror
from lxml import etree
import requests
import numpy as np
import cv2
import time
import json
from tqdm import tqdm


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


# 爬取图片
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
        'user-agent': "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.57"
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


def test():
    file_path = 'D:\\Desktop\\imagenet100\\maps.txt'
    folder2chinese = {}
    with open(file_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for idx, line in enumerate(lines):
        if idx % 2 == 0:
            line = line.strip('\n')
            folder_name = line.split(' ')[1]
            class_name = line.split(' ')[2].strip(',')
            folder2chinese[folder_name] = class_name

    with open('classes.json', 'r', encoding='utf-8') as fr_j:
        infos = json.load(fr_j)
    

    new_infos = {}
    for idx, (k1, v1) in enumerate(folder2chinese.items()):
        new_infos[k1] = {'idx': idx, 'chinese_name': v1, 'name': infos[str(idx)]}
    
    for k, v in new_infos.items():
        print(k, v)

    json_str = json.dumps(new_infos, indent=4, ensure_ascii=False)
    with open('all_classed.json', 'w', encoding='utf-8') as fw:
        fw.write(json_str)


def rename():
    with open('all_classed.json', 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
    root = 'D:\\Desktop\\imagenet100\\imagenet100'
    new_root = 'D:\\Desktop\\imagenet100\\imagenet100_rename'

    class_100 = {}
    for folder_idx, folder in enumerate(os.listdir(root)):
        new_folder = infos[folder]['name'][0].replace(' ', '-')
        class_100[new_folder] = folder_idx
        new_folder_path = os.path.join(new_root, new_folder)
        if not os.path.exists(new_folder_path):
            os.mkdir(new_folder_path)
        for img_idx, fn in enumerate(os.listdir(os.path.join(root, folder))):
            print(folder_idx, img_idx, fn)
            fn_path = os.path.join(root, folder, fn)
            img = cv2.imread(fn_path)
            new_name = new_folder + '_' + str(img_idx).rjust(4, '0') + '.jpg'
            save_path = os.path.join(new_folder_path, new_name)
            cv2.imwrite(save_path, img)
        
    json_str = json.dumps(class_100, indent=4, ensure_ascii=False)
    with open(os.path.join(new_root, 'class_100.json'), 'w') as fw:
        fw.write(json_str)


def proce_val():
    with open('D:\\Desktop\\imagenet100\\all_classes.json', 'r', encoding='utf-8') as fr1:
        infos = json.load(fr1)
    new_infos = {}
    for _, v in infos.items():
        new_infos[str(v['idx'])] = v['name'][0].replace(' ', '-')

    root = 'D:\\Desktop\\imagenet100\\ILSVRC2012_img_val'
    with open('D:\\Desktop\\imagenet100\\val.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
    for line in tqdm(lines):
        line = line.strip('\n')
        fn_name = line.split(' ')[0]
        fn_class = line.split(' ')[1]
        save_folder = new_infos[fn_class]
        if save_folder not in os.listdir('D:\\Desktop\\imagenet100\\imagenet100_train'):
            continue
        save_folder_path = os.path.join('D:\\Desktop\\imagenet100\\imagenet100_val', save_folder)
        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)
        fn_path = os.path.join(root, fn_name)
        new_fn_name = fn_name[:-4] + 'jpg'
        img = cv2.imread(fn_path)
        save_path = os.path.join(save_folder_path, new_fn_name)
        cv2.imwrite(save_path, img)
        


if __name__ == "__main__":
    # main()
    # beautiful()

    # countries()

    # get_resumes()

#    rename()
   proce_val()