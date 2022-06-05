import os
import time
import requests
import re
import numpy as np
import cv2
from lxml import etree

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36'
}

def test_single_img(url):
    # url = 'http://www.thetreefarm.com/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/f/i/file_110_16.jpg'
    # url = 'https://tse1-mm.cn.bing.net/th?id=OIP.pE6ezsi-n9bf6sVqYzYtHAHaE7&amp;pid=15.1'
    # url = 'https://tse2-mm.cn.bing.net/th/id/OIP-C.9Z5j0vG3NC73u4kbxyZCFAHaEK?w=300&amp;h=180&amp;c=7&amp;r=0&amp;o=5&amp;pid=1.7'
    # content 返回二进制类型数据
    # text 返回字符串数据类型数据
    # json 返回对象类型的数据
    img_bin = requests.get(url=url, headers=headers).content
    print(img_bin)
    img_buf = np.frombuffer(img_bin, dtype=np.uint8)
    img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
    # print(img.shape)
    cv2.imshow('res', img)
    cv2.waitKey(0)
    # cv2.imwrite('test.jpg', img)

def main():
    save_root = 'D:\\workspace\\data\\DL\\add_flower\\roses'
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    url = 'https://cn.bing.com/images/search?q=rose&qs=n&form=QBIR&sp=-1&pq=rose&sc=8-4&cvid=6F098FCE048E4918B057A4EFF4947CAB&first=1&tsc=ImageHoverTitle'
    response = requests.get(url=url, headers=headers).text
    # ex = '"purl":(.*?),"murl'
    ex = '<div class="img_cont hoff">.*?<img class=.*?src=(.*?) alt=.*?</div>'
    img_url_list = re.findall(ex, response, re.S)
    c = 0
    for i in img_url_list:
        if len(i) > 150:
            continue
        img_bin = requests.get(url=i.strip('"'), headers=headers).content
        img_buf = np.frombuffer(img_bin, dtype=np.uint8)
        img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)
        save_path = os.path.join(save_root, 'rose_'+str(c).rjust(3, '0')+'.jpg')
        c += 1
        print(c, save_path)
        cv2.imwrite(save_path, img)
        time.sleep(2)

    print(len(img_url_list))

    # '<a class="iusc" style="height:196px;width:295px" 
    # m="{&quot;sid&quot;:&quot;&quot;,&quot;cturl&qu
    # ot;:&quot;&quot;,&quot;cid&quot;:&quot;pE6ezsi+
    # &quot;,&quot;purl&quot;:&quot;
    # http://www.thetreefarm.com/rose-morden-centennial&quot;,
    # &quot;murl&quot;:&quot;
    # http://www.thetreefarm.com/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/f/i/file_110_16.jpg
    # &quot;,&quot;turl&quot;:&quot;https://tse1-mm.cn.bing.net/th?id=OIP.pE6ezsi-n9bf6sVqYzYtHAHaE7&amp;pid=15.1
    # &quot;,&quot;md5&quot;:&quot;a44e9ecec8be9fd6dfeac56a63362d1c&quot;,
    # &quot;shkey&quot;:&quot;j94d2IVrcc4FFrbcvRF5rSW3k/FXRE2jCyf30AA7S40=
    # &quot;,&quot;t&quot;:&quot;Rose, Morden Centennial - TheTreeFarm.com&quot;,
    # &quot;mid&quot;:&quot;C40FF4F584971157DD13539B5B80109032967264&quot;,
    # &quot;desc&quot;:&quot;morden rose centennial thetreefarm zoom views&quot;}"
    #  onclick="sj_evt.fire('IFrame.Navigate', this.href); return false;" 
    # href="/images/search?view=detailV2&amp;

if __name__ == "__main__":
    # test_single_img()
    main()