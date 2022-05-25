from email import header
import os
from urllib import response
import requests

"""
查询数据时, 如果输入查询词条(或者点击回车)之后,页面上方的url不变, 说明这个页面是有
局部刷新, 也就是ajax请求, 那么在 Network 选项里,name 栏里 sug 这一行, 点进去,
'General'里有'Request URL', 'Request URL',,在 response header 里
可以找到返回类型 'Content-Type'
"""

# requests 第一血
def main():
    url = 'https://www.sogou.com/'
    response = requests.get(url)    # 返回一个响应对象
    page_text = response.text   # 获取相应数据

    # with open('test.html', 'a', encoding='utf-8') as f:
    #     f.write(page_text)

    print(page_text)

headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                    AppleWebKit/537.36 (KHTML, like Gecko) \
                    Chrome/101.0.4951.64 Safari/537.36 Edg/101.0.1210.53'
}

# 爬取搜狗指定词条对应的搜索结果(简易网页采集器)
# UA伪装: User-Agent
def requests_get():
    url = 'https://www.sogou.com/web'
    param = {
        'query': '波晓张',
    }
    respone = requests.get(url, param, headers=headers)
    page_text = respone.text
    print(page_text)


# 破解百度翻译
def requests_post():
    url = 'https://fanyi.baidu.com/sug'

    words = ['rabbit', 'response', 'dog', 'love']
    for word in words:
        data = {'kw': word}
        response = requests.post(url=url, data=data, headers=headers)
        res = response.json()   # dict 格式
        print(res)


# 爬取豆瓣电影分类排行榜, https://movie.douban.com/ 中的电影详情数据
def douban():
    url = 'https://movie.douban.com/j/chart/top_list'
    params = {
        'type': '24',
        'interval_id': '100:90',
        'action': '',
        'start': '0',
        'limit': '20'
    }
    response = requests.get(url=url, params=params, headers=headers)
    res = response.json()
    print(res)


# 爬取肯德基餐厅查询 http://www.kfc.com.cn/kfccda/index.aspx 中指定地点的餐厅数量
def kfc():
    url = 'http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx'
    page_indexs = [str(c+1) for c in range(6)]
    for page_index in page_indexs:
        params = {
            'op': 'keyword',
            'cname': '',
            'pid': '', 
            'keyword': '北京',
            'pageIndex': page_index,
            'pageSize': '10'
        }
        response = requests.post(url=url, data=params, headers=headers)

        print(response.text)


# 爬取国家药品监督管理总局中基于中华人民共和国化妆品生产许可证相关数据 http://scxk.nmpa.gov.cn:81/xk/
def medical():
    url = 'http://scxk.nmpa.gov.cn:81/xk/itownet/portalAction.do'
    params = {
        # 'hKHnQfLv': '5DcJ_zIb9XzDA8mMOPJTIaI_P3lCU9la7Tid7neolW0lK69BPG0WjmbSbLMVkhEXz1RdM12eIKZJ_pNEuGajFfqosYTRob0.97utUbmIregLgcJr1sAg1qy0lVUZCVhslOqx66A3PRdMVP178sz9O6pYAJx2c0BENbgPnNlPstYiChbKJUj4cEqor3rq3AmGEhUoVRqaj5NZA_cbOnAmaY6vEt8RApHyNpffCWk06SzVPQ7I4t32olCWM30W0bnAEsfisQ4cyp4VjJHy4D6b1gr_5StuKcJHH3secIGJMRG7',
        # '8X7Yi61c': '4x7RSXV9pH7NwA1r99U66cFwafhzFRgRpfq0M3aTdbrJZZA6z7iKFrdMWXbYGSDaaPYfEGHgRF.uUvvOjERPFWqMobGKxN5DxYIll8Y44S7shche7InznFDEgX_Vaovk0',
        'on': 'true',
        'page': '1',
        'pageSize': '15',
        'productName': '',
        'conditionType': '1',
        'applyname': '',
        'applysn':''
    }
    response = requests.post(url=url, data=params, headers=headers)
    print(response.json())


if __name__ == '__main__':
    # main()
    # requests_get()
    # requests_post()
    # douban()
    # kfc()
    medical()