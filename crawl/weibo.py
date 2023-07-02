import requests
from lxml import etree
import json

common_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) A'
                  'ppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58'
}

cookies = {
    "cookie": 'MLOGIN=0; _T_WM=62246799828; WEIBOCN_FROM=1110106030; XSRF-TOKEN=11c20e; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D102803%26uicode%3D20000174'
}


def get_response(url, params=None, headers=None, proxies=None, timeoue=None, cookies=None):
    response = requests.get(url=url,
                            headers=headers,
                            proxies=proxies,
                            timeout=timeoue,
                            cookies=None)
    print("解析网址: ", response.url)
    print("响应状态码: ", response.status_code)

    return response


def main():
    url = 'https://m.weibo.cn/'
    url = 'https://www.baidu.com/'
    response = get_response(url=url,
                            headers=common_headers,
                            cookies=None)
    page_text = response.text
    page_text = etree.HTML(page_text)

    lis = page_text.xpath('.//ul[@class="s-news-rank-content"]/li')

    dict1 = {'#学生高考成绩被屏蔽老师激动欢呼#': [
        '6月24日，安徽合肥。高考查分时显示学生成绩被屏蔽，老师们激动欢呼疯狂鼓掌：全省前30名，已查到两个学生成绩被屏蔽。',
        '天呐[哈哈][哈哈]像我这种学渣都不知道还有屏蔽成绩这么一回事。这些人真的好优秀吖！怪不得老师们会激动吖！我都替他们激动[鼓掌][鼓掌][鼓掌] ​',
        '那个前几天考了700多的成绩怎么没屏蔽呀？700多都不是省前30名吗？',
        '说实话前30的话，清北应该稳了恭喜啦，大家都来吸吸四川理科状元的喜气吧，祝愿大家都能金榜题名[爱你][爱你]',
        '今天广东公布成绩，侄子642，985稳了。正常发挥不错。关键是没花钱补课[嘻嘻]。',
        '涨知识了，高考查分，原来最令人激动的成绩，是看不到成绩。[笑cry]',
        '各地高考分数纷纷出炉，有喜有忧！我才知道考得好的成绩，分数会被屏蔽，这2G网[允悲][允悲][允悲]',
        '对学子来说，如果人生有一次很愿意被PB的机会，估计就是高考查分的时候吧…… ​',
        '老师们都希望自己的学生厉害，付出了很多心血，陪他们一点点成长，跟自己孩子考上了一样开心 ',
        '北京今年继续不公布高考前20名考生成绩，前20考生查分时将收到一句话',
        '看到这个话题的时候为他们感到开心，看来这位学生可以任意选择学校了，这对于老师来说也是一个激动人心的时刻，他的教学有了成果。 ​',
        '6月24日，安徽合肥。高考查分时显示学生成绩被屏蔽，老师们激动欢呼疯狂鼓掌：全省前30名，已查到两个学生成绩被屏蔽。',
        '学生金榜题名时，老师快乐欢呼生。',
        '这应该是高考的最高境界，当你看不到自己的分数时，证明你可以随便报任何一个学校。不知道大家懂没懂？ ',
        '刚刷到这个视频 恭喜！',
        '分数会比你预期的高上50分',
        '无论怎么说高考还是相对来说最能体现平等的一个门槛',
        '早些时候是不是没有屏蔽这个说法的，也有可能自己学渣，没见识过[允悲] ​',
        '恭喜啊啊啊好牛！ ​',
        '我刚刚也查了一下成绩，这个分数我可以上什么学校呢[doge]',
        '今天广东公布成绩，侄子642，985稳了。正常发挥不错。关键是没花钱补课[嘻嘻]']}
    for k, v in dict1.items():
        print(k)
        for i in v:
            print(i)

    # items = page_text.xpath('.//div[@class="wb-item-wrap"]')
    # print(items)
    # for item in items:
    #     a = item.xpath('.//div[@class="m-avatar-box m-box-center"]/@class')
    #     text = item.xpath('.//div[@class="weibo-text"]')
    #     print(a)


if __name__ == "__main__":
    main()

# import json
# import csv
# import re
# import requests
# import time
#
#
# # 获取网页源码的文本文件
# def get_html(url):
#     headers = {
#         "User-Agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36 Edg/114.0.1823.58',
#         "Referer": "https://m.weibo.cn"
#     }
#     cookies = {
#         "cookie": 'MLOGIN=0; _T_WM=62246799828; WEIBOCN_FROM=1110106030; XSRF-TOKEN=11c20e; M_WEIBOCN_PARAMS=luicode%3D10000011%26lfid%3D102803%26uicode%3D20000174'
#     }
#     response = requests.get(url, headers=headers, cookies=cookies)
#     response.encoding = response.apparent_encoding
#     time.sleep(3)   # 加上3s 的延时防止被反爬
#     return response.text
#
#
# def get_string(text):
#     t = ''
#     flag = 1
#     for i in text:
#         if i == '<':
#             flag = 0
#         elif i == '>':
#             flag = 1
#         elif flag == 1:
#             t += i
#     return t
#
#
# # 保存评论
# def save_text_data(text_data):
#     text_data = get_string(text_data)
#     with open("data.csv", "a", encoding="utf-8", newline="")as fi:
#         fi = csv.writer(fi)
#         fi.writerow([text_data])
#
#
# # 获取二级评论
# def get_second_comments(cid):
#     max_id = 0
#     max_id_type = 0
#     url = 'https://m.weibo.cn/comments/hotFlowChild?cid={}&max_id={}&max_id_type={}'
#     while True:
#         response = get_html(url.format(cid, max_id, max_id_type))
#         content = json.loads(response)
#         comments = content['data']
#         for i in comments:
#             text_data = i['text']
#             save_text_data(text_data)
#         max_id = content['max_id']
#         max_id_type = content['max_id_type']
#         if max_id == 0:		# 如果max_id==0表明评论已经抓取完毕了
#             break
#
#
# # 获取一级评论
# def get_first_comments(mid):
#     max_id = 0
#     max_id_type = 0
#     url = 'https://m.weibo.cn/comments/hotflow?id={}&mid={}&max_id={}&max_id_type={}'
#     while True:
#         response = get_html(url.format(mid, mid, max_id, max_id_type))
#         content = json.loads(response)
#         max_id = content['data']['max_id']
#         max_id_type = content['data']['max_id_type']
#         text_list = content['data']['data']
#         for text in text_list:
#             text_data = text['text']
#             total_number = text['total_number']
#             if int(total_number) != 0:  # 如果有二级评论就去获取二级评论。
#                 get_second_comments(text['id'])
#             save_text_data(text_data)
#         if int(max_id) == 0:    # 如果max_id==0表明评论已经抓取完毕了
#             break
#
#
# if __name__ == '__main__':
#     mid = ["4635408392523906"]
#     for id in mid:
#         get_first_comments(id)    # 爬取一级评论


