import requests
from lxml import etree


def main():
    tree = etree.parse("./htmls/1.html")

    # 查找所有li标签
    lis = tree.xpath("//ul/li")
    print(lis, len(lis))

    # 查找所有带id的li标签
    lis2 = tree.xpath("//ul/li[@id]")
    print(lis2)

    # 查找所有带id的li标签，并查找其内容
    lis3 = tree.xpath("//ul/li[@id]/text()")
    print(lis3)

    # 查找id为1的 li 标签
    lis4 = tree.xpath("//ul/li[@id='1']/text()")
    print(lis4)

    # 查找id为1的li标签的class的属性值
    lis5 = tree.xpath("//ul/li[@id=1]/@class")
    print(lis5)

    # 查找id里包含l的li标签，并显示其内容
    lis6 = tree.xpath("//ul/li[contains(@id, 'l')]/test()")


def test01():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    url = "https://www.baidu.com/"
    res = requests.get(url=url, headers=headers)
    page_text = res.text

    tree = etree.HTML(page_text)
    info = tree.xpath('//a[@id="aging-total-page" and @role="pagedescription"]/@aria-label')
    print(info)

    info2 = tree.xpath('//a[@id="aging-total-page" and @role="pagedescription"]/@aria-label')
    print(info2)

    title = tree.xpath('/html/head/title/text()')
    '/html/head/title/text()'
    '//title/text()'
    print("title = ", title)

    lis = tree.xpath('//div[@class="hot-news-wrapper"]//ul[@class="s-news-rank-content"]//li')
    lis = tree.xpath('//*[@id="s_xmancard_news_new"]/div/div[1]/div/div/ul/li')
    print(len(lis))

    li = tree.xpath('//*[@id="s_xmancard_news_new"]/div/div[1]/div/div/ul/li[1]')
    print(li)


def countries():
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
    }
    url = 'https://www.aqistudy.cn/historydata/'
    res = requests.get(url=url, headers=headers)
    page_text = res.text
    tree = etree.HTML(page_text)

    all_cities = tree.xpath('//div[@class="all"]//ul[@class="unstyled"]//li')
    for i in all_cities:
        print(i.xpath('./a/text()'))


if __name__ == "__main__":
    # countries()
    test01()

