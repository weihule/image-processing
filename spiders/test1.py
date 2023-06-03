import requests
from lxml import etree
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import json


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


def get_driver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    # 无界面
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    options.binary_location = path

    service = Service(r"D:\ProgramFiles\anaconda3\envs\torch12\chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    return driver


def test02():
    driver = get_driver()
    url = "https://www.baidu.com/"
    driver.get(url)
    time.sleep(1)
    driver.save_screenshot("baidu.png")

    input_btn = driver.find_element(By.ID, 'kw')

    # 在文本框中输入周杰伦
    input_btn.send_keys('周杰伦')
    time.sleep(1)

    # 获取百度一下的按钮，并点击回车
    enter_btn = driver.find_element(By.ID, 'su')
    enter_btn.click()
    time.sleep(1)

    # 滑到底部
    js_bottom = 'document.documentElement.scrollTop=100000'
    driver.execute_script(js_bottom)
    time.sleep(1)

    # 点击下一页
    next_btn = driver.find_element(By.XPATH, '//a[@class="n"]')
    next_btn.click()
    time.sleep(3)

    # 回到上一页
    driver.back()
    time.sleep(3)

    # 继续回到下一页
    driver.forward()
    time.sleep(2)

    driver.close()


headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
}


def test03():
    # ? 可以删除
    url = "https://www.baidu.com/s?"
    datas = {'wd': '北京'}
    response = requests.get(url=url, params=datas, headers=headers)
    text = response.text
    print(text)


def test04():
    url = "https://fanyi.baidu.com/sug"
    datas = {'kw': 'spider'}

    response = requests.post(url=url, headers=headers, data=datas)
    print(json.loads(response.text))


def test05():
    url = "https://www.baidu.com/s?"
    datas = {'wd': 'ip'}
    proxy = {'http': '60.167.21.105:1133'}

    response = requests.get(url=url, params=datas, headers=headers, proxies=proxy)
    text = response.text
    print(text)
    with open("./htmls/daili.html", 'w', encoding='utf-8') as fw:
        fw.write(text)


def poetry():
    s = "https://so.gushiwen.cn/user/login.aspx?from=http://so.gushiwen.cn/user/collect.aspx"
    s2 = "https://so.gushiwen.cn/user/collect.aspx"
    url = s
    # response = requests.post(url=url, headers=headers)    # post请求也能实现
    response = requests.get(url=url, headers=headers)
    content = response.text
    tree = etree.HTML(content)
    view_state = tree.xpath('//input[@name="__VIEWSTATE"]/@value')
    view_state_generator = tree.xpath('//input[@name="__VIEWSTATEGENERATOR"]/@value')

    # 获取验证码图片
    code = tree.xpath('//img[@id="imgCode"]/@src')
    code_url = "https://so.gushiwen.cn" + code[0] if code else ''

    # 保存验证码图片(有坑)
    # pic = requests.get(code_url, headers=headers)
    # with open("code.jpg", "wb") as f:
    #     f.write(pic.content)

    # 通过session的返回值，让请求变成一个对象
    session = requests.session()
    pic = session.get(code_url)
    pic_code = pic.content     # 二进制
    with open("code.jpg", "wb") as f:
        f.write(pic_code)

    code_value = input("请输入验证码: ")

    datas = {
        '__VIEWSTATE': view_state,
        '__VIEWSTATEGENERATOR': view_state_generator,
        'from': 'http://so.gushiwen.cn/user/collect.aspx',
        'email': '1849659185@qq.com',
        'pwd': 'whl000113',
        'code': code_value,
        'denglu': '登录'
    }

    url_login = "https://so.gushiwen.cn/user/login.aspx?from=http%3a%2f%2fso.gushiwen.cn%2fuser%2fcollect.aspx"
    # response_login = requests.post(url=url_login, headers=headers, data=datas)
    # TODO: 这里要用上面的session去完成
    response_login = session.post(url=url_login, headers=headers, data=datas)

    text = response_login.text
    with open("./htmls/poetry.html", 'w', encoding='utf-8') as fw:
        fw.write(text)


if __name__ == "__main__":
    # countries()
    # test05()
    poetry()
