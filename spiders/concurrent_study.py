import time
import threading
import requests
from lxml import etree


headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ('
                  'KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36 Edg/102.0.1245.30'
}


def get_response(url, params=None, headers=None, proxies=None, timeoue=None):
    if headers is None:
        headers = headers
    response = requests.get(url=url,
                            headers=headers,
                            proxies=proxies,
                            timeout=timeoue)
    # print("解析网址: ", response.url)
    # print("响应状态码: ", response.status_code)

    return response


def check_url(domains):
    urls = []
    for domain in domains:
        new_url = None
        try:
            new_url = "https://www." + domain
            response = get_response(url=new_url, timeoue=1)
            urls.append(new_url)
        except:
            pass

    print(f"可访问的url数量为：{len(urls)}")
    return urls


def serial_visit(urls):
    """
    串行访问url
    """
    start = time.time()
    for u in urls:
        response = get_response(url=u, headers=headers)
    end = time.time()
    print("串行耗时：{:2f} s".format(end - start))


class MyThread(threading.Thread):
    def __init__(self, name, delay):
        super(MyThread, self).__init__()
        self.name = name
        self.delay = delay

    def run(self):
        print("Starting " + self.name)
        self.print_time(self.name, self.delay)
        print("Exiting " + self.name)

    @staticmethod
    def print_time(thread_name, delay):
        counter = 0
        while counter < 3:
            time.sleep(delay)
            print(thread_name, time.ctime())
            counter += 1


def concurrent_visit():
    threads = []

    # 创建新线程
    thread1 = MyThread(name='Thread-1', delay=1)
    thread2 = MyThread(name='Thread-2', delay=2)

    # 开启新线程
    thread1.start()
    thread2.start()

    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()

    print("Existing Main Thread")


def run():
    url = "https://www.admin5.com/article/20210108/982811.shtml"
    response = get_response(url=url)
    # 写成这种格式，防止中文乱码
    page = etree.HTML(response.content, parser=etree.HTMLParser(encoding='utf8'))
    div_list = page.xpath('.//div[@class="content"]//p')[2:-1]
    domains = []
    for div in div_list:
        domain = div.xpath('./text()')[0].split(' ')[2]
        domains.append(domain)
    urls = check_url(domains)
    serial_visit(urls)


if __name__ == "__main__":
    # run()
    concurrent_visit()
