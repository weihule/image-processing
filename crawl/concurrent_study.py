import time
import threading
import requests
import queue
from lxml import etree
from typing import List


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

    length = len(urls)
    print(f"可访问的url数量为：{length}")
    return urls, length


def serial_visit(urls):
    """
    串行访问url
    """
    start = time.time()
    for u in urls:
        response = get_response(url=u, headers=headers)
    end = time.time()
    print("串行耗时：{:2f} s".format(end - start))


class MyThreadTest(threading.Thread):
    def __init__(self, name, delay):
        super(MyThreadTest, self).__init__()
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


def concurrent_test():
    threads = []

    # 创建新线程
    thread1 = MyThreadTest(name='Thread-1', delay=1)
    thread2 = MyThreadTest(name='Thread-2', delay=2)

    # 开启新线程
    thread1.start()
    thread2.start()

    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()

    print("Existing Main Thread")


class MyThread(threading.Thread):
    def __init__(self, name, link_range, urls):
        super(MyThread, self).__init__()
        self.name = name
        self.link_range = link_range
        self.urls = urls

    def run(self):
        print("Starting " + self.name)
        self.crawl(self.name, self.link_range)
        print("Exiting " + self.name)

    def crawl(self, thread_name, link_range):
        for i in range(link_range[0], link_range[1]):
            try:
                r = get_response(url=self.urls[i], headers=headers)
            except Exception as e:
                print(thread_name, "Error: ", e)


def concurrent_visit(urls, length):
    start = time.time()
    threads = []
    thread_num = 4
    link_range_list = [(s, min(s+length//thread_num, length))
                       for s in range(0, length, length//thread_num)]

    # 创建新线程
    for idx_thread in range(thread_num):
        thread = MyThread(name="Thread-"+str(idx_thread+1),
                          link_range=link_range_list[idx_thread],
                          urls=urls)

        thread.start()

        threads.append(thread)

    # 等待所有线程完成
    for t in threads:
        t.join()

    end = time.time()
    print("简单多线程爬虫耗时: {:.2f} s".format(end - start))
    print("Existing Main Thread")


temp = []


class ThreadQueue(threading.Thread):
    def __init__(self, name, q):
        super(ThreadQueue, self).__init__()
        self.name = name
        self.q = q

    def run(self):
        print("Starting " + self.name)
        while True:
            try:
                self.crawl(self.name, self.q)
            except:
                break
        print("Exiting " + self.name)

    @staticmethod
    def crawl(thread_name, q: queue.Queue):
        url = q.get(timeout=1.5)
        temp.append(url)
        try:
            r = get_response(url=url, timeoue=1.5)
        except Exception as e:
            print(thread_name, "Error: ", e)


def concurrent_queue(urls: List):
    start = time.time()
    # 填充队列
    work_queue = queue.Queue()
    for url in urls:
        work_queue.put(url)

    threads = []
    # 创建4个线程
    for i in range(4):
        thread = ThreadQueue(name="Thread-"+str(i+1),
                             q=work_queue)
        # 开启新线程
        thread.start()

        # 添加新线程到线程列表
        threads.append(thread)

    # 等到所有线程完成
    for t in threads:
        t.join()

    end = time.time()
    print("Queue多线程爬虫耗时: {:.2f} s".format(end - start))

    print(len(temp))


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
    urls, len_urls = check_url(domains)

    # 串行
    serial_visit(urls)

    # 简单多线程
    concurrent_visit(urls, len_urls)

    # queue多线程
    concurrent_queue(urls)


if __name__ == "__main__":
    run()
    # concurrent_test()
