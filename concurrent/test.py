import os
import requests
import threading
import time
import random
import queue
from lxml import etree

urls = [f"https://www.cnblogs.com/#p{page}" for page in range(1, 50)]

def func(a, b):
    pass

def craw(url):
    res = requests.get(url)
    html_info = etree.HTML(res.text)
    titles = html_info.xpath("//div[@class='post-item-text']/a")
    hrefs = html_info.xpath("//div[@class='post-item-text']/a/@href")
    titles = [t.text for t in titles]
    
    # for href, title in zip(hrefs, titles):
    #     print(href, title)

    return zip(hrefs, titles)

def single_thread():
    print("single begin")
    for url in urls:
        per_page_infos = craw(url=url)
        for href, title in per_page_infos:
            print(href, title)
    print("single end")

def craw2(url):
    res = requests.get(url)
    return res.text

def parse2(html):
    html_info = etree.HTML(html)
    titles = html_info.xpath("//div[@class='post-item-text']/a")
    hrefs = html_info.xpath("//div[@class='post-item-text']/a/@href")
    titles = [t.text for t in titles]

    return zip(hrefs, titles)

def do_craw(url_queue: queue.Queue, html_queue: queue.Queue):
    while True:
        url = url_queue.get()
        html = craw2(url)
        html_queue.put(html)
        print(threading.current_thread().name, f"craw {url}",
              f"url_queue.size = {url_queue.qsize()}")
        time.sleep(random.randint(1, 2))

def do_parse(html_queue: queue.Queue):
    while True:
        html = html_queue.get()
        results = parse2(html)
        print(threading.current_thread().name,
              f"url_queue.size = {html_queue.qsize()}")
        time.sleep(random.randint(1, 2))

def multi_thread():
    print("multi begin")
    # threads = []
    # for url in urls:
    #     threads.append(threading.Thread(target=craw, args=(url, )))

    # for thread in threads:
    #     thread.start()

    # for thread in threads:
    #     thread.join()

    url_queue = queue.Queue()
    html_queue = queue.Queue()
    for url in urls:
        url_queue.put(url)
    
    # 3个生产者线程
    for idx in range(3):
        t = threading.Thread(target=do_craw, 
                             args=(url_queue, html_queue),
                             name=f"craw{idx}")
        t.start()

    # 2个消费者线程
    for idx in range(2):
        t = threading.Thread(target=do_parse, 
                             args=(html_queue, ),
                             name=f"parse{idx}")
        t.start()
    
    print("multi end")

def main():
    # start = time.time()
    # single_thread()
    # print(f"single  thread cost: {time.time() - start} s")

    start = time.time()
    multi_thread()
    print(f"multi  thread cost: {time.time() - start} s")


if __name__ == "__main__":
    main()
