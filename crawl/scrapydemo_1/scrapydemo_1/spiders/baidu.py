import scrapy


class BaiduSpider(scrapy.Spider):
    name = "baidu"
    allowed_domains = ["www.baidu.com"]
    start_urls = ["https://www.baidu.com"]

    # Ïàµ±ÓÚ requests.get()
    def parse(self, response):
        print("------")
