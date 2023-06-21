import scrapy


class Tc58Spider(scrapy.Spider):
    name = "tc58"
    allowed_domains = ["cn.58.com"]
    start_urls = ["https://cn.58.com/sou/?key=%E5%89%8D%E7%AB%AF%E5%BC%80%E5%8F%91"]

    def parse(self, response):
        # 获取网页源码
        content = response.text

        # 获取二进制代码
        binary = response.body

        span_list = response.xpath('//div[@id="filter"]/div[@class="tabs"]//span')
        print(len(span_list), span_list[0].extract())
        print("======")
