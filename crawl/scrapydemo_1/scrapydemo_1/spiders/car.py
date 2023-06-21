import scrapy


class CarSpider(scrapy.Spider):
    name = "car"
    allowed_domains = ["sou.autohome.com.cn"]
    start_urls = ["https://sou.autohome.com.cn/zonghe?q=%c6%e6%c8%f0"]

    def parse(self, response):
        cars = response.xpath('//div[@class="brand-rec-box"]/ul')
        print("="*5)
        for ul in cars:
            for li in ul.xpath('./li'):
                ps = li.xpath('./p')
                p1 = ps[0].xpath('./text()').extract()
                p2 = ps[1].xpath('./a/text()').extract()
                p3 = ps[2].xpath('./text()').extract()
                price = ps[2].xpath('./a/text()').extract()
                print(p1, p2, p3, price)
                # break
            # break
        print("=" * 5)
