import scrapy


class D1Spider(scrapy.Spider):
    name = "d1"
    allowed_domains = ["www.dongchedi.com"]
    start_urls = ["https://www.dongchedi.com/sales/sale-energy_1-x-x-x-x-x"]

    def parse(self, response):
        # cars = response.xpath('//div[@class="brand-rec-box"]/ul')
        cars = response.xpath('//ol[@class="tw-mt-12"]/li')
        print(len(cars))
