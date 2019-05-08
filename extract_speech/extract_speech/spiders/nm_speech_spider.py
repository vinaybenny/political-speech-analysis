import os
import scrapy
from scrapy import Request
import json

ALLOWED_DOMAINS = ['www.narendramodi.in']
ROOT_URL = 'https://www.narendramodi.in/speech/loadspeeche?page=%s&language=en'
DATA_DIRECTORY = "../data/data_scraped/"
PAGES_TO_CRAWL = 20
custom_settings = {
        "DOWNLOAD_DELAY": 5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2
    }

class NMSpeechSpider(scrapy.Spider):
    name = "nmspeeches"
    root_url = ROOT_URL

    def start_requests(self):
        for next_page in range(1,PAGES_TO_CRAWL + 1):
            yield scrapy.Request(url = self.root_url % next_page, callback = self.parse_urls)

    def parse_urls(self, response):
        for speech_url in response.xpath('//div/div/*[contains(@class, "speechesItemLink left_class ")]/a/@href').extract():
            yield scrapy.Request(speech_url)

    def parse(self, response):

        path = DATA_DIRECTORY             
        filename = 'speech' + response.url[-30:]
        data = {}        

        with open(path + filename + '.txt', 'w', encoding='utf-8') as datafile:
            data['title'] = response.xpath('//*[@id="detailNews"]//*[@id="article_title"]/descendant-or-self::text()').extract_first()
            data['speechdate'] = response.xpath('//*[@id="detailNews"]/div/div/div/div//*[@class="captionDate"]/text()').extract_first()
            data['url'] = response.url
            data['filename'] = filename
            content = u""           
            for item in response.xpath('//*[@id="detailNews"]/div/div/div//*[@class="articleBody main_article_content"]//*[not(@class)]/descendant-or-self::text()').extract():
                content = content + str(item) + '\r\n'
            
            data['content'] = content
            json.dump(data, datafile, ensure_ascii=False)
                


 