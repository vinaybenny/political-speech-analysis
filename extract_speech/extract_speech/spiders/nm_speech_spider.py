import os
import scrapy
from scrapy import Request

ALLOWED_DOMAINS = ['www.narendramodi.in']
ROOT_URL = 'https://www.narendramodi.in/speech/loadspeeche?page=%s&language=en'

custom_settings = {
        "DOWNLOAD_DELAY": 5,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2
    }

class NMSpeechSpider(scrapy.Spider):
    name = "nmspeeches"
    root_url = ROOT_URL

    def start_requests(self):
        for next_page in range(10):
            yield scrapy.Request(url = self.root_url % next_page, callback = self.parse_urls)

    def parse_urls(self, response):
        for speech_url in response.xpath('//div/div/*[contains(@class, "speechesItemLink left_class ")]/a/@href').extract():
            yield scrapy.Request(speech_url)

    def parse(self, response):

        path = './data/'                
        metafilename = 'mapping.txt'
        filename = 'speech' + response.url[-30:]

        if os.path.exists(path + metafilename):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        with open(path + metafilename, append_write, encoding='utf-8') as metafile:
            metafile.write("%s:%s\r\n" % (filename, response.url) )

        with open(path + filename + '.txt', 'w', encoding='utf-8') as datafile:
            for item in response.xpath('//*[@id="detailNews"]/div/div/div/article/p/descendant-or-self::text()').extract():
                datafile.write("%s\r\n" % item)
                


            


