# -*- coding: utf-8 -*-
"""
@created: 23 April 2019
@author: vinay.benny
@description: This spider is designed to crawl the domain mentioned under "ALLOWED_DOMAINS" declared below, to extract speech 
    transcripts and other related details. The design of the crawler is based on the HTML structure of the target domains as on
    the creation date. The spider parses the HTML content, and creates a JSON dump of the page with the parsed content and
    other metadata. Currently the JSON file will contain the speech title, the date of speech, the URL to the page, the JSON filename,
    and the content of the speech.
    
    The spider accepts the following command line parameters:
        1. "from_date" and "to_date": filters the domain for all speeches between these dates(both inclusive).
        2. "pages_to_crawl": the number of pages to crawl within the result pages fetched based on input from_date and to_date,
        3. "min_paragraph_size": all paragraphs of the speech less than this length is treated as part of the currently parsed paragraph.
"""
import os
import scrapy
import json
from bs4 import BeautifulSoup


# Global constants for the spider
ALLOWED_DOMAINS = ['www.narendramodi.in']
ROOT_URL = 'https://www.narendramodi.in/speech/searchspeeche?language=en&page=%s&keyword=&fromdate=%s&todate=%s'
DATA_DIRECTORY = "../data/data_scraped/"    # Target directory for JSON files
FROM_DATE = '01/01/2018'                    # Default from_date for filtering speeches based on date
TO_DATE = '01/31/2018'                      # Default to_date for filtering speeches based on date
PAGES_TO_CRAWL = 30                         # Default number of pages to crawl, within the results fetched by date filters 
MIN_PARAGRAPH_SIZE = 200                    # Default mininmum paragraph size, below which the paragraphs are treated as part of current paragraph 
custom_settings = {
        "DOWNLOAD_DELAY": 5,                # Delay (in seconds) between successive hits on the domain
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2
    }

class NMSpeechSpider(scrapy.Spider):
    name = "nmspeeches"
    root_url = ROOT_URL
    
    def __init__(self, pages_to_crawl=PAGES_TO_CRAWL, min_paragraph_size=MIN_PARAGRAPH_SIZE, from_date=FROM_DATE, to_date=TO_DATE, *args, **kwargs):
        super(NMSpeechSpider, self).__init__(*args, **kwargs)
        self.pages_to_crawl = pages_to_crawl
        self.min_paragraph_size = min_paragraph_size
        self.from_date = from_date
        self.to_date = to_date        

    def start_requests(self):
        for next_page in range(1,self.pages_to_crawl + 1):
            yield scrapy.Request(url = self.root_url % (next_page, self.from_date, self.to_date), callback = self.parse_urls)

    def parse_urls(self, response):
        for speech_url in response.xpath('//div/div/*[contains(@class, "speechesItemLink left_class ")]/a/@href').extract():
            yield scrapy.Request(speech_url)

    def parse(self, response):
        # Create folders based on dates of speeches
        filepath = DATA_DIRECTORY + self.from_date.replace('/', '') + "-" + self.to_date.replace('/', '')
        filename = 'speech' + response.url[-40:]  
        data = {}
        soup = BeautifulSoup(response.body, 'lxml')        
        
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        with open(filepath + "/" + filename + '.txt', 'w', encoding='utf-8') as datafile:
            data['title'] = response.xpath('//*[@id="detailNews"]//*[@id="article_title"]/descendant-or-self::text()').extract_first()
            data['speechdate'] = response.xpath('//*[@id="detailNews"]/div/div/div/div//*[@class="captionDate"]/text()').extract_first()
            data['url'] = response.url
            data['filename'] = filename
            content = u""
            
            # For speech content, we have 2 rules:
            #   1.Check inside <article> tag with appropriate class name, and find all direct child paragraphs OR
            #   2.Check for div tags, with tag class name "news-bg"
            #   3.Check for unordered list <ul> tags and extract text in these lists.
            #   4.Check for ordered list <ol> tags and extract text in these lists.
            paragraphs = soup.find('article', class_= "articleBody main_article_content").find_all( \
                                  lambda tag: tag and tag.name=="p" \
                                  or (tag.name=="div" and tag.has_attr("class") and "news-bg" in tag["class"]) \
                                  or (tag.name=="ul")
                                  or (tag.name=="ol")
                                  , recursive=False)    
            
            for item in paragraphs:
                content = content + str(item.text) +  ('\r\n' if len(str(item.text)) > self.min_paragraph_size else ' ' )
            
            data['content'] = content
            json.dump(data, datafile, ensure_ascii=False)
                


 