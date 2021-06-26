import scrapy

class WeiBo(scrapy.spiders.Spider):
    name='weibo'
    allowed_domains=['m.weibo.cn']
    start_urls=[
        "https://m.weibo.cn/u/3248617773?uid=3248617773&luicode=10000011&lfid=100103type%3D1%26q%3D%E5%AE%89%E5%BE\
        %BD%E5%A4%A7%E5%AD%A6%E5%9B%BE%E4%B9%A6%E9%A6%86"
    ]


    def parse(self,response):
        filename=response.url.split("/")[-2]
        with open(filename,'wb') as f:
            f.write(response.body)
