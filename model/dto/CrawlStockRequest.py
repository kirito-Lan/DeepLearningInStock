from pydantic import BaseModel


class CrawlStockRequest(BaseModel):

        """
        爬取数据的请求类
        :param stock_code: 股票代码
        :param start_date: 开始日期   可以是各种的时间日期格格式
        :param end_date: 结束日期
        """
        def __init__(self, stock_code: str, start_date: str, end_date: str):
            super().__init__()
            self.stock_code = stock_code
            self.start_date = start_date
            self.end_date = end_date

