from pydantic import BaseModel


class GetStockRequest(BaseModel):
    """
    爬取数据的请求类
    :param stock_code: 股票代码
    :param start_date: format: YYYY-MM-DD
    :param end_date: format: YYYY-MM-DD
    """
    stock_code: str
    start_date: str
    end_date: str


