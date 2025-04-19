from abc import ABC
from datetime import datetime



from constant.ExponentEnum import ExponentEnum
from manager.StockDataManage import  crawl_sock_data
from manager.Template.AbstractBatchUpdater import AbstractBatchUpdater


class StockBatchUpdate(AbstractBatchUpdater):


    async def get_enum(self):
        return ExponentEnum

    async def get_data(self, stock: ExponentEnum) -> int:
        """
        获取指数数据
        :param stock: ExponentEnum
        :return: 指数数据
        """
        return await crawl_sock_data(stock_code=stock.get_code(), start_date="2000-01-01", end_date=datetime.now().strftime("%Y-%m-%d"))