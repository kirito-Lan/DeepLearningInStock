import unittest



from config.LoguruConfig import log
from constant.ExponentEnum import ExponentEnum
from constant.MaroDataEnum import DataTypeEnum
from manager import MacroDataManage, StockDataManage
from model.entity.BaseMeta.BaseMeta import database
from model.entity.StockData import StockData
from utils.ReFormatDate import  format_date


class MyTestCase(unittest.IsolatedAsyncioTestCase):
    """ 宏观数据类测试"""

    async def test_save_or_update_macro_data(self):
        """ 保存或更新数据宏观数据测试"""
        await database.connect()
        await MacroDataManage.save_or_update_macro_data(types=DataTypeEnum.PMI)
        await database.disconnect()
        self.skipTest("test_save_or_update_macro_data")

    async def test_csv(self):
        """ csv导出测试"""
        await database.connect()
        start, end = format_date("2020-01-01", "2023-01-01")
        await MacroDataManage.export_to_csv(types=DataTypeEnum.PMI,start_date=start,end_date=end)
        await database.disconnect()
        self.skipTest("test_csv")


    async def test_get_macro_data(self):
        """ 获取宏观数据测试"""
        await database.connect()
        start, end = format_date("2020-01-01", "2023-01-01")
        res = await MacroDataManage.get_macro_data(types=DataTypeEnum.PMI,start_date=start,end_date=end)
        print(res)
        await database.disconnect()
        self.skipTest("test_get_macro_data")


    """股票数据类测试"""
    async def test_get_stock_data(self):
        """测试获取过股票数据"""
        await database.connect()
        start, end = format_date("2020-01-01", "2023-01-01")
        res = await StockDataManage.get_stock_data(stock_code= ExponentEnum.HS300.get_code(),start_date=start,end_date=end)
        print(res)
        await database.disconnect()
        self.skipTest("test_get_stock_data")

    async def test_crawl_stock_data(self):
        """测试获取股票数据"""
        await database.connect()
        start, end = format_date("2020-01-01", "2025-04-17")
        res = await StockDataManage.crawl_sock_data(stock_code= ExponentEnum.HS300.get_code(),end_date=end)
        print(res)
        await database.disconnect()
        self.skipTest("test_crawl_stock_data")

    async def test_export_to_csv(self):
        """测试导出csv"""
        await database.connect()
        start, end = format_date("2020-01-01", "2025-04-17")
        res = await StockDataManage.export_to_csv(stock_code= ExponentEnum.HS300.get_code(),end_date=end)
        print(res)
        await database.disconnect()
        self.skipTest("test_export_to_csv")

if __name__ == '__main__':
    unittest.main()
