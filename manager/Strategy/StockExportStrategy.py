# 针对股票数据的策略
from constant.ExponentEnum import ExponentEnum
from manager import StockDataManage
from manager.Strategy.ExportStrategy import ExportStrategy
import pandas as pd


class StockExportStrategy(ExportStrategy):
    """获取股票的策略"""
    def get_enum(self):
        return ExponentEnum

    async def get_data(self, stock:ExponentEnum, start_date: str, end_date: str) -> pd.DataFrame:
        return await StockDataManage.get_stock_data(stock_code=stock.get_code(), start_date=start_date, end_date=end_date)

    async def convert_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        return await StockDataManage.stock_colum_name_eng2cn(data_frame)

    def get_sheet_name(self, stock:ExponentEnum) -> str:
        return stock.get_name()