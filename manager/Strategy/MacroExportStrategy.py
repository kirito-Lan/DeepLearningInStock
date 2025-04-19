import pandas as pd

from constant.MaroDataEnum import MacroDataEnum
from manager import MacroDataManage
from manager.Strategy.ExportStrategy import ExportStrategy


class MacroExportStrategy(ExportStrategy):
    """获取宏观数据的策略"""
    def get_enum(self):
        return MacroDataEnum

    async def get_data(self, macro_data:MacroDataEnum, start_date: str, end_date: str) -> pd.DataFrame:
        return await MacroDataManage.get_macro_data(types=macro_data, start_date=start_date, end_date=end_date)

    async def convert_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        return await MacroDataManage.macro_colum_nam_eng2cn(data_frame)

    def get_sheet_name(self, macro_data:MacroDataEnum) -> str:
        return macro_data.display_name