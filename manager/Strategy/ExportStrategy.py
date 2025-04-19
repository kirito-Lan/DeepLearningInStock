import abc
import pandas as pd

class ExportStrategy(abc.ABC):
    """策略接口"""
    @abc.abstractmethod
    def get_enum(self):
        """返回需要迭代的枚举类"""
        pass

    @abc.abstractmethod
    async def get_data(self, export_type, start_date: str, end_date: str) -> pd.DataFrame:
        """获取数据，返回 DataFrame"""
        pass

    @abc.abstractmethod
    async def convert_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """将DataFrame的列名转换成中文"""
        pass

    @abc.abstractmethod
    def get_sheet_name(self, export_type) -> str:
        """返回sheet名称"""
        pass