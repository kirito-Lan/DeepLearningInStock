import abc
import time
from typing import Dict

from config.LoguruConfig import log


class AbstractBatchUpdater(abc.ABC):


    def __init__(self):
        pass


    async def multiple_update_data(self) -> Dict[str, int]:
        """
        批量爬取各个股票指数数据：
          - 遍历 ExponentEnum 中的每个股票指数
          - 每次调用爬取接口，如果成功记录更新条数；
            如果失败则在结果字典中记录为 -1（并不中断其他股票更新）
          - 对初次失败（记录为 -1）的股票进行最后一次重试，
            如果重试成功则更新结果字典中的值
        返回结果字典形如：{"上证指数": 100, "深证成指": -1, ...}
        """
        update_results: Dict[str, int] = {}
        error_stocks = []
        # 初次爬取
        for _enum in await self.get_enum():
            try:
                count = await self.get_data(_enum)
                update_results[_enum.get_name()] = count
                log.info(f"{_enum.get_name()} 更新成功，条数: {count}")
                # 休眠 2 秒，避免触发风控
                time.sleep(2.0)
            except Exception as e:
                log.info(f"{_enum.get_name()} 初次更新失败，错误: {e}")
                update_results[_enum.get_name()] = -1
                error_stocks.append(_enum)

        # 对更新失败的股票进行最后一次重试
        if error_stocks:
            log.info("开始对失败股票进行最后一次重试...")
            for _enum in error_stocks:
                try:
                    count = await self.get_data(_enum)
                    update_results[_enum.get_name()] = count
                    log.info(f"重试成功：{_enum.get_name()} 更新成功，条数: {count}")
                    time.sleep(2.0)
                except Exception as e:
                    log.exception(f"重试失败：{_enum.get_name()} 保持 -1，错误: {e}")
        return update_results

    @abc.abstractmethod
    async def get_enum(self):
        pass


    @abc.abstractmethod
    async def get_data(self,datatype):
        pass