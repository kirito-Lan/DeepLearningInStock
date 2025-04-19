from constant.MaroDataEnum import MacroDataEnum
from manager.MacroDataManage import crawl_macro_data
from manager.Template.AbstractBatchUpdater import AbstractBatchUpdater


class MacroBatchUpdate(AbstractBatchUpdater):
    async def get_enum(self):
        return MacroDataEnum

    async def get_data(self, datatype: MacroDataEnum) -> int:
        return await crawl_macro_data(types=datatype)
