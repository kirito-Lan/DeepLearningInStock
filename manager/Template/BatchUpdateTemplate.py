from exception.BusinessException import BusinessException
from manager.Template.MacroBatchUpdate import MacroBatchUpdate
from manager.Template.StockBatchUpdate import StockBatchUpdate


async def batch_update(el_type: str):
    if el_type == "stock":
        template= StockBatchUpdate()
    elif el_type == "macro":
        template= MacroBatchUpdate()
    else:
        raise BusinessException(400, "type is not support")
    return await template.multiple_update_data()