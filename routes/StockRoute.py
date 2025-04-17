from types import NoneType

from fastapi import APIRouter
from starlette.responses import FileResponse

from config.LoguruConfig import log
from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from constant.ExponentEnum import ExponentEnum
from manager import StockDataManage
from model.dto.GetStockRequest import GetStockRequest
from utils.ReFormatDate import format_date

router = APIRouter(prefix="/stock", tags=["stock"])


@router.post("/getStockCsv", response_model=BaseResponse)
async def get_stock_csv(body: GetStockRequest):  # 参数校验

    exponent = ExponentEnum.get_enum_by_code(body.stock_code)
    if exponent is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间格式
    start_date, end_date = format_date(body.start_date, body.end_date)
    log.info("入口参数: stock_code:【】,start_date:【{}】,end_date:【{}】", body.stock_code, start_date, end_date)
    path:str = await StockDataManage.export_to_csv(stock_code=body.stock_code, start_date=start_date, end_date=end_date)
    if path is None:
        return BaseResponse[NoneType].fail(ErrorCode.OPERATION_ERROR)
    # 使用 FileResponse 返回文件给浏览器下载
    return FileResponse(path, filename="stock_data.csv", media_type="application/octet-stream")
