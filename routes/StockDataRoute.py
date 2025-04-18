from types import NoneType
from typing import Dict

from fastapi import APIRouter
from starlette.responses import FileResponse, JSONResponse

from config.LoguruConfig import log
from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from constant.ExponentEnum import ExponentEnum
from manager import StockDataManage
from model.dto.GetStockRequest import GetStockRequest
from utils.ReFormatDate import format_date

router = APIRouter(prefix="/stock", tags=["stock"])


@router.post("/getStockCsv", response_model=BaseResponse)
async def get_stock_csv(body: GetStockRequest):
    """
    下载股票数据
    :param body: 请求体
    :return: CSV File
    """
    # 校验参数
    exponent = ExponentEnum.get_enum_by_code(body.stock_code)
    if exponent is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间格式
    start_date, end_date = format_date(body.start_date, body.end_date)
    log.info("入口参数: stock_code:【】,start_date:【{}】,end_date:【{}】", body.stock_code, start_date, end_date)
    path: str = await StockDataManage.export_to_csv(stock_code=body.stock_code, start_date=start_date,
                                                    end_date=end_date)
    if path is None:
        return BaseResponse[NoneType].fail(ErrorCode.OPERATION_ERROR)
    # 使用 FileResponse 返回文件给浏览器下载
    return FileResponse(path, filename="stock_data.csv", media_type="application/octet-stream")


@router.post("/getStockData", response_model=BaseResponse)
async def get_stock_data(body: GetStockRequest):
    """
    获取股票数据
    :param body: 请求体
    :return: BaseResponse
    """
    # 校验参数
    exponent = ExponentEnum.get_enum_by_code(body.stock_code)
    if exponent is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间格式
    start_date, end_date = format_date(body.start_date, body.end_date)
    log.info("入口参数: stock_code:【{}】,start_date:【{}】,end_date:【{}】", body.stock_code, start_date, end_date)
    stock_data = await StockDataManage.get_stock_data(stock_code=body.stock_code, start_date=start_date,
                                                      end_date=end_date)
    if stock_data.empty:
        return BaseResponse[NoneType].fail(ErrorCode.NOT_FOUND_ERROR)
    return BaseResponse[list[dict]].success(stock_data.to_dict(orient='records'))


@router.post("/crawlStockData", response_model=BaseResponse)
async def crawl_stock_data(body: GetStockRequest):
    """
    爬取股票数据
    :param body: 请求体
    :return: BaseResponse
    """
    # 校验参数
    exponent = ExponentEnum.get_enum_by_code(body.stock_code)
    if exponent is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间格式
    start_date, end_date = format_date(body.start_date, body.end_date)
    log.info("入口参数: stock_code:【{}】,start_date:【{}】,end_date:【{}】", body.stock_code, start_date, end_date)
    crawl_result = await StockDataManage.crawl_sock_data(stock_code=body.stock_code, start_date=start_date,
                                                         end_date=end_date)

    return BaseResponse[NoneType].success(f"爬取数据成功,共更新{crawl_result}条数据")

@router.get("/batchUpdateStockData", response_model=BaseResponse)
async def batch_update_stock():
    res= await StockDataManage.multiple_update_stock_data()
    return BaseResponse[Dict[str,int]].success(res)