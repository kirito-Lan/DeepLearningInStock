import os
from decimal import Decimal, ROUND_HALF_UP
from types import NoneType
from typing import Dict

from fastapi import APIRouter
from starlette.responses import FileResponse


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
    path: str = await StockDataManage.export_to_csv(stock_code=body.stock_code, start_date=start_date,
                                                    end_date=end_date)
    if not os.path.exists(path):
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
    stock_data = await StockDataManage.get_stock_data(stock_code=body.stock_code, start_date=start_date,
                                                      end_date=end_date)
    #时间降序排列
    stock_data.sort_values('trade_date', ascending=False,inplace=True)
    # 对数据进行处理，保留三位小数HalfUp（数据类型是decimal类型）
    # 里面还有时间列，如果遍历到的是时间列就不处理,反之就是进行数据截断
    for column in stock_data.columns[1:]:
        stock_data[column] = stock_data[column].apply(
            lambda x: float(Decimal(str(x)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)))

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
    crawl_result = await StockDataManage.crawl_sock_data(stock_code=body.stock_code, start_date=start_date,
                                                         end_date=end_date)

    return BaseResponse[NoneType].success(f"爬取数据成功,共更新{crawl_result}条数据")

@router.get("/getStockList", response_model=BaseResponse)
async def get_stock_list():
    """获取所有的股票类型"""
    stock_list = [{"name": item.get_name(), "code": item.get_code()} for item in ExponentEnum]
    return BaseResponse[Dict].success(stock_list)
