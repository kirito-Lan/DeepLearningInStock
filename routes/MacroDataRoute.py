from types import NoneType


from fastapi import APIRouter
from starlette.responses import FileResponse

from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from constant.MaroDataEnum import DataTypeEnum
from manager import MacroDataManage
from model.dto.GetMacroDataRequest import GetMacroDataRequest
from utils.ReFormatDate import format_date

router = APIRouter(prefix="/macro", tags=["macro"])


@router.post("/getMacroData", response_model=BaseResponse)
async def get_macro_data(body: GetMacroDataRequest):
    """
    获取宏观经济数据
    :param body:
    :return:
    """
    # 获取枚举类
    types = DataTypeEnum.get_enum_by_name(body.types)
    if types is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间
    start_date, end_date = format_date(body.start_date, body.end_date)
    macro_data = await MacroDataManage.get_macro_data(types=types, start_date=start_date,
                                                      end_date=end_date)
    if macro_data.empty:
        return BaseResponse[NoneType].fail(ErrorCode.NOT_FOUND_ERROR)
    return BaseResponse.success(macro_data.to_dict(orient='records'))


@router.post("/getMacroCsv", response_model=BaseResponse)
async def get_macro_csv(body: GetMacroDataRequest):
    """
    下载宏观数据文件
    :param body: 请求体
    :return: 文件流
    """
    types = DataTypeEnum.get_enum_by_name(body.types)
    if types is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    start_date, end_date = format_date(body.start_date, body.end_date)
    path = await MacroDataManage.export_to_csv(types=types, start_date=start_date, end_date=end_date)
    if path is None:
        return BaseResponse[NoneType].fail(ErrorCode.NOT_FOUND_ERROR)
    return FileResponse(path=path, filename="macro_data.csv", media_type="application/octet-stream")

@router.post("/crawlMacroData", response_model=BaseResponse)
async def crawl_macro_data(body: GetMacroDataRequest):
    """
    爬取宏观经济数据
    :param body: 请求体
    :return: 更新数据条数
    """

    types = DataTypeEnum.get_enum_by_name(name=body.types)
    if types is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    count = await MacroDataManage.crawl_macro_data(types=types)
    return BaseResponse[NoneType].success(f"爬取数据成功,共更新{count}条数据")