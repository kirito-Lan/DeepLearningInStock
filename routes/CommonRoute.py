import os
from types import NoneType

from fastapi import APIRouter
from starlette.responses import FileResponse

from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from manager.Strategy.ExportStategyManager import batch_export_to_excel_with_strategy
from manager.Template import BatchUpdateTemplate
from model.dto.BatchExportDataRequest import BatchExportDataRequest
from utils.ReFormatDate import format_date

router = APIRouter(prefix="/common", tags=["common"])


@router.post("/batchExportToExcel", response_model=BaseResponse)
async def batch_export_to_excel(body: BatchExportDataRequest):
    """
    批量导出数据到excel
    :param body: 请求体
    :return: BaseResponse
    """
    if body.export_type is None:
        return BaseResponse[NoneType].fail(ErrorCode.PARAMS_ERROR)
    # 格式化时间格式
    start_date, end_date = format_date(body.start_date, body.end_date)
    path = await batch_export_to_excel_with_strategy(export_type=body.export_type,start_date=start_date, end_date=end_date)
    if not os.path.exists(path):
        return BaseResponse[NoneType].fail(ErrorCode.OPERATION_ERROR)

    return FileResponse(path, filename=f"Batch{body.export_type.capitalize()}Data.xlsx", media_type="application/octet-stream")

@router.get("/batchUpdate/{el_type}",response_model=BaseResponse)
async def batch_update_data(el_type:str):
     res = await BatchUpdateTemplate.batch_update(el_type=el_type)
     return BaseResponse[NoneType].success(res)
