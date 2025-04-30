// @ts-ignore
/* eslint-disable */
import request from '@/request'

/** Batch Export To Excel 批量导出数据到excel
:param body: 请求体
:return: BaseResponse POST /common/batchExportToExcel */
export async function batchExportToExcelCommonBatchExportToExcelPost(
  body: API.BatchExportDataRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/common/batchExportToExcel', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Batch Update Data GET /common/batchUpdate/${param0} */
export async function batchUpdateDataCommonBatchUpdateElTypeGet(
  // 叠加生成的Param类型 (非body参数swagger默认没有生成对象)
  params: API.batchUpdateDataCommonBatchUpdateElTypeGetParams,
  options?: { [key: string]: any }
) {
  const { el_type: param0, ...queryParams } = params
  return request<API.BaseResponse>(`/common/batchUpdate/${param0}`, {
    method: 'GET',
    params: { ...queryParams },
    ...(options || {}),
  })
}
