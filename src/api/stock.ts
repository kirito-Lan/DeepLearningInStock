
/* eslint-disable */
import request from '@/request'

/** Crawl Stock Data 爬取股票数据
:param body: 请求体
:return: BaseResponse POST /stock/crawlStockData */
export async function crawlStockDataStockCrawlStockDataPost(
  body: API.GetStockRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/stock/crawlStockData', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Stock Csv 下载股票数据
:param body: 请求体
:return: CSV File POST /stock/getStockCsv */
export async function getStockCsvStockGetStockCsvPost(
  body: API.GetStockRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/stock/getStockCsv', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Stock Data 获取股票数据
:param body: 请求体
:return: BaseResponse POST /stock/getStockData */
export async function getStockDataStockGetStockDataPost(
  body: API.GetStockRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/stock/getStockData', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}
