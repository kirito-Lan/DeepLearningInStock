// @ts-ignore
/* eslint-disable */
import request from '@/request'

/** Crawl Macro Data 爬取宏观经济数据
:param body: 请求体
:return: 更新数据条数 POST /macro/crawlMacroData */
export async function crawlMacroDataMacroCrawlMacroDataPost(
  body: API.GetMacroDataRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/macro/crawlMacroData', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Macro Csv 下载宏观数据文件
:param body: 请求体
:return: 文件流 POST /macro/getMacroCsv */
export async function getMacroCsvMacroGetMacroCsvPost(
  body: API.GetMacroDataRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/macro/getMacroCsv', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Macro Data 获取宏观经济数据
:param body:
:return: POST /macro/getMacroData */
export async function getMacroDataMacroGetMacroDataPost(
  body: API.GetMacroDataRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/macro/getMacroData', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}
