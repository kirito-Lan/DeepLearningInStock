// @ts-ignore
/* eslint-disable */
import request from '@/request'

/** Healthcheck GET /check */
export async function healthCheckCheckGet(options?: { [key: string]: any }) {
  return request<API.BaseResponseStr_>('/check', {
    method: 'GET',
    ...(options || {}),
  })
}
