export interface BaseResponse<T = any> {
  code: number
  msg: string
  data: T
}
