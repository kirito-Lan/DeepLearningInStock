export interface MacroDataRequest {
  types: string
  start_date: string
  end_date: string
}

export interface MacroDataItem {
  report_date: string
  current_value: number
  forecast_value: number
  previous_value: number
}

export interface MacroDataResponse {
  code: number
  msg: string
  data: MacroDataItem[]
}
