declare namespace API {
  type BaseResponse = {}

  type BaseResponseStr_ = {}

  type BatchExportDataRequest = {
    /** Export Type */
    export_type: string
    /** Start Date */
    start_date: string
    /** End Date */
    end_date: string
  }

  type batchUpdateDataCommonBatchUpdateElTypeGetParams = {
    el_type: string
  }

  type GetMacroDataRequest = {
    /** Types */
    types?: string
    /** Start Date */
    start_date?: string
    /** End Date */
    end_date?: string
  }

  type GetStockRequest = {
    /** Stock Code */
    stock_code: string
    /** Start Date */
    start_date: string
    /** End Date */
    end_date: string
  }

  type HTTPValidationError = {
    /** Detail */
    detail?: ValidationError[]
  }

  type ValidationError = {
    /** Location */
    loc: (string | number)[]
    /** Message */
    msg: string
    /** Error Type */
    type: string
  }
}
