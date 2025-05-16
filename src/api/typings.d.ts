declare namespace API {
  type BaseResponse = {}

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

  type exportModelPredictExportModelGetParams = {
    /** 股票代码 */
    code: string
  }

  type GetMacroDataRequest = {
    /** Types */
    types?: string
    /** Start Date */
    start_date?: string
    /** End Date */
    end_date?: string
  }

  type GetPredictRequest = {
    /** Stock Code 股票代码 */
    stock_code?: string
    /** Start Date 开始日期 */
    start_date?: string
    /** End Date 结束日期 */
    end_date?: string
    /** Epoches 训练轮次 */
    Epoches?: number
    /** Reg 正则化 */
    reg?: number
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
