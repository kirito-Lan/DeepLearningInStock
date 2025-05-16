// @ts-ignore
/* eslint-disable */
import request from '@/request'

/** Anomaly Detection 异常检测
Args:
    request: 请求体
Returns:
    dict: 返回异常检测结果 POST /predict/anomaly_detection */
export async function anomalyDetectionPredictAnomalyDetectionPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/anomaly_detection', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Correlation Analysis 相关性分析
Args:
    request: 请求体
Returns:
    dict: 返回相关性分析结果 POST /predict/correlation_analysis */
export async function correlationAnalysisPredictCorrelationAnalysisPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/correlation_analysis', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Decompose Trend 趋势分解
Args:
    request: 请求体
Returns:
    dict: 返回趋势分解结果 POST /predict/decompose_trend */
export async function decomposeTrendPredictDecomposeTrendPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/decompose_trend', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Descript Data Set 描述原始数据集中的数据
Args:
    Close: 收盘价
    Volume: 成交量
    Open: 开盘价
    High: 最高价
    Low: 最低价
    CPI: 消费者价格指数
    PPI: 生产者价格指数
    PMI: 采购经理人指数
Returns:
    dict: 返回数据集的描述信息 GET /predict/description */
export async function descriptDataSetPredictDescriptionGet(options?: { [key: string]: any }) {
  return request<API.BaseResponse>('/predict/description', {
    method: 'GET',
    ...(options || {}),
  })
}

/** Export Model 导出模型
Args:
    code 股票代码
Returns:
    dict: 返回导出结果 GET /predict/export_model */
export async function exportModelPredictExportModelGet(
  // 叠加生成的Param类型 (非body参数swagger默认没有生成对象)
  params: API.exportModelPredictExportModelGetParams,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/export_model', {
    method: 'GET',
    params: {
      ...params,
    },
    ...(options || {}),
  })
}

/** Get Featured File 获取特征工程文件
Args:
    getPredictRequest 请求体
Returns:
    dict: 返回特征工程文件 POST /predict/get_featured_file */
export async function getFeaturedFilePredictGetFeaturedFilePost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/get_featured_file', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Trained History 获取到最近一次的训练数据内含，预测结果。模型的指标参数。最近一次预测的时间
Args:
    request: 请求体
    evaluation_metrics.png 模型评价指标
    prediction_result.png 预测结果
    training_history.png 训练历史
Returns:
    dict: 返回最近一次的训练数据 POST /predict/get_trained_history */
export async function getTrainedHistoryPredictGetTrainedHistoryPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/get_trained_history', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Get Trained Score 获取模型的训练结果
Args:
    request: 请求体
Returns:
    dict: 返回EDA结果 POST /predict/get_trained_score */
export async function getTrainedScorePredictGetTrainedScorePost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/get_trained_score', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Scatter Plot 散点图
Args:
    request: 请求体
Returns:
    dict: 返回散点图结果 POST /predict/scatter_plot */
export async function scatterPlotPredictScatterPlotPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/scatter_plot', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Stock Analysis 对股票进行统计行描述
Args:
    request: 请求体
Returns:
    dict: 返回股票的统计行描述
'样本数': sample_num,
'最小值': min_value,
'最大值': max_value,
'均值': avg,
'标准偏差': std,
'方差': var,
'偏度': bias,
'峰度: sharp, POST /predict/stock_analysis */
export async function stockAnalysisPredictStockAnalysisPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/stock_analysis', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}

/** Train Model 训练模型
Args:
    request: 请求体
Returns:
    dict: 返回训练结果 POST /predict/train_model */
export async function trainModelPredictTrainModelPost(
  body: API.GetPredictRequest,
  options?: { [key: string]: any }
) {
  return request<API.BaseResponse>('/predict/train_model', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    data: body,
    ...(options || {}),
  })
}
