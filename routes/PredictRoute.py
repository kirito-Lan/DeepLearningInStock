import base64
import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from fastapi import APIRouter
from fastapi.params import Query
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from starlette.responses import FileResponse

from config.LoguruConfig import project_root
from constant.BaseResponse import BaseResponse
from constant.ErrorCode import ErrorCode
from constant.ExponentEnum import ExponentEnum
from exception.BusinessException import BusinessException
from manager import StockDataManage
from model.dto.GetPredictRequest import GetPredictRequest
from reasoning.analyse import FeatureEngine, ExploratoryDataAnalysis
from reasoning.train import Improved_Lstm
from utils.ReFormatDate import format_date

router = APIRouter(prefix="/predict", tags=["predict"])


# 数据描述
@router.get("/description", response_model=BaseResponse)
async def descript_data_set():
    """描述原始数据集中的数据
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
            dict: 返回数据集的描述信息
    """
    financial_metrics = {
        "Close": {
            "英文名称": "Close",
            "中文名称": "收盘价",
            "指数解释": "收盘价是指在交易日结束时，某只股票、商品或其他金融工具的最后成交价格。它通常被认为是一天交易的结束价格，并且是计算涨跌幅的重要依据。"
        },
        "Volume": {
            "英文名称": "Volume",
            "中文名称": "成交量",
            "指数解释": "成交量是指在一定时间内，某只股票或商品被买卖的总数量。它是衡量市场活跃度的重要指标，通常高成交量意味着市场对该资产的兴趣较高。"
        },
        "Open": {
            "英文名称": "Open",
            "中文名称": "开盘价",
            "指数解释": "开盘价是指在交易日开始时，某只股票、商品或其他金融工具的第一个成交价格。它通常反映了市场在交易开始时对该资产的预期。"
        },
        "High": {
            "英文名称": "High",
            "中文名称": "最高价",
            "指数解释": "最高价是指在交易日内，某只股票、商品或其他金融工具达到的最高成交价格。它通常用于分析市场的波动性和投资者的情绪。"
        },
        "Low": {
            "英文名称": "Low",
            "中文名称": "最低价",
            "指数解释": "最低价是指在交易日内，某只股票、商品或其他金融工具达到的最低成交价格。它通常用于分析市场的波动性和投资者的情绪。"
        },
        "CPI": {
            "英文名称": "CPI",
            "中文名称": "消费者价格指数",
            "指数解释": "消费者价格指数（CPI）是衡量一篮子消费品和服务价格变化的指标。它通常用于衡量通货膨胀水平，并对货币政策和经济政策有重要影响。"
        },
        "PPI": {
            "英文名称": "PPI",
            "中文名称": "生产者价格指数",
            "指数解释": "生产者价格指数（PPI）是衡量生产者商品价格变化的指标。它通常用于衡量通货膨胀的早期迹象，因为生产者价格的变化通常会传导到消费者价格。"
        },
        "PMI": {
            "英文名称": "PMI",
            "中文名称": "采购经理人指数",
            "指数解释": "采购经理人指数（PMI）是衡量制造业活动的指标，通过调查采购经理人对市场状况的看法来反映经济活动的扩张或收缩。PMI通常包括新订单、生产、就业等子指标。"
        }
    }
    return BaseResponse[Dict].success(financial_metrics)


# 获取股票的统计行描述
@router.post("/stock_analysis", response_model=BaseResponse)
async def stock_analysis(request: GetPredictRequest):
    """
    对股票进行统计行描述
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
    '峰度: sharp,
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if stock is None:
        raise BusinessException(400, "请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    # 获取股票数据
    stock_data = await StockDataManage.get_stock_data(stock.get_code(), start_date, end_date)
    # 浮点转化
    stock_data['close_price'] = stock_data['close_price'].astype(float)
    res = {
        'sample_num': int(stock_data['close_price'].count()),
        'min_value': float(round(stock_data['close_price'].min(), 3)),
        'max_value': float(round(stock_data['close_price'].max(), 3)),
        'avg': float(round(stock_data['close_price'].mean(), 3)),
        'std': float(round(stock_data['close_price'].std(), 3)),
        'var': float(round(stock_data['close_price'].var(), 3)),
        'bias': float(round(stock_data['close_price'].skew(), 3)),
        'sharp': float(round(stock_data['close_price'].kurtosis(), 3))
    }

    return BaseResponse[Dict].success(res)


# 异常检测
@router.post("/anomaly_detection", response_model=BaseResponse)
async def anomaly_detection(request: GetPredictRequest):
    """
    异常检测
    Args:
        request: 请求体
    Returns:
        dict: 返回异常检测结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        raise BusinessException(400, "请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    # 获取股票数据
    stock_data = await StockDataManage.get_stock_data_local(stock.get_code(), start_date, end_date)
    # 浮点转化
    stock_data['close_price'] = stock_data['close_price'].astype(float)
    # stock_data 选择volume和trade_date列
    stock_data = stock_data[["volume", "trade_date"]]
    # 交易量 IQR 异常值检测
    Q1_volume = stock_data['volume'].quantile(0.25)
    Q3_volume = stock_data['volume'].quantile(0.75)
    IQR_volume = Q3_volume - Q1_volume
    lower_bound_volume = Q1_volume - 1.5 * IQR_volume
    upper_bound_volume = Q3_volume + 1.5 * IQR_volume
    # 检测交易量中的异常值 布尔索引会返回Series的布尔值
    stock_data['Volume_outlier'] = ((stock_data['volume'] < lower_bound_volume) |
                                    (stock_data['volume'] > upper_bound_volume))
    # 转换成字段列表
    stock_data = stock_data.to_dict("records")
    return BaseResponse[List[Dict]].success(stock_data)


# 获取散点图
@router.post("/scatter_plot", response_model=BaseResponse)
async def scatter_plot(request: GetPredictRequest):
    """
    散点图
    Args:
        request: 请求体
    Returns:
        dict: 返回散点图结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        raise BusinessException(400, "请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    stock_macro_data = await FeatureEngine.get_merged_data(stock_code=stock.get_code(), start_date=start_date,
                                                           end_date=end_date)
    # 取 CPI、PPI、PMI、Close 并将其转换成float
    for col in ["CPI", "PPI", "PMI", "Close"]:
        stock_macro_data[col] = stock_macro_data[col].astype(float)
    stock_macro_data = stock_macro_data[["CPI", "PPI", "PMI", "Close"]]
    # 转换成字典列表
    stock_macro_data = stock_macro_data.to_dict("records")
    return BaseResponse[List[Dict]].success(stock_macro_data)


# 相关性分析（将股票数据降维权重平均采样）
@router.post("/correlation_analysis", response_model=BaseResponse)
async def correlation_analysis(request: GetPredictRequest):
    """
    相关性分析
    Args:
        request: 请求体
    Returns:
        dict: 返回相关性分析结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        raise BusinessException(400, "请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    stock_data = await StockDataManage.get_stock_data_local(stock_code=stock.get_code(), start_date=start_date,
                                                            end_date=end_date)
    # 转换时间格式
    stock_data['trade_date'] = pd.to_datetime(stock_data['trade_date'], errors='coerce', format="%Y-%m-%d")
    # 将date列设置为索引
    stock_data.set_index('trade_date', inplace=True)
    # 获取降维后的数据
    merged_data = await ExploratoryDataAnalysis._get_macro_stock_data(stock_data=stock_data)
    # 平滑采样
    merged_data['CPI_Inflation_Rate'] = merged_data['CPI_Inflation_Rate'].ewm(span=5, adjust=False).mean()
    # 取 CPI_Inflation_Rate PPI_Inflation_Rate PMI close_price、
    merged_data = merged_data[["CPI_Inflation_Rate", "PPI_Inflation_Rate", "PMI", "close_price"]]
    # 转换成字典列表 包含索引
    merged_data = merged_data.reset_index().rename(columns={'index': 'Date', "CPI_Inflation_Rate": "CPI",
                                                            "PPI_Inflation_Rate": "PPI", "close_price": "Close"})
    # CPI取一位小数,四舍五入
    merged_data['PMI'] = merged_data['PMI'].apply(lambda x: round(x, 1))
    merged_data['Close'] = merged_data['Close'].apply(lambda x: round(x, 3))
    # 格式化时间
    merged_data['Date'] = merged_data['Date'].dt.strftime('%Y-%m-%d')
    merged_data = merged_data.to_dict("records")
    return BaseResponse[List[Dict]].success(merged_data)


# 训练模型
@router.post("/train_model", response_model=BaseResponse)
async def train_model(request: GetPredictRequest):
    """训练模型
        Args:
            request: 请求体
        Returns:
            dict: 返回训练结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    # Epoches: int = Field(default=150, description="训练轮次")
    # reg: float = Field(default=0.001, description="正则化")
    # dropout: float = Field(default=0.3, description="dropout")
    # 判空，如果有一个为空就返回False
    if not all([request.Epoches, request.reg, request.dropout]):
        return BaseResponse[str].fail(errcode=ErrorCode.PARAMS_ERROR,
                                      msg="请输入正确的开始日期和结束日期")
    success = await Improved_Lstm.build_model(stock_code=stock.get_code(), start_date=start_date, end_date=end_date
                                              , epochs=request.Epoches, reg=request.reg, dropout=request.dropout)
    return BaseResponse[bool].success(success)


# 获取训练历史
@router.post("/get_trained_score", response_model=BaseResponse)
async def get_trained_score(request: GetPredictRequest):
    """获取模型的训练结果
        Args:
            request: 请求体
        Returns:
            dict: 返回EDA结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    # 判空
    if not stock:
        return BaseResponse[str].fail(errcode=ErrorCode.FILE_NOT_FOUND_ERROR,
                                      msg="请输入正确的股票代码")
    # 格式化时间
    start_date, end_date = format_date(start_date=request.start_date, end_date=request.end_date)
    # 获取预测结果文件
    predict_file = project_root / "reasoning" / "results" / f"predicted_close{stock.get_code()}.csv"
    # 获取训练损失结果
    loss_file = project_root / "reasoning" / "results" / f"training_losses_{stock.get_code()}.csv"
    # 判断文件是否存在
    if not predict_file.exists() or not loss_file.exists():
        return BaseResponse[str].fail(errcode=ErrorCode.FILE_NOT_FOUND_ERROR,
                                      msg="没有历史预测数据,请先构建模型进行预测分析")
    # 分别获取两个文件的修改日期格式化成%Y-%m-%d。如果修改日期不等于end_date(他是str)也返回煤油文件
    predict_mod_date = datetime.fromtimestamp(predict_file.stat().st_mtime).strftime('%Y-%m-%d')
    loss_mod_date = datetime.fromtimestamp(loss_file.stat().st_mtime).strftime('%Y-%m-%d')
    # 如果任一文件的修改日期与请求中的 end_date 不一致，则返回失败响应
    if predict_mod_date != end_date or loss_mod_date != end_date:
        return BaseResponse[str].fail(errcode=ErrorCode.FILE_NOT_FOUND_ERROR,
                                      msg="文件修改日期与结束日期不匹配，请先重新构建模型")
    # 读取文件使用pandas读取文件
    predict_data = pd.read_csv(predict_file)
    # Date,Actual,Predicted 计算指标
    # 计算指标
    mse = mean_squared_error(predict_data["Actual"], predict_data["Predicted"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(predict_data["Actual"], predict_data["Predicted"])
    r2 = r2_score(predict_data["Actual"], predict_data["Predicted"])
    # 按顺序重命名["Epoch","Train_Loss","Validate_Loss"]
    loss = pd.read_csv(loss_file).rename(
        columns={"Training Loss": "Train_Loss", "Validation Loss": "Validate_Loss"})
    # 将里面的Train_Loss"  "Validate_Loss" 保留五位小数
    loss["Train_Loss"] = loss["Train_Loss"].round(decimals=6)
    loss["Validate_Loss"] = loss["Validate_Loss"].round(decimals=6)
    # 预测结果转换成dict列表
    return BaseResponse[Dict].success({
        "predicted_data": predict_data.to_dict("records"),
        "loss_data": loss.to_dict("records"),
        "metrics": {
            "mse": round(mse, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "r2": round(r2, 2)
        }
    })


# 导出模型
@router.get("/export_model", response_model=BaseResponse)
async def export_model(code: str = Query(..., description="股票代码")):
    """导出模型
        Args:
            code 股票代码
        Returns:
            dict: 返回导出结果
    """
    stock = ExponentEnum.get_enum_by_code(code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    model_file = project_root / "reasoning" / "model" / f"{stock.get_name()}_model.keras"
    logging.info(f'导出的模型路径是{model_file}')
    if not model_file.exists():
        return BaseResponse.fail(ErrorCode.FILE_NOT_FOUND_ERROR.code, msg="当前股票暂无模型可用")
    # 返回文件
    return FileResponse(path=model_file, filename=f"{stock.get_name()}_model.keras",
                        media_type="application/octet-stream")


# 获取特征工程文件
@router.post("/get_featured_file", response_model=BaseResponse)
async def get_featured_file(getPredictRequest: GetPredictRequest):
    """获取特征工程文件
        Args:
            getPredictRequest 请求体
        Returns:
            dict: 返回特征工程文件
    """
    stock = ExponentEnum.get_enum_by_code(getPredictRequest.stock_code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    # 格式化时间
    start, end = format_date(start_date=getPredictRequest.start_date, end_date=getPredictRequest.end_date)
    feature_file = project_root / "reasoning" / "processed_data" / f"{stock.get_code()}" / f"feature_{stock.get_code()}-{start}-{end}.csv"
    logging.info(f'导出的特征工程文件路径是{feature_file}')
    if not feature_file.exists():
        return BaseResponse.fail(errcode=ErrorCode.FILE_NOT_FOUND_ERROR,
                                 msg="当前股票暂无特征工程文件可用，请先构建模型")
    # 返回文件
    return FileResponse(path=feature_file, filename=f"{stock.get_name()}_featured.csv",
                        media_type="application/octet-stream")
