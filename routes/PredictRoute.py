import base64
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter

from config.LoguruConfig import project_root
from constant.BaseResponse import BaseResponse
from constant.ExponentEnum import ExponentEnum
from model.dto.GetPredictRequest import GetPredictRequest

router = APIRouter(prefix="/predict", tags=["predict"])




@router.get("/description", response_model=BaseResponse)
async def data_description():
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




@router.post("/get_trained_history", response_model=BaseResponse)
async def get_trained_history(request:GetPredictRequest):
    """获取到最近一次的训练数据内含，预测结果。模型的指标参数。最近一次预测的时间
        Args:
            request: 请求体
            evaluation_metrics.png 模型评价指标
            prediction_result.png 预测结果
            training_history.png 训练历史
        Returns:
            dict: 返回最近一次的训练数据
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    # 查找保存的图片
    pictures = ["evaluation_metrics.png", "prediction_result.png", "training_history.png"]
    fload= project_root / "reasoning" / "picture" / request.stock_code
    pic_path = [fload / picture for picture in pictures]
    images = {}
    for picture in pic_path:
        # 判断文件存不存在
        try:
            if not picture.exists():
                return BaseResponse[str].fail(msg="最近没有进行过预测请先进行预测")

        except ValueError:
            return BaseResponse[str].fail(msg="最近没有进行过预测请先进行预测")

    for picture in pic_path:
        # 读取图片并编码为Base64
        with open(picture, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            # 获取并格式化修改时间
            modification_time = picture.stat().st_mtime
            formatted_date = datetime.fromtimestamp(modification_time).strftime("%Y-%m-%d")
            images[picture.name] = {
                "data": encoded_image,
                "modified_time": formatted_date
            }
    return BaseResponse[Dict].success(images)


# 训练模型
@router.post("/train_model", response_model=BaseResponse)
async def train_model(request:GetPredictRequest):
    """训练模型
        Args:
            request: 请求体
        Returns:
            dict: 返回训练结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    pass



# EDA部分数据的获取接口
@router.post("/eda", response_model=BaseResponse)
async def eda(request:GetPredictRequest):
    """EDA部分数据的获取接口
        Args:
            request: 请求体
        Returns:
            dict: 返回EDA结果
    """
    stock = ExponentEnum.get_enum_by_code(request.stock_code)
    if not stock:
        return BaseResponse[str].fail(msg="请输入正确的股票代码")
    pass

