import os
import time
from datetime import datetime
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Dict

import akshare as ak
from databases import Database
from tenacity import retry_if_exception_type, stop_after_attempt, wait_fixed, retry

from config.LoguruConfig import log, project_root
from constant.ExponentEnum import ExponentEnum
from constant.MaroDataEnum import PERIOD
from exception.BusinessException import BusinessException
from model.entity.StockData import StockData
from utils.SnowFlake import snowflake_instance
from model.entity.BaseMeta.BaseMeta import database
from model.entity.Indicator import Indicator
import pandas as pd

index_descriptions = {
    "000001": "上证指数是上海证券交易所的主要股票指数，反映了上海证券市场股票价格的总体走势。",
    "399001": "深证成指是深圳证券交易所的主要股票指数，反映了深圳证券市场股票价格的总体走势。",
    "399006": "创业板指数是深圳证券交易所创业板市场的主要指数，反映了创业板市场股票价格的总体走势。",
    "000688": "科创50指数是上海证券交易所科创板市场的主要指数，反映了科创板市场股票价格的总体走势。",
    "000300": "沪深300指数是由上海和深圳证券交易所市值最大的300只股票组成的指数，反映了中国A股市场的整体表现。",
    "930050": "中证A50指数是由中证公司编制的，反映了中国A股市场中50只大盘蓝筹股的表现。",
    "000905": "中证500指数是由中证公司编制的，反映了中国A股市场中500只中小盘股的表现。",
    "000852": "中证1000指数是由中证公司编制的，反映了中国A股市场中1000只小盘股的表现。"
}


async def crawl_sock_data( stock_code: str = None,
                          start_date: str = "2000-01-01", end_date: str = None) -> int:
    """从东方财富网获取数据
    :param stock_code: 股票代码
    :param start_date:  format("YYYY-MM-DD")
    :param end_date:  format("YYYY-MM-DD")  default->now()
    :return: 更新的条数
    :raise BusinessException
    """
    log.info("入口参数:【stock_code={}, start_date={}, end_date={}】", stock_code, start_date, end_date)
    stock_name = ExponentEnum.get_enum_by_code(stock_code).get_name()
    log.info("开始获取【{}】数据", stock_name)
    # 查询indicator是否存在
    exits = await Indicator.objects.filter(code=stock_code).exists()
    if not exits:
        indicator = Indicator(
            id=snowflake_instance.get_id(),
            code=stock_code,
            name=ExponentEnum.get_name_by_code(stock_code),
            description=index_descriptions[stock_code],
            frequency=PERIOD.DAILY
        )
        log.info("入库对象:【" + indicator.__str__() + "】")
        save = await indicator.save()
        if save:
            log.info("新增指标成功")
        else:
            log.info("新增指标失败")
    else:
        indicator = await Indicator.objects.get(code=stock_code)
    # 获取指数数据
    # 时间格式转换
    start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")
    exponent_data: pd.DataFrame = pd.DataFrame()
    try:
        exponent_data = fetch_exponent_data(stock_code, start_date, end_date)
    except RemoteDisconnected:
        raise BusinessException(code=500, msg="网络异常请稍后重试")
    if exponent_data.empty:
        log.info("数据为空不执行入库")
        return 0
    # 替换数据列的名称和对象对应
    exponent_data = exponent_data.rename(columns={"日期": "trade_date", "开盘": "open_price", "收盘": "close_price",
                                                  "最高": "high_price", "最低": "low_price", "成交量": "volume",
                                                  "成交额": "turnover_amount", "涨跌幅": "change_rate",
                                                  "涨跌额": "change_amount", "换手率": "turnover_rate",
                                                  "振幅": "amplitude"})

    # 处理数据，给对象赋值
    # 1.转化日期
    exponent_data["trade_date"] = exponent_data["trade_date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    # 2.处理na值 数据量大直接drop即可
    exponent_data.dropna(inplace=True)
    # 批量生成id
    exponent_data["id"] = exponent_data["trade_date"].apply(lambda x: snowflake_instance.get_id())
    exponent_data["indicator_id"] = indicator.id
    # 查询出最新的一条日期对象
    latest_one: StockData = await (StockData.objects.filter(indicator_id=indicator.id)
                                   .order_by("-trade_date").first_or_none())
    if latest_one is None:
        """全量插入"""
        log.info("开始全量插入【{}】数据", stock_name)
        if not exponent_data.empty:
            try:
                stock_objects: list[StockData] = (exponent_data
                                                  .apply(lambda row: StockData(**row.to_dict()), axis=1).tolist())
                await StockData.objects.bulk_create(stock_objects)
                num: int = len(stock_objects)
                log.info("【{}】数据全量入库成功共【{}】条", stock_name, num)
                return num
            except Exception as e:
                log.exception(e)
                raise BusinessException(code=500, msg="数据入库失败")
        else:
            log.info("数据为空不执行入库")
        return 0
    else:
        """增量更新"""
        try:
            log.info("开始增量更新【{}】数据", stock_name)
            # 获取最新一条数据的日期
            latest_date = pd.to_datetime(latest_one.trade_date)
            # 继续获取出最老的一条数据
            oldest_one = await (StockData.objects.filter(indicator_id=indicator.id)
                                .order_by("trade_date").first_or_none())
            oldest_date = pd.to_datetime(oldest_one.trade_date)
            # 过滤出区间外的数据
            exponent_data = exponent_data[
                (exponent_data["trade_date"] < oldest_date) | (exponent_data["trade_date"] > latest_date)
                ]
            log.info("过滤出日期区间外的结果条数:【{}】", exponent_data.shape[0])
            if exponent_data.empty:
                log.info("【{}】数据已是最新，无需更新", stock_name)
                return 0
            # 处理数据，给对象赋值
            stock_objects: list[StockData] = exponent_data.apply(lambda row: StockData(**row.to_dict()),
                                                                 axis=1).tolist()
            await StockData.objects.bulk_create(stock_objects)
            num: int = len(stock_objects)
            log.info("{}】数据增量入库成功共【{}】条", stock_name, num)
            return num

        except Exception as e:
            log.exception(e)
            raise BusinessException(code=500, msg=e.__str__())


@retry(
    stop=stop_after_attempt(3),  # 最多重试5次
    wait=wait_fixed(3),  # 每次重试前等待3秒
    retry=retry_if_exception_type(RemoteDisconnected)  # 只针对RemoteDisconnected异常重试
)
def fetch_exponent_data(stock_code, start_date, end_date) -> pd.DataFrame:
    """获取指标数据
    :return: pandas.DataFrame
    :raise RemoteDisconnected
    """
    return ak.index_zh_a_hist(symbol=stock_code, period=PERIOD.DAILY,
                              start_date=start_date,
                              end_date=end_date)


async def get_stock_data(stock_code: str = None,
                         start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """从数据库获取数据并返回dataFrame
    :param stock_code: 股票代码
    :param start_date:  format("YYYY-MM-DD")
    :param end_date:  format("YYYY-MM-DD")  default->now()
    :return: pandas.DataFrame
    :raise BusinessException
    """
    log.info("入口参数:【stock_code={}, start_date={}, end_date={}】", stock_code, start_date, end_date)
    try:
        indicator = await Indicator.objects.filter(code=stock_code).get_or_none()
        if indicator is None:
            log.info("指标不存在，返回空列表")
            return pd.DataFrame()
        result: list[StockData] = await StockData.objects.filter(indicator_id=indicator.id, trade_date__gte=start_date,
                                                                 trade_date__lte=end_date).all()
        if len(result) == 0:
            log.info("时间范围内数据为空，返回空列表")
            return pd.DataFrame()
        # 转换成dateFrame
        datalist: list[dict] = [StockData.model_dump(item) for item in result]
        drop_columns=["id", "indicator_id", "created_at", "updated_at"]
        data_frame = pd.DataFrame(datalist).drop(drop_columns, axis=1)
        data_frame.sort_values(by="trade_date", ascending=True, inplace=True)
        return data_frame
    except Exception as e:
        log.exception(e)
        raise BusinessException(code=500, msg="获取数据失败")
async def get_stock_data_local(stock_code: str = None,
                         start_date: str = "2000-01-01", end_date: str = None) -> pd.DataFrame:
    """从该方法用于本地调用，区别于get_stock_data 删除多余的数据列
    :param stock_code: 股票代码
    :param start_date:  format("YYYY-MM-DD")
    :param end_date:  format("YYYY-MM-DD")  default->now()
    :return: pandas.DataFrame
    :raise BusinessException
    """
    date_ = await get_stock_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
    return date_.drop(["turnover_amount","amplitude","change_rate","change_amount","turnover_rate"], axis=1)


async def export_to_csv( stock_code: str = None,
                        start_date: str = "2000-01-01", end_date: str = None) -> str | None:
    """导出数据到csv文件
    :param stock_code: 股票代码
    :param start_date:  format("YYYY-MM-DD")
    :param end_date:  format("YYYY-MM-DD")  default->now()
    :raise BusinessException
    :return: FilePath
    """
    log.info("入口参数: stock_code:【】,start_date:【{}】,end_date:【{}】", stock_code, start_date, end_date)
    file_path = None
    try:
        # 生成路径
        file_path = project_root / "resources" / "csvFiles" / f"stock_{stock_code}_{start_date}_{end_date}.csv"
        # 转化成相对路径返回
        relative_path = file_path.relative_to(project_root)
        if os.path.exists(file_path):
            log.info("路径文件已存在，无需创建:【{}】", relative_path)
            return str(relative_path)
        # 确保目标目录存在，否则创建它
        file_path.parent.mkdir(parents=False, exist_ok=True)
        # 获取数据
        csv_date = await get_stock_data(stock_code=stock_code, start_date=start_date, end_date=end_date)
        if csv_date.empty:
            log.info("数据为空，无需导出")
            return None
        # 转换列名 英文转中文
        csv_date = await stock_colum_name_eng2cn(csv_date)

        csv_date.to_csv(file_path, index=False)
        log.info("导出数据成功 FilePath:【{}】", relative_path)
        return str(relative_path)
    except Exception as e:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except NameError:
            pass
        log.exception(e)
        raise BusinessException(code=500, msg="导出数据失败")


async def stock_colum_name_eng2cn(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame.rename(
        columns={"trade_date": "日期", "open_price": "开盘价", "close_price": "收盘价", "high_price": "最高价",
                 "low_price": "最低价",
                 "volume": "成交量", "turnover_amount": "成交额", "change_amount": "涨跌额",
                 "change_rate": "涨跌幅", "turnover_rate": "换手率", "amplitude": "振幅"}, inplace=False)





