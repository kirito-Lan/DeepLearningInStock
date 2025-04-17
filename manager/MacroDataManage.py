# 获取中国宏观数据 cpi ppi pmi 并且入库
import asyncio
import os
from datetime import datetime
from pathlib import Path

import akshare as ak
from databases import Database
from ormar.exceptions import ModelListEmptyError
from sqlalchemy import text

from typing_extensions import deprecated

from exception.BusinessException import BusinessException
from utils.SnowFlake import snowflake_instance
from config.LoguruConfig import log, project_root
from constant.MaroDataEnum import PERIOD, DataTypeEnum
from repository import BaseSql
from model.entity.BaseMeta.BaseMeta import database
from model.entity.Indicator import Indicator
import pandas as pd

from model.entity.MacroData import MacroData

data_type = ("中国CPI数据,月率报告,数据来自中国官方数据",
             "中国PPI数据,月率报告,数据来自中国官方数据",
             "中国官方制造业PMI,月率报告,数据来自中国官方数据")


async def save_or_update_macro_data(db: Database = database, types: DataTypeEnum = DataTypeEnum.CPI) -> bool:
    """
    该方法用于获取宏观数据数据，如果没有数据那么就全量插入。反之进行增量更新
    :param types: 获取的宏观书据类型 DataTypeEnum
    :param db: 数据库源随项目启动初始化,没有特殊需求无需传入
    :return bool
    """
    try:
        # 根据传入的类型，调用对应的 akshare 接口
        if types == DataTypeEnum.CPI:
            china_macro_data = ak.macro_china_cpi_monthly()
        elif types == DataTypeEnum.PPI:
            china_macro_data = ak.macro_china_ppi_yearly()
        elif types == DataTypeEnum.PMI:
            china_macro_data = ak.macro_china_pmi_yearly()
        else:
            raise ValueError("不支持的数据类型")

        name = china_macro_data["商品"][0]
        # 判断 indicator 是否存在
        count = (await db.fetch_one(query=text(BaseSql.isIndicateExist).bindparams(name=name)))["num"]
        if count == 0:
            # 填充数据 构造主表入库信息
            frequency = PERIOD.MONTHLY
            description = data_type[types.value[0]]
            code = types.value[1]
            field_id = snowflake_instance.get_id()
            # log.info("snowflake获取主键id:【{}】", field_id)
            indicator = Indicator(id=field_id, name=name, description=description, frequency=frequency, code=code)
            # 插入 indicator 表
            indicator_id = (await indicator.save()).id
            log.info("入库对象:【" + indicator.__str__() + "】")
        else:
            indicator_id = (await db.fetch_one(query=text(BaseSql.getIndicateId).bindparams(name=name)))["id"]
            log.info("回查indicator_id【{}】", indicator_id)

        """数据清洗
        1 对日期进行清洗，发布日>20的日期检查下一个月有没有数据，没有归到下个月反之不处理。最后统一将日期刷成01号提取日期，转换成date对象
        2 处理今值 NaN的数据 取下一行的前值填充，如果下一行的前值也为NaN，那么取自己行的前值填充
        3 处理预测值 NaN的数据 直接取同行的今值填充
        4 处理前值 NaN的数据 直接取上一行的今值填充，如果没有就取上一行的预测值填充
        5 将每一行全部转换成对象，然后插入数据库"""
        # 转换日期
        china_macro_data["日期"] = pd.to_datetime(china_macro_data["日期"], format="%Y-%m-%d", errors='coerce')
        # 根据日期去重
        china_macro_data.drop_duplicates(subset=["日期"], keep="first", inplace=True)
        # 数据清洗
        data_set = clean_macro_data(china_macro_data, indicator_id)
        # log.info("入库对象:【macroData】条数:【" + str(len(data_set)) + "】")
        # 查看该指标下的数据条数，决定是全量插入还是新增数据
        count_result = (await db.fetch_one(query=BaseSql.countMacroData,
                                           values={"indicator_id": indicator_id}))["count"]
        if count_result == 0:
            # 全量插入
            await MacroData.objects.bulk_create(data_set)
            log.info("数据进行全量插入")
            return True
        else:
            try:
                # 增量更新 直接将集合截断，存入余量即可
                await MacroData.objects.bulk_create(data_set[count_result:])
                log.info("数据进行增量更新")
                return True
            except ModelListEmptyError:
                log.info("数据已是最新，无需更新")
                return False
        # print(china_macro_data)
    except Exception as e:
        log.error("发生异常 数据清洗失败")
        log.exception(e)
        return False


def clean_macro_data(china_macro_data, indicator_id) -> list[MacroData]:
    """清洗宏观数据方法"""
    data_set: list[MacroData] = []
    for row in china_macro_data.index:
        # 日期
        date = china_macro_data.loc[row, "日期"]
        # 今值
        cur_val = (china_macro_data.loc[row, "今值"] if not pd.isna(china_macro_data.loc[row, "今值"])
                   else (
            china_macro_data.loc[row - 1, "今值"] if row > 0 and not pd.isna(china_macro_data.loc[row - 1, "今值"])
            else china_macro_data.loc[row, "前值"]))
        # 回填一次
        china_macro_data.loc[row, "今值"] = cur_val
        # 预测值
        forecast_val = (china_macro_data.loc[row, "预测值"] if not pd.isna(china_macro_data.loc[row, "预测值"])
                        else cur_val)
        # 前值
        pre_val = (china_macro_data.loc[row, "前值"] if not pd.isna(china_macro_data.loc[row, "前值"])
                   else (
            china_macro_data.loc[row - 1, "今值"] if row > 0 and not pd.isna(china_macro_data.loc[row - 1, "前值"])
            else china_macro_data.loc[row, "今值"]))

        data = MacroData(id=snowflake_instance.get_id(), indicator_id=indicator_id, report_date=date,
                         current_value=cur_val,
                         forecast_value=forecast_val, previous_value=pre_val)
        data_set.append(data)
        # log.info("入库对象:【" + data.__str__() + "】")
    return data_set


async def get_macro_data(*, db: Database = database, types: DataTypeEnum = DataTypeEnum.CPI,
                         start_date: str = "19700101", end_date: str = None) -> pd.DataFrame:
    """
    方法用于从数据库中获取数据并且封装成panda.DateFrame形式
    :param db: 数据库源随项目启动初始化,没有特殊需求无需传入
    :param types: DataTypeEnum
    :param start_date: 起始时间 default->"19700101"
    :param end_date: 结束时间 default->now()
    :return pandas.DataFrame
    """
    log.info("入口参数:【types:{}】,【start_date:{}】,【end_date:{}】")
    try:
        start_date = datetime.strftime(datetime.strptime(start_date, "%Y%m%d"), "%Y-%m-%d")
        if end_date is not None:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
        # 获取指标中的id
        indicator_id = (await db.fetch_one(query=text(BaseSql.getIndicateIdByCode)
                                           .bindparams(code=types.value[1])))["id"]
        res = await MacroData.objects.database.fetch_all(
            query=text(BaseSql.getLimitYearData).bindparams(indicator_id=indicator_id, start_date=start_date,
                                                            end_date=end_date))
        if not res:
            log.info("数据库中不存在该指标,获取数据为空")
            return pd.DataFrame()
        log.info("获取数据成功")
        datas = [MacroData.model_validate(row) for row in res]
        # 直接解包
        # datas = [MacroData(**row) for row in res]
        # 转换成字典
        datas = [row.model_dump() for row in datas]
        data_frame = pd.DataFrame(datas).drop(["id", "indicator_id", "created_at", "updated_at"], axis=1)
        data_frame.sort_values(by="report_date", ascending=False, inplace=True)
    except Exception as e:
        log.error("获取数据失败")
        log.exception(e)
        return pd.DataFrame()
    # 删除指定列
    return data_frame


async def export_to_csv(db: Database = database, types: DataTypeEnum = DataTypeEnum.CPI,
                        start_date: str = "19700101", end_date: str = None) -> str|None:
    """ 导出数据到csv文件
    :param db: 数据库数据源无需传入
    :param types: DataTypeEnum
    :param start_date: 起始时间 default->"19700101"
    :param end_date: 结束时间 default->now()
    :return FilePath
    :raise BusinessException
    """
    file_path:Path=None
    try:
        start=start_date
        end=end_date
        start_date = datetime.strftime(datetime.strptime(start_date, "%Y%m%d"), "%Y-%m-%d")
        if end_date is not None:
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_date = datetime.strftime(datetime.now(), "%Y-%m-%d")
        file_path = project_root / "resources" / "csvFiles" / f"macro_{types.value[1]}_{start_date}_{end_date}.csv"
        #转化成相对路径
        relative_path=file_path.relative_to(project_root)
        if os.path.exists(file_path):
            log.info("路径文件已存在，无需创建:【{}】",relative_path)
            return str(relative_path)
        # 获取数据
        data_frame = await get_macro_data(types=types, start_date=start, end_date=end)
        if data_frame.empty:
            log.info("数据为空")
            return None
        # 重命名列名
        data_frame.rename(columns={"report_date": "日期", "current_value": "今值", "forecast_value": "预测值",
                                   "previous_value": "前值"}, inplace=True)
        # 生成文件
        data_frame.to_csv(file_path, index=False)
        log.info("导出数据成功 FilePath:【{}】",relative_path)
        return str(relative_path)
    except Exception as e:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except NameError:
            pass
        log.exception(e)
        raise BusinessException(code=500, msg="导出数据失败")


def get_next_month(year, month):
    """计算下一月的年、月"""
    if month == 12:
        return year + 1, 1
    else:
        return year, month + 1


@deprecated("清洗日期方法 暂时弃用")
def clean_date(china_macro_data: pd.DataFrame):
    # 先清洗日期
    for i in range(len(china_macro_data)):
        current_date = china_macro_data.loc[i, "日期"]
        # 仅对发布日>20的行处理
        if current_date.day > 20:
            # 计算目标：下个月对应的年和月
            target_year, target_month = get_next_month(current_date.year, current_date.month)
            # 判断下一行是否存在且日期是否为目标月份
            if i == len(china_macro_data) - 1:
                # 当前行是最后一条记录，则必然没有下月数据，修改日期为目标月份
                china_macro_data.loc[i, "日期"] = current_date.replace(year=target_year, month=target_month)
            else:
                next_date = china_macro_data.loc[i + 1, "日期"]
                # 若下一行的日期不等于目标月份，则把当前行的日期更换为目标月份（保留其它信息）
                if not (next_date.year == target_year and next_date.month == target_month):
                    china_macro_data.loc[i, "日期"] = current_date.replace(year=target_year, month=target_month)
    # 最后统一将所有日期的天数设为 1
    china_macro_data["日期"] = china_macro_data["日期"].apply(lambda d: d.replace(day=1))


async def main():
    """ 方法测试"""
    await database.connect()
    # await save_or_update_macro_data(types=DataTypeEnum.PMI)
    # res = await get_macro_data(types=DataTypeEnum.PMI)
    # print(res)
    await export_to_csv(types=DataTypeEnum.PMI)
    await database.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
