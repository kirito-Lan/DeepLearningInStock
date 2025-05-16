import asyncio
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.background import BackgroundScheduler

from config.LoguruConfig import log, project_root
from model.entity.MacroData import MacroData
from model.entity.StockData import StockData
from routes import CommonRoute

# 定时任务
scheduler = AsyncIOScheduler()


async def clean_archive_file():
    """
    异步函数：清理生成的 csv 和 excel 文件。
    Description:
        清理生成的 csv 和 excel 文件，包括位于 project_root/resources/csvFiles 目录下的所有 csv 文件和
        位于 project_root/resources/excelFiles 目录下的所有 excel 文件。

        1. 使用 os.listdir() 获取指定目录下的所有文件名。
        2. 检查文件名是否以 .csv 或 .xlsx 结尾。
        3. 使用 os.remove() 删除符合条件的文件。

    """
    log.info("开始定时清理生成的 csv excel 文件")
    csv_file_path = project_root / "resources" / "csvFiles"
    excel_file_path = project_root / "resources" / "excelFiles"
    for file in os.listdir(csv_file_path):
        if file.endswith(".csv"):
            os.remove(os.path.join(csv_file_path, file))
    for file in os.listdir(excel_file_path):
        if file.endswith(".xlsx"):
            os.remove(os.path.join(excel_file_path, file))


async def clean_db_bad_data():
    """
    清理 Stock Macro_data 库中的异常数据
    偶发的 indicator_id 插入后异常为1的数据
    :return: None
    """
    log.info("开始定时清理 Stock和Macro_data 库中的异常数据")
    try:
        await StockData.objects.filter(indicator_id=1).delete()
        await MacroData.filter(indicator_id=1).delete()
    except Exception as e:
        log.exception(f"定时清理 Stock和Macro_data 库中的异常数据失败: {e}")


async def update_stock_data_schedule():
    """
    异步函数：更新股票数据。
    """
    log.info("定时更新股票数据")
    try:
        res = await CommonRoute.batch_update_data("stock")
        log.info(f"定时更新股票数据成功: {res}")
    except Exception as e:
        log.exception(f"更新股票数据失败: {e}")
        pass


async def update_macro_data_schedule():
    """
    异步函数：更新宏观数据。
    """
    log.info("定时更新宏观数据")
    try:
        res = await CommonRoute.batch_update_data("macro")
        log.info(f"定时更新宏观数据成功: {res}")
    except Exception as e:
        log.exception(f"更新宏观数据失败: {e}")
        pass


# 每天凌晨 0 点执行一次清理任务
scheduler.add_job(clean_archive_file, trigger='cron', hour=0)
# 每天凌晨 1 点执行一次清理任务
scheduler.add_job(clean_db_bad_data, trigger='cron', hour=1)
# 每天18点执行一次更新任务
scheduler.add_job(update_stock_data_schedule, trigger='cron', hour=18)
# 每个月的20号执行一次更新任务
scheduler.add_job(update_macro_data_schedule, trigger='cron', day=10)
