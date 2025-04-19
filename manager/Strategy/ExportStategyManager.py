import os

from config.LoguruConfig import log, project_root
from exception.BusinessException import BusinessException
from manager.Strategy.MacroExportStrategy import MacroExportStrategy
from manager.Strategy.StockExportStrategy import StockExportStrategy
import pandas as pd


async def batch_export_to_excel_with_strategy(export_type:str,start_date: str, end_date: str):
    # 根据传入的类型选择策略
    if export_type == "stock":
        strategy = StockExportStrategy()
    elif export_type == "macro":
        strategy = MacroExportStrategy()
    else:
        raise BusinessException(code=400, msg=f"不支持的导出类型: {export_type}")
    log.info(f"开始导入【{export_type}】数据到Excel")
    # 生成路径
    file_path = project_root / "resources" / "excelFiles" / f"batch{export_type}_{start_date}_{end_date}.xlsx"
    # 转化成相对路径返回
    relative_path = file_path.relative_to(project_root)
    try:
        # 如果文件已存在，则直接返回路径
        if os.path.exists(file_path):
            log.info("路径文件已存在，无需创建:【{}】", relative_path)
            return str(relative_path)

        # 确保目标目录存在，否则创建它
        file_path.parent.mkdir(parents=False, exist_ok=True)
        # 生成数据
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:

            for _enum in strategy.get_enum():

                data_frame = await strategy.get_data(_enum, start_date, end_date)
                if data_frame.empty:
                    log.info("【{}】数据为空，无需导出", _enum.get_name())
                    continue
                # 列名转中文
                data_frame = await strategy.convert_columns(data_frame)
                data_frame.to_excel(writer, sheet_name=_enum.get_name(), index=False)
        return str(relative_path)
    except Exception as e:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except NameError:
            pass
        log.exception(e)
        raise BusinessException(code=500, msg="导出数据失败")