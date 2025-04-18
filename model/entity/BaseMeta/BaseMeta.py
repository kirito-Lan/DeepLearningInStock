import logging

import databases
import sqlalchemy
import ormar
import json

from config.LoguruConfig import project_root, log
from exception.BusinessException import BusinessException


def get_url() -> str:
    """获取数据库连接地址"""
    try:
        path = project_root / "resources" / "properties.json"
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data.get("MysqlUrl")
    except FileNotFoundError as e:
        log.exception(e)
        raise BusinessException(code=500, msg="数据库配置文件不存在，请检查")


DATABASE_URL = get_url()

# 创建数据库连接和元数据对象
database: databases.Database = databases.Database(DATABASE_URL)
metadata: sqlalchemy.MetaData = sqlalchemy.MetaData()

# 创建统一的配置对象
ormar_base_config = ormar.OrmarConfig(
    database=database,
    metadata=metadata
)



