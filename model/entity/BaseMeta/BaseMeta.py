import logging
import os

import databases
import sqlalchemy
import ormar
import json

from config.LoguruConfig import project_root, log
from exception.BusinessException import BusinessException


def get_url() -> str:
    """获取数据库连接地址
    """
    host = os.getenv("DB_HOST", None)
    port = os.getenv("DB_PORT", None)
    user = os.getenv("DB_USER", None)
    password = os.getenv("DB_PASSWORD", None)
    database_name = os.getenv("DB_NAME", None)
    if host is None or port is None or user is None or password is None or database_name is None:
        try:
            path = project_root / "resources" / "properties.json"
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
                return data.get("MysqlUrl")
        except FileNotFoundError as e:
            log.exception(e)
            raise BusinessException(code=500, msg="数据库配置不存在，请检查")
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database_name}"


DATABASE_URL = get_url()

# 创建数据库连接和元数据对象
database: databases.Database = databases.Database(DATABASE_URL)
metadata: sqlalchemy.MetaData = sqlalchemy.MetaData()

# 创建统一的配置对象
ormar_base_config = ormar.OrmarConfig(
    database=database,
    metadata=metadata
)
