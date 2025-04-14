from typing import ClassVar

import databases
import sqlalchemy
import ormar


# 配置数据库连接
DATABASE_URL = "mysql+aiomysql://root:131645@server:3306/macro_data_China"

# 创建数据库连接和元数据对象
database: databases.Database = databases.Database(DATABASE_URL)
metadata: sqlalchemy.MetaData = sqlalchemy.MetaData()

# 创建统一的配置对象
ormar_base_config = ormar.OrmarConfig(
    database=database,
    metadata=metadata
)



