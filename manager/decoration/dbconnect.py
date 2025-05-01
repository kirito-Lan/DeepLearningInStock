import functools

from model.entity.BaseMeta.BaseMeta import database


def db_connection(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 连接数据库
        await database.connect()
        try:
            result = await func(*args, **kwargs)
        finally:
            # 保证无论函数执行结果如何都会断开数据库
            await database.disconnect()
        return result
    return wrapper