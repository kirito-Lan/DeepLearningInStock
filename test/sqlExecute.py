import aiounittest
from dao import BaseSql
from entity.BaseMeta.BaseMeta import database

class MyTestCase(aiounittest.AsyncTestCase):
    async def test_sql(self):
        await database.connect()  # 连接数据库
        query = BaseSql.isIndicateExist
        values = {"name": "中国CPI月率报告"}
        count = await database.fetch_all(query=query, values=values)
        # 打印结果
        for row in count:
            print(row['num'])  # 访问 'num' 字段
        await database.disconnect()  # 断开数据库连接

if __name__ == '__main__':
    aiounittest.main()
