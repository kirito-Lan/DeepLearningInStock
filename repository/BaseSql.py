indicatorBaseColum: str = "id,name,code,frequency,description,created_at,updated_at"
"""indicatorBaseColum"""

isIndicateExist: str = "select count(*) as num from indicator where name=:name"
"""查询当前indicate对象是否存在"""

getIndicateId: str = "select id from indicator where name=:name"
"""通过name获取indicate 对象的id"""

getIndicateIdByCode: str = "select id from indicator where code=:code"
"""通过code获取indicate对象的id"""

countMacroData = "select count(*) as count  from macro_data where indicator_id=:indicator_id"
"""统计指定的指标下的数据条数"""

macro_dataBaseColum: str = "id,indicator_id,report_date,current_value,forecast_value,previous_value,created_at,updated_at"
"""macro_dataBaseColum"""

getLimitYearData: str = "select " + macro_dataBaseColum + (" from macro_data where indicator_id=:indicator_id"
                                                           " and report_date>=DATE_SUB(CURDATE(),INTERVAL :limit YEAR)")
""" 过滤获取limit年前的数据"""

if __name__ == '__main__':
    print(getLimitYearData)
