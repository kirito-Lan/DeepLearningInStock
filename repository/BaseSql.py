

# indicatorBaseColum
indicatorBaseColum = "id,name,code,frequency,description,created_at,updated_at"

# 插叙当前indicate对象是否存在
isIndicateExist= "select count(*) as num from indicator where name=:name"

#获取indicate 对象的id
getIndicateId= "select id from indicator where name=:name"


#获取indicate 对象的id
getIndicateIdByCode= "select id from indicator where code=:code"

#统计指定的指标下的数据条数
countMacroData= "select count(*) as count  from macro_data where indicator_id=:indicator_id"


# macro_dataBaseColum
macro_dataBaseColum = "id,indicator_id,report_date,current_value,forecast_value,previous_value,created_at,updated_at"


""" 过滤获取limit年前的数据"""
getLimitYearData = "select " + macro_dataBaseColum + (" from macro_data where indicator_id=:indicator_id"
                                                      " and report_date>=DATE_SUB(CURDATE(),INTERVAL :limit YEAR)")



if __name__ == '__main__':
    print(getLimitYearData)