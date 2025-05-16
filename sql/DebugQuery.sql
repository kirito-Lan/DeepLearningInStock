

-- DeBug 代码
select *
from macro_data
where indicator_id = 1910667833528946688
  and report_date >= DATE_SUB(CURDATE(), INTERVAL 10 YEAR);

select *
from macro_data
where indicator_id  in (select id from indicator where code in  ('PMI'));


select *
from macro_data
where indicator_id = 1910991963423903744
  and report_date between '2020-01-01' and '2023-01-01';


select count(1) num,max(trade_date) latestday,min(trade_date) oldest_date
from stock_data
where indicator_id=(select id from indicator where code='000688');

select count(1) from stock_data where indicator_id=1912437306833375232;


delete from stock_data where indicator_id=1;

truncate table stock_data;

# 删除科创50股票 000688
# 删除中证1000 000852
delete from stock_data where indicator_id=(select id from indicator where code='000852');

delete from indicator where code='000852';


select * from macro_data where indicator_id=(select id from indicator where code='CPI') order by report_date desc;


delete from macro_data where indicator_id=(select id from indicator where code='CPI') and report_date = '2025-01-12';

