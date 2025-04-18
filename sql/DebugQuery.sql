

-- DeBug 代码
select *
from macro_data
where indicator_id = 1910667833528946688
  and report_date >= DATE_SUB(CURDATE(), INTERVAL 10 YEAR);

select *
from macro_data
where indicator_id not in (select id from indicator where code in ('CPI', 'PPI'));


select *
from macro_data
where indicator_id = 1910991963423903744
  and report_date between '2020-01-01' and '2023-01-01';


select count(1) num,max(trade_date) latestday,min(trade_date) oldest_date
from stock_data
where indicator_id=1912437306833375232;

select count(1) from stock_data where indicator_id=1912437306833375232;


delete from stock_data where indicator_id=1;

truncate table stock_data;