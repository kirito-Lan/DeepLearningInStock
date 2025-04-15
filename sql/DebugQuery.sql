

-- DeBug 代码
select *
from macro_data
where indicator_id = 1910667833528946688
  and report_date >= DATE_SUB(CURDATE(), INTERVAL 10 YEAR);

select *
from macro_data
where indicator_id not in (select id from indicator where code in ('CPI', 'PPI'));