# 创建数据库
CREATE DATABASE IF NOT EXISTS macro_data_China DEFAULT CHARSET = utf8mb4;

use macro_data_China;
drop table if exists indicator;

CREATE TABLE indicator
(
    id          BIGINT PRIMARY KEY,
    name        VARCHAR(50) NOT NULL,          -- 指标名称，如“中国CPI月率报告”
    code        VARCHAR(20) DEFAULT NULL,      -- 可选字段，可存放如“CPI”之类的简写
    description TEXT        DEFAULT NULL,      -- 其他说明信息
    frequency   VARCHAR(20) DEFAULT 'monthly', -- 指标频率，如'monthly' 或 'yearly'
    created_at  datetime    DEFAULT now(),
    updated_at  datetime    DEFAULT now() ON UPDATE now(),
    INDEX idx_name (name)
) ENGINE = InnoDB;

drop table if exists macro_data;

CREATE TABLE macro_data
(
    id             BIGINT PRIMARY KEY,
    indicator_id   bigint NOT NULL,             -- 逻辑上关联 indicator 表的 id（业务中自行维护关联，暂不添加外键约束）
    report_date    DATE   NOT NULL,             -- 数据对应的日期，如 '1996-02-01'
    current_value  DECIMAL(10, 2) DEFAULT NULL, -- 今值
    forecast_value DECIMAL(10, 2) DEFAULT NULL, -- 预测值
    previous_value DECIMAL(10, 2) DEFAULT NULL, -- 前值
    created_at     datetime       DEFAULT now(),
    updated_at     datetime       DEFAULT now() ON UPDATE now(),
    INDEX idx_indicator_id (indicator_id)       -- 加速根据指标查找
) ENGINE = InnoDB;

truncate table macro_data;


select *
from macro_data
where indicator_id = 1910667833528946688
  and report_date >= DATE_SUB(CURDATE(), INTERVAL 10 YEAR)

select *
from macro_data
where indicator_id not in (select id from indicator where code in ("CPI", "PPI"))


delete
from macro_data
where indicator_id=1;