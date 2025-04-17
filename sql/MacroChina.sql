# 创建数据库
CREATE DATABASE IF NOT EXISTS macro_data_China DEFAULT CHARSET = utf8mb4;

use macro_data_China;
drop table if exists indicator;
-- 指标表
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

create index idx_code on indicator(code);

drop table if exists macro_data;
-- 数据表(存储宏观数据)
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

-- truncate table macro_data;
-- 数据表(存储股票数据)
drop table if exists stock_data;
CREATE TABLE stock_data (
    id BIGINT PRIMARY KEY,
    indicator_id BIGINT NOT NULL,  -- 逻辑上关联 indicator 表的 id
    trade_date DATE NOT NULL COMMENT '交易日',
    open_price DECIMAL(15,3) DEFAULT NULL COMMENT '开盘价',
    close_price DECIMAL(15,3) DEFAULT NULL COMMENT '收盘价',
    high_price DECIMAL(15,3) DEFAULT NULL COMMENT '最高价',
    low_price DECIMAL(15,3) DEFAULT NULL COMMENT '最低价',
    volume INT DEFAULT NULL COMMENT '成交量（单位: 手）',
    turnover_amount DECIMAL(20,3) DEFAULT NULL COMMENT '成交额（单位: 元）',
    amplitude DECIMAL(10,3) DEFAULT NULL COMMENT '振幅（单位: %）',
    change_rate DECIMAL(10,3) DEFAULT NULL COMMENT '涨跌幅（单位: %）',
    change_amount DECIMAL(10,3) DEFAULT NULL COMMENT '涨跌额（单位: 元）',
    turnover_rate DECIMAL(10,3) DEFAULT NULL COMMENT '换手率（单位: %）',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_indicator_id (indicator_id),     -- 加速根据指标查找
    index idx_trade_date (trade_date)
) ENGINE=InnoDB;





