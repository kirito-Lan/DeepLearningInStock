from datetime import datetime


import ormar

from model.entity.BaseMeta.BaseMeta import ormar_base_config



class MacroData(ormar.Model):
    # 定义组合唯一约束：同一指标在同一日期只有一条记录
    ormar_config = ormar_base_config.copy(tablename="macro_data")

    id: int = ormar.BigInteger(primary_key=True, autoincrement=False)
    # 使用 ForeignKey 来建立与 Indicator 的关联
    indicator_id: int = ormar.BigInteger(nullable=False)
    report_date: datetime.date = ormar.Date()  # 数据对应的日期，如 '1996-02-01'
    current_value: float = ormar.Float(nullable=True)  # 今值
    forecast_value: float = ormar.Float(nullable=True)  # 预测值
    previous_value: float = ormar.Float(nullable=True)  # 前值
    created_at: datetime = ormar.DateTime(default=lambda: datetime.now())
    updated_at: datetime = ormar.DateTime(default=lambda: datetime.now(), onupdate=lambda: datetime.now())

    def __str__(self):
        created_at_str = self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else 'None'
        updated_at_str = self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else 'None'

        return (f"MacroData(id={self.id}, indicator_id={self.indicator_id}, report_date={self.report_date}, "
                f"current_value={self.current_value}, forecast_value={self.forecast_value}, "
                f"previous_value={self.previous_value}, created_at={created_at_str}, "
                f"updated_at={updated_at_str})")
