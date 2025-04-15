from datetime import date
from datetime import datetime
import ormar
import sqlalchemy
from ormar import Decimal

from entity.BaseMeta.BaseMeta import ormar_base_config


class StockData(ormar.Model):
    ormar_config = ormar_base_config.copy(tablename="stock_data")

    # 必须字段
    id: int = ormar.Integer(primary_key=True, column_type=sqlalchemy.BigInteger, autoincrement=False)
    indicator_id: int = ormar.Integer(nullable=False, column_type=sqlalchemy.BigInteger,
                                      description="逻辑上关联 indicator 表的 id")

    # 时间信息
    report_date: date = ormar.Date(nullable=False, description="数据对应的日期，如 '1996-02-01'")

    # 其他指标字段（你可以根据需要增删字段或调整精度）
    volume: int = ormar.Integer(nullable=True, description="成交量")
    open_price: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="开盘价")
    high_price: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="最高价")
    low_price: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="最低价")
    close_price: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="收盘价")
    change_amount: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="涨跌额")
    change_rate: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="涨跌幅")
    turnover_rate: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="换手率")
    turnover_amount: Decimal = ormar.Decimal(max_digits=15, decimal_places=2, nullable=True, description="成交额")
    pe_ratio: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="市盈率")
    pb_ratio: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="市净率")
    ps_ratio: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="市销率")
    pc_ratio: Decimal = ormar.Decimal(max_digits=10, decimal_places=2, nullable=True, description="市现率")
    market_value: Decimal = ormar.Decimal(max_digits=18, decimal_places=2, nullable=True, description="市值")
    money_flow: Decimal = ormar.Decimal(max_digits=15, decimal_places=2, nullable=True, description="资金流向")

    # 自动记录创建和更新时间
    created_at: datetime = ormar.DateTime(default=datetime.now, nullable=True, description="创建时间")
    updated_at: datetime = ormar.DateTime(default=datetime.now, onupdate=datetime.now, nullable=True,
                                          description="最后更新时间")

    def __str__(self) -> str:
        # 判断 report_date 是否为 None，如果不 None，则格式化为 'YYYY-MM-DD' 格式，否则返回 "None"
        report_date_str = self.report_date.strftime('%Y-%m-%d') if self.report_date is not None else "None"
        return f"StockData(id={self.id}, indicator_id={self.indicator_id}, report_date={report_date_str})"

    def __repr__(self) -> str:
        # 通常 __repr__ 可以调用 __str__ 方法
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StockData):
            return False
        return (self.id == other.id and
                self.indicator_id == other.indicator_id and
                self.report_date == other.report_date)

    def __hash__(self) -> int:
        return hash((self.id, self.indicator_id, self.report_date))