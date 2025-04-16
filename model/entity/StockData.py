from datetime import date
from datetime import datetime
from typing import Optional

import ormar
import sqlalchemy
from ormar import Decimal

from model.entity.BaseMeta.BaseMeta import ormar_base_config


class StockData(ormar.Model):
    ormar_config = ormar_base_config.copy(tablename="stock_data")

    id: int = ormar.BigInteger(primary_key=True, autoincrement=False)
    indicator_id: int = ormar.BigInteger(nullable=False,comment="指标ID")
    trade_date: date = ormar.Date(nullable=False,comment="交易日")
    open_price: Decimal = ormar.Decimal(max_digits=15,decimal_places=3,nullable=True,comment="开盘价")
    close_price: Decimal = ormar.Decimal(max_digits=15,decimal_places=3,nullable=True,comment="收盘价")
    high_price: Decimal = ormar.Decimal(max_digits=15,decimal_places=3,nullable=True,comment="最高价")
    low_price: Decimal = ormar.Decimal(max_digits=15,decimal_places=3,nullable=True,comment="最低价")
    volume: int = ormar.Integer(nullable=True,comment="成交量（单位: 手）")
    turnover_amount: Decimal = ormar.Decimal(max_digits=20,decimal_places=3,nullable=True,comment="成交额（单位: 元）")
    amplitude: Decimal = ormar.Decimal(max_digits=10,decimal_places=3,nullable=True,comment="振幅（单位: %）")
    change_rate: Decimal = ormar.Decimal(max_digits=10,decimal_places=3,nullable=True,comment="涨跌幅（单位: %）")
    change_amount: Decimal = ormar.Decimal(max_digits=10,decimal_places=3,nullable=True,comment="涨跌额（单位: 元）")
    turnover_rate: Decimal = ormar.Decimal(max_digits=10,decimal_places=3,nullable=True,comment="换手率（单位: %）")
    # 自动记录创建和更新时间
    created_at: datetime = ormar.DateTime(default=datetime.now, nullable=True, description="创建时间")
    updated_at: datetime = ormar.DateTime(default=datetime.now, onupdate=datetime.now, nullable=True,
                                          description="最后更新时间")

    def __str__(self) -> str:
        # 判断 trade_date 是否为 None，如果不 None，则格式化为 'YYYY-MM-DD' 格式，否则返回 "None"
        trade_date_str = self.trade_date.strftime('%Y-%m-%d') if self.trade_date is not None else "None"
        return f"StockData(id={self.id}, indicator_id={self.indicator_id}, trade_date={trade_date_str})"

    def __repr__(self) -> str:
        # 通常 __repr__ 可以调用 __str__ 方法
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StockData):
            return False
        return (self.id == other.id and
                self.indicator_id == other.indicator_id and
                self.trade_date == other.trade_date)

    def __hash__(self) -> int:
        return hash((self.id, self.indicator_id, self.trade_date))
