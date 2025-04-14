from datetime import datetime
from typing import ClassVar

import ormar

from entity.BaseMeta.BaseMeta import ormar_base_config


class Indicator(ormar.Model):
    ormar_config = ormar_base_config.copy(tablename="indicator")

    id: int = ormar.BigInteger(primary_key=True, autoincrement=False)
    name: str = ormar.String(max_length=50, nullable=False)  # 如“中国CPI月率报告”
    code: str = ormar.String(max_length=20, nullable=True)  # 可选的简写
    description: str = ormar.Text(nullable=True)
    frequency: str = ormar.String(max_length=20, default="monthly")
    created_at: datetime = ormar.DateTime(default=lambda: datetime.now())
    updated_at: datetime = ormar.DateTime(default=lambda: datetime.now(), onupdate=lambda: datetime.now())

    def __str__(self):
        created_at_str = self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else 'None'
        updated_at_str = self.updated_at.strftime('%Y-%m-%d %H:%M:%S') if self.updated_at else 'None'

        return (
            f"Indicator(id={self.id}, name='{self.name}', code='{self.code}', "
            f"description='{self.description}', frequency='{self.frequency}', "
            f"created_at={created_at_str}, updated_at={updated_at_str})"
        )
