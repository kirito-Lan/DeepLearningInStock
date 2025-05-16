from pydantic import BaseModel, Field


class GetPredictRequest(BaseModel):
    stock_code: str = Field(default=None, description="股票代码")
    start_date: str = Field(default=None, description="开始日期")
    end_date: str = Field(default=None, description="结束日期")
    Epoches: int = Field(default=150, description="训练轮次")
    reg: float = Field(default=0.001, description="正则化")
    dropout: float = Field(default=0.3, description="dropout")
