from pydantic import BaseModel, Field


class GetPredictRequest(BaseModel):
    stock_code: str = Field(default=None, description="股票代码")
    start_date: str = Field(default=None, description="开始日期")
    end_date: str = Field(default=None, description="结束日期")
