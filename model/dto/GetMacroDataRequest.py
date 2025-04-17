from pydantic import BaseModel


class GetMacroDataRequest(BaseModel):
    types: str = None
    start_date: str = None
    end_date: str = None