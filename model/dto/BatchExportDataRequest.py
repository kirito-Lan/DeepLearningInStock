from pydantic import BaseModel


class BatchExportDataRequest(BaseModel):
    export_type: str
    start_date: str
    end_date: str