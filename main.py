import uvicorn
from fastapi import FastAPI
from pydantic import ValidationError

from config.LifeSpanConfig import lifespan
from constant.BaseResponse import BaseResponse
from controller import UserController
from exception.GlobalExceptionHandler import handle_exception, validation_exception_handler, value_error_handler
from routes import StockRoute

# 主启动类添加路由
app = FastAPI(lifespan=lifespan,)
app.include_router(UserController.router)
app.include_router(StockRoute.router)

# 注册全局异常处理器
app.exception_handler(ValueError)(value_error_handler)
app.exception_handler(ValidationError)(validation_exception_handler)
app.exception_handler(Exception)(handle_exception)


@app.get("/healthCheck",tags=["main"],response_model=BaseResponse[str])
def healthCheck()->BaseResponse[str]:
    return BaseResponse[str].success("ok")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)