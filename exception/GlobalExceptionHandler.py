import traceback

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from config.LoguruConfig import log
from exception.BusinessException import BusinessException


# Function to handle all exceptions
async def handle_exception(request: Request, exc: Exception):
    log.info(str(exc))
    traceback.print_exc()
    if isinstance(exc, BusinessException):
        return JSONResponse(status_code=exc.code, content={"code": exc.code, "msg": exc.msg, "data": None})
    return JSONResponse(status_code=500, content={"code": 500, "msg": exc.__str__(), "data": None})


async def validation_exception_handler(request: Request, exc: ValidationError):
    log.info(str(exc))
    return JSONResponse(
        status_code=500,
        content={"code": 500, "msg": exc.errors(), "data": None},
    )
