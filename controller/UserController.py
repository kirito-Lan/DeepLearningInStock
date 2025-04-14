from types import NoneType

from fastapi import APIRouter


from Utils.SnowFlake import Snowflake
from config.LoguruConfig import log
from constant.BaseResponse import BaseResponse
from entity.Indicator import Indicator
from entity.User import User
from exception.BusinessException import BusinessException

router = APIRouter(prefix="/user",tags=["user"])


@router.get("/")
async def userController():
    return {"message": "userController"}


@router.post("/get")
async def getUser(user: User):
    print(user)
    return BaseResponse[list[str]].success(data=["1111111","2222222"])

@router.post("/indicator",response_model=BaseResponse[NoneType])
async def saveIndicator(indicator: Indicator):
    indicator.id = Snowflake().get_id()
    raise BusinessException(400,"test")
    log.info(indicator)
    await indicator.save()
    return BaseResponse[NoneType].success(None)
