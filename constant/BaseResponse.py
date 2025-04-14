from typing import TypeVar, Optional, Type, Generic
from pydantic import BaseModel, ConfigDict, json

from constant.ErrorCode import ErrorCode

T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    __code: int
    __msg: str
    __data: Optional[T] = None

    model_config = ConfigDict(extra='allow')

    def __init__(self, code: int, msg: str, data: Optional[T] = None):
        super().__init__(code=code, msg=msg, data=data)
        self.__code = code
        self.__msg = msg
        self.__data = data

    @classmethod
    def success(cls: Type['BaseResponse'], data: T) -> 'BaseResponse':
        return cls(code=200, msg='success', data=data)

    @classmethod
    def success_with_code(cls: Type['BaseResponse'], code: int, msg: str, data: T) -> 'BaseResponse':
        return cls(code=code, msg=msg, data=data)

    @classmethod
    def fail(cls: Type['BaseResponse'], errcode: ErrorCode) -> 'BaseResponse':
        return cls(code=errcode.code, msg=errcode.message, data=None)

    def get_code(self) -> int:
        return self.__code

    def get_msg(self) -> str:
        return self.__msg

    def get_data(self) -> Optional[T]:
        return self.__data


# 示例用法
if __name__ == "__main__":
    response = BaseResponse.success(data="success")
    print(response.get_code())  # 输出: 200
    print(response.get_msg())  # 输出: success
    print(response.get_data())  # 输出: success
