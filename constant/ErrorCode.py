from enum import Enum


class ErrorCode(Enum):
    # 请求成功
    SUCCESS = (200, "success")
    # 请求参数错误
    PARAMS_ERROR = (40000, "请求参数错误")
    # 未登录
    NOT_LOGIN_ERROR = (40100, "未登录")
    # 无权限
    NO_AUTH_ERROR = (40101, "无权限")
    # 请求数据不存在
    NOT_FOUND_ERROR = (40400, "请求数据不存在")
    # 禁止访问
    FORBIDDEN_ERROR = (40300, "禁止访问")
    # 系统内部异常
    SYSTEM_ERROR = (50000, "系统内部异常")
    # 操作失败
    OPERATION_ERROR = (50001, "操作失败")

    def __init__(self, code: int, message: str):
        self.__code = code
        self.__message = message

    @property
    def code(self) -> int:
        return self.__code

    @property
    def message(self) -> str:
        return self.__message


# 使用示例
if __name__ == "__main__":
    print(ErrorCode.SUCCESS)  # 输出: ErrorCode.SUCCESS
    print(ErrorCode.SUCCESS.code)  # 输出: 0
