from enum import Enum


class ExponentEnum(Enum):
    """股票指数代码枚举类"""
    # 上证股票指数 SSE (Shanghai Stock Exchange)
    SZI = ("上证指数", "000001")
    # 深证成指  SZSE (Shenzhen Stock Exchange)
    SZCZ = ("深证成指", "399001")
    # 创业报指数 Venture News Index
    CYB = ("创业报指数", "399006")
    # 沪深300
    HS300 = ("沪深300", "000300")
    # 中证A50
    ZZA50 = ("中证A50", "930050")
    # 中证500
    ZZ500 = ("中证500", "000905")
    # 中证1000
    ZZ1000 = ("中证1000", "000852")
    # 科创50
    KC50 = ("科创50", "000688")

    def __init__(self, name: str,code: str):
        self.__code = code
        self.__name = name
    @classmethod
    def get_code_by_name(cls, name: str):
        for member in cls:
            if member.name == name:
                return member.value
        return None
    @classmethod
    def get_name_by_code(cls, code: str):
        for member in cls:
            if member.value == code:
                return member.name
        return None