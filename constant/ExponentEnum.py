from enum import Enum


class ExponentEnum(Enum):
    """股票指数代码枚举类"""
    # 上证指数
    SZI = ("上证指数", "000001")
    # 深证成指
    SZCZ = ("深证成指", "399001")
    # 创业报指数
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

    def __init__(self, display_name: str, code: str):
        self._display_name = display_name  # 用单下划线，避免与 Enum 内置的 .name 混淆
        self._code = code

    def get_code(self) -> str:
        return self._code

    def get_name(self) -> str:
        return self._display_name

    @classmethod
    def get_code_by_name(cls, name: str):
        for member in cls:
            if member.get_name() == name:  # 比较的是显示名称
                return member.get_code()
        return None

    @classmethod
    def get_name_by_code(cls, code: str):
        for member in cls:
            if member.get_code() == code:  # 比较的是代码
                return member.get_name()
        return None

    # 获取枚举对象
    @classmethod
    def get_enum_by_code(cls, stock_code):
        for member in cls:
            if member.get_code() == stock_code:  # 比较的是代码
                return member
        return None


# 测试示例
if __name__ == "__main__":
    print(ExponentEnum.get_code_by_name("上证指数"))  # 应该输出 "000001"
    print(ExponentEnum.get_name_by_code("399001"))  # 应该输出 "深证成指"
