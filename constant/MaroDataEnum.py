from enum import Enum

class DataTypeEnum(Enum):
    CPI = (0, "CPI")
    PPI = (1, "PPI")
    PMI = (2, "PMI")

    def __init__(self, code: int, name: str):
        self.__code = code
        self.__name = name

    @classmethod
    def get_name_by_code(cls, code: int):
        for member in cls:
            if member.value[0] == code:
                return member.value[1]
        return None

    @classmethod
    def get_code_by_name(cls, name: str):
        for member in cls:
            if member.value[1] == name:
                return member.value[0]
        return None

    # getcode
    def get_code(self)->int:
        return self.__code

    def get_name(self)->str:
        return self.__name


class PERIOD:
    MONTHLY = "monthly"
    YEARLY = "yearly"
    DAILY= "daily"
    WEEkly = "weekly"




# 示例用法
if __name__ == "__main__":
    print(PERIOD.MONTHLY)
    # 通过 code 找 name
    name = DataTypeEnum.get_name_by_code(1)
    print(name)  # 输出: CPI
    # 通过 name 找 code
    code = DataTypeEnum.get_code_by_name("PPI")
    print(code)  # 输出: 2
