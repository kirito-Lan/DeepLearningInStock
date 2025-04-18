from enum import Enum


class MacroDataEnum(Enum):
    CPI = ("0", "CPI")
    PPI = ("1", "PPI")
    PMI = ("2", "PMI")

    def __init__(self, code: str, display_name: str):
        self._code = code
        self._display_name = display_name

    @property
    def code(self) -> str:
        return self._code

    @property
    def display_name(self) -> str:
        return self._display_name

    @classmethod
    def get_name_by_code(cls, code: str) -> str | None:
        for member in cls:
            if member.code == code:
                return member.display_name
        return None

    @classmethod
    def get_code_by_name(cls, name: str) -> str | None:
        for member in cls:
            if member.display_name == name:
                return member.code
        return None

    @classmethod
    def get_enum_by_name(cls, name: str):
        for member in cls:
            if member.display_name == name:
                return member
        return None


class PERIOD:
    MONTHLY = "monthly"
    YEARLY = "yearly"
    DAILY = "daily"
    WEEKLY = "weekly"
