from datetime import datetime

from exception.BusinessException import BusinessException


def format_date(start_date: str, end_date: str) -> [str, str]:
    """
    格式化时间参数，
    :param start_date:
    :param end_date:
    :return: "%Y-%m-%d"
    """
    if start_date is None:
        start_date = "2000-01-01"
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = re_format(start_date)
    end_date = re_format(end_date)
    return start_date, end_date


def re_format(date_str: str) -> str:
    """
    尝试将日期字符串按两种不同的格式解析，并统一为 YYYY-MM-DD 格式返回。
    如果都无法解析，则抛出 ValueError。
    """
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise BusinessException(code=500, msg="日期格式 {date_str} 无法识别，请提供 YYYY-MM-DD 或 YYYYMMDD 格式的日期")


__all__ = ["format_date"]
