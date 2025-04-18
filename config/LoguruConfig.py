# logger_config.py
from loguru import logger
from pathlib import Path

# 构造日志文件的绝对路径（跨平台）
project_root = Path(__file__).parent.parent
log_file_path = project_root / "resources" /"logs"/"app.log"

logger.add(
    str(log_file_path),
    rotation="20 MB",
    retention="7 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level}| {file}:{line} | {function} | {message}"
)

log = logger

if __name__ == '__main__':
    log.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
