import logging
import os
from logging.handlers import RotatingFileHandler

# --- 基本配置 ---
LOG_DIR = './logs'  # 日志文件存放目录
LOG_FILENAME = 'app.log'  # 日志文件名
MAX_BYTES_EACH = 0  # 不控制日志文件的最大大小
BACKUP_COUNT = 10  # 最多保留 10 个日志文件

# --- 创建日志目录 ---
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, LOG_FILENAME)

def setup_logger(log_dir: str):
    """
    配置并返回一个全局唯一的logger实例
    """
    # 1. 获取一个logger实例
    # 使用一个固定的名字，确保在项目中任何地方获取的都是同一个logger实例
    logger = logging.getLogger('log')
    
    # 2. 设置日志级别（全局最低级别）
    # 只有高于或等于这个级别的日志才会被处理
    logger.setLevel(logging.DEBUG)

    # 3. 防止重复添加handler
    # 如果logger已经有handlers，说明已经配置过了，直接返回
    if logger.handlers:
        return logger

    # 4. 创建一个格式化器(Formatter)
    # 定义日志的输出格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        # %(asctime)s: 日志时间
        # %(name)s: logger名称
        # %(levelname)s: 日志级别
        # %(message)s: 日志内容
    )

    # 5. 创建并配置一个文件处理器(FileHandler)
    file_handler = RotatingFileHandler(
        filename=log_dir,
        maxBytes=MAX_BYTES_EACH,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  
    file_handler.setFormatter(formatter)

    # 6. 创建并配置一个控制台处理器(StreamHandler)
    # 让日志信息也能在控制台输出，方便调试
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  
    console_handler.setFormatter(formatter)

    # 7. 将处理器(Handler)添加到logger中
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# --- 创建全局可用的logger实例和print-like函数 ---

# 全局logger实例，项目中所有模块都可以导入并使用它
# logger = setup_logger()


