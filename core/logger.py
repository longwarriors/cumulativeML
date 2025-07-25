"""
统一的日志管理功能，支持控制台和文件日志输出。
"""
import logging, os, sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from .base import SingletonMeta


class LoggerManager(metaclass=SingletonMeta):
    """
    日志管理器：提供统一的日志记录功能，支持控制台和文件输出。
    """

    def __init__(self):
        self._loggers = {}
        self._log_dir = Path(__file__).parent / 'logs'
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._default_level = logging.INFO
        self._default_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    def get_logger(self, name: str,
                   level: Optional[int] = None,
                   log_file: Optional[str] = None,
                   console: bool = True) -> logging.Logger:
        """
        获取或创建一个日志记录器。
        :param name: 日志记录器的名称。
        :param level: 日志级别，默认为 INFO。
        :param log_file: 日志文件路径，如果为 None 则不输出到文件。
        :param console: 是否输出到控制台，默认为 True。
        :return: 一个配置好的日志记录器实例。
        """
        if name in self._loggers:
            return self._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level or self._default_level)
        logger.propagate = False  # 避免日志重复
        if logger.handlers:  # 清除现有处理器
            logger.handlers.clear()
        formatter = logging.Formatter(self._default_format)

        # 控制台处理器
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # 文件处理器
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            log_file = self._log_dir / f'{name}-{timestamp}.log'
        else:
            log_file = Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 缓存日志记录器
        self._loggers[name] = logger
        return logger

    def setup_root_logger(self,
                          level: Optional[int] = None,
                          log_file: Optional[str] = None,
                          console: bool = True) -> logging.Logger:
        """设置根日志记录器，通常用于全局配置。"""
        return self.get_logger('root', level, log_file, console)

    def setup_pipline_logger(self, pipline_name: str, level: Optional[int] = None) -> logging.Logger:
        """设置管道日志记录器，通常用于特定数据处理管道的日志记录。"""
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_file = self._log_dir / f'{pipline_name}-{timestamp}.log'
        return self.get_logger(f'pipline-{pipline_name}', level, log_file)

    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """获取所有已创建的日志记录器。"""
        return self._loggers.copy()

    def set_default_level(self, level: int):
        """设置默认日志级别。"""
        self._default_level = level
        for logger in self._loggers.values():
            logger.setLevel(level)

    def set_default_format(self, format: str):
        """设置默认日志格式。"""
        self._default_format = format
        formatter = logging.Formatter(format)
        for logger in self._loggers.values():
            for handler in logger.handlers:
                handler.setFormatter(formatter)

    def set_log_dir(self, log_dir: str):
        """设置日志文件存储目录。"""
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        for logger in self._loggers.values():
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            for handler in file_handlers:
                old_file_path = Path(handler.baseFilename)
                old_file_name = old_file_path.name
                new_file_path = self._log_dir / old_file_name
                formatter = handler.formatter
                level = handler.level
                handler.close()
                logger.removeHandler(handler)
                new_handler = logging.FileHandler(new_file_path, encoding='utf-8')
                new_handler.setFormatter(formatter)
                new_handler.setLevel(level)
                logger.addHandler(new_handler)

    def set_logger_level(self, name: str, level: int):
        """设置指定日志记录器的日志级别。"""
        if name in self._loggers:
            logger = self._loggers[name]
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)

    def set_logger_format(self, name: str, format: str):
        """设置指定日志记录器的日志格式。"""
        if name in self._loggers:
            logger = self._loggers[name]
            formatter = logging.Formatter(format)
            for handler in logger.handlers:
                handler.setFormatter(formatter)

    def disable_logger(self, name: str):
        """禁用指定的日志记录器。"""
        if name in self._loggers:
            logger = self._loggers[name]
            logger.handlers.clear()
            logger.addHandler(logging.NullHandler())
            logger.propagate = False

    def enable_logger(self, name: str):
        """启用指定的日志记录器。"""
        if name in self._loggers:
            logger = self._loggers[name]
            logger.propagate = True
            for handler in logger.handlers:
                if isinstance(handler, logging.NullHandler):
                    logger.handlers.remove(handler)
            self.get_logger(name)

    def add_handler_to_logger(self, name: str, handler: logging.Handler):
        """向指定日志记录器添加处理器。"""
        if name in self._loggers:
            logger = self._loggers[name]
            handler.setFormatter(logging.Formatter(self._default_format))
            logger.addHandler(handler)

    def remove_handler_from_logger(self, name: str, handler_type: type):
        """从指定日志记录器移除处理器。"""
        if name in self._loggers:
            logger = self._loggers[name]
            handlers_to_remove = [h for h in logger.handlers if isinstance(h, handler_type)]
            for handler in handlers_to_remove:
                logger.removeHandler(handler)

    def set_handler_level(self, name: str, handler_type: type, level: int):
        """设置指定日志记录器中处理器的日志级别。"""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers:
                if isinstance(handler, handler_type):
                    handler.setLevel(level)

    def set_console_level(self, name: str, level: int):
        """设置指定日志记录器的控制台处理器的日志级别。"""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(level)

    def set_file_level(self, name: str, level: int):
        """设置指定日志记录器的文件处理器的日志级别。"""
        if name in self._loggers:
            logger = self._loggers[name]
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(level)