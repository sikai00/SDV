import logging
import sys


# Configure and manages loggers, handlers and formatters.
class LogsManager:
    FORMATTER = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    LOG_FILE_NAME = r"quality_collector.log"
    CURRENT_LOG_LEVEL = logging.INFO
    CURRENT_HANDLER_LEVEL = logging.INFO

    stdout_handler = None
    file_handler = None

    # Create a logger of the input string name.
    @classmethod
    def get_logger(cls, name):
        logger = logging.getLogger(name)
        logger.setLevel(cls.CURRENT_LOG_LEVEL)

        cls._add_file_handler(logger)
        cls._add_stdout_handler(logger)

        return logger

    # Add stdout handlers to the input logger. If handler is not created yet, create it.
    @classmethod
    def _add_stdout_handler(cls, logger):
        if cls.stdout_handler is None:
            cls.stdout_handler = cls._create_stdout_handler()
        logger.addHandler(cls.stdout_handler)

    # Add file handlers to the input logger. If handler is not created yet, create it.
    @classmethod
    def _add_file_handler(cls, logger):
        if cls.file_handler is None:
            cls.file_handler = cls._create_file_handler(cls.LOG_FILE_NAME)
        logger.addHandler(cls.file_handler)

    # Create handler to output to stdout
    @classmethod
    def _create_stdout_handler(cls):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(cls._get_formatter())
        handler.setLevel(cls.CURRENT_HANDLER_LEVEL)
        return handler

    # Create handler to output to log file
    @classmethod
    def _create_file_handler(cls, log_file_dir):
        handler = logging.FileHandler(log_file_dir, mode='w')
        handler.setFormatter(cls._get_formatter())
        handler.setLevel(cls.CURRENT_HANDLER_LEVEL)
        return handler

    # Return the singleton standard formatter
    @classmethod
    def _get_formatter(cls):
        return cls.FORMATTER
