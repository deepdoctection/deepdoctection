# -*- coding: utf-8 -*-
# File: logger.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/logger.py>

The logger module itself has the common logging functions of Python's
`logging.Logger`.

Example:
    ```python
    from deepdoctection.utils.logger import logger

    logger.set_logger_dir("path/to/dir")
    logger.info("Something has happened")
    logger.warning("Attention!")
    logger.error("Error happened!")
    ```

Log levels can be set via the environment variable `LOG_LEVEL` (default: INFO).
`STD_OUT_VERBOSE` will print a verbose message to the terminal (default: False).
"""

from __future__ import annotations

import errno
import json
import logging
import logging.config
import os
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, no_type_check

from termcolor import colored

from .env_info import ENV_VARS_TRUE
from .types import PathLikeOrStr

__all__ = ["logger", "set_logger_dir", "auto_set_dir", "get_logger_dir"]


@dataclass
class LoggingRecord:
    """
    `LoggingRecord` to pass to the logger in order to distinguish from third party libraries.

    Note:
        `log_dict` will be added to the log record as a dict.

    Args:
        msg: The log message.
        log_dict: Optional dictionary to add to the log record.

    """

    msg: str
    log_dict: Optional[dict[Union[int, str], Any]] = field(default=None)

    def __post_init__(self) -> None:
        """log_dict will be added to the log record as a dict."""
        if self.log_dict is not None:
            self.log_dict["msg"] = self.msg

    def __str__(self) -> str:
        """Return the message."""
        return self.msg


class CustomFilter(logging.Filter):
    """A custom filter"""

    def filter(self, record: logging.LogRecord) -> bool:
        if os.environ["FILTER_THIRD_PARTY_LIB"] in ENV_VARS_TRUE:
            if not isinstance(record.msg, LoggingRecord):
                return False

        return True


class StreamFormatter(logging.Formatter):
    """A custom formatter to produce unified LogRecords"""

    @no_type_check
    def format(self, record: logging.LogRecord) -> str:
        date = colored("[%(asctime)s @%(filename)s:%(lineno)d]", "green")
        msg = colored("%(message)s", "white")

        if os.environ["STD_OUT_VERBOSE"] in ENV_VARS_TRUE:
            if isinstance(record.msg, LoggingRecord):
                verbose_info = f" Additional verbose infos: {repr(record.msg.log_dict)}"
                msg += verbose_info

        if record.levelno == logging.WARNING:
            fmt = f"{date}  {colored('WRN', 'magenta', attrs=['blink'])}  {msg}"
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:  # pylint: disable=R1714
            fmt = f"{date}  {colored('ERR', 'red', attrs=['blink', 'underline'])}  {msg}"
        elif record.levelno == logging.DEBUG:
            fmt = f"{date}  {colored('DBG', 'green', attrs=['blink'])}  {msg}"
        elif record.levelno == logging.INFO:
            fmt = f"{date}  {colored('INF', 'green')}  {msg}"
        else:
            fmt = f"{date} {msg}"
        self._style._fmt = fmt  # pylint: disable=W0212
        self._fmt = fmt
        return super().format(record)


class FileFormatter(logging.Formatter):
    """A custom formatter to produce a loggings in json format"""

    @no_type_check
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        log_dict = {
            "level_no": record.levelno,
            "level_name": record.levelname,
            "module_name": record.filename,
            "line_number": record.lineno,
            "time": datetime.now().strftime("%m%d-%H%M%S"),
            "message": message,
        }
        if isinstance(record.msg, LoggingRecord):
            if record.msg.log_dict:
                log_dict.update(record.msg.log_dict)
                log_dict.pop("msg")
        elif not os.environ["FILTER_THIRD_PARTY_LIB"] in ENV_VARS_TRUE:
            log_dict = {"message": record.msg}
        return json.dumps(log_dict)


_LOG_DIR = None


def _coerce_log_level(val: Any) -> Union[int, str]:
    """Normalize environment log level values.

    Accepts integer values (e.g., ``20``), numeric strings (``"20"``),
    or names case-insensitively (``"info"``, ``"Warn"``, ...). Returns
    either an integer level number or a valid uppercase level name
    accepted by the :mod:`logging` module.

    Args:
        val: The raw value from the environment variable ``LOG_LEVEL``.

    Returns:
        int | str: The corresponding logging level as an int or an
        uppercase string. Defaults to ``"INFO"`` if the input is invalid.
    """
    if isinstance(val, int):
        return val
    if val is None:
        return "INFO"
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    name = s.upper()
    if name == "WARN":
        name = "WARNING"
    if name in logging._nameToLevel:
        return name
    lvl = logging.getLevelName(name)
    return lvl if isinstance(lvl, int) else "INFO"


def _get_logger() -> logging.Logger:

    # resolve level from LOG_LEVEL only
    resolved_log_level = _coerce_log_level(os.environ["LOG_LEVEL"])

    config_dict: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"customfilter": {"()": lambda: CustomFilter()}},  # pylint: disable=W0108
        "formatters": {
            "streamformatter": {"()": lambda: StreamFormatter(datefmt="%m%d %H:%M.%S")},
        },
        "handlers": {
            "streamhandler": {
                "filters": ["customfilter"],
                "formatter": "streamformatter",
                "class": "logging.StreamHandler",
            }
        },
        "root": {
            "handlers": ["streamhandler"],
            "level": resolved_log_level,
            "propagate": os.environ["LOG_PROPAGATE"] in ENV_VARS_TRUE,
        },
    }

    logging.config.dictConfig(config_dict)
    _logger = logging.getLogger(__name__)
    return _logger


logger = _get_logger()

_LOGGING_METHOD = ["info", "warning", "error", "critical", "debug"]
for func in _LOGGING_METHOD:
    locals()[func] = getattr(logger, func)
    __all__.append(func)


_FILE_HANDLER = None


def _get_time_str() -> str:
    return datetime.now().strftime("%m%d-%H%M%S")


def _set_file(path: PathLikeOrStr) -> None:
    path = os.fspath(path)
    global _FILE_HANDLER  # pylint: disable=W0603
    if os.path.isfile(path):
        backup_name = path + "." + _get_time_str()
        shutil.move(path, backup_name)
        logger.info("Existing log file %s backuped to %s", path, backup_name)
    hdl = logging.FileHandler(filename=path, encoding="utf-8", mode="w")
    hdl.setFormatter(FileFormatter(datefmt="%m%d %H:%M:%S"))
    hdl.addFilter(CustomFilter())

    _FILE_HANDLER = hdl
    logger.addHandler(hdl)
    logger.info("Argv: %s ", sys.argv)


def set_logger_dir(dir_name: PathLikeOrStr, action: Optional[str] = None) -> None:
    """
    Set the directory for global logging.

    Args:
        dir_name: Log directory.
        action: An action of ["k", "d", "q"] to be performed when the directory exists. Will ask user by default.
            "d": Delete the directory. Note that the deletion may fail when the directory is used by tensorboard.
            "k": Keep the directory. This is useful when you resume from a previous training and want the directory to
                 look as if the training was not interrupted.
            Note that this option does not load old models or any other old states for you. It simply does nothing.

    Raises:
        OSError: If the directory exists and an invalid action is selected.
    """
    if isinstance(dir_name, Path):
        dir_name = dir_name.as_posix()
    dir_name = os.path.normpath(dir_name)
    global _LOG_DIR, _FILE_HANDLER  # pylint: disable=W0603
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    def dir_nonempty(directory: PathLikeOrStr) -> int:
        return os.path.isdir(directory) and len([x for x in os.listdir(directory) if x[0] != "."])

    if dir_nonempty(dir_name):
        if not action:
            logger.warning("Log directory %s exists! Use 'd' to delete it. ", dir_name)
            logger.warning(
                "If you're resuming from a previous run, you can choose to keep it. Press any other key to exit. "
            )
        while not action:
            action = input("Select Action: k (keep) / d (delete) / q (quit):").lower().strip()
        act = action
        if act == "b":
            backup_name = dir_name + _get_time_str()
            shutil.move(dir_name, backup_name)
            logger.info("Directory %s backuped to %s", dir_name, backup_name)
        elif act == "d":
            shutil.rmtree(dir_name, ignore_errors=True)
            if dir_nonempty(dir_name):
                shutil.rmtree(dir_name, ignore_errors=False)
        elif act == "n":
            dir_name = dir_name + _get_time_str()
            logger.info("Use a new log directory %s ", dir_name)
        elif act == "k":
            pass
        else:
            raise OSError(f"Directory {dir_name} exits!")
    _LOG_DIR = os.path.join(dir_name, "log.jsonl")
    try:
        os.makedirs(dir_name)
    except OSError as err:
        if err.errno != errno.EEXIST:
            raise err

    _set_file(_LOG_DIR)


def auto_set_dir(action: Optional[str] = None, name: Optional[str] = None) -> None:
    """
    Will set the log directory to './train_log/{script_name}:{name}'.
    `script_name` is the name of the main python file currently running.

    Args:
        action: An action of ["k", "d", "q"] to be performed (see also `set_logger_dir`).
        name: Optional suffix of file name.
    """

    mod = sys.modules["__main__"]
    basename = str(os.path.basename(mod.__file__))  # type: ignore  # pylint: disable=E1101
    auto_dir_name = os.path.join("train_log", basename[: basename.rfind(".")])
    if name:
        auto_dir_name += "_%s" % name if os.name == "nt" else ":%s" % name  # pylint: disable=C0209
    set_logger_dir(auto_dir_name, action=action)


def get_logger_dir() -> Optional[PathLikeOrStr]:
    """
    The logger directory, or `None` if not set.

    Returns:
        The directory used for general logging, tensorboard events, checkpoints, etc.
    """
    return _LOG_DIR


_logged_once_keys: set[str] = set()


def log_once(message: Union[str,LoggingRecord] , function: str = "info") -> None:
    """
    Log certain message only once. Calling this function more than once with
    the same message will result in no operation.

    Example:
        ```python
        log_once("This will only be logged once", "info")
        ```

    Args:
        message: Message to log.
        function: The name of the logger method. For example, "info", "warn", "error".
    """
    key = str(message)
    if key in _logged_once_keys:
        return
    _logged_once_keys.add(key)
    getattr(logger, function)(message)
