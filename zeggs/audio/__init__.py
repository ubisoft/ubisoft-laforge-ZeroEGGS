import logging


def get_logger_from_arg(logger=None):
    """Return `logger` if it is a Logger, else this package's logger."""
    return logger if isinstance(logger, logging.Logger) else logging.getLogger(__name__)
