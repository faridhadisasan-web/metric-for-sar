import logging
from rich.logging import RichHandler


def setup_logger(name: str = "rriq", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a rich logger for the specified name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()
