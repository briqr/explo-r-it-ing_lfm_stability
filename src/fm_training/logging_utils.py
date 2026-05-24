from __future__ import annotations

import logging
import os


def create_logger(log_dir: str | None, rank: int = 0) -> logging.Logger:
    logger = logging.getLogger("lfm_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"), mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
