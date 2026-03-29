import logging
import sys


def get_logger(name: str = "rag") -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	handler = logging.StreamHandler(sys.stdout)
	fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
	handler.setFormatter(fmt)

	logger.setLevel(logging.INFO)
	logger.addHandler(handler)
	logger.propagate = False
	return logger
