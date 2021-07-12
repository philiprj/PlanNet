import contextlib
import platform
import multiprocess
import logging
import sys
import xxhash
import numpy as np
import os
from logging.handlers import RotatingFileHandler
from logging import StreamHandler

date_format = "%Y-%m-%d-%H-%M-%S"
hash_instance = xxhash.xxh64()


def get_hash(obj_list):
    # Adds object in list to 64 hash instance
    for obj in obj_list:
        hash_instance.update(obj)
    # Convert hash integer, reset and return
    result = hash_instance.intdigest()
    hash_instance.reset()
    return result


def get_multiprocessing_context():
    # Check OS and applies appropriate MultiProcess approach
    if platform.system() != 'Windows':
        ctx = multiprocess.get_context("forkserver")
    else:
        ctx = multiprocess.get_context()
    return ctx


def get_device_placement():
    # Returns either CPU or GPU depending on available options
    return os.getenv("RELNET_DEVICE_PLACEMENT", "CPU")


@contextlib.contextmanager
def local_seed(seed):
    # Not sure what this does - may not be important
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_logger_instance(filename):
    # Helper function for logging results
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)

    has_stdout = False
    has_file = False
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler):
            has_file = True
        if isinstance(handler, StreamHandler):
            has_stdout = True

    if not has_stdout:
        sh = StreamHandler(sys.stdout)
        sh.addFilter(HostnameFilter())
        root_logger.addHandler(sh)

    if filename is not None and not has_file:
        fh = RotatingFileHandler(filename,
                                 maxBytes=32*1024*1024,
                                 backupCount=10)
        fh.addFilter(HostnameFilter())
        formatter = logging.Formatter(fmt='%(hostname)s %(asctime)s - PID%(process)d %(message)s', datefmt=date_format)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)

    return root_logger


class HostnameFilter(logging.Filter):
    # Custom class to keep track of the host performing task
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True
