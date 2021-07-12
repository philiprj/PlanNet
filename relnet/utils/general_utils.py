import time
from datetime import datetime
import numpy as np

date_format = "%Y-%m-%d-%H-%M-%S"


def find_latest_file(files_dir):
    """
    Finds thr latest file edited file in the directory
    :param files_dir: Directory where files are stored
    :return: Path to the last modified file
    """
    filenames = [f.name for f in list(files_dir.glob("*.pickle"))]
    latest_file_date = datetime.utcfromtimestamp(0)
    latest_file = None
    for f in filenames:
        datestring = "".join(f.split(".")[:-1])
        file_date = datetime.strptime(datestring, date_format)
        if file_date > latest_file_date:
            latest_file = f
            latest_file_date = file_date
    if latest_file is not None:
        return files_dir / latest_file
    else:
        return None


def is_time_expired(time_started, time_allowed):
    """
    Checks if time elapsed in running program has extended beyond expected
    :param time_started: start time for program
    :param time_allowed: Max time allowed for running
    :return: Bool - true if time has expired
    """
    return get_current_time_millis() - time_started > time_allowed


def get_current_time_millis():
    # Gets the current time - times 1000
    return int(time.time() * 1000)
