import time
import warnings

from enum import Enum


warnings.filterwarnings("ignore")


# support variables
_time = 0
_called = False
results_path = ""
log_path = "{}/log_{}.txt"


class Reason(Enum):
    INFO_TRAINING = 2
    SETUP_TRAINING = 3
    LIGHT_INFO_TRAINING = 4
    WARNING = 5
    OUTPUT_TRAINING = 6
    OTHER = 7
    NONE = 8


def get_time_in_millis():
    return int(round(time.time() * 1000))


def clprint(text, reason=Reason.NONE, loggable=False):
    global _called
    global _time
    if not _called:
        _called = True
        _time = get_time_in_millis()

    if reason == Reason.INFO_TRAINING:
        code_color = "\033[94m"

    elif reason == Reason.SETUP_TRAINING:
        code_color = "\033[32m"

    elif reason == Reason.LIGHT_INFO_TRAINING:
        code_color = "\033[92m"

    elif reason == Reason.WARNING:
        code_color = "\033[91m"

    elif reason == Reason.OUTPUT_TRAINING:
        code_color = "\033[95m"

    elif reason == Reason.OTHER:
        code_color = "\033[95m"

    else:
        code_color = "\033[0m"

    if loggable:
        path = log_path.format(results_path, _time)
        file = open(path, "a+")
        file.write(text + "\n")
        file.close()

    print(code_color + str(text) + "\033[0m")
