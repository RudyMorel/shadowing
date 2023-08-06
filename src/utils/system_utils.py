from typing import *
import os
from time import time


def make_dir_if_not_there(dir_path):
    """Check if directory exists, creates it if not"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class TimeTracker:
    """ TimeTracker to track the time of different repeated steps. Example usage:
    ```
    tt.start("dataloading")
    for x in dataset_name():
        tt.start("processing")
        f(x)
        tt.start("dataloading")
    times = tt.stop()
    ```
    (Adapted from Florentin Guth)
    """
    def __init__(self):
        self.category = None  # Current category
        self.t = None  # Time at which we entered the current category
        self.times: Dict[str, float] = {}  # category -> sum of durations

    def _update(self):
        """ Private method, which should not be called from user code. """
        if self.category is not None:
            self.times[self.category] = self.times.get(self.category, 0) + time() - self.t

    def start(self, category: str) -> None:
        """ Switch to a new category. """
        self._update()
        self.category = category
        self.t = time()

    def stop(self) -> Dict[str, float]:
        """ Returns a dictionary of category -> durations (in seconds). """
        self._update()
        self.category = None
        self.t = None
        return {**self.times, 'TOTAL': sum(self.times.values())}

