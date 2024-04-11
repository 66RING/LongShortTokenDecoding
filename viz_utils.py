import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from typing import List, Optional

class StablePool:
    def __init__(self, max_len: int, threshold: float):
        self.pool = []
        self.max_len = max_len
        self.threshold = threshold

    def prefill(self, data: List):
        for i in range(self.max_len):
            self.pool.append(data[i])

    def visit(self, item) -> bool:
        # filter out out of distribution
        if item > self.threshold * np.mean(self.pool):
            return False

        if len(self.pool) < self.max_len:
            self.pool.append(item)
        else:
            self.pool.pop(0)
            self.pool.append(item)

        return True


def draw_line_char(y_data: List, x_data: Optional[List] = None, title: Optional[str] = None, save_path: Optional[str] = None, show: bool = False, filter: bool = True):
    if filter is True:
        # filter out out of distribution
        pool = StablePool(max_len=100, threshold=1.2)
        pool.prefill(y_data)

        filtered_data = [x for x in y_data if pool.visit(x)]
        y_data = filtered_data

    # x axis range
    if x_data is None:
        x_data = range(1, len(y_data) + 1)

    assert len(x_data) == len(y_data)

    plt.figure()

    # maker o
    plt.plot(x_data, y_data, marker='o', markersize=3, linestyle='None')

    if title is not None:
        plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

