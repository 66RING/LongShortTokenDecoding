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


def draw_line_char(data: List, title: Optional[str], save_path: Optional[str], show: bool = False, filter: bool = True):
    if filter is True:
        # filter out out of distribution
        pool = StablePool(max_len=100, threshold=1.2)
        pool.prefill(data)

        filtered_data = [x for x in data if pool.visit(x)]
        data = filtered_data

    # x axis range
    x_data = range(1, len(data) + 1)

    plt.figure()

    # maker o
    plt.plot(x_data, data, marker='o', markersize=3, linestyle='None')

    if title is not None:
        plt.title(title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)


