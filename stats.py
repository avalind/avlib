#!/usr/bin/env python
import numpy as np


def bootstrap(dataset, number_of_samples, statistic, alpha):
    """
        General method for calculating bootstrapped confidence
        intervals from a sample. Inspired by
        https://gist.github.com/atmb4u/6a4e5c0a032486a9b50f
    """
    n = len(dataset)
    idx = np.random(0, n, (number_of_samples, n))
    samples = dataset[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha / 2.0) * number_of_samples)],
            stat[int((1 - alpha / 2.0) * number_of_samples)])
