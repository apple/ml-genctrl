# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

"""
This is a file containing some simple model functions.
"""

import numpy as np

# Import mock models for testing
from genctrl.factory.mock_models import MOCK_FACTORY


# Random walk model (not input dependent)
def random_walk(x: float, u: float):
    return x + np.random.random()


# Random walk with drift model (input dependent)
def random_walk_drift(x: float, u: float):
    return x + np.random.random() + u


FACTORY = {
    "random_walk": random_walk,
    "random_walk_drift": random_walk_drift,
    **MOCK_FACTORY,  # Add mock models
}
