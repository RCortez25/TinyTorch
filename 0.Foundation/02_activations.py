#| default_exp core.activations
#| export

import numpy as np
from typing import Optional

# Import from TinyTorch package (previous modules must be completed and exported)
from tinytorch.core.tensor import Tensor

# Constants for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for floating-point comparisons in tests

# Export only activation classes
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']